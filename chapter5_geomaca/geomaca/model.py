"""
GeoMACA Core Module

Geographic Multi-Agent Credit Assignment for spatially-structured environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class GeoMACAConfig:
    """Configuration for GeoMACA module."""
    # Architecture
    state_dim: int = 20
    action_dim: int = 5
    hidden_dim: int = 128
    num_zones: int = 9
    num_turns: int = 9
    num_agents: int = 3
    num_heads: int = 4
    
    # Feature flags
    use_temporal: bool = True
    use_geographic: bool = True
    use_agent: bool = True
    
    # Combination method: 'multiplicative', 'learned', 'factorized'
    combination: str = 'multiplicative'
    
    # Temperature for softmax (lower = sharper attention)
    temperature: float = 1.0
    
    # Training
    learning_rate: float = 1e-4
    entropy_weight: float = 0.1  # Prevent credit collapse
    consistency_weight: float = 0.05  # Developed zones should have higher credit

    # How the head-averaged attention weights p become the distributions that
    # enter the credit: 'renormalize' divides p by its sum (temperature 1);
    # 'softmax' applies a second softmax, alpha = softmax(p / temperature).
    # Chapter 5 uses 'softmax'.
    attention_normalization: str = 'renormalize'


class ZoneEncoder(nn.Module):
    """Encodes per-zone state features."""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, zone_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            zone_states: [batch, Z, state_dim] or [batch, T, Z, state_dim]
        Returns:
            [batch, Z, hidden_dim] or [batch, T, Z, hidden_dim]
        """
        return self.net(zone_states)


class TemporalEncoder(nn.Module):
    """Encodes per-timestep features (state + all agent actions)."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, temporal_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_input: [batch, T, input_dim]
        Returns:
            [batch, T, hidden_dim]
        """
        return self.net(temporal_input)


class AgentEncoder(nn.Module):
    """Encodes per-agent action features."""
    
    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, agent_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_actions: [batch, A, action_dim]
        Returns:
            [batch, A, hidden_dim]
        """
        return self.net(agent_actions)


class RewardEncoder(nn.Module):
    """Encodes the final reward as a query vector."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reward: [batch, 1] or [batch]
        Returns:
            [batch, 1, hidden_dim] - query for attention
        """
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        encoded = self.net(reward)
        return encoded.unsqueeze(1)  # [batch, 1, hidden_dim]


class GeoMACA(nn.Module):
    """
    Geographic Multi-Agent Credit Assignment
    
    Decomposes end-of-episode rewards across:
    - Geographic zones (which location?)
    - Time steps (which turn?)
    - Agents (which government agent?)
    
    Novel contribution: First method to handle TRUE geographic spatial
    decomposition in multi-agent credit assignment, as opposed to
    "spatial" referring to agent coalition space (like STAS).
    """
    
    def __init__(self, config: GeoMACAConfig):
        super().__init__()
        
        self.config = config
        self.num_zones = config.num_zones
        self.num_turns = config.num_turns
        self.num_agents = config.num_agents
        self.hidden_dim = config.hidden_dim
        self.temperature = config.temperature
        if config.attention_normalization not in ('renormalize', 'softmax'):
            raise ValueError(
                f"attention_normalization must be 'renormalize' or 'softmax', "
                f"got {config.attention_normalization!r}"
            )
        # Kept on the module (like the temperature) so that it does not change
        # if a caller later replaces ``self.config``.
        self.attention_normalization = config.attention_normalization
        
        # Feature encoders
        self.zone_encoder = ZoneEncoder(config.state_dim, config.hidden_dim)
        
        # Temporal encoder input: state + all agent actions
        temporal_input_dim = config.state_dim + config.action_dim * config.num_agents
        self.temporal_encoder = TemporalEncoder(temporal_input_dim, config.hidden_dim)
        
        self.agent_encoder = AgentEncoder(config.action_dim, config.hidden_dim)
        self.reward_encoder = RewardEncoder(config.hidden_dim)
        
        # Attention mechanisms (using MultiheadAttention for efficiency)
        if config.use_geographic:
            self.geographic_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                batch_first=True,
                dropout=0.1
            )
        
        if config.use_temporal:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                batch_first=True,
                dropout=0.1
            )
        
        if config.use_agent:
            self.agent_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                batch_first=True,
                dropout=0.1
            )
        
        # For learned combination mode
        if config.combination == 'learned':
            combo_input_dim = config.num_turns + config.num_zones + config.num_agents
            combo_output_dim = config.num_turns * config.num_zones * config.num_agents
            self.combination_mlp = nn.Sequential(
                nn.Linear(combo_input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, combo_output_dim)
            )
        
        # For factorized combination mode
        if config.combination == 'factorized':
            self.temporal_conditioner = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.geo_conditioner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

    def _apply_temperature(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply temperature to already-normalized attention probabilities."""
        weights = weights.clamp_min(1e-8)
        if abs(self.temperature - 1.0) > 1e-8:
            weights = weights.pow(1.0 / self.temperature)
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    def _normalize_attention(self, weights: torch.Tensor) -> torch.Tensor:
        """Turn head-averaged attention weights into a distribution.

        'renormalize': ``_apply_temperature`` (division by the sum);
        'softmax': a second softmax over the weights, ``softmax(p / temperature)``.
        """
        if self.attention_normalization == 'softmax':
            return F.softmax(weights / self.temperature, dim=-1)
        return self._apply_temperature(weights)
    
    def forward(
        self,
        trajectory: Dict[str, torch.Tensor],
        final_reward: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute credit assignment across (time, zone, agent) dimensions.
        
        Args:
            trajectory: dict containing:
                - zone_states: [batch, T, Z, state_dim] - per-zone state at each timestep
                - agent_actions: [batch, T, A, action_dim] - actions of each agent per turn
                - global_state: [batch, T, state_dim] - global state at each timestep
                - zone_developed: [batch, T] - which zone was developed at each turn (optional)
            final_reward: [batch] or [batch, 1] - the terminal reward (e.g., -segregation)
        
        Returns:
            credit: [batch, T, Z, A] - credit assignment tensor (sums to 1)
            attention_weights: tuple of (alpha_temp, alpha_geo, alpha_agent)
        """
        device = final_reward.device
        batch_size = final_reward.shape[0] if final_reward.dim() > 0 else 1
        
        if final_reward.dim() == 0:
            final_reward = final_reward.unsqueeze(0)
        if final_reward.dim() == 1:
            final_reward = final_reward.unsqueeze(-1)
        
        # Encode reward as query
        reward_query = self.reward_encoder(final_reward)  # [batch, 1, hidden]
        
        # Initialize uniform attention weights as defaults
        alpha_temp = torch.ones(batch_size, self.num_turns, device=device) / self.num_turns
        alpha_geo = torch.ones(batch_size, self.num_zones, device=device) / self.num_zones
        alpha_agent = torch.ones(batch_size, self.num_agents, device=device) / self.num_agents
        
        # Geographic attention
        if self.config.use_geographic and 'zone_states' in trajectory:
            zone_states = trajectory['zone_states']  # [batch, T, Z, state_dim]
            # Aggregate across time to get per-zone features
            zone_features = zone_states.mean(dim=1)  # [batch, Z, state_dim]
            zone_encoded = self.zone_encoder(zone_features)  # [batch, Z, hidden]
            
            _, geo_attn_weights = self.geographic_attention(
                query=reward_query,
                key=zone_encoded,
                value=zone_encoded,
                need_weights=True,
                average_attn_weights=True
            )
            alpha_geo = geo_attn_weights.squeeze(1)  # [batch, Z]
            alpha_geo = self._normalize_attention(alpha_geo)
        
        # Temporal attention
        if self.config.use_temporal and 'global_state' in trajectory and 'agent_actions' in trajectory:
            global_state = trajectory['global_state']  # [batch, T, state_dim]
            agent_actions = trajectory['agent_actions']  # [batch, T, A, action_dim]
            
            # Flatten agent actions across agents
            agent_actions_flat = agent_actions.reshape(batch_size, self.num_turns, -1)
            temporal_input = torch.cat([global_state, agent_actions_flat], dim=-1)
            temporal_encoded = self.temporal_encoder(temporal_input)  # [batch, T, hidden]
            
            _, temp_attn_weights = self.temporal_attention(
                query=reward_query,
                key=temporal_encoded,
                value=temporal_encoded,
                need_weights=True,
                average_attn_weights=True
            )
            alpha_temp = temp_attn_weights.squeeze(1)  # [batch, T]
            alpha_temp = self._normalize_attention(alpha_temp)
        
        # Agent attention
        if self.config.use_agent and 'agent_actions' in trajectory:
            agent_actions = trajectory['agent_actions']  # [batch, T, A, action_dim]
            # Aggregate across time to get per-agent features
            agent_features = agent_actions.mean(dim=1)  # [batch, A, action_dim]
            agent_encoded = self.agent_encoder(agent_features)  # [batch, A, hidden]
            
            _, agent_attn_weights = self.agent_attention(
                query=reward_query,
                key=agent_encoded,
                value=agent_encoded,
                need_weights=True,
                average_attn_weights=True
            )
            alpha_agent = agent_attn_weights.squeeze(1)  # [batch, A]
            alpha_agent = self._normalize_attention(alpha_agent)
        
        # Combine credit using selected method
        credit = self._combine_credit(alpha_temp, alpha_geo, alpha_agent)
        
        return credit, (alpha_temp, alpha_geo, alpha_agent)
    
    def _combine_credit(
        self,
        alpha_temp: torch.Tensor,
        alpha_geo: torch.Tensor,
        alpha_agent: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine the three attention distributions into a single credit tensor.
        
        Args:
            alpha_temp: [batch, T]
            alpha_geo: [batch, Z]
            alpha_agent: [batch, A]
        
        Returns:
            credit: [batch, T, Z, A] - normalized credit tensor
        """
        if self.config.combination == 'multiplicative':
            # Outer product: credit[t,z,a] = alpha_temp[t] * alpha_geo[z] * alpha_agent[a]
            credit = torch.einsum('bt,bz,ba->btza', alpha_temp, alpha_geo, alpha_agent)
        
        elif self.config.combination == 'learned':
            # Concatenate and learn the combination
            batch_size = alpha_temp.shape[0]
            combined = torch.cat([alpha_temp, alpha_geo, alpha_agent], dim=-1)
            credit_flat = self.combination_mlp(combined)
            credit = credit_flat.reshape(
                batch_size, self.num_turns, self.num_zones, self.num_agents
            )
            credit = F.softmax(credit.reshape(batch_size, -1), dim=-1)
            credit = credit.reshape(
                batch_size, self.num_turns, self.num_zones, self.num_agents
            )
            return credit  # Already normalized
        
        else:  # Default to multiplicative
            credit = torch.einsum('bt,bz,ba->btza', alpha_temp, alpha_geo, alpha_agent)
        
        # Normalize to sum to 1
        credit = credit / (credit.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)
        
        return credit
    
    def compute_per_step_credit(
        self, credit: torch.Tensor, zone_developed: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-timestep, per-agent credit that preserves geographic information.

        Instead of marginalizing over time and geography (which cancels them out),
        extract the credit for the specific zone developed at each timestep.

        Args:
            credit: [B, T, Z, A] full credit tensor from forward()
            zone_developed: [B, T] index of zone developed at each timestep

        Returns:
            per_step_credit: [B, T, A] credit for each agent at each timestep,
                             incorporating geographic attention for the actual zone
        """
        B, T, Z, A = credit.shape
        # Gather the credit for the zone that was actually developed at each step
        # zone_developed: [B, T] -> [B, T, 1, 1] -> expand to [B, T, 1, A]
        zone_idx = zone_developed.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, A)
        per_step_credit = credit.gather(dim=2, index=zone_idx).squeeze(2)  # [B, T, A]

        # Normalize so credits sum to 1 across time and agents
        per_step_credit = per_step_credit / (
            per_step_credit.sum(dim=(1, 2), keepdim=True) + 1e-8
        )

        return per_step_credit

    def compute_step_agent_advantages(
        self,
        credit: torch.Tensor,
        zone_developed: torch.Tensor,
        final_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Convert gathered credit into per-timestep, per-agent advantages.

        Under uniform credit the scale matches an episode-level update: if
        every cell has credit ``1 / (T * A)``, each per-step advantage is
        ``final_reward / A``. Non-uniform temporal/geographic attention then
        changes which timestep log-probabilities receive more weight.

        Args:
            credit: [B, T, Z, A] full credit tensor from ``forward``.
            zone_developed: [B, T] zone index developed at each timestep.
            final_reward: [B] or [B, 1] terminal reward.

        Returns:
            [B, T, A] per-step advantages for policy updates.
        """
        per_step_credit = self.compute_per_step_credit(credit, zone_developed)
        if final_reward.dim() == 0:
            final_reward = final_reward.view(1, 1)
        elif final_reward.dim() == 1:
            final_reward = final_reward.unsqueeze(-1)

        num_turns = credit.shape[1]
        return per_step_credit * final_reward.unsqueeze(1) * num_turns

    def compute_agent_rewards(
        self,
        trajectory: Dict[str, torch.Tensor],
        final_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute credit-weighted rewards for each agent.
        
        Args:
            trajectory: Trajectory data
            final_reward: [batch] or [batch, 1] - terminal reward
        
        Returns:
            agent_rewards: [batch, num_agents] - credit-weighted reward per agent
        """
        credit, _ = self.forward(trajectory, final_reward)
        
        if 'zone_developed' in trajectory:
            # Preserve the actually visited geographic path before summing to
            # agent totals. The training loop uses the richer per-step variant.
            agent_credit = self.compute_per_step_credit(
                credit, trajectory['zone_developed']
            ).sum(dim=1)
        else:
            # Fallback for callers without zone IDs (e.g. synthetic trajectories).
            agent_credit = credit.sum(dim=(1, 2))  # [batch, A]
        
        # Ensure final_reward has correct shape
        if final_reward.dim() == 1:
            final_reward = final_reward.unsqueeze(-1)
        
        # Weight rewards by credit
        agent_rewards = agent_credit * final_reward  # [batch, A]
        
        return agent_rewards
    
    def get_credit_interpretation(
        self,
        trajectory: Dict[str, torch.Tensor],
        final_reward: torch.Tensor,
        zone_names: Optional[list] = None,
        agent_names: Optional[list] = None
    ) -> Dict:
        """
        Get human-readable interpretation of credit assignment.
        
        Returns a dict with:
        - top_zones: List of (zone_idx, credit) tuples
        - top_turns: List of (turn_idx, credit) tuples
        - agent_credit: Dict of agent_name -> credit
        - top_cells: List of (turn, zone, agent, credit) tuples
        """
        credit, (alpha_temp, alpha_geo, alpha_agent) = self.forward(trajectory, final_reward)
        
        if zone_names is None:
            zone_names = [f'Zone_{i+1}' for i in range(self.num_zones)]
        if agent_names is None:
            agent_names = ['Economy', 'Environment', 'Society']
        
        # Convert to numpy for easier processing
        alpha_temp_np = alpha_temp[0].detach().cpu().numpy()
        alpha_geo_np = alpha_geo[0].detach().cpu().numpy()
        alpha_agent_np = alpha_agent[0].detach().cpu().numpy()
        credit_np = credit[0].detach().cpu().numpy()
        
        # Top zones
        zone_ranking = sorted(
            enumerate(alpha_geo_np),
            key=lambda x: x[1],
            reverse=True
        )
        top_zones = [(zone_names[i], float(c)) for i, c in zone_ranking[:3]]
        
        # Top turns
        turn_ranking = sorted(
            enumerate(alpha_temp_np),
            key=lambda x: x[1],
            reverse=True
        )
        top_turns = [(f'Turn_{i+1}', float(c)) for i, c in turn_ranking[:3]]
        
        # Agent credit
        agent_credit = {
            agent_names[i]: float(alpha_agent_np[i])
            for i in range(self.num_agents)
        }
        
        # Top cells (flattened credit)
        credit_flat = credit_np.flatten()
        top_indices = credit_flat.argsort()[-5:][::-1]
        top_cells = []
        for idx in top_indices:
            t = idx // (self.num_zones * self.num_agents)
            z = (idx % (self.num_zones * self.num_agents)) // self.num_agents
            a = idx % self.num_agents
            top_cells.append((
                f'Turn_{t+1}',
                zone_names[z],
                agent_names[a],
                float(credit_flat[idx])
            ))
        
        return {
            'top_zones': top_zones,
            'top_turns': top_turns,
            'agent_credit': agent_credit,
            'top_cells': top_cells,
            'total_credit': float(credit_np.sum())
        }

