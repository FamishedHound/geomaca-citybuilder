"""
Urban Development MARL with GeoMACA Credit Assignment
======================================================

This training script integrates GeoMACA (Geographic Multi-Agent Credit Assignment)
with the existing MARL training system.

GeoMACA provides:
- Geographic credit assignment (which zone?)
- Temporal credit assignment (which turn?)
- Agent credit assignment (which government agent?)

This is our NOVEL CONTRIBUTION - first method to handle TRUE geographic spatial
decomposition in multi-agent credit assignment.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from ..citybuilder.urban_simulation import (
    UrbanSimulation, GameState, GovernmentConstraints, DeveloperDecision,
    LAND_USE_EMPTY, LAND_USE_POOR, LAND_USE_MEDIUM, LAND_USE_LUXURY, 
    LAND_USE_GREEN, LAND_USE_WOODLAND
)

# Import existing components from full system
from .train_full_system import (
    Config as BaseConfig, MAPPOGovernment, SpatialDeveloperCNN,
    RewardNormalizer, EarlyStopping,
    AgentCommunication, PillarIntrinsicRewards
)


class SimpleExperienceBuffer:
    """Simple experience buffer for GeoMACA training."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, experience: Dict):
        """Add an experience to the buffer."""
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)

# Import GeoMACA
from ..geomaca import GeoMACA, GeoMACAConfig, GeoMACALosses
from ..geomaca.losses import GeoMACALossComputer
from ..geomaca.visualization import visualize_credit_assignment


@dataclass
class GeoMACATrainingConfig(BaseConfig):
    """Extended config with GeoMACA parameters."""
    # GeoMACA architecture
    geomaca_hidden_dim: int = 128
    geomaca_num_heads: int = 4
    geomaca_temperature: float = 1.0
    geomaca_combination: str = 'multiplicative'  # or 'learned'
    
    # GeoMACA training
    geomaca_lr: float = 1e-4
    geomaca_entropy_weight: float = 0.1
    geomaca_consistency_weight: float = 0.05
    geomaca_causality_weight: float = 0.02
    geomaca_supervised_weight: float = 0.5
    
    # GeoMACA feature flags (for ablation)
    use_temporal: bool = True
    use_geographic: bool = True
    use_agent: bool = True
    credit_mode: str = 'aggregate'  # aggregate | multi_objective | multi_objective_blend | counterfactual
    multi_objective_weight: float = 0.5
    counterfactual_interval: int = 10
    counterfactual_weight: float = 0.5
    
    # TRAINING PARAMETERS (10k episodes)
    # Exploration - prevent policy collapse
    entropy_coef: float = 0.05           # Entropy bonus
    entropy_decay: float = 0.9999        # Slow decay to maintain exploration
    min_entropy_coef: float = 0.01       # Don't go too low
    
    # Reward shaping - encourage balanced development
    vacancy_penalty: float = -0.5        # Penalize empty housing
    population_bonus: float = 0.1        # Reward for attracting residents
    diversity_bonus: float = 0.2         # Reward housing mix diversity
    
    # GeoMACA warmup - train base policy first
    geomaca_warmup: int = 1000           # Start using GeoMACA after this many episodes
    credit_weight: float = 0.5           # Don't let credit dominate early
    
    # Early stopping - off in this dataclass default; the command line
    # (config_from_args) turns it on unless --no-early-stop is given
    early_stop_enabled: bool = False
    early_stop_patience: int = 500       # If enabled, very patient
    skip_validation: bool = False        # Experiment sweeps evaluate separately

    # Method switches. The defaults select the variants this script uses when
    # no switch is given; CHAPTER5_METHOD (command-line flag --chapter5) is the
    # configuration of Chapter 5.
    # terminal_reward: 'pillar_aggregate' (average over the three pillars,
    #   normalised online) or 'neg_segregation' (R_term = -D, used as it is).
    terminal_reward: str = 'pillar_aggregate'
    # government_advantage: 'per_step' (each step gets its own share of the
    #   credit) or 'total_credit' (agent i's total credit C_i = R_term * alpha_agent_i
    #   weights all of its K actions).
    government_advantage: str = 'per_step'
    # zone_keys: 'zone_features' (per-zone state features) or 'city_observation'
    #   (the state recorded for every zone is the city-level observation o_k).
    zone_keys: str = 'zone_features'
    # attention_normalization: see geomaca.model.GeoMACAConfig.
    attention_normalization: str = 'renormalize'
    # agent_entropy_coef: weight of the agent-marginal entropy term, as a
    #   multiple of geomaca_entropy_weight (0 leaves the term out).
    agent_entropy_coef: float = 0.5


# The configuration of Chapter 5: terminal reward -D, each government agent's
# total credit as the advantage of all of its actions, the city-level
# observation as the state of every zone, a second softmax on the
# head-averaged attention weights, and no agent-marginal entropy term.
CHAPTER5_METHOD = {
    'terminal_reward': 'neg_segregation',
    'government_advantage': 'total_credit',
    'zone_keys': 'city_observation',
    'attention_normalization': 'softmax',
    'agent_entropy_coef': 0.0,
}

METHOD_SWITCH_CHOICES = {
    'terminal_reward': ('pillar_aggregate', 'neg_segregation'),
    'government_advantage': ('per_step', 'total_credit'),
    'zone_keys': ('zone_features', 'city_observation'),
    'attention_normalization': ('renormalize', 'softmax'),
}


def check_method_switches(config) -> None:
    """Reject unknown switch values and combinations the trainer does not support."""
    for name, choices in METHOD_SWITCH_CHOICES.items():
        value = getattr(config, name)
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    if config.agent_entropy_coef < 0:
        raise ValueError("agent_entropy_coef must be >= 0")
    credit_mode = getattr(config, 'credit_mode', 'aggregate')
    if credit_mode != 'aggregate' and (
        config.terminal_reward != 'pillar_aggregate'
        or config.government_advantage != 'per_step'
    ):
        raise ValueError(
            "terminal_reward and government_advantage can only be changed with "
            "credit_mode='aggregate'"
        )


class TrajectoryCollector:
    """Collects trajectory data for GeoMACA credit assignment."""
    
    def __init__(self, num_zones: int = 9, state_dim: int = 20, action_dim: int = 5, num_agents: int = 3):
        self.num_zones = num_zones
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.reset()
    
    def reset(self):
        """Reset trajectory for new episode."""
        self.zone_states = []
        self.agent_actions = []
        self.global_states = []
        self.zone_developed = []
        self.developer_rewards = []
    
    def extract_zone_features(self, env) -> np.ndarray:
        """Extract real per-zone features from the environment.

        Returns:
            zone_features: [num_zones, zone_feature_dim] where zone_feature_dim=12
        """
        zone_features = np.zeros((self.num_zones, 12), dtype=np.float32)
        for i, zone in enumerate(env.zones):
            zone_area = max(1.0, float(zone.size * zone.size))
            row = i // 3
            col = i % 3
            zone_features[i] = np.array([
                zone.poor_housing / zone_area,
                zone.medium_housing / zone_area,
                zone.luxury_housing / zone_area,
                zone.green_space / zone_area,
                zone.woodland / zone_area,
                zone.density,
                zone.environmental_score,
                zone.pollution_level,
                col / 2.0,
                row / 2.0,
                1.0 if zone.developed else 0.0,
                zone.total_housing / zone_area,
            ], dtype=np.float32)
        return zone_features

    def record_turn(
        self,
        zone_id: int,
        global_state: np.ndarray,
        zone_state: np.ndarray,
        agent_actions: np.ndarray,
        developer_reward: float,
        env=None
    ):
        """Record data for one turn."""
        self.zone_developed.append(zone_id)
        self.global_states.append(global_state)
        self.developer_rewards.append(developer_reward)
        self.agent_actions.append(agent_actions)

        if env is not None:
            zone_features = self.extract_zone_features(env)
            context_dim = max(0, self.state_dim - zone_features.shape[1])
            global_context = np.tile(
                global_state[:context_dim], (self.num_zones, 1)
            )
            zone_states_this_turn = np.concatenate(
                [zone_features, global_context], axis=1
            )
            if zone_states_this_turn.shape[1] > self.state_dim:
                zone_states_this_turn = zone_states_this_turn[:, :self.state_dim]
        else:
            zone_states_this_turn = np.tile(
                global_state[:self.state_dim], (self.num_zones, 1)
            )
        self.zone_states.append(zone_states_this_turn)
    
    def get_trajectory(self, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """Convert collected data to tensor trajectory dict."""
        num_turns = len(self.global_states)
        
        if num_turns == 0:
            return None
        
        # Pad to expected number of turns if needed
        expected_turns = 9
        while len(self.global_states) < expected_turns:
            self.global_states.append(self.global_states[-1])
            self.agent_actions.append(self.agent_actions[-1])
            self.zone_states.append(self.zone_states[-1])
            self.zone_developed.append(self.zone_developed[-1])
            self.developer_rewards.append(0.0)
        
        # Convert to tensors
        trajectory = {
            'zone_states': torch.tensor(
                np.array(self.zone_states), dtype=torch.float32, device=device
            ).unsqueeze(0),  # [1, T, Z, state_dim]
            
            'agent_actions': torch.tensor(
                np.array(self.agent_actions), dtype=torch.float32, device=device
            ).unsqueeze(0),  # [1, T, A, action_dim]
            
            'global_state': torch.tensor(
                np.array(self.global_states)[:, :self.state_dim], dtype=torch.float32, device=device
            ).unsqueeze(0),  # [1, T, state_dim]
            
            'zone_developed': torch.tensor(
                np.array(self.zone_developed), dtype=torch.long, device=device
            ).unsqueeze(0),  # [1, T]
            
            'developer_reward': torch.tensor(
                np.array(self.developer_rewards), dtype=torch.float32, device=device
            ).unsqueeze(0),  # [1, T]
        }
        
        return trajectory


class GeoMACATrainer:
    """Training system with GeoMACA credit assignment."""

    def __init__(self, config: GeoMACATrainingConfig, save_path: Optional[str] = None):
        check_method_switches(config)
        self.config = config
        self.device = config.device
        self._save_path_override = save_path
        
        print("=" * 80)
        print("GeoMACA TRAINING SYSTEM")
        print("=" * 80)
        print(f"Device: {self.device}")
        
        # Environment
        self.env = UrbanSimulation(map_size=config.map_size, zone_size=config.zone_size)
        
        # Agents (from existing system)
        print("Creating MAPPO Government...")
        self.government = MAPPOGovernment(config)
        
        print("Creating Spatial Developer CNN...")
        self.developer = SpatialDeveloperCNN(
            state_dim=config.state_dim,
            constraint_dim=config.gov_action_dim,
            num_land_types=config.num_land_types
        ).to(self.device)
        # STABILITY: Lower learning rate + weight decay to prevent explosion
        self.dev_optimizer = optim.Adam(
            self.developer.parameters(), 
            lr=config.dev_lr * 0.1,  # Reduce LR by 10x
            weight_decay=1e-5  # Add L2 regularization
        )
        
        # GeoMACA module
        # zone_state_dim == base_state_dim (=20), the state size the trained
        # GeoMACA checkpoints use. The TrajectoryCollector below ensures the
        # per-zone state has the matching shape (see record_turn).
        print("Creating GeoMACA module...")
        geomaca_config = GeoMACAConfig(
            state_dim=config.base_state_dim,
            action_dim=config.gov_action_dim,
            hidden_dim=config.geomaca_hidden_dim,
            num_zones=config.num_zones,
            num_turns=config.num_zones,  # One turn per zone
            num_agents=3,
            num_heads=config.geomaca_num_heads,
            use_temporal=config.use_temporal,
            use_geographic=config.use_geographic,
            use_agent=config.use_agent,
            combination=config.geomaca_combination,
            temperature=config.geomaca_temperature,
            attention_normalization=config.attention_normalization,
        )
        self.geomaca = GeoMACA(geomaca_config).to(self.device)
        self.geomaca_optimizer = optim.Adam(
            self.geomaca.parameters(), lr=config.geomaca_lr
        )
        
        # GeoMACA loss computer
        self.geomaca_losses = GeoMACALossComputer(
            entropy_weight=config.geomaca_entropy_weight,
            consistency_weight=config.geomaca_consistency_weight,
            causality_weight=config.geomaca_causality_weight,
            state_dim=config.base_state_dim,
            device=config.device,
            agent_entropy_coef=config.agent_entropy_coef,
        )
        
        # Trajectory collector
        self.trajectory_collector = TrajectoryCollector(
            num_zones=config.num_zones,
            state_dim=config.base_state_dim,
            action_dim=config.gov_action_dim,
            num_agents=3
        )
        
        # Other components
        self.pillar_rewards = PillarIntrinsicRewards()
        self.reward_normalizer = RewardNormalizer()
        self.pillar_reward_normalizers = [RewardNormalizer() for _ in range(3)]
        self.early_stopping = EarlyStopping(
            patience=config.early_stop_patience,
            min_delta=config.early_stop_min_delta
        )
        self.buffer = SimpleExperienceBuffer()
        
        # Metrics tracking
        self.metrics_history = []
        self.credit_history = []  # Track credit assignments
        
        # Create experiment directory
        if self._save_path_override is not None:
            self.exp_dir = Path(self._save_path_override)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.exp_dir = Path(f'experiments/geomaca/{timestamp}')
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Save config
        config_dict = {k: v for k, v in asdict(config).items()}
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Experiment directory: {self.exp_dir}")
    
    def get_state(self, zone_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state (base and enhanced)."""
        metrics = self.env.metrics
        
        # Base state (20 dims) - for government
        base_state = np.array([
            metrics.population / 5000,
            metrics.gdp / 1e9,
            1.0 - metrics.unemployment_rate,  # Employment rate
            metrics.avg_qol / 5,
            metrics.green_space_ratio,
            metrics.woodland_ratio,
            metrics.total_housing / 2000,  # Density proxy
            metrics.tax_revenue / 1e7,
            metrics.segregation_index,
            metrics.poor_housing / 1000,
            metrics.medium_housing / 1000,
            metrics.luxury_housing / 1000,
            zone_id / 9,
            metrics.housing_vacancy_rate,
            metrics.affordability_index,
            metrics.sustainability_score,
            metrics.reputation,
            metrics.service_level,
            metrics.infrastructure_quality,
            metrics.avg_satisfaction / 5
        ], dtype=np.float32)
        
        # Enhanced state (113 dims) - for developer
        # Add zone-specific info, global developed zones, neighbor awareness
        zone_id_encoded = np.zeros(3, dtype=np.float32)
        if zone_id < 9:
            zone_id_encoded[0] = zone_id / 9  # Zone ID normalized
            zone_id_encoded[1] = (zone_id % 3) / 3  # Column position
            zone_id_encoded[2] = (zone_id // 3) / 3  # Row position
        
        # Global developed zones info (9 zones * 5 features = 45)
        global_zones = np.zeros(45, dtype=np.float32)
        for z_id, zone in enumerate(self.env.zones):
            idx = z_id * 5
            global_zones[idx] = zone.poor_housing / 100
            global_zones[idx + 1] = zone.medium_housing / 100
            global_zones[idx + 2] = zone.luxury_housing / 100
            global_zones[idx + 3] = zone.green_space / 400
            global_zones[idx + 4] = zone.woodland / 400
        
        # Neighbor awareness (9 zones * 5 proximity = 45)
        neighbor_info = np.zeros(45, dtype=np.float32)
        for z_id in range(9):
            idx = z_id * 5
            zone = self.env.zones[z_id]
            neighbor_info[idx] = 1.0 if abs(z_id - zone_id) == 1 else 0.0  # Adjacent horizontally
            neighbor_info[idx + 1] = 1.0 if abs(z_id - zone_id) == 3 else 0.0  # Above/below
            neighbor_info[idx + 2] = zone.total_housing / 200  # Housing density
            neighbor_info[idx + 3] = zone.environmental_score  # Environmental quality
            neighbor_info[idx + 4] = 1.0 if zone.developed else 0.0  # Already developed
        
        enhanced_state = np.concatenate([
            base_state,
            zone_id_encoded,
            global_zones,
            neighbor_info
        ])
        
        return base_state, enhanced_state
    
    def run_episode(self) -> Dict:
        """Run a single episode with GeoMACA credit assignment."""
        # Reset environment and trajectory
        self.env = UrbanSimulation(map_size=self.config.map_size, zone_size=self.config.zone_size)
        self.trajectory_collector.reset()
        self.buffer.clear()
        
        episode_reward = 0
        episode_profit = 0
        prev_seg = 0.0
        
        # Track zone housing distributions for cross-zone segregation
        self.zone_distributions = []
        
        # Store agent actions for trajectory
        all_agent_actions = []
        
        for zone_id in range(self.config.num_zones):
            # Get states
            base_state, enhanced_state = self.get_state(zone_id)
            
            # Government acts (using base state)
            constraints, agent_actions_list, value = self.government.select_action(base_state)
            
            # Convert agent actions to numpy for trajectory
            # Each action in list is [1, action_dim] tensor
            agent_actions = np.zeros((3, self.config.gov_action_dim), dtype=np.float32)
            for i, act in enumerate(agent_actions_list):
                action_np = act.detach().cpu().numpy().flatten()
                # Pad to gov_action_dim if needed
                agent_actions[i, :len(action_np)] = action_np
            
            # Developer acts (using enhanced state)
            state_t = torch.tensor(enhanced_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            constraints_t = torch.tensor([
                constraints.affordable_housing_pct,
                constraints.green_space_pct,
                constraints.woodland_preservation,
                constraints.max_density,
                constraints.tax_rate
            ], dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                land_logits, value = self.developer(state_t, constraints_t)
                # STABILITY: Clamp logits to prevent numerical issues
                land_logits = torch.clamp(land_logits, -50.0, 50.0)
                probs = F.softmax(land_logits, dim=1)
                # Ensure valid probabilities for multinomial (no zeros, no NaNs)
                probs = probs.clamp(min=1e-6)
                probs = probs / probs.sum(dim=1, keepdim=True)  # Re-normalize
                # Handle any NaN values (fallback to uniform)
                if torch.isnan(probs).any():
                    probs = torch.ones_like(probs) / self.config.num_land_types
                # Sample from distribution
                land_use_grid = torch.multinomial(
                    probs.view(-1, self.config.num_land_types),
                    num_samples=1
                ).view(20, 20)
            
            # Create developer decision
            grid_np = land_use_grid.cpu().numpy()
            decision = DeveloperDecision()
            decision.land_use_grid = grid_np.astype(np.int32)
            
            # Execute in environment
            result = self.env.develop_zone(zone_id, constraints, decision)
            
            # Compute developer reward
            profit = result.get('cumulative_profit', 0)
            episode_profit += profit
            
            # Get ACTUAL household wealth class distribution per zone
            # This matches exactly how segregation is calculated in the environment
            self.zone_distributions = self.env.get_zone_household_distributions()
            
            metrics = self.env.metrics
            dev_reward = self._compute_developer_reward(result, metrics, prev_seg, self.zone_distributions)
            prev_seg = metrics.segregation_index
            
            # Record trajectory for GeoMACA
            self.trajectory_collector.record_turn(
                zone_id=zone_id,
                global_state=base_state,
                zone_state=base_state,  # Simplified
                agent_actions=agent_actions,
                developer_reward=dev_reward,
                env=self._zone_state_env()
            )
            
            episode_reward += dev_reward
            
            # Compute old log probs for PPO (simplified and robust)
            with torch.no_grad():
                old_log_probs = F.log_softmax(land_logits, dim=1)  # [1, 6, 20, 20]
                # Flatten for easier indexing
                old_log_probs_flat = old_log_probs.permute(0, 2, 3, 1).reshape(-1, self.config.num_land_types)  # [400, 6]
                land_use_flat = land_use_grid.view(-1)  # [400]
                # Get log prob of each selected action
                selected_log_probs = old_log_probs_flat[torch.arange(400, device=self.device), land_use_flat]  # [400]
                old_action_log_prob = selected_log_probs.sum().item()  # Sum over all 400 pixels
            
            # Store in buffer - INCLUDING the actual constraints and old log probs for PPO!
            self.buffer.add({
                'base_state': base_state,
                'enhanced_state': enhanced_state,
                'gov_actions': agent_actions,
                'land_grid': grid_np,
                'dev_reward': dev_reward,
                'zone_id': zone_id,
                'constraints': [
                    constraints.affordable_housing_pct,
                    constraints.green_space_pct,
                    constraints.woodland_preservation,
                    constraints.max_density,
                    constraints.tax_rate
                ],
                'old_log_prob': old_action_log_prob  # Store for PPO ratio
            })
        
        # Get final metrics
        final_metrics = self.env.metrics
        
        # Get trajectory for GeoMACA
        trajectory = self.trajectory_collector.get_trajectory(self.device)
        
        # Compute terminal aggregate for credit assignment. The raw score is
        # normalized online so the actor update receives a centered advantage
        # instead of blindly reinforcing every completed episode.
        if self.config.terminal_reward == 'neg_segregation':
            # R_term = -D, the negated final segregation index, used as it is.
            terminal_credit_reward = self._raw_terminal_reward(final_metrics)
            normalized_terminal_reward = terminal_credit_reward
        else:
            terminal_credit_reward = self._compute_terminal_credit_reward(final_metrics)
            self.reward_normalizer.update(terminal_credit_reward)
            normalized_terminal_reward = self.reward_normalizer.normalize(
                terminal_credit_reward
            )
        final_reward = torch.tensor(
            [normalized_terminal_reward],
            dtype=torch.float32, device=self.device
        )
        
        # Get current episode count
        current_episode = len(self.metrics_history)
        geomaca_warmup = getattr(self.config, 'geomaca_warmup', 1000)
        use_geomaca = current_episode >= geomaca_warmup
        agent_rewards = None
        
        # Compute GeoMACA credit assignment (ONLY AFTER WARMUP)
        if trajectory is not None and use_geomaca:
            credit_mode = getattr(self.config, 'credit_mode', 'aggregate')
            if credit_mode == 'multi_objective':
                (
                    agent_step_advantages,
                    alpha_temp,
                    alpha_geo,
                    alpha_agent,
                    geomaca_loss_info,
                    normalized_pillar_rewards,
                ) = self._compute_multi_objective_credit(trajectory, final_metrics)
            elif credit_mode == 'multi_objective_blend':
                credit, (alpha_temp, alpha_geo, alpha_agent) = self.geomaca(trajectory, final_reward)
                aggregate_advantages = self.geomaca.compute_step_agent_advantages(
                    credit, trajectory['zone_developed'], final_reward
                )
                aggregate_loss_info = self._update_geomaca(credit, trajectory, final_reward)
                (
                    pillar_advantages,
                    pillar_alpha_temp,
                    pillar_alpha_geo,
                    pillar_alpha_agent,
                    pillar_loss_info,
                    normalized_pillar_rewards,
                ) = self._compute_multi_objective_credit(trajectory, final_metrics)
                blend = float(np.clip(
                    getattr(self.config, 'multi_objective_weight', 0.5),
                    0.0,
                    1.0,
                ))
                agent_step_advantages = (
                    (1.0 - blend) * aggregate_advantages
                    + blend * pillar_advantages
                )
                alpha_temp = (1.0 - blend) * alpha_temp + blend * pillar_alpha_temp
                alpha_geo = (1.0 - blend) * alpha_geo + blend * pillar_alpha_geo
                alpha_agent = (1.0 - blend) * alpha_agent + blend * pillar_alpha_agent
                geomaca_loss_info = {}
                for key in set(aggregate_loss_info) | set(pillar_loss_info):
                    geomaca_loss_info[key] = (
                        (1.0 - blend) * float(aggregate_loss_info.get(key, 0.0))
                        + blend * float(pillar_loss_info.get(key, 0.0))
                    )
            else:
                credit, (alpha_temp, alpha_geo, alpha_agent) = self.geomaca(trajectory, final_reward)

                if self.config.government_advantage == 'total_credit':
                    # Each agent's total credit C_i = R_term * alpha_agent_i: the
                    # normalised credit tensor summed over steps and zones,
                    # times the terminal reward. It weights all K actions.
                    agent_rewards = credit.sum(dim=(1, 2)) * final_reward.view(-1, 1)
                    agent_step_advantages = None
                else:
                    # Preserve temporal/geographic credit in the actor update. Summing
                    # to [B, A] would make the multiplicative temporal/geographic terms
                    # cancel before they can weight the per-zone log-probabilities.
                    agent_step_advantages = self.geomaca.compute_step_agent_advantages(
                        credit, trajectory['zone_developed'], final_reward
                    )

                target_step_credit = None
                if credit_mode == 'counterfactual':
                    cf_result = self._compute_counterfactual_training_signal(
                        list(self.buffer.buffer),
                        terminal_credit_reward,
                        current_episode,
                    )
                    if cf_result is not None:
                        cf_advantages, target_step_credit = cf_result
                        blend = float(np.clip(
                            getattr(self.config, 'counterfactual_weight', 0.5),
                            0.0,
                            1.0,
                        ))
                        agent_step_advantages = (
                            (1.0 - blend) * agent_step_advantages
                            + blend * cf_advantages
                        )

                geomaca_loss_info = self._update_geomaca(
                    credit, trajectory, final_reward, target_step_credit
                )
                normalized_pillar_rewards = None

            credit_record = {
                'episode': current_episode,
                'alpha_temp': alpha_temp[0].detach().cpu().numpy(),
                'alpha_geo': alpha_geo[0].detach().cpu().numpy(),
                'alpha_agent': alpha_agent[0].detach().cpu().numpy(),
                'final_segregation': final_metrics.segregation_index,
                'terminal_credit_reward': terminal_credit_reward,
                'normalized_terminal_reward': normalized_terminal_reward,
                'credit_mode': credit_mode,
            }
            if normalized_pillar_rewards is not None:
                credit_record['normalized_pillar_rewards'] = normalized_pillar_rewards
            self.credit_history.append(credit_record)
        else:
            # During warmup: uniform credit assignment (no GeoMACA learning)
            agent_step_advantages = None
            credit = None
            geomaca_loss_info = {}
            if not use_geomaca and current_episode % 500 == 0:
                print(f"  [Warmup] Episode {current_episode}/{geomaca_warmup} - Base policy training only")
        
        # Update government with credit-weighted rewards
        self._update_government(
            agent_rewards=agent_rewards,
            agent_step_advantages=agent_step_advantages,
        )
        
        # Update developer
        self._update_developer()
        
        return {
            'episode_reward': episode_reward,
            'episode_profit': episode_profit,
            'population': final_metrics.population,
            'segregation': final_metrics.segregation_index,
            'qol': final_metrics.avg_qol,
            'green_space': final_metrics.green_space_ratio,
            'poor_housing': final_metrics.poor_housing,
            'medium_housing': final_metrics.medium_housing,
            'luxury_housing': final_metrics.luxury_housing,
            'terminal_credit_reward': terminal_credit_reward,
            'normalized_terminal_reward': normalized_terminal_reward,
            **geomaca_loss_info
        }

    def _zone_state_env(self):
        """Environment handed to the trajectory collector for the zone states.

        ``None`` (zone_keys='city_observation') makes the state recorded for
        every zone the city-level observation o_k.
        """
        return self.env if self.config.zone_keys == 'zone_features' else None

    def _raw_terminal_reward(self, metrics) -> float:
        """Terminal reward selected by ``terminal_reward``, before any normalisation."""
        if self.config.terminal_reward == 'neg_segregation':
            return -float(metrics.segregation_index)
        return self._compute_terminal_credit_reward(metrics)

    def _compute_terminal_credit_reward(self, metrics) -> float:
        """Balanced terminal reward used by GeoMACA credit assignment.

        An end-of-episode aggregate over the three pillars (economy,
        environment and society), the ``pillar_aggregate`` terminal reward.
        """
        def unit(value: float, scale: float, cap: float = 1.2) -> float:
            return float(np.clip(value / scale, 0.0, cap))

        economy = np.mean([
            unit(metrics.gdp, 250_000_000.0),
            unit(metrics.tax_revenue, 10_000_000.0),
            unit(metrics.developer_profit, 500_000_000.0),
        ])

        environment = np.mean([
            unit(metrics.green_space_ratio, 0.30),
            unit(metrics.woodland_ratio, 0.25),
            float(np.clip(metrics.sustainability_score, 0.0, 1.2)),
        ])

        society = np.mean([
            unit(metrics.population, 1_600.0),
            float(np.clip(1.0 - metrics.segregation_index, 0.0, 1.0)),
            unit(metrics.avg_qol, 5.0),
        ])

        aggregate = float(np.mean([economy, environment, society]))
        return float(np.clip(2.0 * aggregate - 1.0, -1.0, 1.0))

    def _compute_terminal_pillar_rewards(self, metrics) -> np.ndarray:
        """Return separate economy/environment/society terminal rewards."""
        def unit(value: float, scale: float, cap: float = 1.2) -> float:
            return float(np.clip(value / scale, 0.0, cap))

        economy = np.mean([
            unit(metrics.gdp, 250_000_000.0),
            unit(metrics.tax_revenue, 10_000_000.0),
            unit(metrics.developer_profit, 500_000_000.0),
        ])
        environment = np.mean([
            unit(metrics.green_space_ratio, 0.30),
            unit(metrics.woodland_ratio, 0.25),
            float(np.clip(metrics.sustainability_score, 0.0, 1.2)),
        ])
        society = np.mean([
            unit(metrics.population, 1_600.0),
            float(np.clip(1.0 - metrics.segregation_index, 0.0, 1.0)),
            unit(metrics.avg_qol, 5.0),
        ])
        pillars = np.array([economy, environment, society], dtype=np.float32)
        return np.clip(2.0 * pillars - 1.0, -1.0, 1.0)

    def _normalize_pillar_rewards(self, pillar_rewards: np.ndarray) -> np.ndarray:
        normalized = []
        for idx, reward in enumerate(pillar_rewards):
            self.pillar_reward_normalizers[idx].update(float(reward))
            normalized.append(self.pillar_reward_normalizers[idx].normalize(float(reward)))
        return np.asarray(normalized, dtype=np.float32)

    def _compute_multi_objective_credit(self, trajectory: Dict, final_metrics):
        """Query/update GeoMACA once per sustainability pillar.

        Each pillar reward supervises the same GeoMACA module with a different
        scalar query. The corresponding agent receives that pillar's
        per-timestep advantage, rather than every agent sharing one scalar query.
        """
        pillar_rewards = self._compute_terminal_pillar_rewards(final_metrics)
        normalized = self._normalize_pillar_rewards(pillar_rewards)
        final_rewards = torch.tensor(normalized, dtype=torch.float32, device=self.device)

        step_advantages = torch.zeros(
            (1, self.config.num_zones, 3), dtype=torch.float32, device=self.device
        )
        alpha_temps, alpha_geos, alpha_agents = [], [], []
        loss_totals = {
            'geomaca_entropy_loss': 0.0,
            'geomaca_agent_entropy_loss': 0.0,
            'geomaca_consistency_loss': 0.0,
            'geomaca_supervised_loss': 0.0,
            'geomaca_total_loss': 0.0,
        }

        for pillar_idx in range(3):
            reward = final_rewards[pillar_idx:pillar_idx + 1]
            credit, (alpha_temp, alpha_geo, alpha_agent) = self.geomaca(trajectory, reward)
            pillar_adv = self.geomaca.compute_step_agent_advantages(
                credit, trajectory['zone_developed'], reward
            )
            step_advantages[:, :, pillar_idx] = pillar_adv[:, :, pillar_idx]
            info = self._update_geomaca(credit, trajectory, reward)
            for key in loss_totals:
                loss_totals[key] += float(info.get(key, 0.0)) / 3.0
            alpha_temps.append(alpha_temp)
            alpha_geos.append(alpha_geo)
            alpha_agents.append(alpha_agent)

        return (
            step_advantages,
            torch.stack(alpha_temps, dim=0).mean(dim=0),
            torch.stack(alpha_geos, dim=0).mean(dim=0),
            torch.stack(alpha_agents, dim=0).mean(dim=0),
            loss_totals,
            normalized.tolist(),
        )

    def _values_to_constraints(self, values: np.ndarray) -> GovernmentConstraints:
        values = np.asarray(values, dtype=np.float32)
        values = np.clip(
            values,
            np.array([0.10, 0.05, 0.00, 0.50, 0.00], dtype=np.float32),
            np.array([0.50, 0.40, 0.30, 0.95, 0.15], dtype=np.float32),
        )
        return GovernmentConstraints(
            affordable_housing_pct=float(values[0]),
            green_space_pct=float(values[1]),
            woodland_preservation=float(values[2]),
            max_density=float(values[3]),
            tax_rate=float(values[4]),
        )

    def _counterfactual_constraints(
        self, values: np.ndarray, agent_idx: int
    ) -> GovernmentConstraints:
        default = np.array([0.25, 0.15, 0.10, 0.70, 0.02], dtype=np.float32)
        cf_values = np.asarray(values, dtype=np.float32).copy()
        if agent_idx == 0:
            cf_values[[3, 4]] = default[[3, 4]]
        elif agent_idx == 1:
            cf_values[[1, 2]] = default[[1, 2]]
        else:
            cf_values[0] = default[0]
        return self._values_to_constraints(cf_values)

    def _replay_terminal_credit_reward(
        self,
        experiences: List[Dict],
        replace: Optional[Tuple[int, int]] = None,
    ) -> float:
        env = UrbanSimulation(map_size=self.config.map_size, zone_size=self.config.zone_size)
        for step_idx, exp in enumerate(experiences[:self.config.num_zones]):
            values = np.asarray(exp['constraints'], dtype=np.float32)
            if replace is not None and step_idx == replace[0]:
                constraints = self._counterfactual_constraints(values, replace[1])
            else:
                constraints = self._values_to_constraints(values)
            decision = DeveloperDecision(
                land_use_grid=np.asarray(exp['land_grid'], dtype=np.int32)
            )
            env.develop_zone(int(exp['zone_id']), constraints, decision)
        return self._compute_terminal_credit_reward(env.metrics)

    def _compute_counterfactual_training_signal(
        self,
        experiences: List[Dict],
        actual_score: float,
        current_episode: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        interval = max(1, int(getattr(self.config, 'counterfactual_interval', 10)))
        if current_episode % interval != 0:
            return None
        if len(experiences) < self.config.num_zones:
            return None

        contributions = np.zeros((self.config.num_zones, 3), dtype=np.float32)
        for step_idx in range(self.config.num_zones):
            for agent_idx in range(3):
                cf_score = self._replay_terminal_credit_reward(
                    experiences, replace=(step_idx, agent_idx)
                )
                contributions[step_idx, agent_idx] = actual_score - cf_score

        centered = contributions - float(np.mean(contributions))
        scale = float(np.std(centered))
        if scale < 1e-4:
            advantages = np.zeros_like(centered)
        else:
            advantages = np.clip(centered / scale, -3.0, 3.0)

        target_source = np.abs(centered).reshape(-1)
        total = float(target_source.sum())
        if total < 1e-8:
            target = np.full_like(target_source, 1.0 / target_source.size)
        else:
            target = target_source / total
        target = target.reshape(1, self.config.num_zones, 3)

        return (
            torch.tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.tensor(target, dtype=torch.float32, device=self.device),
        )

    def _compute_developer_reward(self, result: Dict, metrics, prev_seg: float, zone_distributions: list) -> float:
        """
        Developer reward: a weighted sum of twelve components (weights below).

        Among them:
        1. Cross-zone segregation penalty (dissimilarity index)
        2. Forced zone diversity (minimum % per housing type)
        3. Weights that put about half of the total on anti-segregation terms

        The cross-zone term penalises layouts such as "all poor in zones 1-6,
        all luxury in zones 7-9".
        """
        
        # === ZONE-SPECIFIC DATA ===
        # Get housing built for development/occupancy calculations
        housing_built = result.get('housing_built', {'poor': 0, 'medium': 0, 'luxury': 0})
        housing_occupied = result.get('housing_occupied', {'poor': 0, 'medium': 0, 'luxury': 0})
        total_built = sum(housing_built.values())
        total_occupied = sum(housing_occupied.values())
        
        # For diversity/segregation: use HOUSEHOLD WEALTH CLASS from zone_distributions
        # This matches exactly how the environment calculates segregation
        current_zone_idx = len(zone_distributions) - 1 if zone_distributions else -1
        if current_zone_idx >= 0 and current_zone_idx < len(zone_distributions):
            zone_hh = zone_distributions[current_zone_idx]
            poor = zone_hh.get('poor', 0)      # Poor HOUSEHOLDS in this zone
            medium = zone_hh.get('middle', 0)  # Middle HOUSEHOLDS in this zone
            luxury = zone_hh.get('rich', 0)    # Rich HOUSEHOLDS in this zone
        else:
            poor, medium, luxury = 0, 0, 0
        zone_housing = poor + medium + luxury  # Total households in zone
        
        zone_capacity = 400  # 20x20 grid
        zone_pop = result.get('immigrants', 0)
        
        # City-wide POPULATION by class (matches segregation calculation)
        # Use pop counts (households who moved in), not housing counts (units built)
        city_poor = metrics.poor_pop
        city_medium = metrics.middle_pop
        city_luxury = metrics.rich_pop
        city_total = city_poor + city_medium + city_luxury
        city_population = metrics.population
        
        current_seg = metrics.segregation_index
        
        # ============================================================
        # COMPONENT 1: PROFIT (5%)
        # ============================================================
        profit = result.get('first_year_profit', 0)
        profit_norm = np.clip(profit / 1e7, -1, 1)
        
        # ============================================================
        # COMPONENT 2: DEVELOPMENT (10%)
        # Based on housing BUILT (not occupied) - developer's output
        # ============================================================
        development_ratio = total_built / zone_capacity
        if total_built == 0:
            development_norm = -1.0
        elif development_ratio < 0.4:
            development_norm = (development_ratio / 0.4) - 1.0
        else:
            development_norm = (development_ratio - 0.4) / 0.6
        
        # ============================================================
        # COMPONENT 3: OCCUPANCY (10%)
        # ============================================================
        if total_built > 0:
            occupancy_rate = total_occupied / total_built
            occupancy_norm = 2.0 * occupancy_rate - 1.0
        else:
            occupancy_norm = -1.0
        
        # ============================================================
        # COMPONENT 4: POPULATION (10%)
        # ============================================================
        pop_target = 250
        if zone_pop == 0:
            population_norm = -1.0
        elif zone_pop < pop_target:
            population_norm = (zone_pop / pop_target) - 0.5
        else:
            excess_ratio = min((zone_pop - pop_target) / pop_target, 1.0)
            population_norm = 0.5 + 0.5 * excess_ratio
        
        # ============================================================
        # COMPONENT 5: WITHIN-ZONE DIVERSITY (5%)
        # ============================================================
        if zone_housing > 0:
            poor_pct = poor / zone_housing
            med_pct = medium / zone_housing
            lux_pct = luxury / zone_housing
            gini_simpson = 1 - (poor_pct**2 + med_pct**2 + lux_pct**2)
            diversity_norm = (gini_simpson / 0.67) * 2 - 1
        else:
            diversity_norm = -1.0
        
        # ============================================================
        # COMPONENT 6: MISSING TYPES PENALTY (5%)
        # ============================================================
        if zone_housing > 0:
            missing_count = (poor == 0) + (medium == 0) + (luxury == 0)
            missing_norm = -missing_count / 3.0
        else:
            missing_norm = -1.0
        
        # ============================================================
        # COMPONENT 7: ZONE-CITY MIXING (5%)
        # ============================================================
        if zone_housing > 0 and city_total > 0:
            zone_poor_pct = poor / zone_housing
            zone_med_pct = medium / zone_housing
            zone_lux_pct = luxury / zone_housing
            city_poor_pct = city_poor / city_total
            city_med_pct = city_medium / city_total
            city_lux_pct = city_luxury / city_total
            deviation = (
                abs(zone_poor_pct - city_poor_pct) +
                abs(zone_med_pct - city_med_pct) +
                abs(zone_lux_pct - city_lux_pct)
            )
            mixing_norm = 1.0 - deviation
        else:
            mixing_norm = -1.0
        
        # ============================================================
        # COMPONENT 8: SEGREGATION DELTA (5%)
        # ============================================================
        seg_delta = current_seg - prev_seg
        if seg_delta > 0:
            delta_penalty = -10.0 * seg_delta
        else:
            delta_penalty = -2.0 * seg_delta
        seg_delta_norm = np.clip(delta_penalty, -1, 1)
        
        # ============================================================
        # COMPONENT 9: SEGREGATION ABSOLUTE (5%)
        # ============================================================
        if current_seg < 0.15:
            seg_absolute_norm = 1.0
        elif current_seg < 0.25:
            seg_absolute_norm = 1.0 - (current_seg - 0.15) / 0.10 * 0.5
        elif current_seg < 0.40:
            seg_absolute_norm = 0.5 - (current_seg - 0.25) / 0.15 * 1.5
        else:
            seg_absolute_norm = -1.0
        
        # ============================================================
        # COMPONENT 10: CROSS-ZONE SEGREGATION (15%)
        # ============================================================
        if len(zone_distributions) > 0:
            # Uses household wealth class (poor/middle/rich), not housing type
            total_poor_city = sum(z.get('poor', 0) for z in zone_distributions)
            total_rich_city = sum(z.get('rich', 0) for z in zone_distributions)
            total_households_city = sum(
                z.get('poor', 0) + z.get('middle', 0) + z.get('rich', 0) 
                for z in zone_distributions
            )
            
            if total_poor_city > 0 and total_rich_city > 0 and total_households_city > 0:
                dissimilarity = 0.0
                for z in zone_distributions:
                    z_poor = z.get('poor', 0)
                    z_rich = z.get('rich', 0)
                    z_total = z_poor + z.get('middle', 0) + z_rich
                    if z_total > 0:
                        poor_share = z_poor / total_poor_city
                        household_share = z_total / total_households_city
                        dissimilarity += abs(poor_share - household_share)
                dissimilarity = dissimilarity / 2.0
                
                if dissimilarity < 0.2:
                    cross_zone_seg_norm = 1.0
                elif dissimilarity < 0.35:
                    cross_zone_seg_norm = 1.0 - (dissimilarity - 0.2) / 0.15
                elif dissimilarity < 0.5:
                    cross_zone_seg_norm = -(dissimilarity - 0.35) / 0.15
                else:
                    cross_zone_seg_norm = -1.0
            else:
                cross_zone_seg_norm = 0.0
        else:
            # Fallback to standard segregation
            if current_seg < 0.2:
                cross_zone_seg_norm = 1.0
            elif current_seg < 0.4:
                cross_zone_seg_norm = 1.0 - (current_seg - 0.2) / 0.2 * 2.0
            else:
                cross_zone_seg_norm = -1.0
        
        # ============================================================
        # COMPONENT 11: FORCED ZONE DIVERSITY (15%)
        # ============================================================
        min_required = 0.15
        if zone_housing > 0:
            poor_pct = poor / zone_housing
            med_pct = medium / zone_housing
            lux_pct = luxury / zone_housing
            
            poor_deficit = max(0, min_required - poor_pct)
            med_deficit = max(0, min_required - med_pct)
            lux_deficit = max(0, min_required - lux_pct)
            total_deficit = poor_deficit + med_deficit + lux_deficit
            
            violations = (poor_pct < min_required) + (med_pct < min_required) + (lux_pct < min_required)
            
            if violations == 0:
                min_pct = min(poor_pct, med_pct, lux_pct)
                max_pct = max(poor_pct, med_pct, lux_pct)
                balance = 1 - (max_pct - min_pct)
                forced_div_norm = 0.5 + 0.5 * balance
            elif violations == 1:
                forced_div_norm = -0.3 - 2.0 * total_deficit
            elif violations == 2:
                forced_div_norm = -0.6 - 1.0 * total_deficit
            else:  # violations == 3
                forced_div_norm = -1.0
        else:
            forced_div_norm = -1.0
        
        # ============================================================
        # COMPONENT 12: POPULATION BALANCE (5%)
        # ============================================================
        if city_population > 0 and zone_pop > 0:
            zone_pop_share = zone_pop / city_population
            ideal_share = 1.0 / 9.0
            pop_deviation = abs(zone_pop_share - ideal_share)
            pop_balance_norm = 1.0 - pop_deviation / 0.2 * 2
            pop_balance_norm = np.clip(pop_balance_norm, -1, 1)
        else:
            pop_balance_norm = 0.0
        
        # ============================================================
        # WEIGHTS (sum to 1.0) - 50% Anti-Segregation Focus
        # ============================================================
        weights = {
            'profit':           0.05,
            'development':      0.10,
            'occupancy':        0.10,
            'population':       0.10,
            'diversity':        0.05,
            'missing':          0.05,
            'mixing':           0.05,
            'seg_delta':        0.05,
            'seg_absolute':     0.05,
            'cross_zone_seg':   0.15,
            'forced_diversity': 0.15,
            'pop_balance':      0.05,
        }
        
        # ============================================================
        # COMBINE
        # ============================================================
        total = (
            weights['profit'] * profit_norm +
            weights['development'] * development_norm +
            weights['occupancy'] * occupancy_norm +
            weights['population'] * population_norm +
            weights['diversity'] * diversity_norm +
            weights['missing'] * missing_norm +
            weights['mixing'] * mixing_norm +
            weights['seg_delta'] * seg_delta_norm +
            weights['seg_absolute'] * seg_absolute_norm +
            weights['cross_zone_seg'] * cross_zone_seg_norm +
            weights['forced_diversity'] * forced_div_norm +
            weights['pop_balance'] * pop_balance_norm
        )
        
        return float(np.clip(total, -1, 1))
    
    def _update_geomaca(
        self,
        credit: torch.Tensor,
        trajectory: Dict,
        final_reward: torch.Tensor,
        target_step_credit: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Update GeoMACA module with auxiliary losses."""
        # Skip if no learnable components (uniform credit)
        if not self.config.use_temporal and not self.config.use_geographic and not self.config.use_agent:
            return {
                'geomaca_entropy_loss': 0.0,
                'geomaca_agent_entropy_loss': 0.0,
                'geomaca_consistency_loss': 0.0,
                'geomaca_supervised_loss': 0.0,
                'geomaca_total_loss': 0.0
            }
        
        # Compute losses
        losses = self.geomaca_losses.compute(credit, trajectory, final_reward)
        supervised_loss = torch.tensor(0.0, device=self.device)
        if target_step_credit is not None and 'zone_developed' in trajectory:
            pred_step_credit = self.geomaca.compute_per_step_credit(
                credit, trajectory['zone_developed']
            ).clamp_min(1e-8)
            target = target_step_credit / (
                target_step_credit.sum(dim=(1, 2), keepdim=True) + 1e-8
            )
            supervised_loss = F.kl_div(
                pred_step_credit.log(),
                target,
                reduction='batchmean',
            )
            losses['total'] = losses['total'] + (
                getattr(self.config, 'geomaca_supervised_weight', 0.5)
                * supervised_loss
            )
        
        # Check if loss requires grad
        if not losses['total'].requires_grad:
            return {
                'geomaca_entropy_loss': losses['entropy'].item() if hasattr(losses['entropy'], 'item') else 0.0,
                'geomaca_agent_entropy_loss': losses['agent_entropy'].item() if hasattr(losses['agent_entropy'], 'item') else 0.0,
                'geomaca_consistency_loss': losses['consistency'].item() if hasattr(losses['consistency'], 'item') else 0.0,
                'geomaca_supervised_loss': supervised_loss.item() if hasattr(supervised_loss, 'item') else 0.0,
                'geomaca_total_loss': losses['total'].item() if hasattr(losses['total'], 'item') else 0.0
            }
        
        # Backprop
        self.geomaca_optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.geomaca.parameters(), 0.5)
        self.geomaca_optimizer.step()
        
        return {
            'geomaca_entropy_loss': losses['entropy'].item(),
            'geomaca_agent_entropy_loss': losses['agent_entropy'].item(),
            'geomaca_consistency_loss': losses['consistency'].item(),
            'geomaca_supervised_loss': supervised_loss.item(),
            'geomaca_total_loss': losses['total'].item()
        }
    
    def _update_government(
        self,
        agent_rewards: Optional[torch.Tensor] = None,
        agent_step_advantages: Optional[torch.Tensor] = None,
    ):
        """Update government agents with credit-weighted rewards."""
        if len(self.buffer) == 0:
            return
        
        # Get experiences
        experiences = list(self.buffer.buffer)
        
        # Convert to tensors
        base_states = torch.tensor(
            np.array([e['base_state'] for e in experiences]),
            dtype=torch.float32, device=self.device
        )
        gov_actions = torch.tensor(
            np.array([e['gov_actions'] for e in experiences]),
            dtype=torch.float32, device=self.device
        )
        
        step_advantages = None
        if agent_step_advantages is not None:
            # [1, T, A] -> [T, A], one advantage per zone-development action.
            step_advantages = agent_step_advantages.squeeze(0).detach()
            step_advantages = torch.clamp(step_advantages, -10.0, 10.0)
            rewards_per_agent = None
        elif agent_rewards is not None:
            # agent_rewards is [1, 3] - one value per agent
            rewards_per_agent = agent_rewards.squeeze(0).detach()  # [3]
            # STABILITY: Clamp rewards to prevent explosion
            rewards_per_agent = torch.clamp(rewards_per_agent, -10.0, 10.0)
        else:
            # Fallback to uniform
            total_reward = sum(e['dev_reward'] for e in experiences)
            rewards_per_agent = torch.full((3,), total_reward / 3, device=self.device)
        
        # Update each agent using actors list
        agent_action_dims = [2, 2, 1]  # Economy, Environment, Society
        action_offset = 0
        
        for agent_idx in range(3):
            actor = self.government.actors[agent_idx]
            optimizer = self.government.actor_optimizers[agent_idx]
            action_dim = agent_action_dims[agent_idx]
            
            # PPO update (simplified)
            for _ in range(self.config.ppo_epochs):
                # Get current actions
                mean, std = actor(base_states)
                
                # STABILITY: Clamp mean and std
                mean = torch.clamp(mean, -5.0, 5.0)
                std = torch.clamp(std, 0.01, 2.0)
                
                dist = torch.distributions.Normal(mean, std)
                
                # Get stored actions for this agent (padded to gov_action_dim)
                agent_actions = gov_actions[:, agent_idx, :action_dim]  # [batch, action_dim]
                
                # STABILITY: Clamp actions before computing log_prob
                agent_actions = torch.clamp(agent_actions, mean - 3*std, mean + 3*std)
                
                new_log_probs = dist.log_prob(agent_actions).sum(dim=-1)
                
                # STABILITY: Clamp log probs
                new_log_probs = torch.clamp(new_log_probs, -20.0, 20.0)
                
                if step_advantages is not None:
                    advantage = step_advantages[:new_log_probs.shape[0], agent_idx]
                    if torch.isnan(advantage).any() or torch.isinf(advantage).any():
                        continue
                else:
                    agent_reward = rewards_per_agent[agent_idx]
                    if torch.isnan(agent_reward) or torch.isinf(agent_reward):
                        continue
                    advantage = agent_reward.expand_as(new_log_probs)
                
                # Policy gradient loss
                actor_loss = -(new_log_probs * advantage).mean()
                entropy_loss = -dist.entropy().mean()
                
                loss = actor_loss + 0.01 * entropy_loss
                
                # Skip if loss is bad
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients
                has_nan = False
                for p in actor.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan = True
                        break
                
                if has_nan:
                    optimizer.zero_grad()
                    continue
                
                # STABILITY: Aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                optimizer.step()
            
            action_offset += action_dim
    
    def _update_developer(self):
        """Update developer network."""
        if len(self.buffer) == 0:
            return
        
        experiences = list(self.buffer.buffer)
        
        enhanced_states = torch.tensor(
            np.array([e['enhanced_state'] for e in experiences]),
            dtype=torch.float32, device=self.device
        )
        land_grids = torch.tensor(
            np.array([e['land_grid'] for e in experiences]),
            dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            np.array([e['dev_reward'] for e in experiences]),
            dtype=torch.float32, device=self.device
        )
        
        # Use the ACTUAL constraints from the episode
        constraints = torch.tensor(
            np.array([e['constraints'] for e in experiences]),
            dtype=torch.float32, device=self.device
        )
        
        # Get old log probs for PPO ratio calculation
        old_log_probs = torch.tensor(
            np.array([e['old_log_prob'] for e in experiences]),
            dtype=torch.float32, device=self.device
        )
        
        # STABILITY: Clamp old log probs to prevent ratio explosion
        old_log_probs = torch.clamp(old_log_probs, -100.0, 100.0)
        
        # Rewards are already in [-1, +1]; scale to PPO-friendly range [-2, +2]
        # Using smaller range for stability
        rewards_scaled = rewards * 2.0
        rewards_normalized = torch.clamp(rewards_scaled, -2.0, 2.0)
        
        for _ in range(self.config.ppo_epochs):
            land_logits, values = self.developer(enhanced_states, constraints)
            
            # STABILITY: Clamp logits more aggressively
            land_logits = torch.clamp(land_logits, -10.0, 10.0)
            
            land_probs = F.softmax(land_logits, dim=1)
            log_probs = F.log_softmax(land_logits, dim=1)
            
            # Get log prob of chosen actions
            batch_size = land_grids.shape[0]
            chosen_log_probs = log_probs.view(batch_size, self.config.num_land_types, -1)
            land_grids_flat = land_grids.view(batch_size, -1)
            
            action_log_probs = chosen_log_probs.gather(
                1, land_grids_flat.unsqueeze(1)
            ).squeeze(1).sum(dim=-1)
            
            # STABILITY: Clamp action log probs
            action_log_probs = torch.clamp(action_log_probs, -100.0, 100.0)
            
            # PPO: Compute ratio with clamped difference
            log_ratio = action_log_probs - old_log_probs
            log_ratio = torch.clamp(log_ratio, -2.0, 2.0)  # Prevent exp explosion
            ratio = torch.exp(log_ratio)
            
            surr1 = ratio * rewards_normalized
            surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip) * rewards_normalized
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipped normalized rewards
            value_loss = F.mse_loss(values.squeeze(), rewards_normalized)
            
            # Entropy bonus to prevent collapse
            entropy = -(land_probs * log_probs).sum(dim=1).mean()

            # Entropy coefficient 0.1
            total_loss = policy_loss + 0.5 * value_loss - 0.1 * entropy
            
            # Skip update if loss is NaN or Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue  # Silent skip - don't spam logs
            
            self.dev_optimizer.zero_grad()
            total_loss.backward()
            
            # Check for NaN gradients and skip if found
            has_nan_grad = False
            for param in self.developer.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                self.dev_optimizer.zero_grad()  # Clear bad gradients
                continue  # Silent skip
            
            # Very aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(self.developer.parameters(), 0.1)
            self.dev_optimizer.step()
    
    def train(self, num_episodes: Optional[int] = None,
              patience_episodes: Optional[int] = None,
              patience_min_episodes: int = 500):
        """Main training loop with optional patience-based early stopping.

        Patience-ES: stop if best (windowed) validation reward has not improved
        for ``patience_episodes`` consecutive episodes after at least
        ``patience_min_episodes`` episodes have elapsed.
        """
        num_episodes = num_episodes or self.config.num_episodes

        print("=" * 80)
        print("GEOMACA TRAINING START")
        print("=" * 80)
        print(f"Episodes: {num_episodes}")
        if patience_episodes:
            print(f"Patience-ES: {patience_episodes} eps after min {patience_min_episodes}")
        print(f"Features: Temporal={self.config.use_temporal}, "
              f"Geographic={self.config.use_geographic}, "
              f"Agent={self.config.use_agent}")
        print("=" * 80)

        best_reward = float('-inf')
        # Patience-ES bookkeeping uses a 100-episode windowed mean as the
        # validation metric (matches the existing logging cadence).
        val_window = 100
        best_val = float('-inf')
        eps_since_best_val = 0

        for episode in range(num_episodes):
            # Run episode
            metrics = self.run_episode()
            self.metrics_history.append(metrics)

            # Update best (per-episode reward, used for 'best.pt')
            if metrics['episode_reward'] > best_reward:
                best_reward = metrics['episode_reward']
                self.save_checkpoint('best')

            # Patience-ES on windowed mean
            if patience_episodes and len(self.metrics_history) >= val_window:
                window_mean = float(np.mean(
                    [m['episode_reward'] for m in self.metrics_history[-val_window:]]
                ))
                if window_mean > best_val + 1e-6:
                    best_val = window_mean
                    eps_since_best_val = 0
                else:
                    eps_since_best_val += 1
                if (episode >= patience_min_episodes and
                        eps_since_best_val >= patience_episodes):
                    print(
                        f"\nPatience-ES triggered at ep {episode}: "
                        f"no windowed-mean improvement for {patience_episodes} eps; "
                        f"best window mean = {best_val:.3f}"
                    )
                    break

            # Per-episode early stopping (if enabled; on unless --no-early-stop)
            if getattr(self.config, 'early_stop_enabled', True):
                if self.early_stopping.check(metrics['episode_reward']):
                    print(f"\nEarly stopping at episode {episode}")
                    break
            
            # Logging
            if episode % self.config.log_interval == 0:
                recent = self.metrics_history[-self.config.log_interval:]
                avg_reward = np.mean([m['episode_reward'] for m in recent])
                avg_profit = np.mean([m['episode_profit'] for m in recent])
                avg_seg = np.mean([m['segregation'] for m in recent])
                avg_pop = np.mean([m['population'] for m in recent])
                
                # Get credit distribution if available
                if self.credit_history:
                    last_credit = self.credit_history[-1]
                    agent_credit_str = ", ".join([
                        f"{name[:3]}={val:.2f}" 
                        for name, val in zip(
                            ['Economy', 'Environment', 'Society'],
                            last_credit['alpha_agent']
                        )
                    ])
                else:
                    agent_credit_str = "N/A"
                
                print(f"Ep {episode:5d} | Reward: {avg_reward:7.2f} | "
                      f"Profit: ${avg_profit/1e6:5.1f}M | "
                      f"Seg: {avg_seg:.3f} | Pop: {avg_pop:5.0f} | "
                      f"Credit: [{agent_credit_str}]")
            
            # Save checkpoint
            if episode % self.config.save_interval == 0 and episode > 0:
                self.save_checkpoint(f'ep_{episode}')
        
        # Final save
        self.save_checkpoint('final')
        
        if self.config.skip_validation:
            validation_passed = None
        else:
            # Validate model before trusting results
            print("\n" + "=" * 80)
            print("VALIDATION PHASE")
            print("=" * 80)
            validation_passed = self._validate_model()
        
        # Generate visualizations
        self._generate_final_visualizations()
        
        print("=" * 80)
        print("TRAINING COMPLETE")
        print(f"Final results saved to: {self.exp_dir}")
        if validation_passed is None:
            print("VALIDATION: SKIPPED - use the evaluation pipeline for metrics")
        elif validation_passed:
            print("VALIDATION: PASSED - Model is ready for paper!")
        else:
            print("VALIDATION: FAILED - Model needs more training or tuning")
        print("=" * 80)
        
        return self.metrics_history
    
    def _validate_model(self, num_episodes: int = 30) -> bool:
        """Validate trained model passes sanity checks before trusting results."""
        print(f"Running {num_episodes} validation episodes...")
        
        results = []
        for ep in range(num_episodes):
            env = UrbanSimulation(map_size=self.config.map_size, zone_size=self.config.zone_size)
            
            total_profit = 0
            has_poor = has_medium = has_luxury = False
            
            for zone_id in range(9):
                # Get state
                base_state = env.get_state_vector()[:self.config.base_state_dim]
                enhanced_state = env.get_developer_state(zone_id)
                
                # Government action
                constraints, _, _ = self.government.select_action(base_state, training=False)
                
                # Developer action
                state_t = torch.tensor(enhanced_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                constraints_t = torch.tensor([
                    constraints.affordable_housing_pct,
                    constraints.green_space_pct,
                    constraints.woodland_preservation,
                    constraints.max_density,
                    constraints.tax_rate
                ], dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    land_logits, _ = self.developer(state_t, constraints_t)
                    probs = F.softmax(land_logits, dim=1)
                    probs = probs.permute(0, 2, 3, 1).reshape(-1, 6)
                    probs = torch.clamp(probs, min=1e-8)
                    grid = torch.multinomial(probs, 1).squeeze(-1).reshape(20, 20).cpu().numpy()
                
                decision = DeveloperDecision(land_use_grid=grid.astype(np.int32))
                result = env.develop_zone(zone_id, constraints, decision)
                total_profit += result.get('profit', 0)
            
            # Check housing types
            m = env.metrics
            has_poor = m.poor_housing > 0
            has_medium = m.medium_housing > 0
            has_luxury = m.luxury_housing > 0
            
            total_housing = m.poor_housing + m.medium_housing + m.luxury_housing
            if total_housing > 0:
                vacancy = 1 - (m.population / (total_housing * 3))
            else:
                vacancy = 1.0
            
            results.append({
                'population': m.population,
                'vacancy': vacancy,
                'profit': total_profit,
                'segregation': m.segregation_index,
                'has_diversity': has_poor and has_medium and has_luxury
            })
        
        # Compute statistics
        avg_pop = np.mean([r['population'] for r in results])
        avg_vacancy = np.mean([r['vacancy'] for r in results])
        avg_profit = np.mean([r['profit'] for r in results])
        avg_seg = np.mean([r['segregation'] for r in results])
        diversity_rate = np.mean([r['has_diversity'] for r in results])
        
        # Sanity checks
        checks = {
            'population > 500': avg_pop > 500,
            'vacancy < 50%': avg_vacancy < 0.5,
            'profit > 0': avg_profit > 0,
            'housing_diversity > 50%': diversity_rate > 0.5,
            'segregation < 0.3': avg_seg < 0.3,
        }
        
        print("\nSanity Checks:")
        all_passed = True
        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")
            if not passed:
                all_passed = False
        
        print(f"\nValidation Stats:")
        print(f"  Avg Population: {avg_pop:.0f}")
        print(f"  Avg Vacancy: {avg_vacancy*100:.1f}%")
        print(f"  Avg Profit: ${avg_profit/1e6:.1f}M")
        print(f"  Avg Segregation: {avg_seg:.3f}")
        print(f"  Housing Diversity: {diversity_rate*100:.0f}%")
        
        return all_passed
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            'government_state': {
                'actors': [actor.state_dict() for actor in self.government.actors],
                'critic': self.government.critic.state_dict(),
            },
            'developer_state': self.developer.state_dict(),
            'geomaca_state': self.geomaca.state_dict(),
            'config': {k: v for k, v in asdict(self.config).items()},
            'metrics_history': self.metrics_history[-100:],  # Last 100
        }
        
        path = self.exp_dir / 'checkpoints' / f'{name}.pt'
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load government actors
        for i, actor_state in enumerate(checkpoint['government_state']['actors']):
            self.government.actors[i].load_state_dict(actor_state)
        self.government.critic.load_state_dict(checkpoint['government_state']['critic'])
        
        self.developer.load_state_dict(checkpoint['developer_state'])
        self.geomaca.load_state_dict(checkpoint['geomaca_state'])
        
        print(f"Loaded checkpoint from {path}")
    
    def _generate_final_visualizations(self):
        """Generate visualizations for paper."""
        if not self.credit_history:
            return
        
        # Run one more episode to get visualization data
        self.env = UrbanSimulation(map_size=self.config.map_size, zone_size=self.config.zone_size)
        self.trajectory_collector.reset()
        
        # Quick episode
        for zone_id in range(self.config.num_zones):
            base_state, enhanced_state = self.get_state(zone_id)
            constraints, agent_actions_list, _ = self.government.select_action(base_state)
            
            # Convert agent actions to numpy
            agent_actions = np.zeros((3, self.config.gov_action_dim), dtype=np.float32)
            for i, act in enumerate(agent_actions_list):
                action_np = act.detach().cpu().numpy().flatten()
                agent_actions[i, :len(action_np)] = action_np
            
            state_t = torch.tensor(enhanced_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            constraints_t = torch.tensor([
                constraints.affordable_housing_pct, constraints.green_space_pct,
                constraints.woodland_preservation, constraints.max_density, constraints.tax_rate
            ], dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                land_logits, _ = self.developer(state_t, constraints_t)
                probs = F.softmax(land_logits, dim=1)
                probs = probs.clamp(min=1e-8)
                probs = probs / probs.sum(dim=1, keepdim=True)
                land_use_grid = torch.multinomial(
                    probs.view(-1, self.config.num_land_types), num_samples=1
                ).view(20, 20)
            
            decision = DeveloperDecision()
            decision.land_use_grid = land_use_grid.cpu().numpy().astype(np.int32)
            result = self.env.develop_zone(zone_id, constraints, decision)
            
            self.trajectory_collector.record_turn(
                zone_id=zone_id,
                global_state=base_state,
                zone_state=base_state,
                agent_actions=agent_actions,
                developer_reward=0,
                env=self._zone_state_env()
            )
        
        # Get final reward
        final_metrics = self.env.metrics
        trajectory = self.trajectory_collector.get_trajectory(self.device)
        final_reward = torch.tensor(
            [self._raw_terminal_reward(final_metrics)],
            dtype=torch.float32,
            device=self.device
        )
        
        # Generate visualization
        try:
            visualize_credit_assignment(
                self.geomaca,
                trajectory,
                final_reward,
                save_path=str(self.exp_dir / 'visualizations' / 'credit_assignment.pdf'),
                show_plot=False
            )
            print(f"Credit visualization saved to {self.exp_dir / 'visualizations'}")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")


def run_ablation_study(num_episodes: int = 5000, num_seeds: int = 3):
    """Run ablation study comparing different GeoMACA configurations."""
    
    ablations = {
        'Full GeoMACA': {
            'use_temporal': True,
            'use_geographic': True,
            'use_agent': True
        },
        'No Geographic': {
            'use_temporal': True,
            'use_geographic': False,
            'use_agent': True
        },
        'No Temporal': {
            'use_temporal': False,
            'use_geographic': True,
            'use_agent': True
        },
        'No Agent': {
            'use_temporal': True,
            'use_geographic': True,
            'use_agent': False
        },
        'STAS-style (Temporal+Agent)': {
            'use_temporal': True,
            'use_geographic': False,
            'use_agent': True
        },
        'Uniform Credit': {
            'use_temporal': False,
            'use_geographic': False,
            'use_agent': False
        },
    }
    
    results = {}
    
    for name, ablation_config in ablations.items():
        print(f"\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"{'='*60}")
        
        seed_results = []
        
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create config with ablation settings
            config = GeoMACATrainingConfig(
                num_episodes=num_episodes,
                use_temporal=ablation_config['use_temporal'],
                use_geographic=ablation_config['use_geographic'],
                use_agent=ablation_config['use_agent']
            )
            
            # Train
            trainer = GeoMACATrainer(config)
            metrics = trainer.train()
            
            # Extract final metrics (last 100 episodes)
            final_metrics = {
                'profit': np.mean([m['episode_profit'] for m in metrics[-100:]]),
                'segregation': np.mean([m['segregation'] for m in metrics[-100:]]),
                'population': np.mean([m['population'] for m in metrics[-100:]]),
                'qol': np.mean([m.get('qol', 0) for m in metrics[-100:]]),
            }
            seed_results.append(final_metrics)
        
        # Aggregate across seeds
        results[name] = {
            'profit_mean': np.mean([r['profit'] for r in seed_results]),
            'profit_std': np.std([r['profit'] for r in seed_results]),
            'segregation_mean': np.mean([r['segregation'] for r in seed_results]),
            'segregation_std': np.std([r['segregation'] for r in seed_results]),
            'population_mean': np.mean([r['population'] for r in seed_results]),
            'qol_mean': np.mean([r['qol'] for r in seed_results]),
        }
    
    # Print results table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Method':<30} {'Profit':>12} {'Segregation':>12} {'Population':>12}")
    print("-"*80)
    
    for name, m in results.items():
        profit_str = f"${m['profit_mean']/1e6:.1f}M"
        seg_str = f"{m['segregation_mean']:.3f}"
        pop_str = f"{m['population_mean']:.0f}"
        print(f"{name:<30} {profit_str:>12} {seg_str:>12} {pop_str:>12}")
    
    return results


def build_parser():
    """Command-line options, including the method switches."""
    import argparse

    parser = argparse.ArgumentParser(description='Train with GeoMACA')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--num_episodes', type=int, default=None, help='Alias for --episodes (overrides if set)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (deterministic torch/np/random)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Override experiment dir (e.g., runs/cb_runs_v4/geomaca_seed42/)')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds for ablation')
    parser.add_argument('--no-early-stop', action='store_true', help='Disable early stopping')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--warmup', type=int, default=1000, help='GeoMACA warmup episodes (base policy only)')
    parser.add_argument('--base-only', action='store_true', help='Train base policy only (no GeoMACA)')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip built-in post-training validation episodes')
    parser.add_argument('--patience', type=int, default=300,
                        help='Patience-ES: stop if no improvement for N episodes (0 to disable)')
    parser.add_argument('--patience_min_episodes', type=int, default=500,
                        help='Minimum episodes before patience-ES can trigger')
    # GeoMACA feature flags (for ablation runs)
    parser.add_argument('--use_temporal', type=str, default=None,
                        help='True/False - override config use_temporal flag')
    parser.add_argument('--use_geographic', type=str, default=None,
                        help='True/False - override config use_geographic flag')
    parser.add_argument('--use_agent', type=str, default=None,
                        help='True/False - override config use_agent flag')
    parser.add_argument('--credit_mode', type=str, default='aggregate',
                        choices=['aggregate', 'multi_objective', 'multi_objective_blend', 'counterfactual'],
                        help='GeoMACA training signal variant')
    parser.add_argument('--multi_objective_weight', type=float, default=0.5,
                        help='Blend weight for multi_objective_blend advantages')
    parser.add_argument('--geomaca_combination', type=str, default='multiplicative',
                        choices=['multiplicative', 'learned', 'factorized'],
                        help='GeoMACA credit tensor combination mode')
    parser.add_argument('--counterfactual_interval', type=int, default=10,
                        help='Counterfactual replay interval in episodes')
    parser.add_argument('--counterfactual_weight', type=float, default=0.5,
                        help='Blend weight for counterfactual advantages')
    parser.add_argument('--geomaca_supervised_weight', type=float, default=0.5,
                        help='Weight for counterfactual credit-target KL')
    # Method switches (see GeoMACATrainingConfig); unset means the config default.
    parser.add_argument('--terminal_reward', type=str, default=None,
                        choices=list(METHOD_SWITCH_CHOICES['terminal_reward']),
                        help="Terminal reward for credit assignment (default 'pillar_aggregate')")
    parser.add_argument('--government_advantage', type=str, default=None,
                        choices=list(METHOD_SWITCH_CHOICES['government_advantage']),
                        help="Advantage of the government update (default 'per_step')")
    parser.add_argument('--zone_keys', type=str, default=None,
                        choices=list(METHOD_SWITCH_CHOICES['zone_keys']),
                        help="State recorded for each zone (default 'zone_features')")
    parser.add_argument('--attention_normalization', type=str, default=None,
                        choices=list(METHOD_SWITCH_CHOICES['attention_normalization']),
                        help="Normalisation of the head-averaged attention weights (default 'renormalize')")
    parser.add_argument('--agent_entropy_coef', type=float, default=None,
                        help='Weight of the agent-marginal entropy term, x entropy weight (default 0.5)')
    parser.add_argument('--chapter5', action='store_true',
                        help='Use the configuration of Chapter 5 for all five method switches: '
                             + ', '.join(f'{k}={v}' for k, v in CHAPTER5_METHOD.items()))
    return parser


def resolve_method_switches(args) -> Dict:
    """Values of the five method switches selected on the command line."""
    from dataclasses import fields

    explicit = {name: getattr(args, name) for name in CHAPTER5_METHOD
                if getattr(args, name, None) is not None}
    if getattr(args, 'chapter5', False):
        if explicit:
            raise ValueError('--chapter5 sets ' + ', '.join(sorted(CHAPTER5_METHOD))
                             + '; do not combine it with '
                             + ', '.join(f'--{k}' for k in sorted(explicit)))
        return dict(CHAPTER5_METHOD)
    defaults = {f.name: f.default for f in fields(GeoMACATrainingConfig) if f.name in CHAPTER5_METHOD}
    defaults.update(explicit)
    return defaults


def _str2bool(v: Optional[str], default: bool) -> bool:
    """Parse the True/False strings of the --use_* flags."""
    if v is None:
        return default
    return str(v).strip().lower() in ('true', '1', 'yes', 'y', 't')


def episodes_from_args(args) -> int:
    """Resolve the episode count (allow --num_episodes alias)."""
    return args.num_episodes if args.num_episodes is not None else args.episodes


def config_from_args(args) -> GeoMACATrainingConfig:
    """The training configuration that ``main`` builds from the parsed options."""
    episodes = episodes_from_args(args)
    # If base-only mode, set warmup very high so GeoMACA never activates
    warmup = 999999 if args.base_only else args.warmup
    switches = resolve_method_switches(args)
    return GeoMACATrainingConfig(
        num_episodes=episodes,
        # TRAINING SETTINGS
        early_stop_enabled=not args.no_early_stop,  # On unless --no-early-stop
        entropy_coef=0.05,           # Prevent policy collapse
        vacancy_penalty=-0.5,        # Penalize empty housing
        population_bonus=0.1,        # Reward residents
        diversity_bonus=0.2,         # Reward balanced mix
        geomaca_warmup=warmup,       # Warmup before GeoMACA kicks in
        credit_weight=0.5,           # Don't dominate early
        use_temporal=_str2bool(args.use_temporal, True),
        use_geographic=_str2bool(args.use_geographic, True),
        use_agent=_str2bool(args.use_agent, True),
        credit_mode=args.credit_mode,
        multi_objective_weight=args.multi_objective_weight,
        geomaca_combination=args.geomaca_combination,
        counterfactual_interval=args.counterfactual_interval,
        counterfactual_weight=args.counterfactual_weight,
        geomaca_supervised_weight=args.geomaca_supervised_weight,
        skip_validation=args.skip_validation,
        **switches,
    )


def patience_from_args(args) -> Optional[int]:
    """Patience-ES setting passed to ``train`` (None disables it)."""
    return args.patience if args.patience and args.patience > 0 else None


def main(argv=None):
    import random as _random

    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve episode count (allow --num_episodes alias)
    episodes = episodes_from_args(args)

    # Deterministic seeding (do this BEFORE constructing trainer)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        _random.seed(args.seed)

    if args.ablation:
        results = run_ablation_study(num_episodes=episodes, num_seeds=args.seeds)
    else:
        config = config_from_args(args)
        warmup = config.geomaca_warmup

        mode = "BASE POLICY ONLY" if args.base_only else f"GeoMACA (warmup={warmup})"

        print("=" * 80)
        print(f"TRAINING MODE: {mode}")
        print("=" * 80)
        print(f"Seed: {args.seed}")
        print(f"Save path: {args.save_path or '(default timestamped)'}")
        print(f"Episodes: {config.num_episodes}")
        print(f"Patience-ES: {args.patience} eps (after min {args.patience_min_episodes})")
        print(f"GeoMACA flags: T={config.use_temporal} G={config.use_geographic} A={config.use_agent}")
        print(f"Credit mode: {config.credit_mode}; combination={config.geomaca_combination}")
        print("Method switches: " + ", ".join(
            f"{name}={getattr(config, name)}" for name in CHAPTER5_METHOD))
        print(f"Entropy: {config.entropy_coef} (decay: {config.entropy_decay})")
        if not args.base_only:
            print(f"GeoMACA Warmup: {config.geomaca_warmup} episodes")
        if args.load_checkpoint:
            print(f"Resuming from: {args.load_checkpoint}")
        print("=" * 80)

        trainer = GeoMACATrainer(config, save_path=args.save_path)

        if args.load_checkpoint:
            trainer.load_checkpoint(args.load_checkpoint)
            print(f"[OK] Loaded checkpoint from {args.load_checkpoint}")

        trainer.train(
            num_episodes=episodes,
            patience_episodes=patience_from_args(args),
            patience_min_episodes=args.patience_min_episodes,
        )


if __name__ == "__main__":
    main()
