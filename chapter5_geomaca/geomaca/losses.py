"""
GeoMACA Auxiliary Losses

Auxiliary training objectives to ensure credit quality:
1. Entropy loss - prevent credit collapse to single cell
2. Consistency loss - developed zones should have higher credit
3. Reward prediction loss - credit-weighted features should predict reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GeoMACALosses:
    """Auxiliary losses to ensure credit quality."""
    
    @staticmethod
    def entropy_loss(
        credit: torch.Tensor,
        target_entropy_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Prevent credit collapse to a single cell.
        
        Higher entropy = more distributed credit (good for exploration)
        Lower entropy = more concentrated credit (may be appropriate for simple cases)
        
        Args:
            credit: [batch, T, Z, A] - credit tensor
            target_entropy_ratio: Target ratio of max entropy (0=collapsed, 1=uniform)
        
        Returns:
            Loss that encourages entropy to be at target level
        """
        batch_size = credit.shape[0]
        
        # Flatten credit
        credit_flat = credit.reshape(batch_size, -1)  # [batch, T*Z*A]
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = -(credit_flat * torch.log(credit_flat + 1e-8)).sum(dim=-1)
        
        # Maximum entropy for uniform distribution over all cells
        num_cells = credit_flat.shape[-1]
        max_entropy = torch.log(torch.tensor(num_cells, dtype=torch.float, device=credit.device))
        
        # Normalized entropy (0 = collapsed, 1 = uniform)
        normalized_entropy = entropy / max_entropy
        
        # Loss: MSE between normalized entropy and target
        target = torch.full_like(normalized_entropy, target_entropy_ratio)
        loss = F.mse_loss(normalized_entropy, target)
        
        return loss
    
    @staticmethod
    def consistency_loss(
        credit: torch.Tensor,
        trajectory: Dict[str, torch.Tensor],
        margin: float = 0.1
    ) -> torch.Tensor:
        """
        Developed zones should have higher credit than non-developed zones.
        
        This provides a weak supervision signal: the zone that was actually
        developed at each turn should receive more credit than others.
        
        Args:
            credit: [batch, T, Z, A] - credit tensor
            trajectory: Must contain 'zone_developed': [batch, T] - zone index per turn
            margin: Margin for the hinge loss
        
        Returns:
            Loss that encourages developed zones to have higher credit
        """
        if 'zone_developed' not in trajectory:
            return torch.tensor(0.0, device=credit.device)
        
        zone_developed = trajectory['zone_developed']  # [batch, T]
        batch_size, num_turns = zone_developed.shape
        num_zones = credit.shape[2]
        
        losses = []
        
        for t in range(num_turns):
            zone_idx = zone_developed[:, t].long()  # [batch]
            
            # Credit for developed zone at this turn (sum over agents)
            # credit[:, t, :, :] is [batch, Z, A]
            credit_at_turn = credit[:, t, :, :].sum(dim=-1)  # [batch, Z]
            
            # Get credit for the developed zone
            developed_credit = credit_at_turn[
                torch.arange(batch_size, device=credit.device), zone_idx
            ]  # [batch]
            
            # Average credit for other zones
            mask = torch.ones(batch_size, num_zones, device=credit.device)
            mask[torch.arange(batch_size), zone_idx] = 0
            other_credit = (credit_at_turn * mask).sum(dim=-1) / (num_zones - 1)  # [batch]
            
            # Hinge loss: developed zone should have margin more credit than others
            loss = F.relu(other_credit - developed_credit + margin)
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    @staticmethod
    def temporal_causality_loss(
        credit: torch.Tensor,
        decay_factor: float = 0.9
    ) -> torch.Tensor:
        """
        Later turns should generally have more credit (causal influence).
        
        This is a soft constraint based on the intuition that later decisions
        have more direct impact on the final outcome.
        
        Args:
            credit: [batch, T, Z, A] - credit tensor
            decay_factor: Expected decay ratio from last to first turn
        
        Returns:
            Loss encouraging temporal credit gradient
        """
        batch_size, num_turns = credit.shape[0], credit.shape[1]
        
        # Sum credit per turn
        turn_credit = credit.sum(dim=(2, 3))  # [batch, T]
        
        # Expected pattern: credit increases toward later turns
        # Create expected weights (later turns should have more weight)
        expected_weights = torch.pow(
            torch.tensor(decay_factor, device=credit.device),
            torch.arange(num_turns - 1, -1, -1, device=credit.device, dtype=torch.float)
        )
        expected_weights = expected_weights / expected_weights.sum()
        expected_weights = expected_weights.unsqueeze(0).expand(batch_size, -1)
        
        # KL divergence from expected pattern
        turn_credit_normalized = turn_credit / (turn_credit.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Symmetric KL for stability
        kl_forward = F.kl_div(
            torch.log(turn_credit_normalized + 1e-8),
            expected_weights,
            reduction='batchmean'
        )
        kl_backward = F.kl_div(
            torch.log(expected_weights + 1e-8),
            turn_credit_normalized,
            reduction='batchmean'
        )
        
        return 0.5 * (kl_forward + kl_backward)

    @staticmethod
    def agent_marginal_entropy_loss(
        credit: torch.Tensor,
        target_entropy_ratio: float = 0.8
    ) -> torch.Tensor:
        """Prevent the learned credit from starving an entire agent/pillar."""
        agent_credit = credit.sum(dim=(1, 2))  # [batch, A]
        agent_credit = agent_credit / (agent_credit.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(agent_credit * torch.log(agent_credit + 1e-8)).sum(dim=-1)
        max_entropy = torch.log(
            torch.tensor(agent_credit.shape[-1], dtype=torch.float, device=credit.device)
        )
        normalized_entropy = entropy / max_entropy
        target = torch.full_like(normalized_entropy, target_entropy_ratio)
        return F.mse_loss(normalized_entropy, target)


class RewardPredictor(nn.Module):
    """
    Auxiliary network to predict final reward from credit-weighted features.
    
    This provides additional supervision: if credit is assigned correctly,
    the credit-weighted features should be predictive of the final reward.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, weighted_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weighted_features: [batch, feature_dim]
        Returns:
            predicted_reward: [batch, 1]
        """
        return self.net(weighted_features)


def compute_reward_prediction_loss(
    credit: torch.Tensor,
    trajectory: Dict[str, torch.Tensor],
    final_reward: torch.Tensor,
    reward_predictor: RewardPredictor,
    feature_dim: int
) -> torch.Tensor:
    """
    Credit should enable prediction of final reward.
    
    Train a small value network that predicts reward from credit-weighted features.
    If the credit assignment is correct, the weighted features should be predictive.
    
    Args:
        credit: [batch, T, Z, A] - credit tensor
        trajectory: Contains feature tensors
        final_reward: [batch, 1] - actual final reward
        reward_predictor: Network to predict reward
        feature_dim: Expected feature dimension
    
    Returns:
        MSE loss between predicted and actual reward
    """
    batch_size = credit.shape[0]
    
    # Extract cell-level features
    # For simplicity, use zone states as features
    if 'zone_states' not in trajectory:
        return torch.tensor(0.0, device=credit.device)
    
    zone_states = trajectory['zone_states']  # [batch, T, Z, state_dim]
    state_dim = zone_states.shape[-1]
    num_agents = credit.shape[-1]
    
    # Expand zone states to match credit dimensions
    # [batch, T, Z, state_dim] -> [batch, T, Z, A, state_dim]
    zone_states_expanded = zone_states.unsqueeze(3).expand(-1, -1, -1, num_agents, -1)
    
    # Weight by credit
    credit_expanded = credit.unsqueeze(-1)  # [batch, T, Z, A, 1]
    weighted_features = (credit_expanded * zone_states_expanded).sum(dim=(1, 2, 3))  # [batch, state_dim]
    
    # Predict reward
    predicted_reward = reward_predictor(weighted_features)
    
    # Ensure final_reward has correct shape
    if final_reward.dim() == 1:
        final_reward = final_reward.unsqueeze(-1)
    
    return F.mse_loss(predicted_reward, final_reward)


class GeoMACALossComputer:
    """
    Computes all GeoMACA auxiliary losses with configurable weights.
    """
    
    def __init__(
        self,
        entropy_weight: float = 0.1,
        consistency_weight: float = 0.05,
        causality_weight: float = 0.02,
        prediction_weight: float = 0.0,  # Optional
        target_entropy_ratio: float = 0.5,
        consistency_margin: float = 0.1,
        causality_decay: float = 0.9,
        state_dim: int = 20,
        device: str = 'cuda',
        agent_entropy_coef: float = 0.5
    ):
        # agent_entropy_coef: the agent-marginal entropy term is added with weight
        # agent_entropy_coef * entropy_weight; 0 leaves it out of the total
        # (Chapter 5 uses 0).
        self.entropy_weight = entropy_weight
        self.agent_entropy_coef = agent_entropy_coef
        self.consistency_weight = consistency_weight
        self.causality_weight = causality_weight
        self.prediction_weight = prediction_weight
        self.target_entropy_ratio = target_entropy_ratio
        self.consistency_margin = consistency_margin
        self.causality_decay = causality_decay
        
        # Optional reward predictor
        if prediction_weight > 0:
            self.reward_predictor = RewardPredictor(state_dim).to(device)
            self.reward_predictor_optimizer = torch.optim.Adam(
                self.reward_predictor.parameters(), lr=1e-4
            )
        else:
            self.reward_predictor = None
    
    def compute(
        self,
        credit: torch.Tensor,
        trajectory: Dict[str, torch.Tensor],
        final_reward: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses.
        
        Returns:
            Dictionary with:
                - 'total': Weighted sum of all losses
                - 'entropy': Entropy loss value
                - 'consistency': Consistency loss value
                - 'causality': Temporal causality loss value
                - 'prediction': Reward prediction loss value (if enabled)
        """
        losses = {}
        total = torch.tensor(0.0, device=credit.device)
        
        # Entropy loss
        entropy_loss = GeoMACALosses.entropy_loss(credit, self.target_entropy_ratio)
        losses['entropy'] = entropy_loss
        total = total + self.entropy_weight * entropy_loss

        agent_entropy_loss = GeoMACALosses.agent_marginal_entropy_loss(credit)
        losses['agent_entropy'] = agent_entropy_loss
        if self.agent_entropy_coef != 0.0:
            total = total + self.agent_entropy_coef * self.entropy_weight * agent_entropy_loss
        
        # Consistency loss
        consistency_loss = GeoMACALosses.consistency_loss(
            credit, trajectory, self.consistency_margin
        )
        losses['consistency'] = consistency_loss
        total = total + self.consistency_weight * consistency_loss
        
        # Causality loss
        causality_loss = GeoMACALosses.temporal_causality_loss(
            credit, self.causality_decay
        )
        losses['causality'] = causality_loss
        total = total + self.causality_weight * causality_loss
        
        # Reward prediction loss (optional)
        if self.reward_predictor is not None and final_reward is not None:
            prediction_loss = compute_reward_prediction_loss(
                credit, trajectory, final_reward,
                self.reward_predictor, 20  # state_dim
            )
            losses['prediction'] = prediction_loss
            total = total + self.prediction_weight * prediction_loss
        else:
            losses['prediction'] = torch.tensor(0.0, device=credit.device)
        
        losses['total'] = total
        return losses

