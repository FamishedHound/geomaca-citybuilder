"""
Urban Development MARL - Complete Research-Grade Implementation
================================================================
Training components of the CityBuilder system, numbered as in the section
headers below:

Reward shaping and constraints
- 1.1 Hard action space constraints (bounds on the government actions)
- 1.2 PPO with Lagrangian constraints
- 1.3 Reward normalization with running statistics

Credit assignment
- 2.1 Predictive Contribution Measurement (PCM)
- 2.2 Spatial-Temporal Return Decomposition (STAS)
- 2.3 Per-agent intrinsic rewards

Exploration
- 3.1 Intrinsic Curiosity Module (ICM)
- 3.2 Adaptive entropy coefficient
- 3.3 Population-Based Training (optional)

Multi-agent design
- 4.1 MAPPO with centralized critic
- 4.2 Communication between agents
- 4.3 Hierarchical RL (optional)

Baselines and validation
- 5.1 All baseline models
- 5.2 Pareto hypervolume calculation
- 5.3 Weight space exploration

Additional:
- Spatial Developer CNN (20x20 grid output)
- Early stopping mechanism
- Comprehensive logging

``training/train_with_geomaca.py`` takes its government policies
(``MAPPOGovernment``), the Spatial Developer CNN and the early-stopping rule
from this module.

Author: Urban Development MARL Research Project
Date: December 2024
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

# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class Config:
    """Complete training configuration - single source of truth."""
    # Environment
    map_size: int = 100
    zone_size: int = 20
    num_zones: int = 9
    
    # Training
    num_episodes: int = 10000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.1  # High enough to prevent policy collapse
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Learning rates
    gov_lr: float = 3e-4
    dev_lr: float = 3e-4
    lagrange_lr: float = 0.01
    
    # Lagrangian constraints
    segregation_limit: float = 0.40  # Max acceptable segregation
    min_population: float = 500      # Min population target
    
    # Curiosity
    curiosity_scale: float = 0.01
    curiosity_feature_dim: int = 64
    
    # Adaptive entropy
    target_entropy: float = -3.0
    entropy_lr: float = 0.001
    
    # Early stopping
    early_stop_patience: int = 200
    early_stop_min_delta: float = 0.01
    
    # Logging
    log_interval: int = 50
    save_interval: int = 500
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # State/Action dimensions
    # Enhanced developer state: 20 base + 3 zone + 45 global zones + 45 neighbors = 113
    state_dim: int = 113
    base_state_dim: int = 20  # For government agent (still uses base state)
    gov_action_dim: int = 5
    dev_grid_size: int = 20  # 20x20 spatial output
    num_land_types: int = 6  # empty, poor, medium, luxury, green, woodland


# ============================================================
# 1.3 REWARD NORMALIZATION
# ============================================================
class RewardNormalizer:
    """Running normalization to prevent reward scale exploitation."""
    
    def __init__(self, clip_range: float = 10.0, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip_range = clip_range
        self.epsilon = epsilon
    
    def update(self, reward: float):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        if self.count > 1:
            self.var += delta * (reward - self.mean)
    
    def normalize(self, reward: float) -> float:
        if self.count < 2:
            return reward
        std = np.sqrt(self.var / (self.count - 1)) + self.epsilon
        normalized = (reward - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def state_dict(self) -> dict:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}
    
    def load_state_dict(self, state: dict):
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


# ============================================================
# SPATIAL DEVELOPER CNN (20x20 Grid Output)
# ============================================================
class SpatialDeveloperCNN(nn.Module):
    """
    CNN that outputs a 20x20 spatial grid of land use decisions.
    Each cell is a probability distribution over 6 land use types.
    
    Architecture:
    - Encodes state and constraints
    - Upsamples to 20x20 spatial resolution
    - Outputs per-pixel land use logits
    """
    
    def __init__(self, state_dim: int, constraint_dim: int, num_land_types: int = 6):
        super().__init__()
        
        self.state_dim = state_dim
        self.constraint_dim = constraint_dim
        self.num_land_types = num_land_types
        
        # Encode state + constraints
        input_dim = state_dim + constraint_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Reshape to spatial format (4x4x16)
        self.to_spatial = nn.Linear(256, 4 * 4 * 64)
        
        # Upsample to 20x20
        self.deconv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16 -> 20x20 (pad to 20)
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Output land use logits per pixel
        self.land_use_head = nn.Conv2d(16, num_land_types, kernel_size=1)
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, state: torch.Tensor, constraints: torch.Tensor):
        """
        Args:
            state: [batch, state_dim]
            constraints: [batch, constraint_dim]
            
        Returns:
            land_logits: [batch, num_land_types, 20, 20]
            value: [batch, 1]
        """
        batch_size = state.shape[0]
        
        # Encode
        x = torch.cat([state, constraints], dim=-1)
        features = self.encoder(x)
        
        # Value estimate
        value = self.value_head(features)
        
        # To spatial
        spatial = self.to_spatial(features)
        spatial = spatial.view(batch_size, 64, 4, 4)
        
        # Upsample
        spatial = self.deconv(spatial)
        
        # Ensure 20x20 output
        if spatial.shape[2] != 20 or spatial.shape[3] != 20:
            spatial = F.interpolate(spatial, size=(20, 20), mode='bilinear', align_corners=False)
        
        # Land use logits
        land_logits = self.land_use_head(spatial)
        
        return land_logits, value
    
    def get_action(self, state: torch.Tensor, constraints: torch.Tensor, 
                   deterministic: bool = False, 
                   temperature: float = 1.0) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample spatial action from policy with temperature-based exploration.
        
        Args:
            state: [batch, state_dim] tensor
            constraints: [batch, 5] government constraints
            deterministic: if True, use argmax instead of sampling
            temperature: exploration temperature (higher = more random, lower = more deterministic)
                        Each zone can have different temperature for diverse exploration
        
        Returns:
            land_use_grid: [20, 20] numpy array of land use types
            log_prob: scalar tensor
            value: scalar tensor
            entropy: scalar tensor (for logging)
        """
        land_logits, value = self.forward(state, constraints)
        
        # [batch, num_types, 20, 20] -> [batch, 20, 20, num_types]
        land_logits = land_logits.permute(0, 2, 3, 1)
        
        # Apply temperature scaling for exploration diversity
        # Higher temperature = softer distribution = more exploration
        scaled_logits = land_logits / max(0.1, temperature)
        probs = F.softmax(scaled_logits, dim=-1)
        
        if deterministic:
            # Argmax per pixel
            grid = probs.argmax(dim=-1)
        else:
            # Sample per pixel with temperature-scaled probabilities
            flat_probs = probs.view(-1, self.num_land_types)
            dist = torch.distributions.Categorical(flat_probs)
            flat_samples = dist.sample()
            grid = flat_samples.view(probs.shape[0], 20, 20)
        
        # Compute log probability
        flat_probs = probs.view(-1, self.num_land_types)
        flat_grid = grid.view(-1)
        dist = torch.distributions.Categorical(flat_probs)
        log_prob = dist.log_prob(flat_grid).view(probs.shape[0], 20, 20).sum(dim=[1, 2])
        
        # Compute entropy (for exploration bonus)
        entropy = dist.entropy().view(probs.shape[0], 20, 20).mean(dim=[1, 2])
        
        return grid.squeeze(0).cpu().numpy().astype(np.int32), log_prob, value, entropy


# ============================================================
# 4.1 MAPPO - Multi-Agent PPO with Centralized Critic
# ============================================================
class MAPPOGovernmentActor(nn.Module):
    """Individual government actor (one per pillar)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output mean and log_std for continuous actions
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.network(state)
        mean = torch.sigmoid(self.mean_head(features))  # [0, 1] range
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0, 1)
        
        return action
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class MAPPOCentralizedCritic(nn.Module):
    """Centralized critic that sees global state + all agent actions."""
    
    def __init__(self, global_state_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        input_dim = global_state_dim + total_action_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, global_state: torch.Tensor, all_actions: List[torch.Tensor]) -> torch.Tensor:
        actions_concat = torch.cat(all_actions, dim=-1)
        x = torch.cat([global_state, actions_concat], dim=-1)
        return self.network(x)


class MAPPOGovernment:
    """
    Multi-Agent PPO for 3 government pillars with centralized critic.
    Based on: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.num_agents = 3  # Economy, Environment, Society
        
        # Government uses BASE state (20 dims) - doesn't need zone-specific info
        gov_state_dim = config.base_state_dim
        
        # Each agent outputs different subset of constraints
        # Economy: max_density, tax_rate
        # Environment: green_space_pct, woodland_preservation
        # Society: affordable_housing_pct
        self.agent_action_dims = [2, 2, 1]  # Per-agent action dimensions
        
        # Decentralized actors (use base state)
        self.actors = nn.ModuleList([
            MAPPOGovernmentActor(gov_state_dim, dim).to(self.device)
            for dim in self.agent_action_dims
        ])
        
        # Centralized critic (also uses base state)
        total_action_dim = sum(self.agent_action_dims)  # 2 + 2 + 1 = 5
        self.critic = MAPPOCentralizedCritic(
            gov_state_dim, total_action_dim
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=config.gov_lr)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.gov_lr)
        
        # Constraint bounds for each agent
        self.constraint_bounds = {
            0: [('max_density', 0.50, 0.95), ('tax_rate', 0.0, 0.15)],
            1: [('green_space_pct', 0.05, 0.40), ('woodland_preservation', 0.0, 0.30)],
            2: [('affordable_housing_pct', 0.10, 0.50)],
        }
        
        # 1.2 Lagrangian constraints
        self.log_lambda_seg = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_lambda_pop = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.lambda_optimizer = optim.Adam([self.log_lambda_seg, self.log_lambda_pop], 
                                            lr=config.lagrange_lr)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[GovernmentConstraints, List[torch.Tensor], torch.Tensor]:
        """Select actions from all agents and combine into constraints."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        actions = []
        for i, actor in enumerate(self.actors):
            action = actor.get_action(state_t, deterministic=not training)
            actions.append(action)
        
        # Get centralized value
        value = self.critic(state_t, actions)
        
        # Combine actions into constraints
        constraints = self._actions_to_constraints(actions)
        
        return constraints, actions, value
    
    def _actions_to_constraints(self, actions: List[torch.Tensor]) -> GovernmentConstraints:
        """Convert per-agent actions to full constraint set."""
        # Agent 0: max_density, tax_rate
        max_density = 0.50 + actions[0][0, 0].item() * 0.45
        tax_rate = actions[0][0, 1].item() * 0.15
        
        # Agent 1: green_space_pct, woodland_preservation
        green_space_pct = 0.05 + actions[1][0, 0].item() * 0.35
        woodland_preservation = actions[1][0, 1].item() * 0.30
        
        # Agent 2: affordable_housing_pct
        affordable_housing_pct = 0.10 + actions[2][0, 0].item() * 0.40
        
        return GovernmentConstraints(
            affordable_housing_pct=affordable_housing_pct,
            green_space_pct=green_space_pct,
            woodland_preservation=woodland_preservation,
            max_density=max_density,
            tax_rate=tax_rate
        )
    
    def compute_ppo_loss(self, 
                         states: torch.Tensor,
                         actions: List[torch.Tensor],
                         old_log_probs: List[torch.Tensor],
                         advantages: torch.Tensor,
                         returns: torch.Tensor,
                         cost_advantages: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute PPO loss with Lagrangian constraints."""
        
        total_actor_loss = 0
        total_entropy = 0
        
        for i, actor in enumerate(self.actors):
            log_prob, entropy = actor.evaluate_action(states, actions[i])
            ratio = torch.exp(log_prob - old_log_probs[i])
            
            # Clipped objective
            clipped_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            total_actor_loss += policy_loss
            total_entropy += entropy.mean()
        
        # Value loss
        value_pred = self.critic(states, actions)
        value_loss = F.mse_loss(value_pred.squeeze(), returns)
        
        # Lagrangian constraint term (1.2)
        lagrange_loss = 0
        if cost_advantages is not None:
            lambda_seg = torch.exp(self.log_lambda_seg)
            lambda_pop = torch.exp(self.log_lambda_pop)
            
            # Segregation constraint
            lagrange_loss += lambda_seg * (cost_advantages[:, 0].mean() - self.config.segregation_limit)
            # Population constraint
            lagrange_loss += lambda_pop * (self.config.min_population - cost_advantages[:, 1].mean())
        
        # Combined loss
        total_loss = (
            total_actor_loss / self.num_agents +
            self.config.value_coef * value_loss -
            self.config.entropy_coef * total_entropy / self.num_agents +
            lagrange_loss
        )
        
        return {
            'total_loss': total_loss,
            'actor_loss': total_actor_loss / self.num_agents,
            'value_loss': value_loss,
            'entropy': total_entropy / self.num_agents,
            'lagrange_loss': lagrange_loss if isinstance(lagrange_loss, torch.Tensor) else torch.tensor(0.0),
        }
    
    def update_lagrange_multipliers(self, avg_segregation: float, avg_population: float):
        """Dual ascent for Lagrange multipliers."""
        self.lambda_optimizer.zero_grad()
        
        # Gradient for segregation (want to minimize)
        seg_violation = avg_segregation - self.config.segregation_limit
        lambda_seg_loss = -self.log_lambda_seg * seg_violation
        
        # Gradient for population (want to maximize)
        pop_violation = self.config.min_population - avg_population
        lambda_pop_loss = -self.log_lambda_pop * pop_violation
        
        total_lambda_loss = lambda_seg_loss + lambda_pop_loss
        total_lambda_loss.backward()
        self.lambda_optimizer.step()


# ============================================================
# 4.2 COMMUNICATION BETWEEN AGENTS
# ============================================================
class AgentCommunication(nn.Module):
    """
    Attention-based communication between government agents.
    Based on: CommNet, TarMAC
    """
    
    def __init__(self, hidden_dim: int = 64, num_agents: int = 3, num_rounds: int = 2):
        super().__init__()
        
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
            for _ in range(num_rounds)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_rounds)
        ])
    
    def forward(self, agent_hiddens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_hiddens: [batch, num_agents, hidden_dim]
        Returns:
            communicated: [batch, num_agents, hidden_dim]
        """
        h = agent_hiddens
        
        for attn, norm in zip(self.attention_layers, self.norm_layers):
            h_comm, _ = attn(h, h, h)
            h = norm(h + h_comm)  # Residual
        
        return h


# ============================================================
# 2.1 PREDICTIVE CONTRIBUTION MEASUREMENT (PCM)
# ============================================================
class PredictiveContributionMeasurement(nn.Module):
    """
    Measures each agent's contribution via state prediction errors.
    Based on: "Credit assignment with predictive contribution measurement" (Neural Networks, 2023)
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 128):
        super().__init__()
        
        self.num_agents = num_agents
        
        # One predictor per agent
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )
            for _ in range(num_agents)
        ])
    
    def compute_contributions(self, 
                             states: torch.Tensor, 
                             actions: List[torch.Tensor], 
                             next_states: torch.Tensor) -> torch.Tensor:
        """
        Compute contribution of each agent based on prediction errors.
        Higher prediction accuracy with agent's action = higher contribution.
        
        Returns:
            contributions: [batch, num_agents] normalized to sum to 1
        """
        errors = []
        
        for i, predictor in enumerate(self.predictors):
            agent_input = torch.cat([states, actions[i]], dim=-1)
            predicted = predictor(agent_input)
            error = F.mse_loss(predicted, next_states, reduction='none').mean(dim=-1)
            errors.append(error)
        
        errors = torch.stack(errors, dim=-1)  # [batch, num_agents]
        
        # Lower error = higher contribution
        # Use inverse softmax
        contributions = F.softmax(-errors, dim=-1)
        
        return contributions
    
    def assign_credit(self, global_reward: torch.Tensor, contributions: torch.Tensor) -> torch.Tensor:
        """Distribute global reward based on contributions."""
        return global_reward.unsqueeze(-1) * contributions


# ============================================================
# 2.3 PER-AGENT INTRINSIC REWARDS
# ============================================================
class PillarIntrinsicRewards:
    """Each government agent gets immediate feedback on their pillar."""
    
    def __init__(self):
        self.pillar_weights = {
            'economy': {'gdp_growth': 0.4, 'tax_revenue': 0.3, 'employment': 0.3},
            'environment': {'green_ratio': 0.4, 'woodland_ratio': 0.3, 'pollution_reduction': 0.3},
            'society': {'segregation_delta': 0.5, 'qol_delta': 0.3, 'affordability': 0.2}
        }
    
    def compute_intrinsic_reward(self, agent_type: str, prev_metrics, curr_metrics) -> float:
        """Immediate per-step reward for each agent."""
        weights = self.pillar_weights[agent_type]
        reward = 0.0
        
        if agent_type == 'economy':
            gdp_growth = (curr_metrics.gdp - prev_metrics.gdp) / max(1, prev_metrics.gdp)
            tax_growth = (curr_metrics.tax_revenue - prev_metrics.tax_revenue) / max(1e6, prev_metrics.tax_revenue)
            emp_change = prev_metrics.unemployment_rate - curr_metrics.unemployment_rate
            
            reward = (weights['gdp_growth'] * np.clip(gdp_growth, -1, 1) +
                      weights['tax_revenue'] * np.clip(tax_growth, -1, 1) +
                      weights['employment'] * np.clip(emp_change, -0.5, 0.5))
            
        elif agent_type == 'environment':
            green_delta = curr_metrics.green_space_ratio - prev_metrics.green_space_ratio
            wood_delta = curr_metrics.woodland_ratio - prev_metrics.woodland_ratio
            poll_delta = prev_metrics.avg_pollution - curr_metrics.avg_pollution
            
            reward = (weights['green_ratio'] * green_delta * 10 +
                      weights['woodland_ratio'] * wood_delta * 10 +
                      weights['pollution_reduction'] * np.clip(poll_delta, -1, 1))
            
        elif agent_type == 'society':
            # CRITICAL: Immediate segregation feedback
            seg_delta = prev_metrics.segregation_index - curr_metrics.segregation_index
            qol_delta = (curr_metrics.avg_qol - prev_metrics.avg_qol) / 5.0
            afford = curr_metrics.poor_housing / max(1, curr_metrics.total_housing)
            
            reward = (weights['segregation_delta'] * seg_delta * 5 +
                      weights['qol_delta'] * qol_delta +
                      weights['affordability'] * max(0, afford - 0.2))
        
        return reward


# ============================================================
# 2.2 SPATIAL-TEMPORAL RETURN DECOMPOSITION (STAS)
# ============================================================
class SpatialTemporalAttribution(nn.Module):
    """
    Spatial-Temporal Action Shaping for MARL credit assignment.
    Based on: "Credit Assignment in Multi-Agent RL" (ICML 2022)
    
    Decomposes global rewards to individual agents based on:
    1. Spatial influence (which zones each agent affected)
    2. Temporal influence (delayed effects of actions)
    """
    
    def __init__(self, state_dim: int, num_agents: int, num_zones: int = 9, hidden_dim: int = 128):
        super().__init__()
        
        self.num_agents = num_agents
        self.num_zones = num_zones
        
        # Spatial attention: which agents influenced which zones
        self.spatial_attention = nn.Sequential(
            nn.Linear(state_dim + num_agents * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * num_zones),
        )
        
        # Temporal attention: how much did past actions matter
        self.temporal_encoder = nn.GRU(
            input_size=state_dim + num_agents * 5,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.temporal_attention = nn.Linear(hidden_dim, num_agents)
        
    def compute_spatial_credit(self, state: torch.Tensor, actions: List[torch.Tensor], 
                                zone_rewards: torch.Tensor) -> torch.Tensor:
        """
        Attribute zone-level rewards to agents based on spatial influence.
        
        Args:
            state: [batch, state_dim]
            actions: list of [batch, action_dim] for each agent
            zone_rewards: [batch, num_zones]
            
        Returns:
            agent_rewards: [batch, num_agents]
        """
        actions_flat = torch.cat(actions, dim=-1)
        x = torch.cat([state, actions_flat], dim=-1)
        
        # Compute spatial attention weights [batch, agents, zones]
        attn = self.spatial_attention(x)
        attn = attn.view(-1, self.num_agents, self.num_zones)
        attn = F.softmax(attn, dim=1)  # Normalize over agents
        
        # Weight zone rewards by attention
        # [batch, agents, zones] x [batch, zones, 1] -> [batch, agents]
        agent_rewards = torch.bmm(attn, zone_rewards.unsqueeze(-1)).squeeze(-1)
        
        return agent_rewards
    
    def compute_temporal_credit(self, trajectory_states: torch.Tensor, 
                                 trajectory_actions: List[torch.Tensor],
                                 final_reward: torch.Tensor) -> torch.Tensor:
        """
        Attribute final episode reward to agents based on temporal contribution.
        
        Args:
            trajectory_states: [batch, timesteps, state_dim]
            trajectory_actions: [batch, timesteps, agents * action_dim]
            final_reward: [batch]
            
        Returns:
            agent_credits: [batch, num_agents]
        """
        actions_cat = torch.cat(trajectory_actions, dim=-1)
        x = torch.cat([trajectory_states, actions_cat], dim=-1)
        
        # Encode trajectory
        _, hidden = self.temporal_encoder(x)
        hidden = hidden.squeeze(0)  # [batch, hidden_dim]
        
        # Compute temporal attention
        attn = F.softmax(self.temporal_attention(hidden), dim=-1)  # [batch, agents]
        
        # Distribute reward
        agent_credits = attn * final_reward.unsqueeze(-1)
        
        return agent_credits


# ============================================================
# 3.3 POPULATION-BASED TRAINING (PBT)
# ============================================================
class PopulationMember:
    """Single member of the population with its own hyperparameters."""
    
    def __init__(self, member_id: int, config: 'Config', device: str):
        self.id = member_id
        self.device = device
        
        # Mutable hyperparameters
        self.lr = config.gov_lr * np.random.uniform(0.5, 2.0)
        self.entropy_coef = config.entropy_coef * np.random.uniform(0.5, 2.0)
        self.gamma = config.gamma * np.random.uniform(0.95, 1.0)
        
        # Performance tracking
        self.fitness = 0.0
        self.generations = 0
        
    def exploit(self, other: 'PopulationMember'):
        """Copy hyperparameters from a better-performing member."""
        self.lr = other.lr
        self.entropy_coef = other.entropy_coef
        self.gamma = other.gamma
    
    def explore(self, perturb_factor: float = 0.2):
        """Randomly perturb hyperparameters."""
        self.lr *= np.random.uniform(1 - perturb_factor, 1 + perturb_factor)
        self.entropy_coef *= np.random.uniform(1 - perturb_factor, 1 + perturb_factor)
        self.gamma = np.clip(self.gamma * np.random.uniform(0.99, 1.01), 0.9, 0.999)


class PopulationBasedTraining:
    """
    Train a population of agents in parallel with hyperparameter exploration.
    Based on: Jaderberg et al. "Population Based Training of Neural Networks"
    """
    
    def __init__(self, config: 'Config', population_size: int = 8):
        self.config = config
        self.population_size = population_size
        self.population = [
            PopulationMember(i, config, config.device) 
            for i in range(population_size)
        ]
        
        self.generation = 0
        self.exploit_every = 50  # Episodes between exploit/explore
        self.truncation_pct = 0.2  # Bottom 20% copies from top 20%
    
    def ready_for_evolution(self) -> bool:
        """Check if it's time to do exploit/explore."""
        return self.generation > 0 and self.generation % self.exploit_every == 0
    
    def evolve(self):
        """Perform population evolution: exploit + explore."""
        # Sort by fitness
        self.population.sort(key=lambda m: m.fitness, reverse=True)
        
        n_truncate = max(1, int(self.population_size * self.truncation_pct))
        
        # Bottom performers copy from top performers
        for i in range(n_truncate):
            bottom_idx = self.population_size - 1 - i
            top_idx = i
            
            self.population[bottom_idx].exploit(self.population[top_idx])
            self.population[bottom_idx].explore()
            self.population[bottom_idx].fitness = 0
            self.population[bottom_idx].generations = 0
        
        self.generation += 1
    
    def get_best_member(self) -> PopulationMember:
        return max(self.population, key=lambda m: m.fitness)


# ============================================================
# 4.3 HIERARCHICAL RL (FeUdal Networks Inspired)
# ============================================================
class HierarchicalGovernment(nn.Module):
    """
    Two-level hierarchy:
    - Manager: Sets high-level goals (target segregation, growth rate)
    - Workers: Three pillar agents execute to achieve goals
    
    Based on: FeUdal Networks (Vezhnevets et al., ICML 2017)
    """
    
    def __init__(self, state_dim: int, goal_dim: int = 8, hidden_dim: int = 128, device: str = 'cpu'):
        super().__init__()
        
        self.goal_dim = goal_dim
        self.device = device
        
        # Manager: operates at slow timescale, sets goals
        self.manager = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.manager_goal_head = nn.Linear(hidden_dim, goal_dim)
        self.manager_value_head = nn.Linear(hidden_dim, 1)
        
        # Manager operates at c-step intervals
        self.manager_horizon = 3  # Manager sets goals every 3 zones
        
        # Workers: condition on manager goals
        self.economy_worker = self._make_worker(state_dim + goal_dim, 2)  # max_density, tax_rate
        self.environment_worker = self._make_worker(state_dim + goal_dim, 2)  # green, woodland
        self.society_worker = self._make_worker(state_dim + goal_dim, 1)  # affordable
        
        # Goal embedding for transition model
        self.goal_transition = nn.GRU(
            input_size=state_dim,
            hidden_size=goal_dim,
            num_layers=1,
            batch_first=True
        )
    
    def _make_worker(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    
    def get_manager_goal(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Manager sets goal direction."""
        features = self.manager(state)
        goal = self.manager_goal_head(features)
        goal = F.normalize(goal, dim=-1)  # Unit direction
        value = self.manager_value_head(features)
        return goal, value
    
    def get_worker_actions(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Workers produce actions conditioned on goal."""
        x = torch.cat([state, goal], dim=-1)
        
        econ_action = self.economy_worker(x)
        env_action = self.environment_worker(x)
        soc_action = self.society_worker(x)
        
        return econ_action, env_action, soc_action
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor = None, 
                step: int = 0) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: [batch, state_dim]
            goal: [batch, goal_dim] - current goal from manager (optional)
            step: current zone step in episode
            
        Returns:
            Dict with goal, worker actions, manager value
        """
        # Check if manager should set new goal
        if goal is None or step % self.manager_horizon == 0:
            goal, manager_value = self.get_manager_goal(state)
        else:
            manager_value = None
        
        # Workers produce actions
        econ, env, soc = self.get_worker_actions(state, goal)
        
        return {
            'goal': goal,
            'economy_action': econ,
            'environment_action': env,
            'society_action': soc,
            'manager_value': manager_value
        }
    
    def compute_intrinsic_reward(self, state_delta: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Intrinsic reward for workers: did state move toward goal?
        Cosine similarity between state change and goal direction.
        """
        state_direction = F.normalize(state_delta, dim=-1)
        reward = F.cosine_similarity(state_direction, goal, dim=-1)
        return reward


# ============================================================
# 5.3 WEIGHT SPACE EXPLORATION (Complete)
# ============================================================
class WeightSpaceExplorer:
    """
    Systematically explore the weight space to map Pareto frontier.
    Uses 66 weight combinations for 3 objectives with n_grid=10.
    """
    
    def __init__(self, n_grid: int = 10):
        self.n_grid = n_grid
        self.weights = self._generate_simplex_grid()
        self.results = {}
        self.current_idx = 0
    
    def _generate_simplex_grid(self) -> List[Tuple[float, float, float]]:
        """Generate uniform samples on the 2-simplex."""
        weights = []
        for i in range(self.n_grid + 1):
            for j in range(self.n_grid + 1 - i):
                k = self.n_grid - i - j
                w_econ = i / self.n_grid
                w_env = j / self.n_grid
                w_soc = k / self.n_grid
                weights.append((w_econ, w_env, w_soc))
        return weights
    
    def get_next_weights(self) -> Optional[Tuple[float, float, float]]:
        """Get next weight combination to try."""
        if self.current_idx >= len(self.weights):
            return None
        weights = self.weights[self.current_idx]
        self.current_idx += 1
        return weights
    
    def record_result(self, weights: Tuple[float, float, float], 
                      metrics: Dict[str, float]):
        """Record results for a weight combination."""
        self.results[weights] = metrics
    
    def get_pareto_frontier(self) -> List[Tuple[Tuple[float, float, float], Dict]]:
        """Extract Pareto-optimal weight configurations."""
        frontier = []
        
        for w1, m1 in self.results.items():
            dominated = False
            for w2, m2 in self.results.items():
                if w1 == w2:
                    continue
                # Check if m2 dominates m1
                if (m2['economy'] >= m1['economy'] and 
                    m2['environment'] >= m1['environment'] and 
                    m2['society'] >= m1['society'] and
                    (m2['economy'] > m1['economy'] or 
                     m2['environment'] > m1['environment'] or 
                     m2['society'] > m1['society'])):
                    dominated = True
                    break
            
            if not dominated:
                frontier.append((w1, m1))
        
        return frontier
    
    def export_results(self, filepath: str):
        """Export all results for analysis."""
        import json
        
        export_data = {
            'weights': self.weights,
            'results': {str(k): v for k, v in self.results.items()},
            'pareto_frontier': [(str(w), m) for w, m in self.get_pareto_frontier()],
            'n_grid': self.n_grid,
            'total_combinations': len(self.weights)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


def run_weight_space_exploration(config: 'Config', episodes_per_weight: int = 100) -> Dict:
    """
    Run full weight space exploration experiment.
    
    This trains a government agent with each weight combination
    to map the Pareto frontier of achievable objectives.
    """
    explorer = WeightSpaceExplorer(n_grid=10)
    
    print("=" * 80)
    print("WEIGHT SPACE EXPLORATION")
    print(f"Total combinations: {len(explorer.weights)}")
    print(f"Episodes per weight: {episodes_per_weight}")
    print("=" * 80)
    
    while True:
        weights = explorer.get_next_weights()
        if weights is None:
            break
        
        w_econ, w_env, w_soc = weights
        print(f"\n[{explorer.current_idx}/{len(explorer.weights)}] "
              f"Economy={w_econ:.1f}, Environment={w_env:.1f}, Society={w_soc:.1f}")
        
        # Create agent with these weights
        gov = SingleAgentGovernment(
            state_dim=config.state_dim,
            weights={'economy': w_econ, 'environment': w_env, 'society': w_soc},
            device=config.device
        )
        
        # Train for episodes_per_weight episodes
        metrics_sum = {'gdp': 0, 'green': 0, 'segregation': 0, 'qol': 0, 'population': 0}
        
        for ep in range(episodes_per_weight):
            env = UrbanSimulation(map_size=config.map_size, zone_size=config.zone_size)
            
            for zone_id in range(config.num_zones):
                state = env.get_state_vector()
                constraints = gov.select_action(state)
                decision = DeveloperDecision()  # Default
                result = env.develop_zone(zone_id, constraints, decision)
                
                if env.game_state != GameState.RUNNING:
                    break
            
            # Accumulate metrics
            metrics_sum['gdp'] += env.metrics.gdp
            metrics_sum['green'] += env.metrics.green_space_ratio
            metrics_sum['segregation'] += env.metrics.segregation_index
            metrics_sum['qol'] += env.metrics.avg_qol
            metrics_sum['population'] += env.metrics.population
        
        # Average
        final_metrics = {
            'economy': metrics_sum['gdp'] / episodes_per_weight / 1e8,  # Normalized
            'environment': metrics_sum['green'] / episodes_per_weight,
            'society': 1 - metrics_sum['segregation'] / episodes_per_weight,  # Invert
            'qol': metrics_sum['qol'] / episodes_per_weight,
            'population': metrics_sum['population'] / episodes_per_weight
        }
        
        explorer.record_result(weights, final_metrics)
        print(f"  Results: GDP={final_metrics['economy']:.2f}, "
              f"Green={final_metrics['environment']:.2%}, "
              f"AntiSeg={final_metrics['society']:.2%}")
    
    # Export results
    explorer.export_results(str(config.exp_dir / 'weight_space_results.json') if hasattr(config, 'exp_dir') 
                           else 'weight_space_results.json')
    
    frontier = explorer.get_pareto_frontier()
    print(f"\nPareto frontier has {len(frontier)} solutions")
    
    return explorer.results


# ============================================================
# 3.1 INTRINSIC CURIOSITY MODULE (ICM)
# ============================================================
class IntrinsicCuriosityModule(nn.Module):
    """
    Curiosity-driven exploration via prediction errors.
    Based on: Pathak et al. (ICML 2017)
    """
    
    def __init__(self, state_dim: int, action_dim: int, feature_dim: int = 64):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Inverse model: predict action from state transition
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Forward model: predict next state features
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
    
    def compute_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor, 
                                  next_state: torch.Tensor, scale: float = 0.01) -> torch.Tensor:
        """Intrinsic reward = forward model prediction error."""
        with torch.no_grad():
            phi_s = self.encoder(state)
            phi_s_next = self.encoder(next_state)
            
            pred_phi_next = self.forward_model(torch.cat([phi_s, action], dim=-1))
            
            error = F.mse_loss(pred_phi_next, phi_s_next, reduction='none').mean(dim=-1)
            intrinsic_reward = error * scale
        
        return intrinsic_reward
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, 
               next_states: torch.Tensor) -> float:
        """Train forward and inverse models."""
        phi_s = self.encoder(states)
        phi_s_next = self.encoder(next_states)
        
        # Inverse loss
        pred_actions = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=-1))
        inverse_loss = F.mse_loss(pred_actions, actions)
        
        # Forward loss
        pred_phi_next = self.forward_model(torch.cat([phi_s, actions], dim=-1))
        forward_loss = F.mse_loss(pred_phi_next, phi_s_next.detach())
        
        total_loss = inverse_loss + forward_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


# ============================================================
# 3.2 ADAPTIVE ENTROPY COEFFICIENT
# ============================================================
class AdaptiveEntropyCoefficient:
    """Automatically adjusts entropy coefficient based on policy entropy."""
    
    def __init__(self, target_entropy: float, lr: float = 0.001, device: str = 'cpu'):
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)
    
    def update(self, entropy: torch.Tensor) -> float:
        """Update alpha based on current entropy."""
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()
        
        return self.alpha.item()


# ============================================================
# 5.1 BASELINE MODELS
# ============================================================
class SingleAgentGovernment(nn.Module):
    """Baseline A: Single agent with weighted sum objective."""
    
    def __init__(self, state_dim: int, weights: Dict[str, float] = None, device: str = 'cpu'):
        super().__init__()
        
        self.weights = weights or {'economy': 0.33, 'environment': 0.33, 'society': 0.34}
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Sigmoid()
        ).to(device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        
        # Hard constraint bounds
        self.bounds = [
            (0.10, 0.50),  # affordable_housing_pct
            (0.05, 0.40),  # green_space_pct
            (0.00, 0.30),  # woodland_preservation
            (0.50, 0.95),  # max_density
            (0.00, 0.15),  # tax_rate
        ]
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.network(state)
        # Scale to constraint bounds
        scaled = torch.zeros_like(raw)
        for i, (low, high) in enumerate(self.bounds):
            scaled[:, i] = low + raw[:, i] * (high - low)
        return scaled
    
    def select_action(self, state: np.ndarray) -> GovernmentConstraints:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.forward(state_t).squeeze(0)
        
        return GovernmentConstraints(
            affordable_housing_pct=action[0].item(),
            green_space_pct=action[1].item(),
            woodland_preservation=action[2].item(),
            max_density=action[3].item(),
            tax_rate=action[4].item()
        )


class FixedPolicyGovernment:
    """Baseline B: Fixed policy (no learning)."""
    
    def select_action(self, state: np.ndarray = None) -> GovernmentConstraints:
        return GovernmentConstraints(
            affordable_housing_pct=0.25,
            green_space_pct=0.15,
            woodland_preservation=0.10,
            max_density=0.60,
            tax_rate=0.10
        )


class RandomPolicyGovernment:
    """Baseline C: Random policy."""
    
    def select_action(self, state: np.ndarray = None) -> GovernmentConstraints:
        return GovernmentConstraints(
            affordable_housing_pct=np.random.uniform(0.10, 0.50),
            green_space_pct=np.random.uniform(0.05, 0.40),
            woodland_preservation=np.random.uniform(0.00, 0.30),
            max_density=np.random.uniform(0.50, 0.95),
            tax_rate=np.random.uniform(0.00, 0.15)
        )


class NoGovernment:
    """Baseline D: No government constraints."""
    
    def select_action(self, state: np.ndarray = None) -> GovernmentConstraints:
        return GovernmentConstraints(
            affordable_housing_pct=0.0,
            green_space_pct=0.0,
            woodland_preservation=0.0,
            max_density=1.0,
            tax_rate=0.0
        )


# ============================================================
# 5.2 PARETO HYPERVOLUME CALCULATION
# ============================================================
class ParetoAnalysis:
    """Compute hypervolume to measure quality of Pareto frontier."""
    
    def __init__(self, reference_point: Tuple[float, float, float] = (0, 0, 0)):
        self.reference_point = np.array(reference_point)
    
    def compute_hypervolume(self, solutions: List[Tuple[float, float, float]]) -> float:
        """
        Compute hypervolume indicator.
        Solutions: list of (economy_score, environment_score, society_score)
        """
        try:
            from pymoo.indicators.hv import HV
            front = -np.array(solutions)  # Negate for minimization
            ref = -self.reference_point
            hv = HV(ref_point=ref)
            return hv(front)
        except ImportError:
            # Fallback: simple dominated hypervolume approximation
            return self._approximate_hypervolume(solutions)
    
    def _approximate_hypervolume(self, solutions: List[Tuple]) -> float:
        """Simple hypervolume approximation if pymoo not available."""
        if not solutions:
            return 0.0
        
        solutions = np.array(solutions)
        # Sum of products above reference
        hv = 0.0
        for sol in solutions:
            vol = np.prod(np.maximum(0, sol - self.reference_point))
            hv += vol
        
        return hv / len(solutions)  # Average


# ============================================================
# 5.3 WEIGHT SPACE EXPLORATION
# ============================================================
def generate_weight_samples(n_grid: int = 10) -> List[Tuple[float, float, float]]:
    """
    Generate uniform samples on the 2-simplex.
    For n_grid=10: (10+2 choose 2) = 66 combinations
    """
    weights = []
    
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            w_econ = i / n_grid
            w_env = j / n_grid
            w_soc = k / n_grid
            weights.append((w_econ, w_env, w_soc))
    
    return weights


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    """Stop training if reward doesn't improve for patience episodes."""
    
    def __init__(self, patience: int = 200, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
    
    def check(self, reward: float) -> bool:
        """Returns True if training should stop."""
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        return self.episodes_without_improvement >= self.patience
    
    def reset(self):
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0


# ============================================================
# EXPERIENCE BUFFER
# ============================================================
class ExperienceBuffer:
    """Buffer for storing episode experience for PPO updates."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.gov_actions = []
        self.dev_actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.prev_metrics = []
        self.curr_metrics = []
    
    def add(self, state, gov_actions, dev_action, reward, value, log_prob, done,
            prev_metrics=None, curr_metrics=None):
        self.states.append(state)
        self.gov_actions.append(gov_actions)
        self.dev_actions.append(dev_action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.prev_metrics.append(prev_metrics)
        self.curr_metrics.append(curr_metrics)
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float, 
                                        last_value: float = 0) -> Tuple[List, List]:
        """Compute GAE advantages and returns."""
        rewards = self.rewards + [last_value]
        values = self.values + [last_value]
        dones = self.dones + [True]
        
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return returns, advantages
    
    def __len__(self):
        return len(self.states)


# ============================================================
# MAIN TRAINER
# ============================================================
class FullSystemTrainer:
    """Complete training system with the components listed in the module docstring."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        print(f"Initializing FullSystemTrainer on {self.device}")
        
        # Environment
        self.env = UrbanSimulation(map_size=config.map_size, zone_size=config.zone_size)
        
        # Agents
        print("Creating MAPPO Government...")
        self.government = MAPPOGovernment(config)
        
        print("Creating Spatial Developer CNN...")
        self.developer = SpatialDeveloperCNN(
            state_dim=config.state_dim,
            constraint_dim=config.gov_action_dim,
            num_land_types=config.num_land_types
        ).to(self.device)
        self.dev_optimizer = optim.Adam(self.developer.parameters(), lr=config.dev_lr)
        
        # Communication module (4.2)
        self.communication = AgentCommunication(hidden_dim=64, num_agents=3).to(self.device)
        
        # Credit assignment (2.1) - uses base state
        self.pcm = PredictiveContributionMeasurement(
            state_dim=config.base_state_dim,  # Use base state (20 dims)
            action_dim=2,  # Per-agent action dim
            num_agents=3
        ).to(self.device)
        self.pcm_optimizer = optim.Adam(self.pcm.parameters(), lr=1e-4)
        
        # Intrinsic rewards (2.3)
        self.pillar_rewards = PillarIntrinsicRewards()
        
        # Curiosity module (3.1) - uses BASE state for efficiency
        self.icm = IntrinsicCuriosityModule(
            state_dim=config.base_state_dim,  # Use base state (20 dims)
            action_dim=config.gov_action_dim,
            feature_dim=config.curiosity_feature_dim
        ).to(self.device)
        
        # Adaptive entropy (3.2)
        self.adaptive_entropy = AdaptiveEntropyCoefficient(
            target_entropy=config.target_entropy,
            lr=config.entropy_lr,
            device=self.device
        )
        
        # Reward normalization (1.3)
        self.reward_normalizer = RewardNormalizer()
        
        # Pareto analysis (5.2)
        self.pareto = ParetoAnalysis(reference_point=(0, 0, 0))
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stop_patience,
            min_delta=config.early_stop_min_delta
        )
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Metrics tracking
        self.metrics_history = []
        self.pareto_solutions = []
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f'experiments/full_system/{timestamp}')
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        print(f"Experiment directory: {self.exp_dir}")
    
    def run_episode(self) -> Dict:
        """Run a single episode with full system."""
        # Reset environment
        self.env = UrbanSimulation(map_size=self.config.map_size, zone_size=self.config.zone_size)
        self.buffer.clear()
        
        episode_reward = 0
        episode_profit = 0
        prev_metrics = None
        prev_seg = 0.0
        
        # Generate random temperature per zone for diverse exploration (NOISE1)
        # Each zone gets a different exploration level per episode
        zone_temperatures = np.random.uniform(0.5, 2.0, size=self.config.num_zones)
        
        for zone_id in range(self.config.num_zones):
            # Government uses BASE state (20 dims)
            gov_state = self.env.get_state_vector()
            
            # Developer uses ENHANCED state with zone info (113 dims)
            dev_state = self.env.get_developer_state(zone_id)
            dev_state_t = torch.tensor(dev_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Store previous metrics for delta calculation
            if prev_metrics is None:
                prev_metrics = self.env.metrics
            
            # Government action (MAPPO) - uses base state
            constraints, gov_actions, gov_value = self.government.select_action(gov_state, training=True)
            
            # Convert constraints to tensor for developer
            constraints_t = torch.tensor([
                constraints.affordable_housing_pct,
                constraints.green_space_pct,
                constraints.woodland_preservation,
                constraints.max_density,
                constraints.tax_rate
            ], dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Developer action (Spatial CNN) - uses enhanced state with zone info
            # Use per-zone temperature for diverse exploration
            zone_temp = zone_temperatures[zone_id]
            land_grid, dev_log_prob, dev_value, dev_entropy = self.developer.get_action(
                dev_state_t, constraints_t, deterministic=False, temperature=zone_temp
            )
            
            # Create developer decision
            decision = DeveloperDecision(land_use_grid=land_grid)
            
            # Execute in environment
            result = self.env.develop_zone(zone_id, constraints, decision)
            
            # Get current metrics
            curr_metrics = self.env.metrics
            
            # === REWARD CALCULATION (shaped against degenerate behaviour) ===
            
            # Get development statistics
            profit = result.get('first_year_profit', 0)
            episode_profit += profit
            
            # Count what was built vs zone capacity
            zone_pixels = 400  # 20x20
            housing_built = result.get('poor_built', 0) + result.get('medium_built', 0) + result.get('luxury_built', 0)
            green_built = result.get('green_built', 0) + result.get('woodland_built', 0)
            total_developed = housing_built + green_built
            development_rate = total_developed / zone_pixels
            
            # Get occupancy metrics
            occupied = result.get('occupied_poor', 0) + result.get('occupied_medium', 0) + result.get('occupied_luxury', 0)
            occupancy_rate = occupied / max(1, housing_built)
            
            # Housing type diversity (penalize building only one type)
            poor_built = result.get('poor_built', 0)
            medium_built = result.get('medium_built', 0)
            luxury_built = result.get('luxury_built', 0)
            poor_pct = poor_built / max(1, housing_built)
            medium_pct = medium_built / max(1, housing_built)
            luxury_pct = luxury_built / max(1, housing_built)
            
            # Entropy-based diversity (higher = more balanced)
            diversity = 1 - (poor_pct**2 + medium_pct**2 + luxury_pct**2)  # 0 = mono, 0.67 = balanced
            
            # DEMAND MATCHING: Immigrants want 20% poor, 60% medium, 20% luxury
            # Penalty for mismatch with demand distribution
            target_poor, target_medium, target_luxury = 0.2, 0.6, 0.2
            demand_mismatch = abs(poor_pct - target_poor) + abs(medium_pct - target_medium) + abs(luxury_pct - target_luxury)
            demand_penalty = -0.3 * demand_mismatch  # Penalize mismatch up to -0.6
            
            # Population change
            pop_change = curr_metrics.population - prev_metrics.population
            
            # === DEVELOPER REWARD ===

            # 1: Minimum development requirement - penalize zones >50% empty
            underdevelopment_penalty = 0.0
            if development_rate < 0.5:
                underdevelopment_penalty = -2.0 * (0.5 - development_rate)  # -1.0 if totally empty
            
            # 2: Occupancy is MULTIPLICATIVE with profit
            # Zero occupancy = zero reward regardless of cost savings
            profit_component = np.clip(profit / 200000, -2, 2)
            occupancy_multiplier = max(0.1, occupancy_rate)  # At least 0.1 so negative profits still hurt
            profit_reward = profit_component * occupancy_multiplier
            
            # 3: Population growth term for the developer
            pop_growth_reward = np.clip(pop_change / 100, -1, 1)  # +100 people = +1 reward

            # Diversity bonus (encourage mixed housing), weight 1.0
            diversity_bonus = 1.0 * diversity
            
            # Combined developer reward
            dev_reward = (
                profit_reward +              # Profit × occupancy (multiplicative)
                underdevelopment_penalty +   # Penalize sparse development
                pop_growth_reward +          # Reward population growth
                diversity_bonus +            # Reward mixed housing types
                demand_penalty               # Penalize mismatch with immigrant demand
            )
            
            # 2. Government pillar rewards (2.3)
            econ_reward = self.pillar_rewards.compute_intrinsic_reward('economy', prev_metrics, curr_metrics)
            env_reward = self.pillar_rewards.compute_intrinsic_reward('environment', prev_metrics, curr_metrics)
            soc_reward = self.pillar_rewards.compute_intrinsic_reward('society', prev_metrics, curr_metrics)
            
            # 3. Combined government reward
            gov_reward = 0.33 * econ_reward + 0.33 * env_reward + 0.34 * soc_reward
            
            # 4. Curiosity bonus (3.1) - uses base state for efficiency
            gov_state_t = torch.tensor(gov_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            next_gov_state = self.env.get_state_vector()
            next_gov_state_t = torch.tensor(next_gov_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Flatten gov actions for ICM
            gov_action_flat = torch.cat([a.squeeze(0) for a in gov_actions]).unsqueeze(0)
            curiosity_reward = self.icm.compute_intrinsic_reward(
                gov_state_t, gov_action_flat, next_gov_state_t, scale=self.config.curiosity_scale
            ).item()
            
            # 5. Normalize and combine rewards
            total_reward = dev_reward + gov_reward + curiosity_reward
            self.reward_normalizer.update(total_reward)
            normalized_reward = self.reward_normalizer.normalize(total_reward)
            
            episode_reward += normalized_reward
            
            # Log segregation delta (QW4)
            seg_delta = curr_metrics.segregation_index - prev_seg
            prev_seg = curr_metrics.segregation_index
            
            # Store experience (using developer state for PPO updates)
            done = (self.env.game_state != GameState.RUNNING)
            self.buffer.add(
                state=dev_state,  # Use enhanced developer state (113 dims)
                gov_actions=gov_actions,
                dev_action=land_grid,
                reward=normalized_reward,
                value=gov_value.item(),
                log_prob=dev_log_prob.item(),
                done=done,
                prev_metrics=prev_metrics,
                curr_metrics=curr_metrics
            )
            
            prev_metrics = curr_metrics
            
            if done:
                break
        
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            self.config.gamma, self.config.gae_lambda
        )
        
        # Store Pareto solution
        final_metrics = self.env.metrics
        self.pareto_solutions.append((
            final_metrics.gdp / 1e8,
            final_metrics.green_space_ratio + final_metrics.woodland_ratio,
            1 - final_metrics.segregation_index
        ))
        
        return {
            'episode_reward': episode_reward,
            'profit': episode_profit,
            'population': final_metrics.population,
            'segregation': final_metrics.segregation_index,
            'qol': final_metrics.avg_qol,
            'green_ratio': final_metrics.green_space_ratio,
            'returns': returns,
            'advantages': advantages,
            'game_state': self.env.game_state,
            'seg_delta': seg_delta
        }
    
    def update_networks(self, episode_data: Dict):
        """Update all networks using PPO."""
        if len(self.buffer) == 0:
            return {}
        
        # Convert buffer to tensors
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        returns = torch.tensor(episode_data['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(episode_data['advantages'], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        total_loss = 0
        for _ in range(self.config.ppo_epochs):
            # Update developer
            dev_loss = self._update_developer(states, advantages, returns)
            
            # Update government via MAPPO
            gov_loss = self._update_government(states, advantages, returns)
            
            total_loss += dev_loss + gov_loss
        
        # Update ICM (uses BASE state - first 20 dims)
        if len(self.buffer) > 1:
            # Extract base state (first 20 dims) for ICM
            base_states = np.array([s[:self.config.base_state_dim] for s in self.buffer.states[:-1]])
            next_base_states = np.array([s[:self.config.base_state_dim] for s in self.buffer.states[1:]])
            states_t = torch.tensor(base_states, dtype=torch.float32, device=self.device)
            next_states_t = torch.tensor(next_base_states, dtype=torch.float32, device=self.device)
            
            # Flatten gov actions
            actions_flat = []
            for i in range(len(self.buffer.gov_actions) - 1):
                flat = torch.cat([a.squeeze(0) for a in self.buffer.gov_actions[i]])
                actions_flat.append(flat)
            actions_t = torch.stack(actions_flat).to(self.device)
            
            icm_loss = self.icm.update(states_t, actions_t, next_states_t)
        else:
            icm_loss = 0
        
        # Update adaptive entropy (3.2)
        current_entropy = torch.tensor(self.config.entropy_coef)  # Placeholder
        new_alpha = self.adaptive_entropy.update(current_entropy)
        
        # Update Lagrange multipliers
        self.government.update_lagrange_multipliers(
            episode_data['segregation'],
            episode_data['population']
        )
        
        return {
            'total_loss': total_loss / self.config.ppo_epochs,
            'icm_loss': icm_loss,
            'entropy_alpha': new_alpha
        }
    
    def _update_developer(self, states: torch.Tensor, advantages: torch.Tensor, 
                          returns: torch.Tensor) -> float:
        """Update developer network."""
        # Get constraints from buffer
        constraints_list = []
        for gov_actions in self.buffer.gov_actions:
            c = self.government._actions_to_constraints(gov_actions)
            constraints_list.append([
                c.affordable_housing_pct, c.green_space_pct, c.woodland_preservation,
                c.max_density, c.tax_rate
            ])
        constraints = torch.tensor(constraints_list, dtype=torch.float32, device=self.device)
        
        # Get old log probs
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        
        # Forward pass
        land_logits, values = self.developer(states, constraints)
        
        # Compute new log probs
        land_logits_flat = land_logits.permute(0, 2, 3, 1).reshape(-1, self.config.num_land_types)
        probs = F.softmax(land_logits_flat, dim=-1)
        
        # Get actions from buffer
        actions = np.array(self.buffer.dev_actions)
        actions_flat = torch.tensor(actions.reshape(-1), dtype=torch.long, device=self.device)
        
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions_flat).view(len(self.buffer), -1).sum(dim=1)
        entropy = dist.entropy().view(len(self.buffer), -1).mean(dim=1)
        
        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_loss
        
        # Backprop with gradient clipping
        self.dev_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.developer.parameters(), self.config.max_grad_norm)
        self.dev_optimizer.step()
        
        return loss.item()
    
    def _update_government(self, states: torch.Tensor, advantages: torch.Tensor,
                           returns: torch.Tensor) -> float:
        """Update government networks via MAPPO."""
        # Government uses BASE state (first 20 dims of enhanced state)
        base_states = states[:, :self.config.base_state_dim]  # [batch, 20]
        
        # Collect old log probs using base state
        old_log_probs = []
        for i in range(3):
            log_probs = []
            for j, gov_actions in enumerate(self.buffer.gov_actions):
                # Extract base state (first 20 dims) from buffer
                base_state = torch.tensor(
                    self.buffer.states[j][:self.config.base_state_dim], 
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                _, lp = self.government.actors[i].evaluate_action(base_state, gov_actions[i])
                log_probs.append(lp)
            old_log_probs.append(torch.stack(log_probs))
        
        # Compute PPO loss using base states
        losses = self.government.compute_ppo_loss(
            base_states,  # Use base state for government
            [torch.stack([ga[i] for ga in self.buffer.gov_actions]).squeeze(1) for i in range(3)],
            [lp.squeeze() for lp in old_log_probs],
            advantages,
            returns
        )
        
        # Backprop
        total_loss = losses['total_loss']
        
        for opt in self.government.actor_optimizers:
            opt.zero_grad()
        self.government.critic_optimizer.zero_grad()
        
        total_loss.backward()
        
        for actor in self.government.actors:
            nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.government.critic.parameters(), self.config.max_grad_norm)
        
        for opt in self.government.actor_optimizers:
            opt.step()
        self.government.critic_optimizer.step()
        
        return total_loss.item()
    
    def save_checkpoint(self, episode: int, final: bool = False):
        """Save all model checkpoints."""
        suffix = 'final' if final else f'ep{episode}'
        
        checkpoint = {
            'episode': episode,
            'developer': self.developer.state_dict(),
            'gov_actors': [a.state_dict() for a in self.government.actors],
            'gov_critic': self.government.critic.state_dict(),
            'icm': self.icm.state_dict(),
            'pcm': self.pcm.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict(),
            'metrics_history': self.metrics_history,
            'pareto_solutions': self.pareto_solutions,
        }
        
        torch.save(checkpoint, self.exp_dir / 'checkpoints' / f'checkpoint_{suffix}.pth')
        print(f"Saved checkpoint: {suffix}")
    
    def train(self):
        """Main training loop."""
        print("=" * 80)
        print("FULL SYSTEM TRAINING - RESEARCH-GRADE IMPLEMENTATION")
        print("=" * 80)
        print(f"Episodes: {self.config.num_episodes}")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {self.config.early_stop_patience}")
        print("=" * 80)
        
        best_reward = float('-inf')
        running_reward = 0
        
        for episode in range(1, self.config.num_episodes + 1):
            # Run episode
            episode_data = self.run_episode()
            
            # Update networks
            update_info = self.update_networks(episode_data)
            
            # Track metrics
            self.metrics_history.append({
                'episode': episode,
                'reward': episode_data['episode_reward'],
                'profit': episode_data['profit'],
                'population': episode_data['population'],
                'segregation': episode_data['segregation'],
                'qol': episode_data['qol'],
                'green_ratio': episode_data['green_ratio'],
                'game_state': episode_data['game_state'].name,
                'seg_delta': episode_data['seg_delta'],
                **update_info
            })
            
            # Update running reward
            running_reward = 0.99 * running_reward + 0.01 * episode_data['episode_reward']
            
            # Logging
            if episode % self.config.log_interval == 0:
                print(f"Ep {episode:5d} | Reward: {episode_data['episode_reward']:7.2f} | "
                      f"Profit: ${episode_data['profit']:,.0f} | Pop: {episode_data['population']:4d} | "
                      f"Seg: {episode_data['segregation']:.3f} | QoL: {episode_data['qol']:.2f} | "
                      f"State: {episode_data['game_state'].name}")
            
            # Save checkpoints
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)
                
                # Compute Pareto hypervolume
                if len(self.pareto_solutions) >= 10:
                    hv = self.pareto.compute_hypervolume(self.pareto_solutions[-100:])
                    print(f"  Pareto Hypervolume (last 100): {hv:.4f}")
            
            # Best model
            if episode_data['episode_reward'] > best_reward:
                best_reward = episode_data['episode_reward']
                self.save_checkpoint(episode, final=False)
            
            # Early stopping check
            if self.early_stopping.check(running_reward):
                print(f"\nEarly stopping triggered at episode {episode}")
                print(f"No improvement for {self.config.early_stop_patience} episodes")
                break
        
        # Final save
        self.save_checkpoint(episode, final=True)
        
        # Save metrics history
        with open(self.exp_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Final Pareto hypervolume: {self.pareto.compute_hypervolume(self.pareto_solutions):.4f}")
        print("=" * 80)
        
        return self.metrics_history


# ============================================================
# BASELINE COMPARISON
# ============================================================
def run_baseline_comparison(config: Config, num_episodes: int = 100):
    """Compare trained model against all baselines."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    
    baselines = {
        'Fixed': FixedPolicyGovernment(),
        'Random': RandomPolicyGovernment(),
        'NoGov': NoGovernment(),
    }
    
    results = {}
    
    for name, gov in baselines.items():
        print(f"\nEvaluating {name}...")
        
        episode_results = []
        for ep in range(num_episodes):
            env = UrbanSimulation(map_size=config.map_size, zone_size=config.zone_size)
            
            for zone_id in range(config.num_zones):
                state = env.get_state_vector()
                constraints = gov.select_action(state)
                
                # Fixed developer for baselines
                decision = DeveloperDecision()  # Default balanced
                
                result = env.develop_zone(zone_id, constraints, decision)
                
                if env.game_state != GameState.RUNNING:
                    break
            
            episode_results.append({
                'profit': env.cumulative_developer_profit,
                'population': env.metrics.population,
                'segregation': env.metrics.segregation_index,
                'qol': env.metrics.avg_qol,
                'game_state': env.game_state.name
            })
        
        results[name] = {
            'profit_mean': np.mean([r['profit'] for r in episode_results]),
            'population_mean': np.mean([r['population'] for r in episode_results]),
            'segregation_mean': np.mean([r['segregation'] for r in episode_results]),
            'qol_mean': np.mean([r['qol'] for r in episode_results]),
            'win_rate': sum(1 for r in episode_results if 'WIN' in r['game_state']) / num_episodes
        }
        
        print(f"  Profit: ${results[name]['profit_mean']:,.0f}")
        print(f"  Population: {results[name]['population_mean']:.0f}")
        print(f"  Segregation: {results[name]['segregation_mean']:.3f}")
        print(f"  QoL: {results[name]['qol_mean']:.2f}")
    
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    config = Config()
    
    print("=" * 80)
    print("TRAINING WITH FIXED REWARD FUNCTION")
    print("=" * 80)
    print("Fixes applied:")
    print("  1. Minimum development requirement (penalize >50% empty)")
    print("  2. Occupancy MULTIPLICATIVE with profit (not additive)")
    print("  3. Population growth term added to developer reward")
    print("  4. Diversity bonus for mixed housing types")
    print("  5. Entropy coefficient increased to 0.1")
    print("=" * 80)
    
    # Run full 10k training directly
    config.num_episodes = 10000  # Full training
    config.log_interval = 100
    config.save_interval = 500  # Save every 500 episodes
    config.early_stop_patience = 999999  # Disabled - train full 10k
    
    trainer = FullSystemTrainer(config)
    
    try:
        metrics = trainer.train()
        
        # Check final results
        if len(metrics) > 100:
            final_100 = metrics[-100:]
            avg_profit = np.mean([m['profit'] for m in final_100])
            avg_pop = np.mean([m['population'] for m in final_100])
            avg_seg = np.mean([m['segregation'] for m in final_100])
            
            print("\n" + "=" * 80)
            print("FINAL RESULTS (last 100 episodes)")
            print("=" * 80)
            print(f"  Average Profit: ${avg_profit:,.0f}")
            print(f"  Average Population: {avg_pop:.0f}")
            print(f"  Average Segregation: {avg_seg:.3f}")
            print("=" * 80)
        
        # Run baseline comparison
        print("\nRunning baseline comparison...")
        baseline_results = run_baseline_comparison(config, num_episodes=50)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

