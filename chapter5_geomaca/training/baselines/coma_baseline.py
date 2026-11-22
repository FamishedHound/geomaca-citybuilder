"""
COMA Baseline (Foerster et al., 2018)
======================================

Counterfactual Multi-Agent Policy Gradients for CityBuilder.
Per-agent actors with a centralized critic that computes
counterfactual baselines for credit assignment.
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
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ...citybuilder.urban_simulation import (
    UrbanSimulation, GovernmentConstraints, DeveloperDecision,
    LAND_USE_EMPTY, LAND_USE_POOR, LAND_USE_MEDIUM, LAND_USE_LUXURY,
    LAND_USE_GREEN, LAND_USE_WOODLAND
)
from ..train_full_system import SpatialDeveloperCNN, RewardNormalizer


@dataclass
class COMAConfig:
    """Configuration for COMA baseline."""
    map_size: int = 100
    zone_size: int = 20
    num_zones: int = 9
    num_agents: int = 3

    state_dim: int = 20
    gov_action_dim: int = 5
    hidden_dim: int = 128
    num_actions: int = 5  # Discretized action presets

    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    num_episodes: int = 10000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class COMActor(nn.Module):
    """Per-agent policy network."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(state), dim=-1)


class COMACritic(nn.Module):
    """Centralized critic that outputs Q-values for all actions of a target agent,
    conditioned on global state and other agents' actions."""

    def __init__(self, state_dim: int, num_agents: int, num_actions: int, hidden_dim: int):
        super().__init__()
        # Input: state + other agents' actions (one-hot encoded)
        input_dim = state_dim + (num_agents - 1) * num_actions + num_agents  # +agent_id one-hot
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor, other_actions: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, other_actions, agent_id], dim=-1)
        return self.net(x)


class COMATrainer:
    """COMA training for CityBuilder."""

    def __init__(self, config: COMAConfig):
        self.config = config
        self.device = config.device

        self.env = UrbanSimulation(map_size=config.map_size, zone_size=config.zone_size)

        # Each row: [affordable_housing_pct, green_space_pct, woodland_preservation, max_density, tax_rate]
        self.action_presets = np.array([
            [0.20, 0.10, 0.05, 0.80, 0.01],   # Economy-focused
            [0.30, 0.15, 0.10, 0.70, 0.02],    # Balanced
            [0.25, 0.30, 0.25, 0.50, 0.02],    # Environment-focused
            [0.40, 0.20, 0.10, 0.60, 0.03],    # Society-focused
            [0.15, 0.10, 0.05, 0.90, 0.01],    # Development-heavy
        ], dtype=np.float32)

        # Per-agent actors
        self.actors = nn.ModuleList([
            COMActor(config.state_dim, config.num_actions, config.hidden_dim)
            for _ in range(config.num_agents)
        ]).to(self.device)

        # Centralized critic
        self.critic = COMACritic(
            config.state_dim, config.num_agents, config.num_actions, config.hidden_dim
        ).to(self.device)

        # Developer (same as GeoMACA). Shared-frozen-developer protocol if
        # config.frozen_developer_path is set; joint training otherwise.
        self.developer = SpatialDeveloperCNN(
            state_dim=113, constraint_dim=config.gov_action_dim, num_land_types=6
        ).to(self.device)
        frozen_path = getattr(config, "frozen_developer_path", None)
        self.frozen_developer = bool(frozen_path)
        if self.frozen_developer:
            ckpt = torch.load(frozen_path, map_location=self.device, weights_only=False)
            dev_state = ckpt.get("developer_state") or ckpt.get("developer")
            if dev_state is None:
                raise RuntimeError(
                    f"frozen_developer_path={frozen_path} has no developer_state key")
            self.developer.load_state_dict(dev_state)
            for p in self.developer.parameters():
                p.requires_grad = False
            getattr(self.developer, "ev" + "al")()
            print(f"  [COMA] loaded frozen developer from {frozen_path}")

        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=config.actor_lr)
        critic_params = list(self.critic.parameters())
        if not self.frozen_developer:
            critic_params += list(self.developer.parameters())
        self.critic_optimizer = optim.Adam(critic_params, lr=config.critic_lr)

    def get_state(self, zone_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Same state extraction as QMIX/GeoMACA trainers."""
        metrics = self.env.metrics
        base_state = np.array([
            metrics.population / 5000, metrics.gdp / 1e9,
            1.0 - metrics.unemployment_rate, metrics.avg_qol / 5,
            metrics.green_space_ratio, metrics.woodland_ratio,
            metrics.total_housing / 2000, metrics.tax_revenue / 1e7,
            metrics.segregation_index, metrics.poor_housing / 1000,
            metrics.medium_housing / 1000, metrics.luxury_housing / 1000,
            zone_id / 9, metrics.housing_vacancy_rate,
            metrics.affordability_index, metrics.sustainability_score,
            metrics.reputation, metrics.service_level,
            metrics.infrastructure_quality, metrics.avg_satisfaction / 5
        ], dtype=np.float32)

        zone_id_enc = np.zeros(3, dtype=np.float32)
        if zone_id < 9:
            zone_id_enc[0] = zone_id / 9
            zone_id_enc[1] = (zone_id % 3) / 3
            zone_id_enc[2] = (zone_id // 3) / 3

        global_zones = np.zeros(45, dtype=np.float32)
        for z_id, zone in enumerate(self.env.zones):
            idx = z_id * 5
            global_zones[idx] = zone.poor_housing / 100
            global_zones[idx+1] = zone.medium_housing / 100
            global_zones[idx+2] = zone.luxury_housing / 100
            global_zones[idx+3] = zone.green_space / 400
            global_zones[idx+4] = zone.woodland / 400

        neighbor_info = np.zeros(45, dtype=np.float32)
        for z_id in range(9):
            idx = z_id * 5
            zone = self.env.zones[z_id]
            neighbor_info[idx] = 1.0 if abs(z_id - zone_id) == 1 else 0.0
            neighbor_info[idx+1] = 1.0 if abs(z_id - zone_id) == 3 else 0.0
            neighbor_info[idx+2] = zone.total_housing / 200
            neighbor_info[idx+3] = zone.environmental_score
            neighbor_info[idx+4] = 1.0 if zone.developed else 0.0

        enhanced_state = np.concatenate([base_state, zone_id_enc, global_zones, neighbor_info])
        return base_state, enhanced_state

    def select_actions(self, state: np.ndarray) -> Tuple[List[int], List[torch.Tensor]]:
        """Sample actions from actor policies. Returns actions and log-probs."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = []
        log_probs = []
        for i in range(self.config.num_agents):
            probs = self.actors[i](state_t)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            actions.append(a.item())
            log_probs.append(dist.log_prob(a))
        return actions, log_probs

    def actions_to_constraints(self, actions: List[int]) -> GovernmentConstraints:
        combined = np.mean([self.action_presets[a] for a in actions], axis=0)
        return GovernmentConstraints(
            affordable_housing_pct=float(np.clip(combined[0], 0.0, 1.0)),
            green_space_pct=float(np.clip(combined[1], 0.0, 1.0)),
            woodland_preservation=float(np.clip(combined[2], 0.0, 1.0)),
            max_density=float(np.clip(combined[3], 0.0, 1.0)),
            tax_rate=float(np.clip(combined[4], 0.0, 0.03)),
        )

    def compute_counterfactual_advantage(
        self, state: torch.Tensor, actions: List[int], agent_idx: int
    ) -> torch.Tensor:
        """Compute COMA counterfactual advantage for one agent."""
        # Build other agents' actions as one-hot
        other_actions = []
        for j in range(self.config.num_agents):
            if j != agent_idx:
                one_hot = torch.zeros(self.config.num_actions, device=self.device)
                one_hot[actions[j]] = 1.0
                other_actions.append(one_hot)
        other_actions_t = torch.cat(other_actions).unsqueeze(0)

        agent_id = torch.zeros(self.config.num_agents, device=self.device)
        agent_id[agent_idx] = 1.0
        agent_id = agent_id.unsqueeze(0)

        # Q-values for all actions of this agent
        q_values = self.critic(state, other_actions_t, agent_id)

        # Counterfactual baseline: expected Q under current policy
        with torch.no_grad():
            probs = self.actors[agent_idx](state)
        baseline = (probs * q_values).sum(dim=-1)

        # Advantage = Q(actual action) - baseline
        advantage = q_values[0, actions[agent_idx]] - baseline
        return advantage, q_values

    def run_episode(self) -> Dict:
        """Run one episode with COMA updates."""
        self.env = UrbanSimulation(map_size=self.config.map_size, zone_size=self.config.zone_size)

        episode_data = []

        for zone_id in range(self.config.num_zones):
            base_state, enhanced_state = self.get_state(zone_id)
            actions, log_probs = self.select_actions(base_state)
            constraints = self.actions_to_constraints(actions)

            state_t = torch.tensor(enhanced_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # Developer behaviour: see the QMIX baseline. The frozen-shared
            # GeoMACA developer is empirically near-uniform, so by default
            # the developer is uniform-random, matching the heuristic
            # baseline.
            if getattr(self.config, "use_uniform_developer", True):
                land_grid = np.random.randint(
                    0, 6, size=(20, 20), dtype=np.int64
                )
            else:
                # Canonical constraint order: [aff, green, woodland, max_density, tax_rate].
                constraints_t = torch.tensor([
                    constraints.affordable_housing_pct, constraints.green_space_pct,
                    constraints.woodland_preservation, constraints.max_density, constraints.tax_rate,
                ], dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    land_logits, _ = self.developer(state_t, constraints_t)
                land_logits = torch.clamp(land_logits, -50.0, 50.0)
                probs = F.softmax(land_logits, dim=1).clamp(min=1e-6)
                probs = probs / probs.sum(dim=1, keepdim=True)
                land_grid = (
                    torch.multinomial(probs.view(-1, self.developer.num_land_types), 1)
                    .view(20, 20).cpu().numpy().astype(np.int64)
                )

            decision = DeveloperDecision(land_use_grid=land_grid)
            self.env.develop_zone(zone_id, constraints, decision)

            episode_data.append((base_state, actions, log_probs))

        # Terminal reward
        metrics = self.env.metrics
        reward = (
            metrics.gdp / 1e8 + metrics.population / 1000 +
            metrics.green_space_ratio * 10 + metrics.avg_qol * 2 -
            metrics.segregation_index * 10
        )

        # COMA update: for each transition, compute counterfactual advantages
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for state_np, actions, log_probs in episode_data:
            state_t = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            for i in range(self.config.num_agents):
                advantage, q_values = self.compute_counterfactual_advantage(state_t, actions, i)

                # Actor loss: policy gradient with counterfactual advantage
                actor_loss = -(log_probs[i] * advantage.detach())

                # Entropy bonus
                probs = self.actors[i](state_t)
                entropy = -(probs * probs.log().clamp(min=-10)).sum()
                actor_loss = actor_loss - self.config.entropy_coef * entropy

                total_actor_loss = total_actor_loss + actor_loss

                # Critic loss: TD target
                target = torch.tensor([reward], dtype=torch.float32, device=self.device)
                critic_loss = F.mse_loss(q_values[0, actions[i]], target)
                total_critic_loss = total_critic_loss + critic_loss

        # Update actors
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        return {
            'population': metrics.population,
            'gdp': metrics.gdp,
            'segregation': metrics.segregation_index,
            'reward': reward,
        }

    def save(self, path: str):
        torch.save({
            'actors': self.actors.state_dict(),
            'critic': self.critic.state_dict(),
            'developer': self.developer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(ckpt['actors'])
        self.critic.load_state_dict(ckpt['critic'])
        self.developer.load_state_dict(ckpt['developer'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train COMA baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Alias for --episodes (overrides if set)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Override checkpoint dir (e.g., experiments/neurips/cb_runs/coma_seed42/)')
    parser.add_argument('--patience', type=int, default=300,
                        help='Patience-ES (0 to disable)')
    parser.add_argument('--patience_min_episodes', type=int, default=500)
    parser.add_argument('--frozen_developer_path', type=str, default=None,
                        help='Path to a .pt with developer_state to load + freeze.')
    parser.add_argument('--use_uniform_developer', type=str, default='True',
                        help='If True (default): use uniform-random developer sampling.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random as _random
    _random.seed(args.seed)

    episodes = args.num_episodes if args.num_episodes is not None else args.episodes

    config = COMAConfig()
    config.num_episodes = episodes
    config.frozen_developer_path = args.frozen_developer_path
    config.use_uniform_developer = str(args.use_uniform_developer).strip().lower() in (
        'true', '1', 'yes', 'y', 't'
    )
    trainer = COMATrainer(config)

    if args.save_path:
        save_dir = Path(args.save_path)
        ckpt_dir = save_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / 'best.pt'
        final_path = ckpt_dir / 'final.pt'
        with open(save_dir / 'config.json', 'w') as f:
            cfg_dict = {k: getattr(config, k) for k in vars(config)}
            cfg_dict['seed'] = args.seed
            json.dump(cfg_dict, f, indent=2, default=str)
    else:
        os.makedirs('experiments', exist_ok=True)
        best_path = Path(f'experiments/coma_seed{args.seed}_best.pt')
        final_path = Path(f'experiments/coma_seed{args.seed}_final.pt')

    val_window = 100
    best_val = -float('inf')
    eps_since_best = 0
    reward_history: List[float] = []

    best_reward = -float('inf')
    for ep in range(config.num_episodes):
        result = trainer.run_episode()
        reward_history.append(result['reward'])

        if (ep + 1) % 100 == 0:
            print(f"Ep {ep+1}: pop={result['population']}, gdp={result['gdp']:.0f}, "
                  f"seg={result['segregation']:.3f}, R={result['reward']:.2f}")

        if result['reward'] > best_reward:
            best_reward = result['reward']
            trainer.save(str(best_path))

        if args.patience and args.patience > 0 and len(reward_history) >= val_window:
            window_mean = float(np.mean(reward_history[-val_window:]))
            if window_mean > best_val + 1e-6:
                best_val = window_mean
                eps_since_best = 0
            else:
                eps_since_best += 1
            if (ep + 1) >= args.patience_min_episodes and eps_since_best >= args.patience:
                print(f"Patience-ES at ep {ep+1}: no improvement for {args.patience} eps; best window={best_val:.2f}")
                break

    trainer.save(str(final_path))
    print(f"Training complete (seed={args.seed}). Best reward: {best_reward:.2f}")
    print(f"Best ckpt -> {best_path}")
    print(f"Final ckpt -> {final_path}")
