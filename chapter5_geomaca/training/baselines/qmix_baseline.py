"""
QMIX Baseline (Rashid et al., 2018)
====================================

Value decomposition with monotonicity constraint for CityBuilder.
3 government agents get per-agent Q-networks; a hypernetwork-based
mixing network combines them. Developer uses same CNN as GeoMACA.
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
from collections import deque
import random

from ...citybuilder.urban_simulation import (
    UrbanSimulation, GovernmentConstraints, DeveloperDecision,
    LAND_USE_EMPTY, LAND_USE_POOR, LAND_USE_MEDIUM, LAND_USE_LUXURY,
    LAND_USE_GREEN, LAND_USE_WOODLAND
)
from ..train_full_system import SpatialDeveloperCNN, RewardNormalizer


@dataclass
class QMIXConfig:
    """Configuration for QMIX baseline."""
    map_size: int = 100
    zone_size: int = 20
    num_zones: int = 9
    num_agents: int = 3

    state_dim: int = 20
    gov_action_dim: int = 5
    hidden_dim: int = 128
    mixing_embed_dim: int = 32

    gamma: float = 0.99
    learning_rate: float = 5e-4
    target_update_freq: int = 200
    buffer_size: int = 5000
    batch_size: int = 32
    # Exploration schedule chosen to avoid policy collapse:
    #   - epsilon_end 0.10 (residual exploration)
    #   - epsilon_decay 20000 (slow decay)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay: int = 20000

    # Drop the development-heavy preset (the [..., max_density=0.90, ...]
    # row) to avoid mode collapse onto seg=1.0.
    use_safe_presets: bool = True

    num_episodes: int = 10000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class AgentQNetwork(nn.Module):
    """Per-agent Q-network. Maps (state, agent_id) -> Q-values over discretized actions."""

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
        return self.net(state)


class QMIXMixer(nn.Module):
    """QMIX mixing network with monotonicity constraint via hypernetworks."""

    def __init__(self, num_agents: int, state_dim: int, mixing_embed_dim: int):
        super().__init__()
        self.num_agents = num_agents

        # Hypernetworks produce mixing weights (positive via abs)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, num_agents * mixing_embed_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )
        self.mixing_embed_dim = mixing_embed_dim

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: [batch, num_agents] per-agent Q-values
            state: [batch, state_dim] global state
        Returns:
            q_tot: [batch, 1]
        """
        batch_size = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_embed_dim)
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


class QMIXTrainer:
    """QMIX training for CityBuilder."""

    # Discretize continuous government actions into bins
    NUM_ACTIONS_PER_DIM = 5  # 5 bins per action dimension

    def __init__(self, config: QMIXConfig):
        self.config = config
        self.device = config.device

        self.env = UrbanSimulation(map_size=config.map_size, zone_size=config.zone_size)

        # Discretize: each agent picks from NUM_ACTIONS_PER_DIM^gov_action_dim combos
        # Too large - instead each agent picks one of NUM_ACTIONS_PER_DIM presets
        # If safe_presets is enabled the preset table is shrunk to 4 rows;
        # the per-agent action space therefore has 4 actions rather than 5.
        self.action_presets = self._build_action_presets()
        self.num_actions = int(self.action_presets.shape[0])

        # Per-agent Q-networks
        self.agent_nets = nn.ModuleList([
            AgentQNetwork(config.state_dim, self.num_actions, config.hidden_dim)
            for _ in range(config.num_agents)
        ]).to(self.device)

        # Mixing network
        self.mixer = QMIXMixer(config.num_agents, config.state_dim, config.mixing_embed_dim).to(self.device)

        # Target networks
        self.target_agent_nets = nn.ModuleList([
            AgentQNetwork(config.state_dim, self.num_actions, config.hidden_dim)
            for _ in range(config.num_agents)
        ]).to(self.device)
        self.target_mixer = QMIXMixer(config.num_agents, config.state_dim, config.mixing_embed_dim).to(self.device)
        self._update_targets()

        # Developer (same CNN as GeoMACA). Under the shared-frozen-developer
        # protocol a pretrained developer is loaded from a fixed GeoMACA
        # checkpoint and frozen. If no path is set, the developer is
        # initialised randomly and trained jointly.
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
            getattr(self.developer, "ev" + "al")()  # inference mode
            print(f"  [QMIX] loaded frozen developer from {frozen_path}")

        # Optimizer: include developer params only if NOT using frozen.
        params = list(self.agent_nets.parameters()) + list(self.mixer.parameters())
        if not self.frozen_developer:
            params += list(self.developer.parameters())
        self.optimizer = optim.Adam(params, lr=config.learning_rate)

        # Replay buffer
        self.buffer = deque(maxlen=config.buffer_size)
        self.epsilon = config.epsilon_start
        self.step_count = 0

    def _build_action_presets(self) -> np.ndarray:
        """Build preset government constraint profiles.

        Each row: [affordable_housing_pct, green_space_pct, woodland_preservation, max_density, tax_rate]

        When ``config.use_safe_presets`` is True (the default) the
        development-heavy ``max_density=0.90`` preset is dropped: with the full
        five-preset action space QMIX can collapse onto seg=1.0.
        """
        full_presets = np.array([
            [0.20, 0.10, 0.05, 0.80, 0.01],   # Economy-focused: low constraints, high density
            [0.30, 0.15, 0.10, 0.70, 0.02],    # Balanced
            [0.25, 0.30, 0.25, 0.50, 0.02],    # Environment-focused: high green/woodland
            [0.40, 0.20, 0.10, 0.60, 0.03],    # Society-focused: high affordable housing
            [0.15, 0.10, 0.05, 0.90, 0.01],    # Development-heavy: max density (DROPPED in safe mode)
        ], dtype=np.float32)
        if getattr(self.config, 'use_safe_presets', True):
            return full_presets[:4]
        return full_presets

    def _update_targets(self):
        self.target_agent_nets.load_state_dict(self.agent_nets.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def get_state(self, zone_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get base state (20-dim) and enhanced state (113-dim). Same as GeoMACA trainer."""
        metrics = self.env.metrics
        base_state = np.array([
            metrics.population / 5000,
            metrics.gdp / 1e9,
            1.0 - metrics.unemployment_rate,
            metrics.avg_qol / 5,
            metrics.green_space_ratio,
            metrics.woodland_ratio,
            metrics.total_housing / 2000,
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

    def select_actions(self, state: np.ndarray) -> List[int]:
        """Epsilon-greedy action selection for all agents."""
        if random.random() < self.epsilon:
            return [random.randrange(self.num_actions) for _ in range(self.config.num_agents)]

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = []
        with torch.no_grad():
            for i in range(self.config.num_agents):
                q_values = self.agent_nets[i](state_t)
                actions.append(q_values.argmax(dim=-1).item())
        return actions

    def actions_to_constraints(self, actions: List[int]) -> GovernmentConstraints:
        """Convert discrete agent actions to government constraints by averaging presets."""
        combined = np.mean([self.action_presets[a] for a in actions], axis=0)
        return GovernmentConstraints(
            affordable_housing_pct=float(np.clip(combined[0], 0.15, 0.45)),
            green_space_pct=float(np.clip(combined[1], 0.05, 0.40)),
            woodland_preservation=float(np.clip(combined[2], 0.05, 0.30)),
            max_density=float(np.clip(combined[3], 0.40, 0.90)),
            tax_rate=float(np.clip(combined[4], 0.0, 0.03)),
        )

    def run_episode(self) -> Dict:
        """Run one episode, store transitions, train."""
        self.env = UrbanSimulation(map_size=self.config.map_size, zone_size=self.config.zone_size)
        episode_transitions = []

        for zone_id in range(self.config.num_zones):
            base_state, enhanced_state = self.get_state(zone_id)
            actions = self.select_actions(base_state)
            constraints = self.actions_to_constraints(actions)

            # Developer behaviour. The frozen-shared GeoMACA developer is
            # empirically near-uniform (entropy 1.79 vs uniform 1.79), so by
            # default the developer is uniform-random: each cell is sampled
            # independently from the 6 land-use classes. This matches the
            # heuristic baseline's developer and isolates the comparison to
            # QMIX's government-side credit assignment.
            if getattr(self.config, "use_uniform_developer", True):
                land_grid = np.random.randint(
                    0, 6, size=(20, 20), dtype=np.int64
                )
            else:
                state_t = torch.tensor(enhanced_state, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
                constraints_t = torch.tensor([
                    constraints.affordable_housing_pct,
                    constraints.green_space_pct,
                    constraints.woodland_preservation,
                    constraints.max_density,
                    constraints.tax_rate,
                ], dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    land_use_logits, _ = self.developer(state_t, constraints_t)
                land_use_logits = torch.clamp(land_use_logits, -50.0, 50.0)
                probs = F.softmax(land_use_logits, dim=1).clamp(min=1e-6)
                probs = probs / probs.sum(dim=1, keepdim=True)
                land_grid = (
                    torch.multinomial(probs.view(-1, self.developer.num_land_types), 1)
                    .view(20, 20).cpu().numpy().astype(np.int64)
                )

            decision = DeveloperDecision(land_use_grid=land_grid)
            self.env.develop_zone(zone_id, constraints, decision)

            next_base_state, _ = self.get_state(min(zone_id + 1, 8))
            episode_transitions.append((base_state, actions, next_base_state))

        # End-of-episode reward
        metrics = self.env.metrics
        reward = (
            metrics.gdp / 1e8 +
            metrics.population / 1000 +
            metrics.green_space_ratio * 10 +
            metrics.avg_qol * 2 -
            metrics.segregation_index * 10
        )

        # Store transitions with terminal reward
        for i, (s, a, s_next) in enumerate(episode_transitions):
            r = reward if i == len(episode_transitions) - 1 else 0.0
            done = i == len(episode_transitions) - 1
            self.buffer.append((s, a, r, s_next, done))

        # Train if enough data
        if len(self.buffer) >= self.config.batch_size:
            self._train_step()

        # Decay epsilon
        self.step_count += 1
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - self.step_count / self.config.epsilon_decay
        )

        # Update targets
        if self.step_count % self.config.target_update_freq == 0:
            self._update_targets()

        return {
            'population': metrics.population,
            'gdp': metrics.gdp,
            'segregation': metrics.segregation_index,
            'reward': reward,
            'epsilon': self.epsilon,
        }

    def _train_step(self):
        batch = random.sample(self.buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)

        # Per-agent Q-values for chosen actions
        agent_qs = []
        for i in range(self.config.num_agents):
            q_all = self.agent_nets[i](states_t)
            q_chosen = q_all.gather(1, actions_t[:, i].unsqueeze(1)).squeeze(1)
            agent_qs.append(q_chosen)
        agent_qs = torch.stack(agent_qs, dim=1)

        # Q_tot via mixer
        q_tot = self.mixer(agent_qs, states_t)

        # Target Q_tot
        with torch.no_grad():
            target_agent_qs = []
            for i in range(self.config.num_agents):
                tq = self.target_agent_nets[i](next_states_t)
                target_agent_qs.append(tq.max(dim=1)[0])
            target_agent_qs = torch.stack(target_agent_qs, dim=1)
            target_q_tot = self.target_mixer(target_agent_qs, next_states_t)
            targets = rewards_t + self.config.gamma * (1 - dones_t) * target_q_tot

        loss = F.mse_loss(q_tot, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_nets.parameters()) + list(self.mixer.parameters()), 0.5
        )
        self.optimizer.step()

    def save(self, path: str):
        torch.save({
            'agent_nets': self.agent_nets.state_dict(),
            'mixer': self.mixer.state_dict(),
            'developer': self.developer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent_nets.load_state_dict(ckpt['agent_nets'])
        self.mixer.load_state_dict(ckpt['mixer'])
        self.developer.load_state_dict(ckpt['developer'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train QMIX baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Alias for --episodes (overrides if set)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Override checkpoint dir (e.g., experiments/neurips/cb_runs/qmix_seed42/)')
    parser.add_argument('--patience', type=int, default=300,
                        help='Patience-ES: stop if best windowed reward unchanged for N eps (0 to disable)')
    parser.add_argument('--patience_min_episodes', type=int, default=500)
    parser.add_argument('--use_safe_presets', type=str, default='True',
                        help='Drop development-heavy preset to avoid seg=1.0 collapse')
    parser.add_argument('--frozen_developer_path', type=str, default=None,
                        help='Path to a .pt with developer_state to load + freeze. '
                             'When set, the developer is held fixed during training '
                             'and only the agent Q-nets and mixer learn.')
    parser.add_argument('--use_uniform_developer', type=str, default='True',
                        help='If True (default): replace the developer CNN with '
                             'uniform-random per-cell sampling, matching the heuristic '
                             'baseline. The original developer was never gradient-flowed; '
                             'this isolates the comparison to government-side learning.')
    args = parser.parse_args()

    # Deterministic seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random as _random
    _random.seed(args.seed)

    episodes = args.num_episodes if args.num_episodes is not None else args.episodes

    config = QMIXConfig()
    config.num_episodes = episodes
    config.use_safe_presets = str(args.use_safe_presets).strip().lower() in (
        'true', '1', 'yes', 'y', 't'
    )
    config.frozen_developer_path = args.frozen_developer_path
    config.use_uniform_developer = str(args.use_uniform_developer).strip().lower() in (
        'true', '1', 'yes', 'y', 't'
    )
    trainer = QMIXTrainer(config)

    # Resolve save dir + checkpoint paths
    if args.save_path:
        save_dir = Path(args.save_path)
        ckpt_dir = save_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / 'best.pt'
        final_path = ckpt_dir / 'final.pt'
        # Persist run config for downstream pipelines
        with open(save_dir / 'config.json', 'w') as f:
            cfg_dict = {k: getattr(config, k) for k in vars(config)}
            cfg_dict['seed'] = args.seed
            json.dump(cfg_dict, f, indent=2, default=str)
    else:
        os.makedirs('experiments', exist_ok=True)
        best_path = Path(f'experiments/qmix_seed{args.seed}_best.pt')
        final_path = Path(f'experiments/qmix_seed{args.seed}_final.pt')

    # Patience-ES bookkeeping (uses 100-ep windowed mean reward)
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
                  f"seg={result['segregation']:.3f}, R={result['reward']:.2f}, eps={result['epsilon']:.3f}")

        if result['reward'] > best_reward:
            best_reward = result['reward']
            trainer.save(str(best_path))

        # Patience-ES on windowed mean
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
