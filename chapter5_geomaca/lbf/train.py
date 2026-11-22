"""
Level-Based Foraging: the six methods of Chapter 5
==================================================

Environment: ``Foraging-8x8-3p-4f-v3`` from ``lbforaging`` (standard mode, not
``force_coop``): an 8 x 8 grid, 3 agents, 4 food items, maximum food level 2.
A single agent can load level-1 food; higher-level food needs agents to load
together. Episodes last at most 50 steps. The grid is overlaid with 16 zones of
2 x 2 cells for the credit module.

Methods (result-file names in brackets):
  GeoMACA  [geomaca_fixed] - independent PPO learners; after the warm-up the
                             per-step GeoMACA credit, scaled by the episode return
                             and ``credit_weight``, is added to each agent's reward
  IPPO     [independent]   - independent PPO learners, each with the team reward
  Uniform  [uniform]       - independent PPO learners, each with a third of it
  QMIX     [qmix]          - per-agent Q-networks + monotonic mixing
  VDN      [vdn]           - per-agent Q-networks, Q_tot = sum(Q_i)
  COMA     [coma]          - per-agent actors + centralised counterfactual critic

During training every method's reward also contains the proximity-shaping term
(weight ``proximity_shaping``); the held-out evaluation counts the food reward
only.

Usage (from the folder that contains the package):
    python -m chapter5_geomaca.lbf.train --chapter5 --method all \
        --seeds 42 123 456 789 1024 1337 2048 3141 4096 5555 \
        --episodes 50000 --eval-episodes 100 --results-dir <folder>
    python -m chapter5_geomaca.lbf.train --chapter5 --method geomaca_fixed --seeds 42 --episodes 50000
"""

import argparse
import json
import logging
import sys
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium
import lbforaging  # noqa: F401 - registers Foraging envs with gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

from ..geomaca.losses import GeoMACALossComputer
from ..geomaca.model import GeoMACA, GeoMACAConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FairComparisonConfig:
    """Configuration for the fair comparison experiment."""

    # Environment -- standard (non-cooperative) LBF
    env_id: str = "Foraging-8x8-3p-4f-v3"
    field_size: int = 8
    zone_size: int = 2
    num_zones: int = 16  # (8 / 2) ** 2
    num_agents: int = 3
    num_actions: int = 6  # none, up, down, left, right, load
    max_steps: int = 50
    obs_dim: int = 21  # 4 food * 3 + 3 players * 3

    # GeoMACA
    geomaca_state_dim: int = 4  # zone features: food_present, food_level, agent_present, agent_level
    geomaca_hidden_dim: int = 64
    geomaca_num_heads: int = 2
    geomaca_temperature: float = 1.0
    geomaca_warmup: int = 5000  # episodes of plain MAPPO before GeoMACA credit
    # Normalisation of the head-averaged attention weights and weight of the
    # agent-marginal entropy term (see geomaca.model / geomaca.losses);
    # --chapter5 sets 'softmax' and 0.0, the configuration of Chapter 5.
    geomaca_attention_normalization: str = 'renormalize'
    geomaca_agent_entropy_coef: float = 0.5

    # PPO hyperparameters (methods A, B, C)
    policy_lr: float = 3e-4
    geomaca_lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    ppo_hidden_dim: int = 128

    # Q-learning hyperparameters (methods D, E)
    q_lr: float = 5e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 20000
    replay_buffer_size: int = 50000
    q_batch_size: int = 64
    target_update_interval: int = 200
    q_hidden_dim: int = 64

    # COMA hyperparameters (method F)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Environment kwargs
    env_max_food_level: int = 2

    # Reward shaping
    max_food_level: float = 2.0
    max_agent_level: float = 2.0
    proximity_shaping: float = 0.05

    # Credit weight for GeoMACA per-step rewards
    credit_weight: float = 0.3

    # Experiment
    num_episodes: int = 50000
    log_interval: int = 1000
    eval_episodes: int = 100

    # Every this many episodes after the warm-up, GeoMACA records the three
    # attention distributions of the episode with its zones and rewards
    # (0 = off).  The record reuses the forward pass that training makes
    # anyway, so it does not change the run.
    log_attention_every: int = 0


# ---------------------------------------------------------------------------
# Zone Mapping Utilities
# ---------------------------------------------------------------------------

def position_to_zone(row: int, col: int, field_size: int = 8, zone_size: int = 2) -> int:
    """Map a grid position (row, col) to a zone index."""
    zones_per_row = field_size // zone_size
    zone_row = row // zone_size
    zone_col = col // zone_size
    return zone_row * zones_per_row + zone_col


def extract_zone_states(
    field: np.ndarray,
    player_positions: List[Tuple[int, int, int]],
    field_size: int = 8,
    zone_size: int = 2,
    max_food_level: float = 2.0,
    max_agent_level: float = 2.0,
) -> np.ndarray:
    """Extract per-zone state features: [food_present, food_level, agent_present, agent_level]."""
    num_zones = (field_size // zone_size) ** 2
    states = np.zeros((num_zones, 4), dtype=np.float32)

    for r in range(field_size):
        for c in range(field_size):
            if field[r, c] > 0:
                zone_idx = position_to_zone(r, c, field_size, zone_size)
                states[zone_idx, 0] = 1.0
                normalized_level = float(field[r, c]) / max_food_level
                states[zone_idx, 1] = max(states[zone_idx, 1], normalized_level)

    for row, col, level in player_positions:
        zone_idx = position_to_zone(int(row), int(col), field_size, zone_size)
        states[zone_idx, 2] = 1.0
        normalized_level = float(level) / max_agent_level
        states[zone_idx, 3] = max(states[zone_idx, 3], normalized_level)

    return states


def compute_global_state(zone_states: np.ndarray) -> np.ndarray:
    """Compute a global state summary from zone states."""
    total_food = zone_states[:, 0].sum()
    avg_food = zone_states[:, 1].mean()
    total_agents = zone_states[:, 2].sum()
    avg_agents = zone_states[:, 3].mean()
    return np.array([total_food, avg_food, total_agents, avg_agents], dtype=np.float32)


def compute_proximity_reward(
    field: np.ndarray,
    player_positions: List[Tuple[int, int, int]],
    field_size: int = 8,
) -> float:
    """Proximity-based shaping reward for cooperative foraging."""
    food_positions = []
    for r in range(field_size):
        for c in range(field_size):
            if field[r, c] > 0:
                food_positions.append((r, c))

    if not food_positions or not player_positions:
        return 0.0

    max_dist = 2.0 * (field_size - 1)

    total_min_dist = 0.0
    for pr, pc, _ in player_positions:
        min_dist = min(abs(pr - fr) + abs(pc - fc) for fr, fc in food_positions)
        total_min_dist += min_dist / max_dist
    proximity = -total_min_dist / len(player_positions)

    cluster_bonus = 0.0
    for fr, fc in food_positions:
        nearby_count = sum(
            1 for pr, pc, _ in player_positions
            if abs(pr - fr) + abs(pc - fc) <= 2
        )
        if nearby_count >= 2:
            cluster_bonus += (nearby_count - 1) * 0.2
        adjacent_count = sum(
            1 for pr, pc, _ in player_positions
            if abs(pr - fr) + abs(pc - fc) <= 1
        )
        if adjacent_count >= 2:
            cluster_bonus += (adjacent_count - 1) * 0.3

    return proximity + cluster_bonus


def get_player_positions(env) -> List[Tuple[int, int, int]]:
    """Extract player positions from environment."""
    ue = env.unwrapped
    return [
        (int(p.position[0]), int(p.position[1]), int(p.level))
        for p in ue.players
    ]


def get_agent_zones(env, field_size: int = 8, zone_size: int = 2) -> List[int]:
    """Get zone index for each agent's current position."""
    positions = get_player_positions(env)
    return [position_to_zone(r, c, field_size, zone_size) for r, c, _ in positions]


# ---------------------------------------------------------------------------
# Actor-Critic Policy Network (PPO methods A, B, C)
# ---------------------------------------------------------------------------

class ActorCriticPolicy(nn.Module):
    """Actor-Critic MLP for PPO: obs -> (action logits, value)."""

    def __init__(self, obs_dim: int = 21, num_actions: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        return self.actor(features), self.critic(features)


# ---------------------------------------------------------------------------
# Q-Network (QMIX, VDN)
# ---------------------------------------------------------------------------

class AgentQNetwork(nn.Module):
    """Per-agent Q-network: obs -> Q-values."""

    def __init__(self, obs_dim: int = 21, num_actions: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# QMIX Mixing Network
# ---------------------------------------------------------------------------

class QMIXMixingNetwork(nn.Module):
    """Hypernetwork-based mixing with monotonicity constraint."""

    def __init__(self, num_agents: int = 3, state_dim: int = 63, mixing_embed_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, num_agents * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, mixing_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim), nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(self, agent_q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = agent_q_values.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_embed_dim)
        q = agent_q_values.unsqueeze(1)
        hidden = F.elu(torch.bmm(q, w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(-1).squeeze(-1)


# ---------------------------------------------------------------------------
# COMA Networks
# ---------------------------------------------------------------------------

class COMAActorNetwork(nn.Module):
    """Per-agent actor: obs -> action logits."""

    def __init__(self, obs_dim: int = 21, num_actions: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class COMAcriticNetwork(nn.Module):
    """COMA centralized critic: state + agent_id + other_actions -> Q(all actions)."""

    def __init__(self, state_dim: int, num_agents: int = 3, num_actions: int = 6, hidden_dim: int = 128):
        super().__init__()
        input_dim = state_dim + num_agents + (num_agents - 1) * num_actions
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor, agent_id: int, other_actions: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        agent_id_onehot = torch.zeros(batch_size, self.num_agents, device=state.device)
        agent_id_onehot[:, agent_id] = 1.0
        return self.net(torch.cat([state, agent_id_onehot, other_actions], dim=-1))


# ---------------------------------------------------------------------------
# GAE & Replay Buffer
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: List[float], values: List[float], dones: List[bool],
    gamma: float = 0.99, lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if dones[t] else values[t + 1]
        if dones[t]:
            last_gae = 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    values_tensor = torch.tensor(values[:T], dtype=torch.float32)
    returns = advantages + values_tensor
    return advantages, returns


@dataclass
class Transition:
    obs: np.ndarray
    actions: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    global_state: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


def batch_transitions(transitions: List[Transition]) -> Dict[str, torch.Tensor]:
    obs = torch.tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32)
    actions = torch.tensor(np.stack([t.actions for t in transitions]), dtype=torch.long)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    next_obs = torch.tensor(np.stack([t.next_obs for t in transitions]), dtype=torch.float32)
    dones = torch.tensor([float(t.done) for t in transitions], dtype=torch.float32)
    global_state = torch.tensor(np.stack([t.global_state for t in transitions]), dtype=torch.float32)
    return {
        "obs": obs, "actions": actions, "rewards": rewards,
        "next_obs": next_obs, "dones": dones, "global_state": global_state,
    }


def get_epsilon(episode: int, start: float, end: float, decay_eps: int) -> float:
    if episode >= decay_eps:
        return end
    return start + (end - start) * (episode / decay_eps)


def get_global_state(observations: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(observations, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Method A: GeoMACA (``geomaca_fixed``) with per-step credit
# ---------------------------------------------------------------------------

def _attention_record(
    episode: int,
    attention: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    trajectory: Dict[str, torch.Tensor],
    agent_rewards: List[List[float]],
    total_reward: float,
) -> Dict:
    """One logged GeoMACA episode: the three attention distributions, the inputs
    of the geographic head (each zone's features averaged over the episode), the
    zone credited at each step and each agent's food reward at each step."""
    alpha_temp, alpha_geo, alpha_agent = (a.detach().squeeze(0).tolist() for a in attention)
    zone_features = trajectory["zone_states"].mean(dim=1).squeeze(0).tolist()

    def rounded(values):
        return [round(float(v), 6) for v in values]

    return {
        "episode": int(episode),
        "steps": len(alpha_temp),
        "total_reward": float(total_reward),
        "alpha_temp": rounded(alpha_temp),
        "alpha_geo": rounded(alpha_geo),
        "alpha_agent": rounded(alpha_agent),
        "zone_features": [rounded(row) for row in zone_features],
        "zone_developed": [int(z) for z in trajectory["zone_developed"].squeeze(0).tolist()],
        "agent_rewards": [rounded(step) for step in agent_rewards],
    }


def train_geomaca(seed: int, config: FairComparisonConfig, results_dir: Path) -> Dict:
    """Train GeoMACA with the per-step geographic credit."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gymnasium.make(config.env_id, max_food_level=config.env_max_food_level)
    env.reset(seed=seed)

    # Per-agent actor-critic policies
    policies = [
        ActorCriticPolicy(config.obs_dim, config.num_actions, config.ppo_hidden_dim)
        for _ in range(config.num_agents)
    ]
    policy_optimizers = [
        torch.optim.Adam(p.parameters(), lr=config.policy_lr)
        for p in policies
    ]

    # GeoMACA model
    geomaca_config = GeoMACAConfig(
        state_dim=config.geomaca_state_dim,
        action_dim=config.num_actions,
        hidden_dim=config.geomaca_hidden_dim,
        num_zones=config.num_zones,
        num_turns=config.max_steps,
        num_agents=config.num_agents,
        num_heads=config.geomaca_num_heads,
        temperature=config.geomaca_temperature,
        attention_normalization=config.geomaca_attention_normalization,
    )
    geomaca_model = GeoMACA(geomaca_config)
    geomaca_optimizer = torch.optim.Adam(geomaca_model.parameters(), lr=config.geomaca_lr)

    history = {"episode_returns": [], "food_collected": [], "coop_events": [], "episode_lengths": []}
    attention_log: List[Dict] = []

    logger.info("Starting GeoMACA-Fixed (seed=%d) for %d episodes", seed, config.num_episodes)

    for ep in range(config.num_episodes):
        for p in policies:
            p.train()

        obs, _ = env.reset()
        ue = env.unwrapped

        per_agent_obs = [[] for _ in range(config.num_agents)]
        per_agent_actions = [[] for _ in range(config.num_agents)]
        per_agent_log_probs = [[] for _ in range(config.num_agents)]
        per_agent_values = [[] for _ in range(config.num_agents)]

        zone_states_list = []
        agent_actions_list = []
        global_state_list = []
        zone_developed_list = []
        per_step_agent_rewards = []
        per_step_rewards = []
        per_step_dones = []

        total_reward = 0.0
        food_collected = 0
        coop_events = 0

        for step_idx in range(config.max_steps):
            positions = get_player_positions(env)
            zone_states = extract_zone_states(
                ue.field, positions, config.field_size, config.zone_size,
                config.max_food_level, config.max_agent_level,
            )
            global_state = compute_global_state(zone_states)
            zone_states_list.append(zone_states)
            global_state_list.append(global_state)

            # Determine which zone is "developed" this step
            # Use the zone of the first agent that performs a load action, or agent 0's zone
            agent_zones = get_agent_zones(env, config.field_size, config.zone_size)
            zone_developed_list.append(agent_zones[0])  # will be updated after action

            actions = []
            action_onehots = []
            for i, policy in enumerate(policies):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                logits, value = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                per_agent_obs[i].append(obs_tensor.squeeze(0))
                per_agent_actions[i].append(action.item())
                per_agent_log_probs[i].append(log_prob.squeeze().detach())
                per_agent_values[i].append(value.squeeze().item())

                actions.append(int(action.item()))
                onehot = np.zeros(config.num_actions, dtype=np.float32)
                onehot[int(action.item())] = 1.0
                action_onehots.append(onehot)

            # Update zone_developed based on which agent loaded (action=5)
            for i, a in enumerate(actions):
                if a == 5:  # load action
                    zone_developed_list[-1] = agent_zones[i]
                    break

            agent_actions_list.append(np.stack(action_onehots))

            obs, rewards, terminated, truncated, info = env.step(tuple(actions))
            per_step_agent_rewards.append([float(r) for r in rewards])
            raw_reward = sum(float(r) for r in rewards)
            step_reward = raw_reward

            if config.proximity_shaping > 0:
                current_positions = get_player_positions(env)
                shaping = compute_proximity_reward(ue.field, current_positions, config.field_size)
                step_reward += config.proximity_shaping * shaping

            total_reward += step_reward
            per_step_rewards.append(step_reward)

            done = terminated or truncated
            per_step_dones.append(done)

            if raw_reward > 0:
                food_collected += 1
                # Check if multiple agents were adjacent to collected food
                if any(float(r) > 0 for r in rewards):
                    contributing = sum(1 for r in rewards if float(r) > 0)
                    if contributing >= 2:
                        coop_events += 1

            if done:
                break

        actual_steps = len(zone_states_list)

        # Bootstrap values
        for i, policy in enumerate(policies):
            if per_step_dones[-1]:
                per_agent_values[i].append(0.0)
            else:
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, bootstrap_value = policy(obs_tensor)
                per_agent_values[i].append(bootstrap_value.squeeze().item())

        # Build trajectory for GeoMACA
        trajectory = {
            "zone_states": torch.tensor(np.stack(zone_states_list), dtype=torch.float32).unsqueeze(0),
            "agent_actions": torch.tensor(np.stack(agent_actions_list), dtype=torch.float32).unsqueeze(0),
            "global_state": torch.tensor(np.stack(global_state_list), dtype=torch.float32).unsqueeze(0),
            "zone_developed": torch.tensor(zone_developed_list, dtype=torch.long).unsqueeze(0),
        }

        # Update GeoMACA num_turns for this episode length
        geomaca_model.num_turns = actual_steps
        geomaca_model.config = GeoMACAConfig(
            state_dim=config.geomaca_state_dim,
            action_dim=config.num_actions,
            hidden_dim=config.geomaca_hidden_dim,
            num_zones=config.num_zones,
            num_turns=actual_steps,
            num_agents=config.num_agents,
            num_heads=config.geomaca_num_heads,
            temperature=config.geomaca_temperature,
            attention_normalization=config.geomaca_attention_normalization,
        )

        use_geomaca = ep >= config.geomaca_warmup

        # Compute per-step credit-weighted rewards for GAE
        if use_geomaca and total_reward != 0:
            reward_tensor = torch.tensor([total_reward], dtype=torch.float32)
            credit, attention = geomaca_model(trajectory, reward_tensor)
            every = config.log_attention_every
            if every and (ep - config.geomaca_warmup) % every == 0:
                attention_log.append(
                    _attention_record(ep, attention, trajectory, per_step_agent_rewards, total_reward)
                )

            # Per-step credit preserving geographic attention
            per_step_credit = geomaca_model.compute_per_step_credit(
                credit, trajectory["zone_developed"]
            )  # [1, T, A]
            per_step_credit = per_step_credit.squeeze(0).detach()  # [T, A]

            # Build per-agent reward lists with credit-weighted bonus
            per_agent_rewards = []
            for i in range(config.num_agents):
                agent_rewards_list = []
                for k in range(actual_steps):
                    base_r = per_step_rewards[k]
                    credit_bonus = config.credit_weight * per_step_credit[k, i].item() * total_reward
                    agent_rewards_list.append(base_r + credit_bonus)
                per_agent_rewards.append(agent_rewards_list)

            # GeoMACA auxiliary loss update
            loss_computer = GeoMACALossComputer(
                entropy_weight=0.1, consistency_weight=0.0,
                causality_weight=0.02, device="cpu",
                agent_entropy_coef=config.geomaca_agent_entropy_coef,
            )
            credit_fresh, _ = geomaca_model(trajectory, reward_tensor)
            aux_losses = loss_computer.compute(credit_fresh, trajectory)
            geomaca_loss = aux_losses["total"]
            geomaca_optimizer.zero_grad()
            geomaca_loss.backward()
            torch.nn.utils.clip_grad_norm_(geomaca_model.parameters(), max_norm=0.5)
            geomaca_optimizer.step()
        else:
            # During warmup or zero reward: uniform per-agent rewards
            per_agent_rewards = [per_step_rewards[:] for _ in range(config.num_agents)]

        # PPO update for each agent
        total_policy_loss = 0.0
        for agent_idx in range(config.num_agents):
            agent_obs = torch.stack(per_agent_obs[agent_idx])
            agent_actions_t = torch.tensor(per_agent_actions[agent_idx], dtype=torch.long)
            agent_old_lp = torch.stack(per_agent_log_probs[agent_idx])
            values_for_gae = list(per_agent_values[agent_idx])
            rewards_list = per_agent_rewards[agent_idx]
            dones_list = per_step_dones

            advantages, returns = compute_gae(
                rewards_list, values_for_gae, dones_list,
                config.gamma, config.gae_lambda,
            )

            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(config.ppo_epochs):
                logits, values_pred = policies[agent_idx](agent_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(agent_actions_t)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - agent_old_lp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred.squeeze(-1), returns)
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                policy_optimizers[agent_idx].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policies[agent_idx].parameters(), max_norm=config.max_grad_norm)
                policy_optimizers[agent_idx].step()

            total_policy_loss += policy_loss.item()

        history["episode_returns"].append(total_reward)
        history["food_collected"].append(food_collected)
        history["coop_events"].append(coop_events)
        history["episode_lengths"].append(actual_steps)

        if (ep + 1) % config.log_interval == 0:
            recent = slice(-config.log_interval, None)
            avg_ret = np.mean(history["episode_returns"][recent])
            avg_food = np.mean(history["food_collected"][recent])
            avg_coop = np.mean(history["coop_events"][recent])
            phase = "GeoMACA" if use_geomaca else "Warmup"
            logger.info(
                "[GeoMACA-Fixed seed=%d] ep %d/%d [%s] | return=%.3f | food=%.2f | coop=%.2f",
                seed, ep + 1, config.num_episodes, phase, avg_ret, avg_food, avg_coop,
            )

    eval_results = _evaluate_ppo_agents(policies, env, config)
    env.close()

    results = _build_results("geomaca_fixed", seed, config, history, eval_results)
    if attention_log:
        results["attention_log"] = attention_log
    _save_results(results, results_dir, f"geomaca_fixed_seed{seed}.json")
    return results


# ---------------------------------------------------------------------------
# Method B: MAPPO-Independent
# ---------------------------------------------------------------------------

def train_mappo_independent(seed: int, config: FairComparisonConfig, results_dir: Path) -> Dict:
    """MAPPO with shared reward to all agents (independent learning)."""
    return _train_ppo_method("independent", seed, config, results_dir)


# ---------------------------------------------------------------------------
# Method C: MAPPO-Uniform
# ---------------------------------------------------------------------------

def train_mappo_uniform(seed: int, config: FairComparisonConfig, results_dir: Path) -> Dict:
    """MAPPO with reward/N to each agent."""
    return _train_ppo_method("uniform", seed, config, results_dir)


def _train_ppo_method(method: str, seed: int, config: FairComparisonConfig, results_dir: Path) -> Dict:
    """Generic PPO training for independent and uniform methods."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gymnasium.make(config.env_id, max_food_level=config.env_max_food_level)
    env.reset(seed=seed)

    policies = [
        ActorCriticPolicy(config.obs_dim, config.num_actions, config.ppo_hidden_dim)
        for _ in range(config.num_agents)
    ]
    policy_optimizers = [
        torch.optim.Adam(p.parameters(), lr=config.policy_lr) for p in policies
    ]

    history = {"episode_returns": [], "food_collected": [], "coop_events": [], "episode_lengths": []}
    logger.info("Starting %s (seed=%d) for %d episodes", method, seed, config.num_episodes)

    for ep in range(config.num_episodes):
        for p in policies:
            p.train()
        obs, _ = env.reset()
        ue = env.unwrapped

        per_agent_obs = [[] for _ in range(config.num_agents)]
        per_agent_actions = [[] for _ in range(config.num_agents)]
        per_agent_log_probs = [[] for _ in range(config.num_agents)]
        per_agent_values = [[] for _ in range(config.num_agents)]
        per_step_rewards = []
        per_step_dones = []

        total_reward = 0.0
        food_collected = 0
        coop_events = 0

        for step_idx in range(config.max_steps):
            actions = []
            for i, policy in enumerate(policies):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                logits, value = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                per_agent_obs[i].append(obs_tensor.squeeze(0))
                per_agent_actions[i].append(action.item())
                per_agent_log_probs[i].append(log_prob.squeeze().detach())
                per_agent_values[i].append(value.squeeze().item())
                actions.append(int(action.item()))

            obs, rewards, terminated, truncated, info = env.step(tuple(actions))
            raw_reward = sum(float(r) for r in rewards)
            step_reward = raw_reward

            if config.proximity_shaping > 0:
                positions = get_player_positions(env)
                shaping = compute_proximity_reward(ue.field, positions, config.field_size)
                step_reward += config.proximity_shaping * shaping

            total_reward += step_reward
            per_step_rewards.append(step_reward)

            done = terminated or truncated
            per_step_dones.append(done)

            if raw_reward > 0:
                food_collected += 1
                contributing = sum(1 for r in rewards if float(r) > 0)
                if contributing >= 2:
                    coop_events += 1

            if done:
                break

        actual_steps = len(per_step_rewards)

        for i, policy in enumerate(policies):
            if per_step_dones[-1]:
                per_agent_values[i].append(0.0)
            else:
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, bv = policy(obs_tensor)
                per_agent_values[i].append(bv.squeeze().item())

        # PPO update
        for agent_idx in range(config.num_agents):
            agent_obs = torch.stack(per_agent_obs[agent_idx])
            agent_actions_t = torch.tensor(per_agent_actions[agent_idx], dtype=torch.long)
            agent_old_lp = torch.stack(per_agent_log_probs[agent_idx])
            values_for_gae = list(per_agent_values[agent_idx])

            if method == "uniform":
                rewards_list = [r / config.num_agents for r in per_step_rewards]
            else:
                rewards_list = per_step_rewards

            advantages, returns = compute_gae(
                rewards_list, values_for_gae, per_step_dones,
                config.gamma, config.gae_lambda,
            )
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(config.ppo_epochs):
                logits, values_pred = policies[agent_idx](agent_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(agent_actions_t)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - agent_old_lp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred.squeeze(-1), returns)
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                policy_optimizers[agent_idx].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policies[agent_idx].parameters(), max_norm=config.max_grad_norm)
                policy_optimizers[agent_idx].step()

        history["episode_returns"].append(total_reward)
        history["food_collected"].append(food_collected)
        history["coop_events"].append(coop_events)
        history["episode_lengths"].append(actual_steps)

        if (ep + 1) % config.log_interval == 0:
            recent = slice(-config.log_interval, None)
            avg_ret = np.mean(history["episode_returns"][recent])
            avg_food = np.mean(history["food_collected"][recent])
            avg_coop = np.mean(history["coop_events"][recent])
            logger.info(
                "[%s seed=%d] ep %d/%d | return=%.3f | food=%.2f | coop=%.2f",
                method, seed, ep + 1, config.num_episodes, avg_ret, avg_food, avg_coop,
            )

    eval_results = _evaluate_ppo_agents(policies, env, config)
    env.close()

    results = _build_results(method, seed, config, history, eval_results)
    _save_results(results, results_dir, f"{method}_seed{seed}.json")
    return results


# ---------------------------------------------------------------------------
# Method D: QMIX
# ---------------------------------------------------------------------------

def train_qmix(seed: int, config: FairComparisonConfig, results_dir: Path) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gymnasium.make(config.env_id, max_food_level=config.env_max_food_level)
    env.reset(seed=seed)

    state_dim = config.num_agents * config.obs_dim

    q_networks = [AgentQNetwork(config.obs_dim, config.num_actions, config.q_hidden_dim) for _ in range(config.num_agents)]
    target_q_networks = [AgentQNetwork(config.obs_dim, config.num_actions, config.q_hidden_dim) for _ in range(config.num_agents)]
    for i in range(config.num_agents):
        target_q_networks[i].load_state_dict(q_networks[i].state_dict())

    mixing_net = QMIXMixingNetwork(config.num_agents, state_dim, 32)
    target_mixing_net = QMIXMixingNetwork(config.num_agents, state_dim, 32)
    target_mixing_net.load_state_dict(mixing_net.state_dict())

    all_params = list(mixing_net.parameters())
    for qn in q_networks:
        all_params += list(qn.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config.q_lr)

    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    history = {"episode_returns": [], "food_collected": [], "coop_events": [], "episode_lengths": []}

    logger.info("Starting QMIX (seed=%d) for %d episodes", seed, config.num_episodes)

    for ep in range(config.num_episodes):
        obs_list, _ = env.reset()
        ue = env.unwrapped
        total_reward = 0.0
        food_collected = 0
        coop_events = 0
        epsilon = get_epsilon(ep, config.epsilon_start, config.epsilon_end, config.epsilon_decay_episodes)

        for step_idx in range(config.max_steps):
            global_state = get_global_state(obs_list)
            actions = []
            for i in range(config.num_agents):
                if np.random.random() < epsilon:
                    action = np.random.randint(config.num_actions)
                else:
                    obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        q_vals = q_networks[i](obs_tensor)
                    action = q_vals.argmax(dim=-1).item()
                actions.append(action)

            next_obs_list, rewards, terminated, truncated, info = env.step(tuple(actions))
            raw_reward = sum(float(r) for r in rewards)
            step_reward = raw_reward

            if config.proximity_shaping > 0:
                positions = get_player_positions(env)
                shaping = compute_proximity_reward(ue.field, positions, config.field_size)
                step_reward += config.proximity_shaping * shaping

            total_reward += step_reward
            if raw_reward > 0:
                food_collected += 1
                contributing = sum(1 for r in rewards if float(r) > 0)
                if contributing >= 2:
                    coop_events += 1

            done = terminated or truncated

            replay_buffer.push(Transition(
                obs=np.stack(obs_list), actions=np.array(actions),
                reward=step_reward, next_obs=np.stack(next_obs_list),
                done=done, global_state=global_state,
            ))

            obs_list = next_obs_list
            if done:
                break

        # Training
        if len(replay_buffer) >= config.q_batch_size:
            batch = replay_buffer.sample(config.q_batch_size)
            bd = batch_transitions(batch)

            agent_qs = []
            for i in range(config.num_agents):
                q_vals = q_networks[i](bd["obs"][:, i])
                chosen_q = q_vals.gather(1, bd["actions"][:, i].unsqueeze(1)).squeeze(1)
                agent_qs.append(chosen_q)
            agent_qs = torch.stack(agent_qs, dim=1)
            q_tot = mixing_net(agent_qs, bd["global_state"])

            with torch.no_grad():
                target_qs = []
                for i in range(config.num_agents):
                    tq = target_q_networks[i](bd["next_obs"][:, i])
                    target_qs.append(tq.max(dim=-1).values)
                target_qs = torch.stack(target_qs, dim=1)
                target_q_tot = target_mixing_net(target_qs, bd["global_state"])

            targets = bd["rewards"] + config.gamma * (1 - bd["dones"]) * target_q_tot
            loss = F.mse_loss(q_tot, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            optimizer.step()

        if (ep + 1) % config.target_update_interval == 0:
            for i in range(config.num_agents):
                target_q_networks[i].load_state_dict(q_networks[i].state_dict())
            target_mixing_net.load_state_dict(mixing_net.state_dict())

        history["episode_returns"].append(total_reward)
        history["food_collected"].append(food_collected)
        history["coop_events"].append(coop_events)
        history["episode_lengths"].append(step_idx + 1)

        if (ep + 1) % config.log_interval == 0:
            recent = slice(-config.log_interval, None)
            avg_ret = np.mean(history["episode_returns"][recent])
            avg_food = np.mean(history["food_collected"][recent])
            avg_coop = np.mean(history["coop_events"][recent])
            logger.info(
                "[QMIX seed=%d] ep %d/%d | return=%.3f | food=%.2f | coop=%.2f | eps=%.3f",
                seed, ep + 1, config.num_episodes, avg_ret, avg_food, avg_coop, epsilon,
            )

    eval_results = _evaluate_q_agents(q_networks, env, config)
    env.close()
    results = _build_results("qmix", seed, config, history, eval_results)
    _save_results(results, results_dir, f"qmix_seed{seed}.json")
    return results


# ---------------------------------------------------------------------------
# Method E: VDN
# ---------------------------------------------------------------------------

def train_vdn(seed: int, config: FairComparisonConfig, results_dir: Path) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gymnasium.make(config.env_id, max_food_level=config.env_max_food_level)
    env.reset(seed=seed)

    q_networks = [AgentQNetwork(config.obs_dim, config.num_actions, config.q_hidden_dim) for _ in range(config.num_agents)]
    target_q_networks = [AgentQNetwork(config.obs_dim, config.num_actions, config.q_hidden_dim) for _ in range(config.num_agents)]
    for i in range(config.num_agents):
        target_q_networks[i].load_state_dict(q_networks[i].state_dict())

    all_params = []
    for qn in q_networks:
        all_params += list(qn.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config.q_lr)

    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    history = {"episode_returns": [], "food_collected": [], "coop_events": [], "episode_lengths": []}

    logger.info("Starting VDN (seed=%d) for %d episodes", seed, config.num_episodes)

    for ep in range(config.num_episodes):
        obs_list, _ = env.reset()
        ue = env.unwrapped
        total_reward = 0.0
        food_collected = 0
        coop_events = 0
        epsilon = get_epsilon(ep, config.epsilon_start, config.epsilon_end, config.epsilon_decay_episodes)

        for step_idx in range(config.max_steps):
            global_state = get_global_state(obs_list)
            actions = []
            for i in range(config.num_agents):
                if np.random.random() < epsilon:
                    action = np.random.randint(config.num_actions)
                else:
                    obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        q_vals = q_networks[i](obs_tensor)
                    action = q_vals.argmax(dim=-1).item()
                actions.append(action)

            next_obs_list, rewards, terminated, truncated, info = env.step(tuple(actions))
            raw_reward = sum(float(r) for r in rewards)
            step_reward = raw_reward

            if config.proximity_shaping > 0:
                positions = get_player_positions(env)
                shaping = compute_proximity_reward(ue.field, positions, config.field_size)
                step_reward += config.proximity_shaping * shaping

            total_reward += step_reward
            if raw_reward > 0:
                food_collected += 1
                contributing = sum(1 for r in rewards if float(r) > 0)
                if contributing >= 2:
                    coop_events += 1

            done = terminated or truncated

            replay_buffer.push(Transition(
                obs=np.stack(obs_list), actions=np.array(actions),
                reward=step_reward, next_obs=np.stack(next_obs_list),
                done=done, global_state=global_state,
            ))

            obs_list = next_obs_list
            if done:
                break

        if len(replay_buffer) >= config.q_batch_size:
            batch = replay_buffer.sample(config.q_batch_size)
            bd = batch_transitions(batch)

            agent_qs = []
            for i in range(config.num_agents):
                q_vals = q_networks[i](bd["obs"][:, i])
                chosen_q = q_vals.gather(1, bd["actions"][:, i].unsqueeze(1)).squeeze(1)
                agent_qs.append(chosen_q)
            q_tot = torch.stack(agent_qs, dim=1).sum(dim=1)

            with torch.no_grad():
                target_qs = []
                for i in range(config.num_agents):
                    tq = target_q_networks[i](bd["next_obs"][:, i])
                    target_qs.append(tq.max(dim=-1).values)
                target_q_tot = torch.stack(target_qs, dim=1).sum(dim=1)

            targets = bd["rewards"] + config.gamma * (1 - bd["dones"]) * target_q_tot
            loss = F.mse_loss(q_tot, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            optimizer.step()

        if (ep + 1) % config.target_update_interval == 0:
            for i in range(config.num_agents):
                target_q_networks[i].load_state_dict(q_networks[i].state_dict())

        history["episode_returns"].append(total_reward)
        history["food_collected"].append(food_collected)
        history["coop_events"].append(coop_events)
        history["episode_lengths"].append(step_idx + 1)

        if (ep + 1) % config.log_interval == 0:
            recent = slice(-config.log_interval, None)
            avg_ret = np.mean(history["episode_returns"][recent])
            avg_food = np.mean(history["food_collected"][recent])
            avg_coop = np.mean(history["coop_events"][recent])
            logger.info(
                "[VDN seed=%d] ep %d/%d | return=%.3f | food=%.2f | coop=%.2f | eps=%.3f",
                seed, ep + 1, config.num_episodes, avg_ret, avg_food, avg_coop, epsilon,
            )

    eval_results = _evaluate_q_agents(q_networks, env, config)
    env.close()
    results = _build_results("vdn", seed, config, history, eval_results)
    _save_results(results, results_dir, f"vdn_seed{seed}.json")
    return results


# ---------------------------------------------------------------------------
# Method F: COMA
# ---------------------------------------------------------------------------

def train_coma(seed: int, config: FairComparisonConfig, results_dir: Path) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gymnasium.make(config.env_id, max_food_level=config.env_max_food_level)
    env.reset(seed=seed)

    state_dim = config.num_agents * config.obs_dim

    actors = [COMAActorNetwork(config.obs_dim, config.num_actions, config.q_hidden_dim) for _ in range(config.num_agents)]
    actor_optimizers = [torch.optim.Adam(a.parameters(), lr=config.actor_lr) for a in actors]

    critic = COMAcriticNetwork(state_dim, config.num_agents, config.num_actions, 128)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    target_critic = COMAcriticNetwork(state_dim, config.num_agents, config.num_actions, 128)
    target_critic.load_state_dict(critic.state_dict())

    history = {"episode_returns": [], "food_collected": [], "coop_events": [], "episode_lengths": []}
    logger.info("Starting COMA (seed=%d) for %d episodes", seed, config.num_episodes)

    for ep in range(config.num_episodes):
        obs_list, _ = env.reset()
        ue = env.unwrapped
        total_reward = 0.0
        food_collected = 0
        coop_events = 0

        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        ep_states = []

        for step_idx in range(config.max_steps):
            global_state = get_global_state(obs_list)
            ep_states.append(global_state)
            ep_obs.append(np.stack(obs_list))

            actions = []
            for i in range(config.num_agents):
                obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
                logits = actors[i](obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                actions.append(action.item())

            ep_actions.append(np.array(actions))

            next_obs_list, rewards, terminated, truncated, info = env.step(tuple(actions))
            raw_reward = sum(float(r) for r in rewards)
            step_reward = raw_reward

            if config.proximity_shaping > 0:
                positions = get_player_positions(env)
                shaping = compute_proximity_reward(ue.field, positions, config.field_size)
                step_reward += config.proximity_shaping * shaping

            total_reward += step_reward
            ep_rewards.append(step_reward)

            if raw_reward > 0:
                food_collected += 1
                contributing = sum(1 for r in rewards if float(r) > 0)
                if contributing >= 2:
                    coop_events += 1

            done = terminated or truncated
            ep_dones.append(done)
            obs_list = next_obs_list
            if done:
                break

        actual_steps = len(ep_rewards)

        # Compute discounted returns
        returns = []
        running_return = 0.0
        for t in reversed(range(actual_steps)):
            if ep_dones[t]:
                running_return = 0.0
            running_return = ep_rewards[t] + config.gamma * running_return
            returns.insert(0, running_return)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        obs_tensor = torch.tensor(np.stack(ep_obs), dtype=torch.float32)
        actions_tensor = torch.tensor(np.stack(ep_actions), dtype=torch.long)
        states_tensor = torch.tensor(np.stack(ep_states), dtype=torch.float32)

        # Critic update
        critic_loss_total = torch.tensor(0.0)
        for i in range(config.num_agents):
            other_actions_list = []
            for j in range(config.num_agents):
                if j != i:
                    onehot = F.one_hot(actions_tensor[:, j], config.num_actions).float()
                    other_actions_list.append(onehot)
            other_actions = torch.cat(other_actions_list, dim=-1)
            q_values = critic(states_tensor, i, other_actions)
            chosen_q = q_values.gather(1, actions_tensor[:, i].unsqueeze(1)).squeeze(1)
            critic_loss_total = critic_loss_total + F.mse_loss(chosen_q, returns_tensor)

        critic_optimizer.zero_grad()
        critic_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
        critic_optimizer.step()

        # Actor update with counterfactual baseline
        for i in range(config.num_agents):
            other_actions_list = []
            for j in range(config.num_agents):
                if j != i:
                    onehot = F.one_hot(actions_tensor[:, j], config.num_actions).float()
                    other_actions_list.append(onehot)
            other_actions = torch.cat(other_actions_list, dim=-1)

            with torch.no_grad():
                q_values = critic(states_tensor, i, other_actions)
                chosen_q = q_values.gather(1, actions_tensor[:, i].unsqueeze(1)).squeeze(1)
                obs_i = obs_tensor[:, i]
                logits_i = actors[i](obs_i)
                pi_i = F.softmax(logits_i, dim=-1)
                baseline = (pi_i * q_values).sum(dim=-1)
                advantage = chosen_q - baseline

            obs_i = obs_tensor[:, i]
            logits_i = actors[i](obs_i)
            dist = torch.distributions.Categorical(logits=logits_i)
            log_probs = dist.log_prob(actions_tensor[:, i])
            entropy = dist.entropy().mean()
            actor_loss = -(log_probs * advantage).mean() - config.entropy_coef * entropy

            actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actors[i].parameters(), max_norm=10.0)
            actor_optimizers[i].step()

        if (ep + 1) % config.target_update_interval == 0:
            target_critic.load_state_dict(critic.state_dict())

        history["episode_returns"].append(total_reward)
        history["food_collected"].append(food_collected)
        history["coop_events"].append(coop_events)
        history["episode_lengths"].append(actual_steps)

        if (ep + 1) % config.log_interval == 0:
            recent = slice(-config.log_interval, None)
            avg_ret = np.mean(history["episode_returns"][recent])
            avg_food = np.mean(history["food_collected"][recent])
            avg_coop = np.mean(history["coop_events"][recent])
            logger.info(
                "[COMA seed=%d] ep %d/%d | return=%.3f | food=%.2f | coop=%.2f",
                seed, ep + 1, config.num_episodes, avg_ret, avg_food, avg_coop,
            )

    eval_results = _evaluate_actor_agents(actors, env, config)
    env.close()
    results = _build_results("coma", seed, config, history, eval_results)
    _save_results(results, results_dir, f"coma_seed{seed}.json")
    return results


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def _evaluate_ppo_agents(policies: List[ActorCriticPolicy], env, config: FairComparisonConfig) -> Dict:
    eval_returns, eval_food, eval_coop = [], [], []
    for p in policies:
        p.eval()

    for _ in range(config.eval_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_food = 0
        ep_coop = 0

        for _ in range(config.max_steps):
            actions = []
            for i, policy in enumerate(policies):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions.append(dist.sample().item())

            obs, rewards, terminated, truncated, _ = env.step(tuple(actions))
            raw_reward = sum(float(r) for r in rewards)
            ep_reward += raw_reward
            if raw_reward > 0:
                ep_food += 1
                contributing = sum(1 for r in rewards if float(r) > 0)
                if contributing >= 2:
                    ep_coop += 1
            if terminated or truncated:
                break

        eval_returns.append(ep_reward)
        eval_food.append(ep_food)
        eval_coop.append(ep_coop)

    return {
        "mean_return": float(np.mean(eval_returns)),
        "std_return": float(np.std(eval_returns)),
        "mean_food": float(np.mean(eval_food)),
        "std_food": float(np.std(eval_food)),
        "mean_coop": float(np.mean(eval_coop)),
        "std_coop": float(np.std(eval_coop)),
    }


def _evaluate_q_agents(q_networks: List[AgentQNetwork], env, config: FairComparisonConfig) -> Dict:
    eval_returns, eval_food, eval_coop = [], [], []

    for _ in range(config.eval_episodes):
        obs_list, _ = env.reset()
        ep_reward = 0.0
        ep_food = 0
        ep_coop = 0

        for _ in range(config.max_steps):
            actions = []
            for i in range(config.num_agents):
                obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_vals = q_networks[i](obs_tensor)
                actions.append(q_vals.argmax(dim=-1).item())

            obs_list, rewards, terminated, truncated, _ = env.step(tuple(actions))
            raw_reward = sum(float(r) for r in rewards)
            ep_reward += raw_reward
            if raw_reward > 0:
                ep_food += 1
                contributing = sum(1 for r in rewards if float(r) > 0)
                if contributing >= 2:
                    ep_coop += 1
            if terminated or truncated:
                break

        eval_returns.append(ep_reward)
        eval_food.append(ep_food)
        eval_coop.append(ep_coop)

    return {
        "mean_return": float(np.mean(eval_returns)),
        "std_return": float(np.std(eval_returns)),
        "mean_food": float(np.mean(eval_food)),
        "std_food": float(np.std(eval_food)),
        "mean_coop": float(np.mean(eval_coop)),
        "std_coop": float(np.std(eval_coop)),
    }


def _evaluate_actor_agents(actors: List[COMAActorNetwork], env, config: FairComparisonConfig) -> Dict:
    eval_returns, eval_food, eval_coop = [], [], []

    for _ in range(config.eval_episodes):
        obs_list, _ = env.reset()
        ep_reward = 0.0
        ep_food = 0
        ep_coop = 0

        for _ in range(config.max_steps):
            actions = []
            for i in range(config.num_agents):
                obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = actors[i](obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions.append(dist.sample().item())

            obs_list, rewards, terminated, truncated, _ = env.step(tuple(actions))
            raw_reward = sum(float(r) for r in rewards)
            ep_reward += raw_reward
            if raw_reward > 0:
                ep_food += 1
                contributing = sum(1 for r in rewards if float(r) > 0)
                if contributing >= 2:
                    ep_coop += 1
            if terminated or truncated:
                break

        eval_returns.append(ep_reward)
        eval_food.append(ep_food)
        eval_coop.append(ep_coop)

    return {
        "mean_return": float(np.mean(eval_returns)),
        "std_return": float(np.std(eval_returns)),
        "mean_food": float(np.mean(eval_food)),
        "std_food": float(np.std(eval_food)),
        "mean_coop": float(np.mean(eval_coop)),
        "std_coop": float(np.std(eval_coop)),
    }


# ---------------------------------------------------------------------------
# Results Helpers
# ---------------------------------------------------------------------------

def _build_results(method: str, seed: int, config: FairComparisonConfig, history: Dict, eval_results: Dict) -> Dict:
    return {
        "method": method,
        "seed": seed,
        "env_id": config.env_id,
        "config": asdict(config),
        "history": history,
        "eval": eval_results,
    }


def _save_results(results: Dict, results_dir: Path, filename: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", filepath)


# ---------------------------------------------------------------------------
# Run All & Summary
# ---------------------------------------------------------------------------

METHOD_TRAINERS = {
    "geomaca_fixed": train_geomaca,
    "independent": train_mappo_independent,
    "uniform": train_mappo_uniform,
    "qmix": train_qmix,
    "vdn": train_vdn,
    "coma": train_coma,
}

DISPLAY_NAMES = {
    "geomaca_fixed": "GeoMACA (fixed)",
    "independent": "MAPPO-Independent",
    "uniform": "MAPPO-Uniform",
    "qmix": "QMIX",
    "vdn": "VDN",
    "coma": "COMA",
}


def run_all_experiments(
    methods: List[str], seeds: List[int], config: FairComparisonConfig, results_dir: Path
) -> Dict:
    """Run all method x seed combinations and produce a summary."""
    all_results = {}

    total_runs = len(methods) * len(seeds)
    run_idx = 0

    for method in methods:
        all_results[method] = {}
        trainer = METHOD_TRAINERS[method]
        for seed in seeds:
            run_idx += 1
            logger.info("=" * 60)
            logger.info("Run %d/%d: %s (seed=%d)", run_idx, total_runs, DISPLAY_NAMES[method], seed)
            logger.info("=" * 60)
            result = trainer(seed, config, results_dir)
            all_results[method][seed] = result

    # Aggregate summary
    summary = {}
    for method in methods:
        evals = [all_results[method][s]["eval"] for s in seeds]
        mean_returns = [e["mean_return"] for e in evals]
        mean_food = [e["mean_food"] for e in evals]
        mean_coop = [e["mean_coop"] for e in evals]

        summary[method] = {
            "display_name": DISPLAY_NAMES[method],
            "mean_return": float(np.mean(mean_returns)),
            "std_return": float(np.std(mean_returns)),
            "mean_food": float(np.mean(mean_food)),
            "std_food": float(np.std(mean_food)),
            "mean_coop": float(np.mean(mean_coop)),
            "std_coop": float(np.std(mean_coop)),
        }

    # Save summary
    summary_path = results_dir / "fair_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 90)
    print("LBF Fair Comparison Results - Standard (Non-Cooperative) Mode")
    print(f"Environment: {config.env_id} | Episodes: {config.num_episodes} | Seeds: {seeds}")
    print("=" * 90)
    print(f"{'Method':<22} {'Return':>14} {'Food Collected':>18} {'Coop Events':>15}")
    print("-" * 90)
    for method, s in summary.items():
        print(
            f"{s['display_name']:<22}"
            f" {s['mean_return']:>8.3f} +/- {s['std_return']:<5.3f}"
            f" {s['mean_food']:>8.2f} +/- {s['std_food']:<5.2f}"
            f" {s['mean_coop']:>8.2f} +/- {s['std_coop']:<5.2f}"
        )
    print("=" * 90)

    return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

# Settings of the GeoMACA module that --chapter5 selects (Chapter 5).
CHAPTER5_GEOMACA = {
    "geomaca_attention_normalization": "softmax",
    "geomaca_agent_entropy_coef": 0.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LBF Fair Comparison: All 6 Methods")
    parser.add_argument(
        "--method", type=str, default="all",
        choices=list(METHOD_TRAINERS.keys()) + ["all"],
        help="Which method(s) to run.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument(
        "--results-dir", type=str,
        default=str(PACKAGE_ROOT / "results" / "lbf"),
    )
    parser.add_argument(
        "--chapter5", action="store_true",
        help="GeoMACA module as in Chapter 5: "
             + ", ".join(f"{k}={v}" for k, v in CHAPTER5_GEOMACA.items()),
    )
    parser.add_argument(
        "--log-attention-every", type=int, default=0, metavar="N",
        help="GeoMACA only: record the attention distributions of every N-th episode "
             "after the warm-up (0 = off; the run itself is unchanged).",
    )
    return parser


def config_from_args(args) -> FairComparisonConfig:
    extra = dict(CHAPTER5_GEOMACA) if args.chapter5 else {}
    return FairComparisonConfig(
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
        log_attention_every=args.log_attention_every,
        **extra,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)

    config = config_from_args(args)

    results_dir = Path(args.results_dir)

    if args.method == "all":
        methods = list(METHOD_TRAINERS.keys())
    else:
        methods = [args.method]

    run_all_experiments(methods, args.seeds, config, results_dir)


if __name__ == "__main__":
    main()
