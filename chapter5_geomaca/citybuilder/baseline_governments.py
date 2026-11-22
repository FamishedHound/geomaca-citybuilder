#!/usr/bin/env python3
"""
Baseline Government Models
==========================

Four baseline government models:
1. Single-Agent Baseline - One network optimizing weighted sum
2. Fixed Policy Baseline - Static hand-tuned constraints  
3. No Government Baseline - Developer acts alone
4. Random Policy Baseline - Random constraints each turn

These are compared against our Three-Agent Architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class BaselineConstraints:
    """Simple constraint class for baselines"""
    affordable_housing_pct: float = 0.30
    green_space_pct: float = 0.20
    woodland_min_cluster: int = 6
    max_density: float = 0.60
    tax_rate: float = 0.0
    
    def to_array(self):
        return np.array([
            self.affordable_housing_pct,
            self.green_space_pct,
            self.woodland_min_cluster / 12.0,  # Normalize
            self.max_density,
            self.tax_rate
        ])


# =============================================================================
# BASELINE B: Single-Agent Government
# =============================================================================

class SingleAgentGovernment(nn.Module):
    """
    Single network that outputs all 5 constraints directly.
    Optimizes weighted sum of all three objectives.
    
    Difference from Three-Agent:
    - One network instead of three
    - No competing proposals
    - Directly outputs constraints
    """
    
    def __init__(self, state_dim=20, hidden_dim=128, device='cpu'):
        super().__init__()
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 5),  # 5 constraints
            nn.Sigmoid()  # Output in [0, 1]
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.experiences = []
        
        # Weights for combined reward
        self.weights = {'economy': 0.33, 'environment': 0.33, 'society': 0.34}
    
    def set_scenario(self, scenario):
        """Set weights based on scenario"""
        scenarios = {
            'balanced': {'economy': 0.33, 'environment': 0.33, 'society': 0.34},
            'growth':   {'economy': 0.50, 'environment': 0.25, 'society': 0.25},
            'green':    {'economy': 0.25, 'environment': 0.50, 'society': 0.25},
            'social':   {'economy': 0.25, 'environment': 0.25, 'society': 0.50},
        }
        self.weights = scenarios[scenario]
    
    def select_action(self, state, training=True, epsilon=0.0):
        """Get constraints from single network"""
        if training and np.random.random() < epsilon:
            constraints = BaselineConstraints()
            constraints.affordable_housing_pct = np.random.uniform(0.2, 0.4)
            constraints.green_space_pct = np.random.uniform(0.1, 0.3)
            constraints.woodland_min_cluster = np.random.randint(4, 13)
            constraints.max_density = np.random.uniform(0.4, 0.8)
            constraints.tax_rate = np.random.uniform(0.0, 0.15)  # NO SUBSIDIES
            return constraints, {'exploration': True}
        
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        
        with torch.no_grad():
            output = self.network(state_t).squeeze(0).cpu().numpy()
        
        constraints = BaselineConstraints(
            affordable_housing_pct=float(output[0] * 0.2 + 0.2),  # [0.2, 0.4]
            green_space_pct=float(output[1] * 0.2 + 0.1),         # [0.1, 0.3]
            woodland_min_cluster=int(output[2] * 8 + 4),          # [4, 12]
            max_density=float(output[3] * 0.4 + 0.4),             # [0.4, 0.8]
            tax_rate=float(output[4] * 0.15)                      # [0.0, 0.15] NO SUBSIDIES
        )
        
        return constraints, {'exploration': False}
    
    def store_experience(self, state, action, rewards, next_state, done):
        """Store experience for training"""
        # Compute combined reward
        combined_reward = (
            self.weights['economy'] * rewards['economy'] +
            self.weights['environment'] * rewards['environment'] +
            self.weights['society'] * rewards['society']
        )
        self.experiences.append({
            'state': state,
            'action': action,
            'reward': combined_reward,
            'next_state': next_state,
            'done': done
        })
        
        if len(self.experiences) > 10000:
            self.experiences = self.experiences[-10000:]
    
    def train_step(self, states, actions, rewards_dict, next_states, dones, gamma=0.99):
        """Train single network on combined reward"""
        # Compute combined rewards
        combined_rewards = (
            self.weights['economy'] * rewards_dict['economy'] +
            self.weights['environment'] * rewards_dict['environment'] +
            self.weights['society'] * rewards_dict['society']
        )
        
        proposals = self.network(states)
        
        # Policy gradient
        log_probs = -0.5 * ((proposals - actions) ** 2).sum(dim=-1)
        advantages = combined_rewards - combined_rewards.mean()
        loss = -(log_probs * advantages).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
    
    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'weights': self.weights
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.weights = checkpoint.get('weights', self.weights)
    
    def eval(self):
        self.network.eval()
    
    def train(self):
        self.network.train()


# =============================================================================
# BASELINE C: Fixed Policy Government
# =============================================================================

class FixedPolicyGovernment:
    """
    No learning. Returns static, hand-tuned constraints every turn.
    
    These are "reasonable" constraints that a human planner might set.
    """
    
    def __init__(self, scenario='balanced'):
        self.scenario = scenario
        self.weights = {'economy': 0.33, 'environment': 0.33, 'society': 0.34}
        
        # Hand-tuned constraints based on scenario
        self.scenario_constraints = {
            'balanced': BaselineConstraints(
                affordable_housing_pct=0.30,
                green_space_pct=0.20,
                woodland_min_cluster=6,
                max_density=0.60,
                tax_rate=0.0
            ),
            'growth': BaselineConstraints(
                affordable_housing_pct=0.25,
                green_space_pct=0.15,
                woodland_min_cluster=4,
                max_density=0.75,
                tax_rate=0.05  # Low tax to encourage development
            ),
            'green': BaselineConstraints(
                affordable_housing_pct=0.30,
                green_space_pct=0.30,
                woodland_min_cluster=10,
                max_density=0.50,
                tax_rate=0.0
            ),
            'social': BaselineConstraints(
                affordable_housing_pct=0.40,
                green_space_pct=0.20,
                woodland_min_cluster=6,
                max_density=0.60,
                tax_rate=0.02  # Low tax (NO SUBSIDIES!) to encourage housing
            )
        }
        self.constraints = self.scenario_constraints[scenario]
    
    def set_scenario(self, scenario):
        self.scenario = scenario
        self.constraints = self.scenario_constraints[scenario]
    
    def select_action(self, state, training=True, epsilon=0.0):
        """Always returns the same constraints"""
        return self.constraints, {'exploration': False, 'fixed': True}
    
    def store_experience(self, *args, **kwargs):
        """No learning, so no experience storage"""
        pass
    
    def train_step(self, *args, **kwargs):
        """No learning"""
        pass
    
    def save(self, path):
        """Nothing to save"""
        pass
    
    def load(self, path):
        """Nothing to load"""
        pass
    
    def eval(self):
        pass
    
    def train(self):
        pass


# =============================================================================
# BASELINE D: No Government
# =============================================================================

class NoGovernment:
    """
    Developer operates with NO constraints.
    
    This tests: Does government regulation matter at all?
    Expected: Higher developer profit, but worse social/environmental outcomes.
    """
    
    def __init__(self, scenario='balanced'):
        self.weights = {'economy': 0.33, 'environment': 0.33, 'society': 0.34}
        
        # Minimal/no constraints
        self.constraints = BaselineConstraints(
            affordable_housing_pct=0.0,   # No requirement
            green_space_pct=0.0,           # No requirement
            woodland_min_cluster=0,        # No requirement
            max_density=1.0,               # No limit
            tax_rate=0.0                   # No tax
        )
    
    def set_scenario(self, scenario):
        pass  # No effect for no-government baseline
    
    def select_action(self, state, training=True, epsilon=0.0):
        """Always returns no constraints"""
        return self.constraints, {'exploration': False, 'no_government': True}
    
    def store_experience(self, *args, **kwargs):
        pass
    
    def train_step(self, *args, **kwargs):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def eval(self):
        pass
    
    def train(self):
        pass


# =============================================================================
# BASELINE E: Random Policy Government
# =============================================================================

class RandomPolicyGovernment:
    """
    Samples random constraints each turn.
    No learning, no adaptation.
    
    This tests: Do learned policies outperform random?
    """
    
    def __init__(self, scenario='balanced'):
        self.weights = {'economy': 0.33, 'environment': 0.33, 'society': 0.34}
    
    def set_scenario(self, scenario):
        pass  # Scenario doesn't affect random policy
    
    def select_action(self, state, training=True, epsilon=0.0):
        """Sample random constraints"""
        constraints = BaselineConstraints(
            affordable_housing_pct=np.random.uniform(0.0, 0.5),
            green_space_pct=np.random.uniform(0.0, 0.4),
            woodland_min_cluster=np.random.randint(0, 13),
            max_density=np.random.uniform(0.3, 1.0),
            tax_rate=np.random.uniform(0.0, 0.15)  # NO SUBSIDIES
        )
        return constraints, {'exploration': False, 'random': True}
    
    def store_experience(self, *args, **kwargs):
        pass
    
    def train_step(self, *args, **kwargs):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def eval(self):
        pass
    
    def train(self):
        pass


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

# The three government agents of Chapter 5 are not built by this factory.
THREE_AGENT_MESSAGE = (
    "create_government('three_agent') is not part of the chapter5_geomaca package. "
    "The three government agents of Chapter 5 are MAPPOGovernment in "
    "chapter5_geomaca.training.train_full_system, trained by "
    "chapter5_geomaca.training.train_with_geomaca."
)
AVAILABLE_MODEL_TYPES = ('single_agent', 'fixed', 'no_gov', 'random')


def create_government(model_type='three_agent', scenario='balanced', device='cpu'):
    """
    Factory function to create government agents.
    
    Args:
        model_type: One of 'single_agent', 'fixed', 'no_gov', 'random'
            ('three_agent' is not part of this package; see THREE_AGENT_MESSAGE)
        scenario: One of 'balanced', 'growth', 'green', 'social'
        device: 'cpu' or 'cuda'
    
    Returns:
        Government agent instance
    """
    if model_type == 'three_agent':
        raise NotImplementedError(THREE_AGENT_MESSAGE)
    elif model_type == 'single_agent':
        gov = SingleAgentGovernment(device=device)
        gov.set_scenario(scenario)
        return gov
    elif model_type == 'fixed':
        return FixedPolicyGovernment(scenario=scenario)
    elif model_type == 'no_gov':
        return NoGovernment()
    elif model_type == 'random':
        return RandomPolicyGovernment()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Baseline Government Models")
    print("=" * 70)
    
    state = np.random.randn(20)
    
    print(f"\nTHREE_AGENT: {THREE_AGENT_MESSAGE}")
    for model_type in AVAILABLE_MODEL_TYPES:
        print(f"\n{model_type.upper()}:")
        gov = create_government(model_type, 'balanced')
        constraints, info = gov.select_action(state)
        print(f"  Affordable: {constraints.affordable_housing_pct:.2f}")
        print(f"  Green: {constraints.green_space_pct:.2f}")
        print(f"  Woodland: {constraints.woodland_min_cluster}")
        print(f"  Max Density: {constraints.max_density:.2f}")
        print(f"  Tax: {constraints.tax_rate:+.2f}")
    
    print("\n" + "=" * 70)
    print("Baseline tests complete!")
    print("=" * 70)

