"""The attention-normalisation switch of the GeoMACA module."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from chapter5_geomaca.geomaca.model import GeoMACA, GeoMACAConfig

CITY = dict(state_dim=20, action_dim=5, hidden_dim=128, num_zones=9, num_turns=9,
            num_agents=3, num_heads=4)


def _trajectory(seed: int = 0, identical_zones: bool = False):
    g = torch.Generator().manual_seed(seed)
    global_state = torch.randn(1, 9, 20, generator=g)
    zone_states = (global_state.unsqueeze(2).expand(1, 9, 9, 20).clone() if identical_zones
                   else torch.randn(1, 9, 9, 20, generator=g))
    return {"zone_states": zone_states,
            "agent_actions": torch.rand(1, 9, 3, 5, generator=g),
            "global_state": global_state,
            "zone_developed": torch.arange(9).unsqueeze(0)}


def _head_weights(model: GeoMACA, traj, reward):
    """The head-averaged attention weights p of the three streams (before normalisation)."""
    q = model.reward_encoder(reward.unsqueeze(-1))
    zone = model.zone_encoder(traj["zone_states"].mean(dim=1))
    _, p_geo = model.geographic_attention(q, zone, zone, need_weights=True, average_attn_weights=True)
    temporal_in = torch.cat([traj["global_state"], traj["agent_actions"].reshape(1, 9, -1)], dim=-1)
    temp = model.temporal_encoder(temporal_in)
    _, p_temp = model.temporal_attention(q, temp, temp, need_weights=True, average_attn_weights=True)
    agent = model.agent_encoder(traj["agent_actions"].mean(dim=1))
    _, p_agent = model.agent_attention(q, agent, agent, need_weights=True, average_attn_weights=True)
    return p_temp.squeeze(1), p_geo.squeeze(1), p_agent.squeeze(1)


def _model(mode: str, seed: int = 0) -> GeoMACA:
    torch.manual_seed(seed)
    m = GeoMACA(GeoMACAConfig(**CITY, attention_normalization=mode))
    m.eval()
    return m


def test_default_is_renormalize():
    assert GeoMACAConfig().attention_normalization == "renormalize"
    assert GeoMACA(GeoMACAConfig(**CITY)).attention_normalization == "renormalize"


def test_softmax_mode_applies_a_second_softmax():
    model = _model("softmax")
    traj, r = _trajectory(), torch.tensor([-0.1])
    with torch.no_grad():
        _, (a_temp, a_geo, a_agent) = model(traj, r)
        p_temp, p_geo, p_agent = _head_weights(model, traj, r)
    assert torch.allclose(a_temp, F.softmax(p_temp, dim=-1), atol=0, rtol=0)
    assert torch.allclose(a_geo, F.softmax(p_geo, dim=-1), atol=0, rtol=0)
    assert torch.allclose(a_agent, F.softmax(p_agent, dim=-1), atol=0, rtol=0)


def test_renormalize_mode_divides_by_the_sum():
    model = _model("renormalize")
    traj, r = _trajectory(), torch.tensor([-0.1])
    with torch.no_grad():
        _, (a_temp, a_geo, a_agent) = model(traj, r)
        p_temp, p_geo, p_agent = _head_weights(model, traj, r)
    for a, p in ((a_temp, p_temp), (a_geo, p_geo), (a_agent, p_agent)):
        expected = p.clamp_min(1e-8)
        expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-8)
        assert torch.equal(a, expected)


def test_second_softmax_bounds_every_weight():
    """p lies in [0, 1] (in [0, 1/0.9] under dropout), so softmax(p) is bounded."""
    eval_model = _model("softmax")
    with torch.no_grad():
        _, (a_temp, a_geo, a_agent) = eval_model(_trajectory(1), torch.tensor([-0.12]))
    assert a_temp.max() <= math.e / (math.e + 8) + 1e-6
    assert a_agent.max() <= math.e / (math.e + 2) + 1e-6
    train_model = _model("softmax")
    train_model.train()
    e = math.exp(1 / 0.9)
    with torch.no_grad():
        for seed in range(20):
            _, (a_temp, a_geo, a_agent) = train_model(_trajectory(seed), torch.tensor([-0.1]))
            assert a_temp.max() <= e / (e + 8) + 1e-6
            assert a_geo.max() <= e / (e + 8) + 1e-6
            assert a_agent.max() <= e / (e + 2) + 1e-6


def test_identical_zone_keys_give_uniform_geographic_weights():
    """With every zone's state equal to the city observation, the nine keys coincide."""
    model = _model("softmax")
    with torch.no_grad():
        _, (_, a_geo, _) = model(_trajectory(2, identical_zones=True), torch.tensor([-0.1]))
    assert torch.allclose(a_geo, torch.full_like(a_geo, 1 / 9), atol=1e-6)


def test_modes_share_parameters():
    a, b = _model("renormalize"), _model("softmax")
    assert list(a.state_dict()) == list(b.state_dict())
    b.load_state_dict(a.state_dict())


def test_mode_survives_config_replacement():
    model = _model("softmax")
    model.config = GeoMACAConfig(**CITY)  # as the Level-Based Foraging trainer does every episode
    traj, r = _trajectory(), torch.tensor([-0.1])
    with torch.no_grad():
        _, (a_temp, _, _) = model(traj, r)
        p_temp, _, _ = _head_weights(model, traj, r)
    assert torch.allclose(a_temp, F.softmax(p_temp, dim=-1), atol=0, rtol=0)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        GeoMACA(GeoMACAConfig(**CITY, attention_normalization="sparsemax"))
