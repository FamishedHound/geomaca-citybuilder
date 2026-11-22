"""The memory measurement of the credit module (scaling/memory.py), on the CPU."""

from __future__ import annotations

import pytest

from chapter5_geomaca.scaling import memory


def test_update_uses_the_chapter5_credit_module():
    model, optimizer, losses, traj, reward = memory.build(9)
    assert model.attention_normalization == "softmax" and model.training
    assert model.num_zones == model.num_turns == 9 and model.num_agents == 3
    assert model.hidden_dim == 128 and model.config.num_heads == 4
    assert losses.agent_entropy_coef == 0.0
    assert (losses.entropy_weight, losses.consistency_weight, losses.causality_weight) == (0.1, 0.05, 0.02)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert traj["zone_states"].shape == (1, 9, 9, 20) and traj["agent_actions"].shape == (1, 9, 3, 5)
    assert memory.one_update(model, optimizer, losses, traj, reward) > 0


def test_allocator_events_balance():
    row = memory.measure(9)
    assert row["end_bytes"] == 0  # every allocation inside the update was freed again
    assert 1e6 < row["peak_bytes"] < 1e7 and row["allocator_events"] > 0


def test_peak_at_1000_zones():
    """About 0.2 GB with 1,000 zones (the first figure of the Chapter 6 scalability analysis)."""
    row = memory.measure(1000)
    assert 0.15e9 < row["peak_bytes"] < 0.25e9
    assert row["credit_entries"] == 3_000_000


def test_fit_extrapolation_and_crossing():
    a, b, c = 172.0, 5592.0, 1105224.0
    zones = [1000, 2000, 3000]
    fit = memory.fit_quadratic(zones, [a * n * n + b * n + c for n in zones])
    assert fit["a"] == pytest.approx(a) and fit["b"] == pytest.approx(b, rel=1e-6)
    assert fit["c"] == pytest.approx(c, rel=1e-6)
    assert memory.predict(fit, 10000) == pytest.approx(17.257e9, rel=1e-3)
    n32 = memory.zones_at(fit, 32e9)
    assert memory.predict(fit, n32) == pytest.approx(32e9)
    assert 13_500 < n32 < 13_700
    assert memory.fit_quadratic([1000, 2000], [1.0, 2.0]) is None
