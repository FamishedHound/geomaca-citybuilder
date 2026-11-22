"""The code paths the Chapter 5 configuration selects in a training episode (CPU)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from chapter5_geomaca.training.train_with_geomaca import (
    CHAPTER5_METHOD,
    GeoMACATrainer,
    GeoMACATrainingConfig,
)


def _trainer(tmp_path, **switches) -> GeoMACATrainer:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    cfg = GeoMACATrainingConfig(num_episodes=2, geomaca_warmup=1, device="cpu",
                                early_stop_enabled=True, **switches)
    return GeoMACATrainer(cfg, save_path=str(tmp_path / "run"))


def _instrument(trainer: GeoMACATrainer) -> dict:
    """Record the government updates, credit-module calls and losses of each episode."""
    log = {"gov": [], "forward": [], "losses": []}
    gov, forward, compute = trainer._update_government, trainer.geomaca.forward, trainer.geomaca_losses.compute

    def gov_spy(agent_rewards=None, agent_step_advantages=None):
        log["gov"].append({
            "agent_rewards": None if agent_rewards is None else agent_rewards.detach().clone(),
            "agent_step_advantages": (None if agent_step_advantages is None
                                      else agent_step_advantages.detach().clone())})
        return gov(agent_rewards=agent_rewards, agent_step_advantages=agent_step_advantages)

    def forward_spy(trajectory, final_reward):
        out = forward(trajectory, final_reward)
        log["forward"].append({"trajectory": {k: v.detach().clone() for k, v in trajectory.items()},
                               "final_reward": final_reward.detach().clone(),
                               "credit": out[0].detach().clone(),
                               "alphas": tuple(a.detach().clone() for a in out[1])})
        return out

    def compute_spy(credit, trajectory, final_reward=None):
        out = compute(credit, trajectory, final_reward)
        log["losses"].append({k: v.detach().clone() for k, v in out.items()})
        return out

    trainer._update_government = gov_spy
    trainer.geomaca.forward = forward_spy
    trainer.geomaca_losses.compute = compute_spy
    return log


def _play(trainer: GeoMACATrainer, n: int):
    metrics = []
    for _ in range(n):
        m = trainer.run_episode()
        trainer.metrics_history.append(m)  # as train() does: sets the episode index
        metrics.append(m)
    return metrics


@pytest.fixture(scope="module")
def chapter5_run(tmp_path_factory):
    trainer = _trainer(tmp_path_factory.mktemp("c5"), **CHAPTER5_METHOD)
    log = _instrument(trainer)
    metrics = _play(trainer, 2)  # episode 0: warm-up; episode 1: GeoMACA
    return trainer, log, metrics


def test_warmup_episode_takes_the_developer_reward_fallback(chapter5_run):
    """No credit in the warm-up: the update falls back to the developer reward (sum / 3)."""
    _, log, _ = chapter5_run
    first = log["gov"][0]
    assert first["agent_rewards"] is None and first["agent_step_advantages"] is None
    assert len(log["forward"]) == 1  # the credit module is idle in the warm-up episode


def test_terminal_reward_is_negative_segregation(chapter5_run):
    trainer, log, metrics = chapter5_run
    m = metrics[1]
    assert m["terminal_credit_reward"] == -m["segregation"]
    assert m["normalized_terminal_reward"] == -m["segregation"]
    r = log["forward"][0]["final_reward"]
    assert r.shape == (1,) and r.item() == pytest.approx(-m["segregation"], abs=1e-7)
    assert trainer.reward_normalizer.count == 0  # no online normalisation


def test_every_zone_state_is_the_city_observation(chapter5_run):
    _, log, _ = chapter5_run
    traj = log["forward"][0]["trajectory"]
    zs = traj["zone_states"]
    assert zs.shape == (1, 9, 9, 20)
    for z in range(9):
        assert torch.equal(zs[:, :, z], traj["global_state"])
    assert torch.equal(traj["zone_developed"], torch.arange(9).unsqueeze(0))


def test_government_advantage_is_the_total_credit(chapter5_run):
    _, log, _ = chapter5_run
    call = log["gov"][1]
    assert call["agent_step_advantages"] is None
    fwd = log["forward"][0]
    expected = fwd["credit"].sum(dim=(1, 2)) * fwd["final_reward"].view(-1, 1)
    assert torch.equal(call["agent_rewards"], expected)
    a_agent = fwd["alphas"][2]
    assert torch.allclose(call["agent_rewards"], a_agent * fwd["final_reward"], atol=1e-7)
    assert call["agent_rewards"].sum().item() == pytest.approx(fwd["final_reward"].item(), abs=1e-6)


def test_second_softmax_is_used(chapter5_run):
    trainer, log, _ = chapter5_run
    assert trainer.geomaca.attention_normalization == "softmax"
    e = math.exp(1 / 0.9)
    a_temp, a_geo, a_agent = log["forward"][0]["alphas"]
    assert a_temp.max() <= e / (e + 8) + 1e-6
    assert a_geo.max() <= e / (e + 8) + 1e-6 and a_geo.min() >= 1 / (e + 8) - 1e-6
    assert a_agent.max() <= e / (e + 2) + 1e-6


def test_objective_is_entropy_consistency_causality(chapter5_run):
    trainer, log, _ = chapter5_run
    assert trainer.geomaca_losses.agent_entropy_coef == 0.0
    loss = log["losses"][0]
    expected = (torch.tensor(0.0) + 0.1 * loss["entropy"] + 0.05 * loss["consistency"]
                + 0.02 * loss["causality"])
    assert torch.equal(loss["total"], expected)


def test_default_switches_select_the_other_paths(tmp_path):
    trainer = _trainer(tmp_path)
    log = _instrument(trainer)
    metrics = _play(trainer, 2)
    call = log["gov"][1]
    assert call["agent_rewards"] is None and call["agent_step_advantages"].shape == (1, 9, 3)
    zs = log["forward"][0]["trajectory"]["zone_states"]
    assert not all(torch.equal(zs[:, :, z], zs[:, :, 0]) for z in range(1, 9))
    assert trainer.reward_normalizer.count == 2
    assert metrics[1]["terminal_credit_reward"] != -metrics[1]["segregation"]
    loss = log["losses"][0]
    expected = (torch.tensor(0.0) + 0.1 * loss["entropy"] + 0.5 * 0.1 * loss["agent_entropy"]
                + 0.05 * loss["consistency"] + 0.02 * loss["causality"])
    assert torch.equal(loss["total"], expected)
    assert trainer.geomaca.attention_normalization == "renormalize"


def test_early_stopping_rule(tmp_path, monkeypatch):
    """Stop after 500 consecutive episodes whose summed developer reward does not beat
    the best so far by more than 0.01, counting from the first episode; best.pt is the
    episode with the highest reward (any improvement)."""
    trainer = _trainer(tmp_path, **CHAPTER5_METHOD)
    rewards = iter([0.0] * 10 + [0.5] + [0.505] * 2000)
    monkeypatch.setattr(trainer, "run_episode", lambda: {
        "episode_reward": next(rewards), "episode_profit": 0.0, "segregation": 0.1,
        "population": 1500})
    saved = []
    monkeypatch.setattr(trainer, "save_checkpoint",
                        lambda name: saved.append((name, len(trainer.metrics_history) - 1)))
    monkeypatch.setattr(trainer, "_validate_model", lambda num_episodes=30: True)
    monkeypatch.setattr(trainer, "_generate_final_visualizations", lambda: None)
    trainer.train(num_episodes=2000, patience_episodes=None)
    assert len(trainer.metrics_history) == 511  # last improvement > 0.01 at episode 10
    assert [i for name, i in saved if name == "best"] == [0, 10, 11]
    assert saved[-1] == ("final", 510)
