"""The attention record of the LBF trainer and the analysis of what the heads learned."""

import json

import numpy as np
import pytest
import torch

from chapter5_geomaca.lbf.attention_analysis import (
    episode_metrics,
    main,
    reproduction_check,
    seed_summary,
)
from chapter5_geomaca.lbf.train import _attention_record, build_parser, config_from_args


def _record(episode=5000, temp=None, geo=None, agent=None, zones=None, rewards=None):
    steps = 4
    return {
        "episode": episode,
        "steps": steps,
        "total_reward": 1.0,
        "alpha_temp": temp or [0.25] * steps,
        "alpha_geo": geo or [1 / 16] * 16,
        "alpha_agent": agent or [1 / 3] * 3,
        "zone_features": [[0.0, 0.0, 0.0, 0.0]] * 16,
        "zone_developed": zones or [0, 1, 2, 3],
        "agent_rewards": rewards or [[0, 0, 0], [0, 0.5, 0], [0, 0, 0], [0, 0, 0]],
    }


def test_attention_record_keeps_the_three_heads_and_the_zone_inputs():
    attention = (torch.full((1, 3), 1 / 3), torch.full((1, 16), 1 / 16), torch.tensor([[0.5, 0.25, 0.25]]))
    trajectory = {"zone_states": torch.ones(1, 3, 16, 4), "zone_developed": torch.tensor([[0, 5, 5]])}
    record = _attention_record(5000, attention, trajectory, [[0, 0, 0], [0, 1, 0], [0, 0, 0]], 1.0)
    assert record["steps"] == 3
    assert record["zone_developed"] == [0, 5, 5]
    assert len(record["alpha_geo"]) == 16 and len(record["zone_features"]) == 16
    assert record["alpha_agent"] == [0.5, 0.25, 0.25]


def test_logging_is_off_unless_asked():
    args = build_parser().parse_args(["--chapter5"])
    assert config_from_args(args).log_attention_every == 0
    args = build_parser().parse_args(["--chapter5", "--log-attention-every", "25"])
    assert config_from_args(args).log_attention_every == 25


def test_a_uniform_head_has_no_lift_over_the_uniform_share():
    m = episode_metrics(_record())
    assert m["temp_food_share"] == pytest.approx(m["temp_food_uniform"])
    assert m["geo_food_share"] == pytest.approx(m["geo_food_uniform"])
    assert m["credit_food_share"] == pytest.approx(m["credit_food_uniform"])
    assert m["geo_entropy"] == pytest.approx(1.0)


def test_a_head_that_tracks_the_food_step_has_lift():
    m = episode_metrics(_record(temp=[0.1, 0.7, 0.1, 0.1], agent=[0.1, 0.8, 0.1]))
    assert m["temp_food_share"] == pytest.approx(0.7)
    assert m["temp_food_uniform"] == pytest.approx(0.25)
    assert m["agent_top_match"] == 1.0


def test_episodes_without_food_leave_the_food_quantities_undefined():
    m = episode_metrics(_record(rewards=[[0, 0, 0]] * 4))
    assert m["temp_food_share"] is None and m["agent_top_match"] is None


def test_seed_summary_uses_only_the_window():
    records = [_record(episode=5000), _record(episode=45000, temp=[0.1, 0.7, 0.1, 0.1])]
    late = seed_summary(records, (40000, 50000))
    assert late["episodes"] == 1.0 and late["temp_food_share"] == pytest.approx(0.7)


def test_reproduction_check_reports_the_first_differing_episode():
    run = {"history": {"episode_returns": [1, 2, 3], "food_collected": [0, 0, 0], "episode_lengths": [5, 5, 5]}, "eval": {"a": 1}}
    ref = {"history": {"episode_returns": [1, 2, 4], "food_collected": [0, 0, 0], "episode_lengths": [5, 5, 5]}, "eval": {"a": 1}}
    out = reproduction_check(run, ref)
    assert out["episode_returns"] == "differs from episode 2"
    assert out["food_collected"] == "identical" and out["eval"] == "identical"


def test_main_writes_the_summary_and_the_table(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for seed in (1, 2):
        log = [_record(episode=e, temp=[0.1, 0.7, 0.1, 0.1]) for e in (5000, 5025, 45000)]
        payload = {"seed": seed, "history": {"episode_returns": [], "food_collected": [], "episode_lengths": []},
                   "eval": {}, "attention_log": log}
        (runs / f"geomaca_fixed_seed{seed}.json").write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--runs", str(runs), "--out", str(tmp_path / "out")]) == 0
    summary = json.loads((tmp_path / "out" / "attention_summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == [1, 2]
    assert np.isclose(summary["summary"]["late"]["temp_food_share"]["mean"], 0.7)
    assert (tmp_path / "out" / "attention_table.tex").read_text(encoding="utf-8").startswith("\\begin{tabular}")
