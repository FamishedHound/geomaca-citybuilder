"""The Chapter 5 configuration of the GeoMACA trainer and the launcher commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from chapter5_geomaca.training import launch
from chapter5_geomaca.training.train_with_geomaca import (
    CHAPTER5_METHOD,
    GeoMACATrainingConfig,
    build_parser,
    check_method_switches,
    config_from_args,
    patience_from_args,
)

SWITCHES = ("terminal_reward", "government_advantage", "zone_keys",
            "attention_normalization", "agent_entropy_coef")


def _config(argv):
    args = build_parser().parse_args(argv)
    return args, config_from_args(args)


def test_chapter5_method():
    assert CHAPTER5_METHOD == {
        "terminal_reward": "neg_segregation",
        "government_advantage": "total_credit",
        "zone_keys": "city_observation",
        "attention_normalization": "softmax",
        "agent_entropy_coef": 0.0,
    }


def test_defaults_are_unchanged_without_switches():
    cfg = GeoMACATrainingConfig()
    assert (cfg.terminal_reward, cfg.government_advantage, cfg.zone_keys,
            cfg.attention_normalization, cfg.agent_entropy_coef) == (
        "pillar_aggregate", "per_step", "zone_features", "renormalize", 0.5)
    _, cfg = _config([])
    assert {k: getattr(cfg, k) for k in SWITCHES} == {
        "terminal_reward": "pillar_aggregate", "government_advantage": "per_step",
        "zone_keys": "zone_features", "attention_normalization": "renormalize",
        "agent_entropy_coef": 0.5}


def test_chapter5_flag_selects_the_chapter_configuration():
    _, cfg = _config(["--chapter5"])
    assert {k: getattr(cfg, k) for k in SWITCHES} == CHAPTER5_METHOD


def test_explicit_switches_equal_the_flag():
    _, a = _config(["--chapter5"])
    _, b = _config(["--terminal_reward", "neg_segregation", "--government_advantage", "total_credit",
                    "--zone_keys", "city_observation", "--attention_normalization", "softmax",
                    "--agent_entropy_coef", "0"])
    assert a == b


def test_flag_and_explicit_switch_conflict():
    args = build_parser().parse_args(["--chapter5", "--zone_keys", "zone_features"])
    with pytest.raises(ValueError):
        config_from_args(args)


def test_launcher_command_gives_the_chapter_protocol(tmp_path):
    """The command of a Table 5.2 run: Chapter 5 switches, 2,000 episodes at most,
    500 warm-up episodes, early stopping after 500 episodes without an improvement
    of more than 0.01 (counted from the first episode), windowed patience rule off."""
    cmd = launch.geomaca_command(42, tmp_path / "geomaca_seed42", python="python")
    assert cmd[:4] == ["python", "-u", "-m", "chapter5_geomaca.training.train_with_geomaca"]
    args, cfg = _config(cmd[4:])
    assert {k: getattr(cfg, k) for k in SWITCHES} == CHAPTER5_METHOD
    assert cfg.num_episodes == 2000 and cfg.geomaca_warmup == 500
    assert cfg.early_stop_enabled is True
    assert cfg.early_stop_patience == 500 and cfg.early_stop_min_delta == 0.01
    assert patience_from_args(args) is None
    assert (cfg.use_temporal, cfg.use_geographic, cfg.use_agent) == (True, True, True)
    assert args.seed == 42 and Path(args.save_path) == tmp_path / "geomaca_seed42"
    # credit-module weights of Sec. 5.4.3 and learning rates of Table 5.1
    assert (cfg.geomaca_entropy_weight, cfg.geomaca_consistency_weight,
            cfg.geomaca_causality_weight) == (0.1, 0.05, 0.02)
    assert cfg.geomaca_lr == 1e-4 and cfg.gov_lr == 3e-4 and cfg.dev_lr * 0.1 == pytest.approx(3e-5)
    assert cfg.ppo_epochs == 4 and cfg.geomaca_hidden_dim == 128 and cfg.geomaca_num_heads == 4
    assert cfg.credit_mode == "aggregate" and cfg.geomaca_combination == "multiplicative"


def test_launcher_groups(tmp_path):
    geo = launch.build_jobs("geomaca", tmp_path, 2000)
    assert [j.label for j in geo] == [f"geomaca_seed{s}" for s in launch.ALL_SEEDS]
    abl = launch.build_jobs("ablations", tmp_path, 2000)
    assert len(abl) == 21
    flags = {}
    for j in abl:
        args, cfg = _config(j.cmd[4:])
        flags[j.label.rsplit("_seed", 1)[0]] = (cfg.use_temporal, cfg.use_geographic, cfg.use_agent)
        assert {k: getattr(cfg, k) for k in SWITCHES} == CHAPTER5_METHOD
        assert args.seed in launch.ABLATION_SEEDS
        assert j.save_path.parent == tmp_path / "cb_runs_v4"
    assert flags == {"abl_uniform": (False, False, False), "abl_temporal": (True, False, False),
                     "abl_temporal_agent": (True, False, True)}
    base = launch.build_jobs("baselines", tmp_path, 2000)
    assert len(base) == 20
    q = base[0].cmd
    assert q[3] == "chapter5_geomaca.training.baselines.qmix_baseline"
    assert q[4:] == ["--seed", "42", "--num_episodes", "2000", "--save_path",
                     str(tmp_path / "cb_runs_v3" / "qmix_seed42"), "--patience", "0",
                     "--patience_min_episodes", "0", "--use_uniform_developer", "True"]


@pytest.mark.parametrize("field,value", [("terminal_reward", "mean"), ("government_advantage", "gae"),
                                         ("zone_keys", "grid"), ("attention_normalization", "none")])
def test_unknown_switch_values_are_rejected(field, value):
    cfg = GeoMACATrainingConfig(**{field: value})
    with pytest.raises(ValueError):
        check_method_switches(cfg)


def test_switches_need_the_aggregate_credit_mode():
    cfg = GeoMACATrainingConfig(credit_mode="multi_objective", government_advantage="total_credit")
    with pytest.raises(ValueError):
        check_method_switches(cfg)
