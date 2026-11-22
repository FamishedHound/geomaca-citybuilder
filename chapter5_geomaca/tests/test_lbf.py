"""Level-Based Foraging: the Chapter 5 settings of the credit module and a short run."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("lbforaging")

from chapter5_geomaca.lbf import train as lbf  # noqa: E402


def test_defaults_and_chapter5_flag():
    default = lbf.config_from_args(lbf.build_parser().parse_args([]))
    assert (default.geomaca_attention_normalization, default.geomaca_agent_entropy_coef) == ("renormalize", 0.5)
    c5 = lbf.config_from_args(lbf.build_parser().parse_args(["--chapter5"]))
    assert (c5.geomaca_attention_normalization, c5.geomaca_agent_entropy_coef) == ("softmax", 0.0)
    # protocol of the chapter's runs
    assert (c5.num_episodes, c5.eval_episodes, c5.geomaca_warmup) == (50000, 100, 5000)
    assert c5.env_id == "Foraging-8x8-3p-4f-v3" and c5.num_zones == 16
    assert (c5.proximity_shaping, c5.credit_weight) == (0.05, 0.3)


def test_short_geomaca_run(tmp_path, monkeypatch):
    """Three episodes with the credit active from the second: the run uses the second
    softmax and an objective without the consistency and agent-entropy terms."""
    built, models = [], []
    real_lc, real_model = lbf.GeoMACALossComputer, lbf.GeoMACA

    def loss_spy(**kw):
        built.append(kw)
        return real_lc(**kw)

    def model_spy(cfg):
        m = real_model(cfg)
        models.append(m)
        return m

    monkeypatch.setattr(lbf, "GeoMACALossComputer", loss_spy)
    monkeypatch.setattr(lbf, "GeoMACA", model_spy)
    cfg = lbf.FairComparisonConfig(num_episodes=3, eval_episodes=2, geomaca_warmup=1,
                                   log_interval=1, **lbf.CHAPTER5_GEOMACA)
    res = lbf.train_geomaca(7, cfg, tmp_path)
    assert len(res["history"]["episode_returns"]) == 3
    saved = json.loads((tmp_path / "geomaca_fixed_seed7.json").read_text())
    assert saved["config"]["geomaca_attention_normalization"] == "softmax"
    assert models and models[0].attention_normalization == "softmax"
    assert built  # one loss computer per credit episode with a non-zero return
    for kw in built:
        assert kw["consistency_weight"] == 0.0 and kw["agent_entropy_coef"] == 0.0
        assert kw["entropy_weight"] == 0.1 and kw["causality_weight"] == 0.02
