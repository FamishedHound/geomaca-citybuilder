"""Evaluation pipeline: tables and figure from the records, and one checkpoint end to end."""

from __future__ import annotations

import csv
import json

import pytest

from chapter5_geomaca import reproduce
from chapter5_geomaca.checkpoints.verify import read_manifest, verify
from chapter5_geomaca.evaluation import evaluate
from chapter5_geomaca.evaluation.ablation_table import best_checkpoint_phase
from chapter5_geomaca.tests.expected_tables import CAPTIONS, FIGURE_5_3, TABLE_ROWS

OUTPUTS = {
    "table5.2": "citybuilder_main_table.tex",
    "table5.3": "citybuilder_ranks_table.tex",
    "table5.4": "citybuilder_ablation_table.tex",
    "table5.5": "citybuilder_no_government_table.tex",
    "table5.6": "lbf_train_window_table.tex",
    "table5.7": "lbf_eval_table.tex",
}


def _rows(text: str):
    return [l for l in text.splitlines() if " & " in l and l.rstrip().endswith("\\\\")]


@pytest.fixture(scope="module")
def from_records(tmp_path_factory):
    out = tmp_path_factory.mktemp("from_records")
    reproduce.main(["all", "--from-records", "--results", str(out)])
    return out


@pytest.mark.parametrize("target", sorted(OUTPUTS))
def test_tables_from_records_match_the_thesis(from_records, target):
    text = (from_records / "tables" / OUTPUTS[target]).read_text(encoding="utf-8")
    assert _rows(text) == TABLE_ROWS[target]
    assert [l for l in text.splitlines() if l.startswith("\\caption{")] == [CAPTIONS[target]]


def test_figure_5_3_from_records(from_records):
    heights = json.loads((from_records / "figures" / "cb_pillar_aggregate_bars.json").read_text())
    assert heights == FIGURE_5_3
    assert (from_records / "figures" / "cb_pillar_aggregate_bars.pdf").stat().st_size > 0


def test_records_refer_to_the_manifest_checkpoints(records):
    manifest = {r["path"]: r for r in read_manifest()}
    data = json.loads((records / "citybuilder_evaluation.json").read_text(encoding="utf-8"))
    n = 0
    for method, block in data["methods"].items():
        assert block["seeds_used"] == list(evaluate.DEFAULT_SEEDS[method])
        if method == "heuristic":
            continue
        folder = "cb_runs_v3" if method in ("qmix", "coma") else "cb_runs_v4"
        for run, sha in zip(block["runs"], block["checkpoint_sha256"]):
            assert manifest[f"{folder}/{run}/checkpoints/best.pt"]["sha256"] == sha
            n += 1
    checkpoints = [r for r in manifest.values() if r["role"] == "checkpoint"]
    assert n == len(checkpoints) == 51
    with open(records / "lbf_per_seed.csv", newline="") as f:
        lbf = {f"lbf/{row['file']}" for row in csv.DictReader(f)}
    assert lbf == {p for p, r in manifest.items() if r["role"] == "LBF run record"}


def test_one_checkpoint_end_to_end(archive, tmp_path):
    """GeoMACA seed 789, three episodes: the copied pipeline gives values in the range of
    the 500 GeoMACA evaluation episodes behind Table 5.2."""
    record = evaluate.run_evaluation(archive / "cb_runs_v4", archive / "cb_runs_v3", tmp_path,
                                     methods=("geomaca",), seeds={"geomaca": (789,)},
                                     episodes=3, workers=1, seed_all=True)
    block = record["methods"]["geomaca"]
    manifest = {r["path"]: r["sha256"] for r in read_manifest()}
    assert block["checkpoint_sha256"] == [manifest["cb_runs_v4/geomaca_seed789/checkpoints/best.pt"]]
    episodes = block["per_replicate"][0]
    ranges = {
        "gdp": (2.0e8, 2.8e8), "tax_revenue": (0.0, 2.0e7), "developer_profit": (3.8e8, 5.7e8),
        "green_ratio": (0.12, 0.32), "woodland_ratio": (0.12, 0.32), "density": (0.43, 0.49),
        "population": (1450, 1760), "segregation": (0.03, 0.17), "qol": (4.9, 5.3),
    }
    for key, (lo, hi) in ranges.items():
        assert len(episodes[key]) == 3
        assert all(lo <= v <= hi for v in episodes[key]), (key, episodes[key])
    # this policy sets a tax: its revenue is far from zero (Table 5.2's spread comes from seeds)
    assert min(episodes["tax_revenue"]) > 5e6
    # same seeds, same numbers: the evaluation is repeatable with --seed-all
    again = evaluate.run_evaluation(archive / "cb_runs_v4", archive / "cb_runs_v3", tmp_path / "again",
                                    methods=("geomaca",), seeds={"geomaca": (789,)},
                                    episodes=3, workers=1, seed_all=True)
    assert again["methods"]["geomaca"]["per_replicate"] == block["per_replicate"]
    assert best_checkpoint_phase(archive / "cb_runs_v4" / "geomaca_seed789") == "GeoMACA"


def test_archive_checkpoints_match_the_manifest(archive):
    res = verify(archive, roles=("checkpoint", "run configuration"))
    assert not (res["missing"] or res["size"] or res["sha256"])
    assert len(res["ok"]) == 82
