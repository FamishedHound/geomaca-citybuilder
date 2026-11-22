"""Table 5.4: the GeoMACA component ablation on seven seeds.

All four rows (Uniform credit, Temporal only, Temporal + Agent, Full GeoMACA)
use the seeds 789, 1024, 1337, 2048, 3141, 4096 and 5555; Full GeoMACA is the
GeoMACA column of Table 5.2 restricted to these seeds.

Table: same columns, cell formatter (``fmt``), rounding and bold-best rule as
Table 5.2 (mean +- SE over seeds, ddof=1).

Statistics:
  * Main family: Welch's two-sided t-test of Full GeoMACA against each ablation
    row on each of the four metrics shown in the table (3 x 4 = 12 contrasts),
    with Holm's step-down adjustment over the 12 (scipy.stats.ttest_ind,
    equal_var=False; cross-checked against tables.welch_p).
  * Supplementary: the same test on all nine CityBuilder metrics (27 contrasts,
    Holm over 27).
  * Warm-up checkpoints: per row, how many of the seven evaluated best.pt
    checkpoints come from the 500-episode warm-up, before any GeoMACA credit was
    used. ``best.pt`` stores the training metrics of the last 100 episodes up
    to the saved one; GeoMACA's loss entries appear in them only after the
    warm-up, so the saved episode is a warm-up episode exactly when its own
    entry has none.

Usage::

    python -m chapter5_geomaca.evaluation.ablation_table --eval <citybuilder_evaluation.json> \\
        --checkpoints <archive> --out <dir>

Without checkpoints the warm-up counts are read from
``records/best_checkpoint_phase.json``.

Writes ``citybuilder_ablation_table.tex`` and ``ablation_stats.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scipy import stats as sstats

from .tables import METRICS, fmt, mean_, sd_, se_, welch_p

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = PACKAGE_ROOT / "results" / "evaluation" / "citybuilder_evaluation.json"
DEFAULT_OUT = PACKAGE_ROOT / "results" / "tables"
PHASE_RECORD = PACKAGE_ROOT / "records" / "best_checkpoint_phase.json"

SEEDS_7: Tuple[int, ...] = (789, 1024, 1337, 2048, 3141, 4096, 5555)
ROWS: Tuple[Tuple[str, str], ...] = (
    ("Uniform credit", "abl_uniform"),
    ("Temporal only", "abl_temporal"),
    ("Temporal + Agent", "abl_temporal_agent"),
    ("Full GeoMACA", "geomaca"),
)
ABLATIONS: Tuple[str, ...] = tuple(k for _, k in ROWS[:3])
# (metric key, cell style, higher is better) -- the four columns of the table
ABL_COLS: Tuple[Tuple[str, str, bool], ...] = (
    ("population", "int", True),
    ("qol", "f3", True),
    ("developer_profit", "money_M", True),
    ("segregation", "f3", False),
)
ALPHA = 0.05
WARMUP_EPISODES = 500


# ---------- data access -------------------------------------------------------------------
def seed_values(data: Dict, method: str, key: str, seeds: Sequence[int]) -> List[float]:
    """Replicate means of ``method`` on ``key`` for ``seeds``, in the order given."""
    m = data["methods"][method]
    used = m["seeds_used"]
    missing = [s for s in seeds if s not in used]
    if missing:
        raise KeyError(f"{method}: seeds {missing} not in the evaluation record")
    vals = m["summary"][key]["values"]
    out = [vals[used.index(s)] for s in seeds]
    # consistency: the stored replicate mean equals the mean of the stored episodes
    for s, v in zip(seeds, out):
        eps = m["per_replicate"][used.index(s)][key]
        if abs(sum(eps) / len(eps) - v) > 1e-9 * max(1.0, abs(v)):
            raise ValueError(f"{method} seed {s} {key}: summary value != per-replicate mean")
    return out


# ---------- statistics --------------------------------------------------------------------
def holm(pvals: List[float]) -> List[float]:
    """Holm's step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def welch_full(a: List[float], b: List[float]) -> Dict:
    r = sstats.ttest_ind(a, b, equal_var=False)
    va, vb = sd_(a) ** 2 / len(a), sd_(b) ** 2 / len(b)
    den = va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)
    df = (va + vb) ** 2 / den if den > 0 else float("nan")
    p_check = welch_p(a, b)
    if abs(p_check - float(r.pvalue)) > 1e-4:
        raise ValueError(f"Welch p mismatch: scipy {r.pvalue} vs tables.welch_p {p_check}")
    return {"t": float(r.statistic), "df": df, "p": float(r.pvalue), "p_tables_welch_p": p_check}


def contrasts(data: Dict, metrics: Sequence[Tuple[str, bool]], family: str) -> List[Dict]:
    rows = []
    for abl in ABLATIONS:
        for key, hb in metrics:
            g = seed_values(data, "geomaca", key, SEEDS_7)
            a = seed_values(data, abl, key, SEEDS_7)
            rows.append({"ablation": abl, "metric": key, "higher_is_better": hb,
                         "n_geomaca": len(g), "n_ablation": len(a),
                         "mean_geomaca": mean_(g), "se_geomaca": se_(g),
                         "mean_ablation": mean_(a), "se_ablation": se_(a),
                         "diff_geomaca_minus_ablation": mean_(g) - mean_(a), **welch_full(g, a)})
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r[f"p_holm_{family}"] = p
    return rows


# ---------- warm-up checkpoints -----------------------------------------------------------
def best_checkpoint_phase(run_dir: Path) -> str:
    """'warm-up' if the run's best.pt was saved during the warm-up, else 'GeoMACA'."""
    from .citybuilder_pipeline import load_checkpoint

    ckpt = load_checkpoint(Path(run_dir) / "checkpoints" / "best.pt")
    last = ckpt["metrics_history"][-1]
    return "GeoMACA" if "geomaca_total_loss" in last else "warm-up"


def phases_from_checkpoints(geomaca_runs: Path) -> Dict[str, Dict[str, str]]:
    return {method: {str(s): best_checkpoint_phase(Path(geomaca_runs) / f"{method}_seed{s}")
                     for s in SEEDS_7}
            for _, method in ROWS}


def phases_from_record(path: Path = PHASE_RECORD) -> Dict[str, Dict[str, str]]:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    return {method: {str(s): rec["phase"][method][str(s)] for s in SEEDS_7} for _, method in ROWS}


def warmup_table(phases: Dict[str, Dict[str, str]]) -> Dict:
    out = {}
    for _, method in ROWS:
        ph = phases[method]
        out[method] = {
            "n": len(SEEDS_7),
            "warmup_count": sum(ph[str(s)] == "warm-up" for s in SEEDS_7),
            "warmup_seeds": [s for s in SEEDS_7 if ph[str(s)] == "warm-up"],
            "phase": {str(s): ph[str(s)] for s in SEEDS_7},
        }
    return out


# ---------- LaTeX -------------------------------------------------------------------------
def caption(warm: Dict, main: List[Dict]) -> str:
    counts = [warm[k]["warmup_count"] for _, k in ROWS]
    min_p = min(c["p"] for c in main)
    min_holm = min(c["p_holm_12"] for c in main)
    if min_p >= ALPHA:
        when = "even before Holm's correction"
    elif min_holm >= ALPHA:
        when = "after Holm's correction"
    else:
        when = None
    n = len(SEEDS_7)
    text = ("GeoMACA component ablation ($n{=}" + str(n) + "$ seeds per row). Each cell is mean "
            "$\\pm$ SE across seeds under the same unified uniform-developer evaluation protocol as "
            "Tab.~\\ref{tab:p3-main}; Full GeoMACA is the GeoMACA column of Tab.~\\ref{tab:p3-main} "
            "restricted to these seven seeds. Bold marks the best mean. The evaluated checkpoint comes "
            "from the warm-up for " + ", ".join(f"${c}$" for c in counts[:-1]) + f", and ${counts[-1]}$ "
            "of the seven seeds in the four rows.")
    if when is not None:
        text += (" No ablation differs significantly from Full GeoMACA on any of the four metrics, "
                 + when + " (Welch's $t$-tests).")
    else:
        sig = [c for c in main if c["p_holm_12"] < ALPHA]
        text += (f" {len(sig)} of the 12 comparisons with Full GeoMACA on these metrics are "
                 "significant after Holm's correction (Welch's $t$-tests).")
    return text


def tex_table(data: Dict, cap: str, source: str = "") -> Tuple[str, Dict]:
    bests = {}
    for key, _, hb in ABL_COLS:
        vals = [mean_(seed_values(data, k, key, SEEDS_7)) for _, k in ROWS]
        bests[key] = max(vals) if hb else min(vals)
    lines = ["% Table 5.4 -- generated by chapter5_geomaca.evaluation.ablation_table"
             + (f" from {source}" if source else ""),
             "% All four rows: seeds " + ", ".join(map(str, SEEDS_7)) + ".",
             "\\begin{table}[t]\\centering\\small",
             "\\caption{" + cap + "}",
             "\\label{tab:p3-ablation}",
             "\\begin{tabular}{l" + "c" * len(ABL_COLS) + "}",
             "\\toprule",
             "\\textbf{Configuration} & \\textbf{Population $\\uparrow$} & \\textbf{QoL $\\uparrow$} & "
             "\\textbf{Profit (\\$M) $\\uparrow$} & \\textbf{Segregation $\\downarrow$} \\\\",
             "\\midrule"]
    cells_out: Dict[str, Dict[str, Dict]] = {}
    for lab, k in ROWS:
        cells, cells_out[k] = [], {}
        for key, sty, _ in ABL_COLS:
            xs = seed_values(data, k, key, SEEDS_7)
            mu, se = mean_(xs), se_(xs)
            cell = fmt(mu, se, sty)
            best = abs(mu - bests[key]) < 1e-9
            if best:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
            cells_out[k][key] = {"cell": cell, "best": best}
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}\\end{table}")
    return "\n".join(lines) + "\n", cells_out


def summary_block(data: Dict) -> Dict:
    out = {}
    for _, k in ROWS:
        out[k] = {}
        for key, _, _, _ in METRICS:
            xs = seed_values(data, k, key, SEEDS_7)
            out[k][key] = {"n": len(xs), "mean": mean_(xs), "sd": sd_(xs), "se": se_(xs),
                           "values_by_seed": {str(s): v for s, v in zip(SEEDS_7, xs)}}
    return out


def build(eval_path: Path, phases: Dict[str, Dict[str, str]]) -> Tuple[str, Dict]:
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    warm = warmup_table(phases)
    main_family = contrasts(data, [(k, hb) for k, _, hb in ABL_COLS], "12")
    supp_family = contrasts(data, [(k, hb) for k, _, _, hb in METRICS], "27")
    cap = caption(warm, main_family)
    tex, cells = tex_table(data, cap, Path(eval_path).name)
    res = {
        "evaluation": Path(eval_path).name,
        "seeds": list(SEEDS_7),
        "rows": {k: lab for lab, k in ROWS},
        "test_main": "Welch two-sided t-test, Full GeoMACA vs each ablation, four table metrics, "
                     "n=7 vs n=7 replicate means; Holm over the 12 contrasts",
        "test_supplementary": "same test on all nine metrics; Holm over the 27 contrasts",
        "summary": summary_block(data),
        "table_cells": cells,
        "caption": cap,
        "contrasts_table_metrics_holm12": main_family,
        "contrasts_all_metrics_holm27_supplementary": supp_family,
        "min_p_table_metrics": min(c["p"] for c in main_family),
        "min_p_all_metrics": min(c["p"] for c in supp_family),
        "warmup_checkpoints": warm,
    }
    return tex, res


def write(eval_path: Path, out_dir: Path, geomaca_runs: Optional[Path] = None) -> List[Path]:
    phases = (phases_from_checkpoints(geomaca_runs)
              if geomaca_runs is not None and Path(geomaca_runs).exists()
              else phases_from_record())
    tex, res = build(eval_path, phases)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_tex = out_dir / "citybuilder_ablation_table.tex"
    p_json = out_dir / "ablation_stats.json"
    p_tex.write_text(tex, encoding="utf-8")
    p_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return [p_tex, p_json]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=str, default=str(DEFAULT_EVAL))
    ap.add_argument("--checkpoints", type=str, default=os.environ.get("CHAPTER5_CHECKPOINTS"),
                    help="checkpoint archive root (for the warm-up counts); default $CHAPTER5_CHECKPOINTS")
    ap.add_argument("--geomaca-runs", type=str, default=None,
                    help="folder with the GeoMACA and ablation runs (default <checkpoints>/cb_runs_v4)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    runs = (Path(args.geomaca_runs) if args.geomaca_runs
            else (Path(args.checkpoints) / "cb_runs_v4" if args.checkpoints else None))
    for p in write(Path(args.eval), Path(args.out), runs):
        print(f"Wrote {p}")
    res = json.loads((Path(args.out) / "ablation_stats.json").read_text(encoding="utf-8"))
    print("Warm-up checkpoints (of 7):",
          {k: (v["warmup_count"], v["warmup_seeds"]) for k, v in res["warmup_checkpoints"].items()})
    print(f"min p (4 table metrics) = {res['min_p_table_metrics']:.4g}; "
          f"min p (9 metrics) = {res['min_p_all_metrics']:.4g}")


if __name__ == "__main__":
    main()
