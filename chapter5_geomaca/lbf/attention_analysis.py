"""What GeoMACA's attention learned in Level-Based Foraging.

Reads the run records written by ``chapter5_geomaca.lbf.train --log-attention-every N``
(one ``geomaca_fixed_seed<S>.json`` per seed, each with an ``attention_log``), and
asks of each head whether it tracks where the food reward was earned:

* temporal head: the share of its weight on the steps at which food was collected,
  against the share a uniform head would give (food steps / episode length);
* geographic head: the share of its weight on the zones credited at those steps
  (the zone in which an agent loaded food), against the uniform share (those
  zones / 16), and the rank correlation of each zone's weight with the zone's
  food and agent presence (the inputs of the head);
* agent head: whether the most-weighted agent is the one that collected the most
  food, against chance (1/3), in episodes where the agents' food rewards differ;
* the per-step credit that enters the agents' rewards: its share on food steps,
  against the uniform share.

Each quantity is averaged within a seed over the logged episodes of one training
window, then summarised over seeds (mean, SE, and a two-sided one-sample t-test of
the difference from the uniform value, n = number of seeds).  With ``--reference``
the script also checks each run against the record of the reported run of the same
seed: per-episode returns, food counts and episode lengths, and the held-out
evaluation.

Usage (from the folder that contains the package)::

    python -m chapter5_geomaca.lbf.attention_analysis \\
        --runs chapter5_geomaca/results/lbf_attention --reference <reported lbf records> \\
        --out chapter5_geomaca/results/lbf_attention/analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WINDOWS: Dict[str, Tuple[int, int]] = {"early": (5000, 10000), "late": (40000, 50000)}
METRICS: Tuple[Tuple[str, str], ...] = (
    ("temp_food_share", "temp_food_uniform"),
    ("geo_food_share", "geo_food_uniform"),
    ("credit_food_share", "credit_food_uniform"),
    ("agent_top_match", "agent_top_chance"),
)
LABELS = {
    "temp_food_share": "Temporal weight on food steps",
    "geo_food_share": "Geographic weight on food zones",
    "credit_food_share": "Per-step credit on food steps",
    "agent_top_match": "Top agent collected most food",
    "geo_entropy": "Geographic entropy",
    "temp_entropy": "Temporal entropy",
    "agent_entropy": "Agent entropy",
    "geo_rho_food": r"Zone weight vs. food presence ($\rho$)",
    "geo_rho_agent": r"Zone weight vs. agent presence ($\rho$)",
}


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def episode_metrics(record: Dict) -> Dict[str, Optional[float]]:
    """The quantities of one logged episode; None where they are undefined."""
    temp = np.asarray(record["alpha_temp"], dtype=float)
    geo = np.asarray(record["alpha_geo"], dtype=float)
    agent = np.asarray(record["alpha_agent"], dtype=float)
    zones = np.asarray(record["zone_developed"], dtype=int)
    rewards = np.asarray(record["agent_rewards"], dtype=float)  # [T, A]
    features = np.asarray(record["zone_features"], dtype=float)  # [Z, 4]
    steps, n_zones, n_agents = len(temp), len(geo), len(agent)
    food = rewards.sum(axis=1) > 0
    out: Dict[str, Optional[float]] = {
        "geo_entropy": _entropy(geo) / math.log(n_zones),
        "temp_entropy": _entropy(temp) / math.log(steps) if steps > 1 else None,
        "agent_entropy": _entropy(agent) / math.log(n_agents),
        "geo_rho_food": _spearman(geo, features[:, 0]),
        "geo_rho_agent": _spearman(geo, features[:, 2]),
    }
    for key in ("temp_food_share", "temp_food_uniform", "geo_food_share", "geo_food_uniform",
                "credit_food_share", "credit_food_uniform", "agent_top_match", "agent_top_chance"):
        out[key] = None
    if food.any():
        food_zones = np.unique(zones[food])
        per_step = temp * geo[zones]
        out.update(
            temp_food_share=float(temp[food].sum()),
            temp_food_uniform=float(food.sum() / steps),
            geo_food_share=float(geo[food_zones].sum()),
            geo_food_uniform=float(len(food_zones) / n_zones),
            credit_food_share=float(per_step[food].sum() / per_step.sum()),
            credit_food_uniform=float(food.sum() / steps),
        )
        totals = rewards.sum(axis=0)
        if np.ptp(totals) > 0 and (totals == totals.max()).sum() == 1:
            out.update(
                agent_top_match=float(int(np.argmax(agent)) == int(np.argmax(totals))),
                agent_top_chance=1.0 / n_agents,
            )
    return out


def seed_summary(records: Sequence[Dict], window: Tuple[int, int]) -> Dict[str, Optional[float]]:
    """Mean of each quantity over the logged episodes of one window."""
    chosen = [episode_metrics(r) for r in records if window[0] <= int(r["episode"]) < window[1]]
    summary: Dict[str, Optional[float]] = {"episodes": float(len(chosen))}
    keys = chosen[0].keys() if chosen else []
    for key in keys:
        values = [m[key] for m in chosen if m[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def _t_test(differences: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    d = np.asarray(differences, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return None, None
    t = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))
    from scipy import stats

    return t, float(2 * stats.t.sf(abs(t), df=len(d) - 1))


def across_seeds(per_seed: Dict[int, Dict[str, Dict]], window: str) -> Dict[str, Dict]:
    """Mean and SE over seeds of every quantity, and tests against the uniform value."""
    rows: Dict[str, Dict] = {}
    seeds = sorted(per_seed)
    names = sorted({k for s in seeds for k in per_seed[s][window] if k != "episodes"})
    for name in names:
        values = [per_seed[s][window].get(name) for s in seeds if per_seed[s][window].get(name) is not None]
        if not values:
            continue
        row = {"mean": float(np.mean(values)), "n": len(values),
               "se": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None}
        rows[name] = row
    for share, uniform in METRICS:
        pairs = [(per_seed[s][window].get(share), per_seed[s][window].get(uniform)) for s in seeds
                 if per_seed[s][window].get(share) is not None and per_seed[s][window].get(uniform) is not None]
        if pairs and share in rows:
            t, p = _t_test([a - b for a, b in pairs])
            rows[share].update(uniform=float(np.mean([b for _, b in pairs])), t=t, p=p)
    for name in ("geo_entropy", "temp_entropy", "agent_entropy", "geo_rho_food", "geo_rho_agent"):
        if name in rows:
            target = 1.0 if name.endswith("entropy") else 0.0
            values = [per_seed[s][window].get(name) for s in seeds if per_seed[s][window].get(name) is not None]
            t, p = _t_test([v - target for v in values])
            rows[name].update(uniform=target, t=t, p=p)
    return rows


def reproduction_check(run: Dict, reference: Dict) -> Dict[str, object]:
    """Does a re-run equal the reported run of the same seed, episode by episode?"""
    out: Dict[str, object] = {}
    for key in ("episode_returns", "food_collected", "episode_lengths"):
        a, b = run["history"][key], reference["history"][key]
        out[key] = "identical" if a == b else f"differs from episode {next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))}"
    out["eval"] = "identical" if run["eval"] == reference["eval"] else "differs"
    return out


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def latex_table(summary: Dict[str, Dict[str, Dict]]) -> str:
    """Booktabs rows: for each window, mean +- SE over seeds and the value for uniform weights.

    The uniform value is per window: the share of food steps and zones changes as
    the agents learn, so each window is compared with its own reference."""
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Episodes 5{,}000--10{,}000} & \multicolumn{2}{c}{Episodes 40{,}000--50{,}000} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Quantity & GeoMACA & Uniform & GeoMACA & Uniform \\",
        r"\midrule",
    ]
    order = ["temp_food_share", "geo_food_share", "credit_food_share", "agent_top_match",
             "temp_entropy", "geo_entropy", "agent_entropy", "geo_rho_food", "geo_rho_agent"]
    for name in order:
        cells = []
        for window in ("early", "late"):
            row = summary[window].get(name)
            if row is None:
                cells += ["--", "--"]
                continue
            star = "^{*}" if row.get("p") is not None and row["p"] < 0.05 else ""
            cells.append(f"${_fmt(row['mean'])} \\pm {_fmt(row['se'])}{star}$")
            cells.append(_fmt(row.get("uniform")))
        lines.append(f"{LABELS[name]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="folder searched for geomaca_fixed_seed*.json")
    parser.add_argument("--reference", type=Path, default=None, help="folder with the reported runs' records")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = sorted(args.runs.rglob("geomaca_fixed_seed*.json"))
    per_seed: Dict[int, Dict[str, Dict]] = {}
    reproduction: Dict[int, Dict] = {}
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        log = run.get("attention_log")
        if not log:
            continue
        seed = int(run["seed"])
        per_seed[seed] = {name: seed_summary(log, window) for name, window in WINDOWS.items()}
        if args.reference is not None:
            ref = args.reference / f"geomaca_fixed_seed{seed}.json"
            if ref.is_file():
                reproduction[seed] = reproduction_check(run, json.loads(ref.read_text(encoding="utf-8")))
    if not per_seed:
        raise SystemExit(f"no run with an attention_log under {args.runs}")
    summary = {window: across_seeds(per_seed, window) for window in WINDOWS}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "attention_summary.json").write_text(
        json.dumps({"seeds": sorted(per_seed), "windows": WINDOWS, "summary": summary,
                    "per_seed": {str(s): v for s, v in per_seed.items()},
                    "reproduction": {str(s): v for s, v in reproduction.items()}}, indent=2),
        encoding="utf-8")
    (args.out / "attention_table.tex").write_text(latex_table(summary), encoding="utf-8")
    for window in WINDOWS:
        print(f"== {window} {WINDOWS[window]}")
        for name, row in summary[window].items():
            print(f"  {name:20s} mean {_fmt(row['mean'])} se {_fmt(row.get('se'))} "
                  f"uniform {_fmt(row.get('uniform'))} t {_fmt(row.get('t'), 2)} p {_fmt(row.get('p'), 4)} n {row['n']}")
    if reproduction:
        print("reproduction:", json.dumps(reproduction))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
