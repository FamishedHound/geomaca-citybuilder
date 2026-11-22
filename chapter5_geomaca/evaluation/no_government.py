"""CityBuilder without a government (the evaluation behind Table 5.5).

Two parts:

(a) ``protocol run`` - the Heuristic protocol of Table 5.2 (uniform-random
    developer; 5 replicates with developer seeds 42 + 1000 s, s = 0..4; 50
    episodes each; same metrics and aggregation) with the NoGovernment
    constraints: no affordable-housing, green-space or woodland minimum, no
    density cap (1.0) and no property tax.
(b) ``paired design`` (common random numbers) - every arm sees identical
    developer grids (the same ``np.random.default_rng(seed)`` stream as the
    Heuristic) and identical environment randomness (Python ``random`` and
    NumPy's global generator reseeded with ``seed * 1000 + episode`` before
    each episode). Arms: ``heuristic`` (the default constraints),
    ``no_government``, and two decomposition arms (the default quantity
    constraints with tax 0; no quantity constraints with the default 2 % tax).
    Replicate-level paired t-tests (n = 5) and episode-level paired t and
    Wilcoxon tests (n = 250); Holm's adjustment across the nine metrics.
    Per-zone diagnostics record which constraints actually bind.

Usage::

    python -m chapter5_geomaca.evaluation.no_government --workers 4 \\
        --out chapter5_geomaca/results/evaluation/no_government_evaluation.json

Table 5.5 is made from this file (``no_government_table``).
"""
from __future__ import annotations

import argparse
import math
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy
import torch
from scipy import stats as sstats

from ..citybuilder.baseline_governments import NoGovernment
from ..citybuilder.urban_simulation import DeveloperDecision, GovernmentConstraints, UrbanSimulation
from .evaluate import (DEVICE, HEURISTIC_SEEDS, METRIC_KEYS, N_EVAL, NUM_LAND, GeoMACAConfig,
                       aggregate, get_metrics, per_seed_metrics_to_dict, uniform_grid)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PACKAGE_ROOT / "results" / "evaluation" / "no_government_evaluation.json"

HEUR_SEEDS = HEURISTIC_SEEDS  # == run_heuristic(5)

PAIRED_ARMS = (
    "heuristic",                # default GovernmentConstraints() == Heuristic row
    "no_government",            # NoGovernment
    "default_quantity_no_tax",  # default floors / cap / quota, tax 0
    "no_quantity_default_tax",  # no floors / cap / quota, default tax 0.02
)
CONTRASTS = (
    # (name, arm_a, arm_b): statistics are for arm_a - arm_b
    ("no_government_minus_heuristic", "no_government", "heuristic"),
    ("no_government_minus_default_quantity_no_tax", "no_government", "default_quantity_no_tax"),
    ("no_government_minus_no_quantity_default_tax", "no_government", "no_quantity_default_tax"),
    ("default_quantity_no_tax_minus_heuristic", "default_quantity_no_tax", "heuristic"),
    ("no_quantity_default_tax_minus_heuristic", "no_quantity_default_tax", "heuristic"),
)


def no_government_constraints() -> GovernmentConstraints:
    """NoGovernment (citybuilder/baseline_governments.py) as env constraints.

    NoGovernment stores a BaselineConstraints whose woodland field is
    ``woodland_min_cluster`` (a cluster size), not the environment's
    ``woodland_preservation`` (a minimum area fraction) that develop_zone
    reads; NoGovernment sets it to 0 (no woodland requirement), which maps to
    woodland_preservation = 0.0. All other fields map one-to-one.
    """
    c = NoGovernment().constraints
    if c.woodland_min_cluster != 0:
        raise ValueError("NoGovernment woodland requirement is no longer 0")
    return GovernmentConstraints(
        affordable_housing_pct=float(c.affordable_housing_pct),
        green_space_pct=float(c.green_space_pct),
        woodland_preservation=0.0,
        max_density=float(c.max_density),
        tax_rate=float(c.tax_rate),
    )


def arm_constraints(arm: str) -> GovernmentConstraints:
    d = GovernmentConstraints()
    n = no_government_constraints()
    if arm == "heuristic":
        return d
    if arm == "no_government":
        return n
    if arm == "default_quantity_no_tax":
        return GovernmentConstraints(d.affordable_housing_pct, d.green_space_pct,
                                     d.woodland_preservation, d.max_density, n.tax_rate)
    if arm == "no_quantity_default_tax":
        return GovernmentConstraints(n.affordable_housing_pct, n.green_space_pct,
                                     n.woodland_preservation, n.max_density, d.tax_rate)
    raise ValueError(f"unknown arm {arm}")


# ---- (a) protocol-identical run: mirrors heuristic_episode / run_heuristic --
def nogov_episode(rng: np.random.Generator, cfg: GeoMACAConfig) -> Dict[str, float]:
    env = UrbanSimulation(map_size=cfg.map_size, zone_size=cfg.zone_size)
    for zone_id in range(cfg.num_zones):
        env.develop_zone(zone_id, no_government_constraints(),
                         DeveloperDecision(land_use_grid=uniform_grid(rng)))
    return get_metrics(env)


def run_no_government(n_seeds: int = 5, n_episodes: int = N_EVAL) -> Dict:
    cfg = GeoMACAConfig()
    cfg.device = DEVICE
    per_replicate = []
    for s_idx in range(n_seeds):
        seed = 42 + 1000 * s_idx
        rng = np.random.default_rng(seed)
        eps = [nogov_episode(rng, cfg) for _ in range(n_episodes)]
        per_replicate.append(per_seed_metrics_to_dict(eps))
        gdp = sum(e["gdp"] for e in eps) / len(eps) / 1e6
        pop = sum(e["population"] for e in eps) / len(eps)
        print(f"  no_government seed={seed} GDP=${gdp:.1f}M Pop={pop:.0f}", flush=True)
    return {"method": "no_government", "seeds": [42 + 1000 * s for s in range(n_seeds)],
            "per_replicate": per_replicate, "summary": aggregate(per_replicate)}


def protocol_run(n_episodes: int = N_EVAL) -> Dict:
    torch.set_num_threads(1)
    t0 = time.time()
    res = run_no_government(5, n_episodes)
    res["seconds"] = time.time() - t0
    return res


# ---- (b) + (c) paired common-random-numbers runs with diagnostics ---------
def diag_episode(rng: np.random.Generator, cfg: GeoMACAConfig,
                 constraints: GovernmentConstraints) -> Tuple[Dict[str, float], Dict[str, float]]:
    env = UrbanSimulation(map_size=cfg.map_size, zone_size=cfg.zone_size)
    d: Dict[str, float] = {
        "zones": 0, "green_area_changed": 0, "woodland_area_changed": 0,
        "housing_total_reduced": 0, "housing_mix_changed": 0, "any_allocation_change": 0,
        "violation_green": 0, "violation_woodland": 0, "violation_density": 0,
        "violation_affordable": 0, "violation_penalty": 0.0, "property_tax": 0.0,
        "first_year_profit_sum": 0.0, "immigrants": 0, "turned_away": 0,
        "raw_housing_cells": 0, "built_housing_units": 0, "raw_poor_cells": 0,
        "built_poor_units": 0, "raw_green_cells": 0, "zone_green_area": 0,
        "raw_woodland_cells": 0, "zone_woodland_area": 0,
    }
    for zone_id in range(cfg.num_zones):
        grid = uniform_grid(rng)
        raw = np.bincount(grid.ravel(), minlength=NUM_LAND)
        res = env.develop_zone(zone_id, constraints, DeveloperDecision(land_use_grid=grid))
        zone = env.zones[zone_id]
        hb = res["housing_built"]
        raw_mix = (int(raw[1]), int(raw[2]), int(raw[3]))
        built_mix = (int(hb["poor"]), int(hb["medium"]), int(hb["luxury"]))
        green_changed = int(zone.green_space) != int(raw[4])
        wood_changed = int(zone.woodland) != int(raw[5])
        mix_changed = built_mix != raw_mix
        viol = res.get("constraint_violations", {})
        d["zones"] += 1
        d["green_area_changed"] += int(green_changed)
        d["woodland_area_changed"] += int(wood_changed)
        d["housing_total_reduced"] += int(sum(built_mix) < sum(raw_mix))
        d["housing_mix_changed"] += int(mix_changed)
        d["any_allocation_change"] += int(green_changed or wood_changed or mix_changed)
        d["violation_green"] += int("green_space" in viol)
        d["violation_woodland"] += int("woodland" in viol)
        d["violation_density"] += int("density" in viol)
        d["violation_affordable"] += int("affordable_housing" in viol)
        d["violation_penalty"] += float(res["violation_penalty"])
        d["property_tax"] += float(zone.monthly_tax) * 12.0
        d["first_year_profit_sum"] += float(res["first_year_profit"])
        d["immigrants"] += int(res["immigrants"])
        d["turned_away"] += int(res["turned_away"])
        d["raw_housing_cells"] += sum(raw_mix)
        d["built_housing_units"] += sum(built_mix)
        d["raw_poor_cells"] += raw_mix[0]
        d["built_poor_units"] += built_mix[0]
        d["raw_green_cells"] += int(raw[4])
        d["zone_green_area"] += int(zone.green_space)
        d["raw_woodland_cells"] += int(raw[5])
        d["zone_woodland_area"] += int(zone.woodland)
    m = env.metrics
    for k in ("reputation", "service_level", "city_budget", "unemployment_rate",
              "infrastructure_quality", "housing_vacancy_rate"):
        d[k] = float(getattr(m, k, float("nan")))
    return get_metrics(env), d


def paired_replicate(arm: str, seed: int, n_episodes: int = N_EVAL) -> Dict:
    torch.set_num_threads(1)
    t0 = time.time()
    cfg = GeoMACAConfig()
    cfg.device = DEVICE
    constraints = arm_constraints(arm)
    rng = np.random.default_rng(seed)  # same developer stream as the Heuristic row
    eps, diags = [], []
    for e in range(n_episodes):
        ep_seed = seed * 1000 + e
        random.seed(ep_seed)       # env dynamics (terrain, migration, placement)
        np.random.seed(ep_seed)    # env emigration draws
        met, dg = diag_episode(rng, cfg, constraints)
        eps.append(met)
        diags.append(dg)
    return {"arm": arm, "seed": seed, "episodes": eps, "diagnostics": diags,
            "seconds": time.time() - t0}


# ---- statistics -------------------------------------------------------------
def welch_test(a: List[float], b: List[float]) -> Dict:
    r = sstats.ttest_ind(a, b, equal_var=False)
    return {"t": float(r.statistic), "p": float(r.pvalue)}


def paired_test(a: List[float], b: List[float]) -> Dict:
    """Statistics of a - b over aligned pairs (two-sided)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = int(diff.size)
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    out = {"n": n, "mean_diff": float(diff.mean()), "sd_diff": sd,
           "se_diff": sd / math.sqrt(n) if n else float("nan"),
           "n_exactly_zero": int(np.sum(diff == 0.0))}
    if np.all(diff == 0.0):
        out.update({"t": None, "p_ttest": None, "p_wilcoxon": None,
                    "note": "identical in every pair (difference exactly 0)"})
        return out
    r = sstats.ttest_rel(a, b)
    out["t"] = float(r.statistic)
    out["p_ttest"] = float(r.pvalue)
    try:
        out["p_wilcoxon"] = float(sstats.wilcoxon(diff).pvalue)
    except ValueError:
        out["p_wilcoxon"] = None
    return out


def holm(pvals: List[Optional[float]]) -> List[Optional[float]]:
    idx = [i for i, p in enumerate(pvals) if p is not None and not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    adj: List[Optional[float]] = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvals[i]))
        adj[i] = running
    return adj


def replicate_means(per_rep_eps: List[List[Dict[str, float]]], key: str) -> List[float]:
    return [float(np.mean([e[key] for e in eps])) for eps in per_rep_eps]


def diag_summary(diags: List[Dict[str, float]]) -> Dict[str, float]:
    zones = sum(d["zones"] for d in diags)
    out = {"episodes": len(diags), "zones": zones}
    for k in ("green_area_changed", "woodland_area_changed", "housing_total_reduced",
              "housing_mix_changed", "any_allocation_change", "violation_green",
              "violation_woodland", "violation_density", "violation_affordable"):
        out[f"frac_zones_{k}"] = sum(d[k] for d in diags) / zones
    for k in ("violation_penalty", "property_tax", "first_year_profit_sum", "immigrants",
              "turned_away", "raw_housing_cells", "built_housing_units", "raw_poor_cells",
              "built_poor_units", "raw_green_cells", "zone_green_area", "raw_woodland_cells",
              "zone_woodland_area", "reputation", "service_level", "city_budget",
              "unemployment_rate", "infrastructure_quality", "housing_vacancy_rate"):
        out[f"mean_per_episode_{k}"] = float(np.mean([d[k] for d in diags]))
    return out


# ---- driver -------------------------------------------------------------------
def run(workers: int = 4, n_episodes: int = N_EVAL, selftest_episodes: int = 2,
        keep_episode_diagnostics: bool = False) -> Dict:
    """Run (a) and (b) and return the evaluation record."""
    t_start = time.time()
    print(f"No-government evaluation: workers={workers} seeds={HEUR_SEEDS} "
          f"episodes={n_episodes}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        fut_protocol = ex.submit(protocol_run, n_episodes)
        fut_paired = {(arm, seed): ex.submit(paired_replicate, arm, seed, n_episodes)
                      for arm in PAIRED_ARMS for seed in HEUR_SEEDS}
        fut_selftest = [ex.submit(paired_replicate, "heuristic", HEUR_SEEDS[0],
                                  selftest_episodes) for _ in range(2)]
        paired_raw = {}
        for key, fut in fut_paired.items():
            paired_raw[key] = fut.result()
            print(f"  paired {key[0]:26s} seed={key[1]:5d} done "
                  f"({paired_raw[key]['seconds']:.0f}s)", flush=True)
        selftest_raw = [f.result() for f in fut_selftest]
        protocol = fut_protocol.result()
    wall = time.time() - t_start

    # ---- determinism self-test: 3 independent processes, same arm/seed ------
    ref = paired_raw[("heuristic", HEUR_SEEDS[0])]["episodes"][:selftest_episodes]
    identical = all(st["episodes"] == ref for st in selftest_raw)

    # ---- (b) paired arms -----------------------------------------------------
    arms_out = {}
    for arm in PAIRED_ARMS:
        reps = [paired_raw[(arm, s)] for s in HEUR_SEEDS]
        per_replicate = [per_seed_metrics_to_dict(r["episodes"]) for r in reps]
        arms_out[arm] = {
            "constraints": arm_constraints(arm).__dict__,
            "per_replicate": per_replicate,
            "summary": aggregate(per_replicate),
            "diagnostics_summary": diag_summary([d for r in reps for d in r["diagnostics"]]),
        }
        if keep_episode_diagnostics:
            arms_out[arm]["diagnostics_per_episode"] = [r["diagnostics"] for r in reps]
    contrasts_out = {}
    for name, a_arm, b_arm in CONTRASTS:
        a_eps = [paired_raw[(a_arm, s)]["episodes"] for s in HEUR_SEEDS]
        b_eps = [paired_raw[(b_arm, s)]["episodes"] for s in HEUR_SEEDS]
        per_metric = {}
        for k in METRIC_KEYS:
            rep = paired_test(replicate_means(a_eps, k), replicate_means(b_eps, k))
            epi = paired_test([e[k] for eps in a_eps for e in eps],
                              [e[k] for eps in b_eps for e in eps])
            per_metric[k] = {"replicate_level_n5": rep, "episode_level_n250": epi}
        adj = holm([per_metric[k]["replicate_level_n5"]["p_ttest"] for k in METRIC_KEYS])
        for k, p in zip(METRIC_KEYS, adj):
            per_metric[k]["replicate_level_n5"]["p_holm_9metrics"] = p
        contrasts_out[name] = {"a": a_arm, "b": b_arm, "stat": "a - b", "metrics": per_metric}

    return {
        "description": "CityBuilder without a government (Table 5.5)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": ("Uniform-random developer (per-cell sampling over 6 land-use classes), "
                     f"n=5 replicates with seeds 42+1000*s (s=0..4), {n_episodes} episodes per "
                     "replicate, metrics as in Table 5.2, ddof=1 aggregation across "
                     "replicate means -- identical to the Heuristic row of Table 5.2."),
        "n_eval_per_replicate": n_episodes,
        "constraints": {"heuristic_default": GovernmentConstraints().__dict__,
                        "no_government": no_government_constraints().__dict__},
        "protocol_run": protocol,
        "paired": {
            "description": ("Common random numbers: every arm uses np.random.default_rng(seed) "
                            "for the developer grids (identical grids across arms) and reseeds "
                            "Python random and global np.random with seed*1000+episode before "
                            "every episode. Replicate-level tests use the 5 replicate means "
                            "(paired t, df=4); episode-level tests use the aligned episodes "
                            "(paired t and Wilcoxon signed-rank). Holm adjustment across the "
                            "9 metrics is applied to the replicate-level paired t-test "
                            "p-values within each contrast."),
            "arms": arms_out,
            "contrasts": contrasts_out,
        },
        "determinism_selftest": {
            "description": ("heuristic arm, seed 42, first episodes recomputed in 2 further "
                            "independent worker processes and compared with the main run"),
            "episodes_compared": selftest_episodes,
            "identical": identical,
        },
        "provenance": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
            "workers": workers, "wall_seconds": wall,
        },
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=N_EVAL)
    ap.add_argument("--selftest_episodes", type=int, default=2)
    ap.add_argument("--keep-episode-diagnostics", action="store_true",
                    help="store the per-episode constraint diagnostics as well")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    out = run(args.workers, args.episodes, args.selftest_episodes, args.keep_episode_diagnostics)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {path}  (wall {out['provenance']['wall_seconds'] / 60:.1f} min; "
          f"determinism self-test identical={out['determinism_selftest']['identical']})")


if __name__ == "__main__":
    main()
