"""CityBuilder evaluation under one protocol for every method.

Every method is rolled out with the same developer: each cell of the active
20 x 20 zone is drawn uniformly at random from the six land-use classes.
Only the government side differs between methods.

Replicates (one replicate = one trained policy, or one Heuristic seed,
evaluated for 50 episodes):

* ``heuristic`` - the fixed constraint vector (0.25, 0.15, 0.10, 0.70, 0.02)
  in every zone; 5 replicates with developer seeds 42 + 1000 s, s = 0..4.
* ``geomaca`` - ``<geomaca-runs>/geomaca_seed<k>/``, k in SEEDS.
* ``abl_uniform``, ``abl_temporal``, ``abl_temporal_agent`` -
  ``<geomaca-runs>/<method>_seed<k>/``, k in ABLATION_SEEDS.
* ``qmix``, ``coma`` - ``<baseline-runs>/<method>_seed<k>/``, k in SEEDS.

A learned policy is evaluated from ``checkpoints/best.pt`` (and, for the
GeoMACA family, the run's ``config.json``). Before its episodes a replicate
calls ``torch.manual_seed(k)`` and ``np.random.seed(k)`` and draws the
developer's cells from ``np.random.default_rng(k)``. The background terrain is
drawn with Python's ``random`` module, which this protocol does not seed, so
two runs give slightly different values; ``--seed-all`` also seeds ``random``
(and, for the Heuristic, NumPy's global generator) with the replicate seed,
which makes a run repeatable.

Usage (from the folder that contains the package)::

    python -m chapter5_geomaca.evaluation.evaluate --checkpoints <archive> --workers 4

Output, under ``--out`` (default ``chapter5_geomaca/results/evaluation``):

* ``parts/<method>_seed<k>.json`` - one file per replicate (the 50 episodes'
  metrics); existing parts are reused unless ``--force`` is given;
* ``citybuilder_evaluation.json`` - all replicates of the requested methods,
  with per-method summaries (mean, SD with ddof = 1, SE across replicate
  means). Tables 5.2-5.4 and Figure 5.3 are made from this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..citybuilder.urban_simulation import (
    DeveloperDecision,
    GovernmentConstraints,
    UrbanSimulation,
)
from ..training.baselines.coma_baseline import COMAConfig, COMATrainer
from ..training.baselines.qmix_baseline import QMIXConfig, QMIXTrainer
from ..training.train_full_system import MAPPOGovernment
from ..training.train_with_geomaca import GeoMACATrainingConfig as GeoMACAConfig
from . import citybuilder_pipeline as cbp

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PACKAGE_ROOT / "results" / "evaluation"
EVALUATION_FILE = "citybuilder_evaluation.json"

DEVICE = "cpu"
N_EVAL = 50
SEEDS = (42, 123, 456, 789, 1024, 1337, 2048, 3141, 4096, 5555)
ABLATION_SEEDS = (789, 1024, 1337, 2048, 3141, 4096, 5555)
HEURISTIC_SEEDS = tuple(42 + 1000 * s_idx for s_idx in range(5))
NUM_LAND = 6

GEOMACA_FAMILY = ("geomaca", "abl_uniform", "abl_temporal", "abl_temporal_agent")
METHODS = ("heuristic",) + GEOMACA_FAMILY + ("qmix", "coma")
DEFAULT_SEEDS: Dict[str, Tuple[int, ...]] = {
    "heuristic": HEURISTIC_SEEDS,
    "geomaca": SEEDS,
    "abl_uniform": ABLATION_SEEDS,
    "abl_temporal": ABLATION_SEEDS,
    "abl_temporal_agent": ABLATION_SEEDS,
    "qmix": SEEDS,
    "coma": SEEDS,
}

METRIC_KEYS = (
    "gdp", "tax_revenue", "developer_profit",
    "green_ratio", "woodland_ratio", "density",
    "population", "segregation", "qol",
)

PROTOCOL = ("All methods evaluated under a uniform-random developer (per-cell sampling "
            "over the 6 land-use classes); 50 episodes per replicate; replicate means "
            "aggregated across replicates with ddof=1.")


# ---------------------------------------------------------------------------
# Episodes (one CityBuilder episode per call)
# ---------------------------------------------------------------------------

def get_metrics(env: UrbanSimulation) -> Dict[str, float]:
    m = env.metrics
    total = env.map_size * env.map_size
    developed = int(np.sum(env.land_grid != 0))
    return {
        "gdp": float(m.gdp),
        "tax_revenue": float(m.tax_revenue),
        "developer_profit": float(m.developer_profit),
        "green_ratio": float(m.green_space_ratio),
        "woodland_ratio": float(m.woodland_ratio),
        "density": developed / total,
        "population": float(m.population),
        "segregation": float(m.segregation_index),
        "qol": float(m.avg_qol),
    }


def uniform_grid(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, NUM_LAND, size=(20, 20), dtype=np.int64)


def heuristic_episode(rng: np.random.Generator, cfg: GeoMACAConfig) -> Dict[str, float]:
    env = UrbanSimulation(map_size=cfg.map_size, zone_size=cfg.zone_size)
    for zone_id in range(cfg.num_zones):
        env.develop_zone(zone_id, GovernmentConstraints(),
                          DeveloperDecision(land_use_grid=uniform_grid(rng)))
    return get_metrics(env)


def geomaca_episode(government: MAPPOGovernment, cfg: GeoMACAConfig,
                    rng: np.random.Generator) -> Dict[str, float]:
    env = UrbanSimulation(map_size=cfg.map_size, zone_size=cfg.zone_size)
    for zone_id in range(cfg.num_zones):
        base_state, _ = cbp.build_states(env, zone_id)
        constraints, _, _ = government.select_action(base_state, training=False)
        env.develop_zone(zone_id, constraints,
                          DeveloperDecision(land_use_grid=uniform_grid(rng)))
    return get_metrics(env)


def qmix_episode(trainer: QMIXTrainer, cfg: QMIXConfig,
                 rng: np.random.Generator) -> Dict[str, float]:
    env = UrbanSimulation(map_size=cfg.map_size, zone_size=cfg.zone_size)
    trainer.env = env
    trainer.epsilon = 0.0
    for zone_id in range(cfg.num_zones):
        base_state, _ = trainer.get_state(zone_id)
        actions = trainer.select_actions(base_state)
        constraints = trainer.actions_to_constraints(actions)
        env.develop_zone(zone_id, constraints,
                          DeveloperDecision(land_use_grid=uniform_grid(rng)))
    return get_metrics(env)


def coma_episode(trainer: COMATrainer, cfg: COMAConfig,
                 rng: np.random.Generator) -> Dict[str, float]:
    env = UrbanSimulation(map_size=cfg.map_size, zone_size=cfg.zone_size)
    trainer.env = env
    for zone_id in range(cfg.num_zones):
        base_state, _ = trainer.get_state(zone_id)
        s_t = torch.tensor(base_state, dtype=torch.float32,
                           device=DEVICE).unsqueeze(0)
        actions = []
        for i in range(cfg.num_agents):
            with torch.no_grad():
                probs = trainer.actors[i](s_t)
            actions.append(int(probs.argmax(dim=-1).item()))
        constraints = trainer.actions_to_constraints(actions)
        env.develop_zone(zone_id, constraints,
                          DeveloperDecision(land_use_grid=uniform_grid(rng)))
    return get_metrics(env)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(per_replicate: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not per_replicate:
        return {k: {"n": 0, "mean": float("nan"), "sd": float("nan"),
                    "se": float("nan")} for k in METRIC_KEYS}
    for k in METRIC_KEYS:
        means = [sum(r[k]) / len(r[k]) for r in per_replicate if r[k]]
        n = len(means)
        if n == 0:
            out[k] = {"n": 0, "mean": float("nan"), "sd": float("nan"),
                      "se": float("nan")}
            continue
        mu = sum(means) / n
        if n > 1:
            sd = math.sqrt(sum((x - mu) ** 2 for x in means) / (n - 1))
        else:
            sd = 0.0
        se = sd / math.sqrt(n) if n > 0 else 0.0
        out[k] = {"n": n, "mean": mu, "sd": sd, "se": se,
                  "ci95_lo": mu - 1.96 * se, "ci95_hi": mu + 1.96 * se,
                  "values": means}
    return out


def per_seed_metrics_to_dict(eps: List[Dict[str, float]]) -> Dict[str, List[float]]:
    if not eps:
        return {k: [] for k in METRIC_KEYS}
    return {k: [e[k] for e in eps] for k in METRIC_KEYS}


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------

def run_dir(method: str, seed: int, geomaca_runs: Path, baseline_runs: Path) -> Optional[Path]:
    """Run folder of a learned replicate (None for the Heuristic)."""
    if method == "heuristic":
        return None
    root = geomaca_runs if method in GEOMACA_FAMILY else baseline_runs
    return Path(root) / f"{method}_seed{seed}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replicate_episodes(method: str, seed: int, geomaca_runs: Path, baseline_runs: Path,
                       episodes: int = N_EVAL, seed_all: bool = False) -> List[Dict[str, float]]:
    """Run the ``episodes`` episodes of one replicate and return their metrics."""
    if method == "heuristic":
        cfg = GeoMACAConfig()
        cfg.device = DEVICE
        if seed_all:
            random.seed(seed)
            np.random.seed(seed)
        rng = np.random.default_rng(seed)
        return [heuristic_episode(rng, cfg) for _ in range(episodes)]

    folder = run_dir(method, seed, geomaca_runs, baseline_runs)
    ckpt = folder / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt}")
    torch.manual_seed(seed); np.random.seed(seed)
    if seed_all:
        random.seed(seed)
    rng = np.random.default_rng(seed)

    if method in GEOMACA_FAMILY:
        cfg_path = folder / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing run configuration {cfg_path}")
        gov, _, base_cfg, _ = cbp.load_trained_models(ckpt, cfg_path)
        return [geomaca_episode(gov, base_cfg, rng) for _ in range(episodes)]

    if method == "qmix":
        cfg = QMIXConfig()
        cfg.device = DEVICE
        cfg.use_safe_presets = True
        cfg.use_uniform_developer = True
        trainer = QMIXTrainer(cfg)
        trainer.load(str(ckpt))
        for net in list(trainer.agent_nets) + [trainer.mixer]:
            getattr(net, "ev" + "al")()
        return [qmix_episode(trainer, cfg, rng) for _ in range(episodes)]

    if method == "coma":
        cfg = COMAConfig()
        cfg.device = DEVICE
        cfg.use_uniform_developer = True
        trainer = COMATrainer(cfg)
        trainer.load(str(ckpt))
        for net in list(trainer.actors) + [trainer.critic]:
            getattr(net, "ev" + "al")()
        return [coma_episode(trainer, cfg, rng) for _ in range(episodes)]

    raise ValueError(f"unknown method {method!r}")


def part_path(out_dir: Path, method: str, seed: int) -> Path:
    return Path(out_dir) / "parts" / f"{method}_seed{seed}.json"


def evaluate_replicate(method: str, seed: int, geomaca_runs: str, baseline_runs: str,
                       out_dir: str, episodes: int = N_EVAL, seed_all: bool = False,
                       single_thread: bool = False) -> Dict:
    """Evaluate one replicate, write its part file and return the record."""
    if single_thread:
        torch.set_num_threads(1)
    geomaca_runs, baseline_runs = Path(geomaca_runs), Path(baseline_runs)
    t0 = time.time()
    eps = replicate_episodes(method, seed, geomaca_runs, baseline_runs, episodes, seed_all)
    folder = run_dir(method, seed, geomaca_runs, baseline_runs)
    record = {
        "method": method,
        "seed": seed,
        "episodes": episodes,
        "seed_all": seed_all,
        "run": folder.name if folder is not None else None,
        "checkpoint_sha256": (sha256(folder / "checkpoints" / "best.pt")
                              if folder is not None else None),
        "per_seed": per_seed_metrics_to_dict(eps),
        "seconds": time.time() - t0,
    }
    path = part_path(Path(out_dir), method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    gdp = sum(e["gdp"] for e in eps) / len(eps) / 1e6
    pop = sum(e["population"] for e in eps) / len(eps)
    print(f"  {method} seed={seed} GDP=${gdp:.1f}M Pop={pop:.0f}  ({record['seconds']:.0f}s)",
          flush=True)
    return record


def _part_matches(path: Path, episodes: int, seed_all: bool) -> bool:
    if not path.exists():
        return False
    rec = json.loads(path.read_text(encoding="utf-8"))
    return rec.get("episodes") == episodes and bool(rec.get("seed_all")) == seed_all


def assemble(out_dir: Path, plan: Dict[str, Sequence[int]], episodes: int,
             seed_all: bool) -> Dict:
    """Combine the part files of ``plan`` into one evaluation record."""
    out: Dict = {
        "protocol": PROTOCOL,
        "n_eval_per_replicate": episodes,
        "seed_all": seed_all,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "methods": {},
    }
    for method in METHODS:
        if method not in plan:
            continue
        per_replicate, seeds_used, runs, shas = [], [], [], []
        for seed in plan[method]:
            rec = json.loads(part_path(out_dir, method, seed).read_text(encoding="utf-8"))
            per_replicate.append(rec["per_seed"])
            seeds_used.append(seed)
            runs.append(rec["run"])
            shas.append(rec["checkpoint_sha256"])
        out["methods"][method] = {"method": method, "seeds_used": seeds_used,
                                  "runs": runs, "checkpoint_sha256": shas,
                                  "per_replicate": per_replicate,
                                  "summary": aggregate(per_replicate)}
    out["provenance"] = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "device": DEVICE,
    }
    return out


def merge_into(path: Path, new: Dict) -> Dict:
    """Add the methods of ``new`` to the evaluation record at ``path``."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if (old.get("n_eval_per_replicate") == new["n_eval_per_replicate"]
                and bool(old.get("seed_all")) == bool(new["seed_all"])):
            merged = dict(new)
            methods = dict(old.get("methods", {}))
            methods.update(new["methods"])
            merged["methods"] = {m: methods[m] for m in METHODS if m in methods}
            new = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new, indent=2, default=str), encoding="utf-8")
    return new


def run_evaluation(geomaca_runs: Path, baseline_runs: Path, out_dir: Path = DEFAULT_OUT,
                   methods: Sequence[str] = METHODS,
                   seeds: Optional[Dict[str, Sequence[int]]] = None,
                   episodes: int = N_EVAL, workers: int = 1, seed_all: bool = False,
                   force: bool = False) -> Dict:
    """Evaluate the missing replicates of ``methods`` and update the evaluation record."""
    out_dir = Path(out_dir)
    plan = {m: tuple((seeds or {}).get(m, DEFAULT_SEEDS[m])) for m in methods}
    for m in plan:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; choose from {METHODS}")
    todo = [(m, s) for m in plan for s in plan[m]
            if force or not _part_matches(part_path(out_dir, m, s), episodes, seed_all)]
    print(f"Evaluation: {len(todo)} replicate(s) to run, {episodes} episodes each, "
          f"workers={workers}", flush=True)
    t0 = time.time()
    args = [(m, s, str(geomaca_runs), str(baseline_runs), str(out_dir), episodes, seed_all)
            for m, s in todo]
    if workers <= 1:
        for a in args:
            evaluate_replicate(*a)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(evaluate_replicate, *a, single_thread=True) for a in args]
            for f in futures:
                f.result()
    if todo:
        print(f"Evaluation done in {(time.time() - t0) / 60:.1f} min", flush=True)
    record = assemble(out_dir, plan, episodes, seed_all)
    return merge_into(out_dir / EVALUATION_FILE, record)


def checkpoint_roots(args) -> Tuple[Path, Path]:
    """The GeoMACA-family and QMIX/COMA run folders selected on the command line."""
    root = Path(args.checkpoints) if args.checkpoints else None
    geomaca_runs = Path(args.geomaca_runs) if args.geomaca_runs else (
        root / "cb_runs_v4" if root else None)
    baseline_runs = Path(args.baseline_runs) if args.baseline_runs else (
        root / "cb_runs_v3" if root else None)
    return geomaca_runs, baseline_runs


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoints", type=str, default=os.environ.get("CHAPTER5_CHECKPOINTS"),
                        help="root of the checkpoint archive (contains cb_runs_v4/ and "
                             "cb_runs_v3/); default: $CHAPTER5_CHECKPOINTS")
    parser.add_argument("--geomaca-runs", type=str, default=None,
                        help="folder with the GeoMACA and ablation runs (default <checkpoints>/cb_runs_v4)")
    parser.add_argument("--baseline-runs", type=str, default=None,
                        help="folder with the QMIX and COMA runs (default <checkpoints>/cb_runs_v3)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_checkpoint_arguments(parser)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="replicate seeds for every requested method (default: the protocol's)")
    parser.add_argument("--episodes", type=int, default=N_EVAL)
    parser.add_argument("--workers", type=int, default=1,
                        help="replicates evaluated in parallel (one CPU thread each)")
    parser.add_argument("--seed-all", action="store_true",
                        help="also seed Python's random module (repeatable runs)")
    parser.add_argument("--force", action="store_true", help="re-evaluate existing parts")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    geomaca_runs, baseline_runs = checkpoint_roots(args)
    learned = [m for m in args.methods if m != "heuristic"]
    if learned and (geomaca_runs is None or baseline_runs is None):
        sys.exit("give --checkpoints (or --geomaca-runs and --baseline-runs)")
    seeds = {m: tuple(args.seeds) for m in args.methods} if args.seeds else None
    record = run_evaluation(geomaca_runs or Path("."), baseline_runs or Path("."), Path(args.out),
                            args.methods, seeds, args.episodes, args.workers, args.seed_all,
                            args.force)
    print(f"Wrote {Path(args.out) / EVALUATION_FILE}")
    for m, block in record["methods"].items():
        s = block["summary"]
        print(f"  {m:19s} n={s['gdp']['n']:2d}  GDP=${s['gdp']['mean'] / 1e6:.1f}M  "
              f"Pop={s['population']['mean']:.0f}  Seg={s['segregation']['mean']:.3f}  "
              f"QoL={s['qol']['mean']:.3f}")


if __name__ == "__main__":
    main()
