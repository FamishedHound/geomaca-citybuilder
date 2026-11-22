"""One command per table and figure of Chapter 5.

    python -m chapter5_geomaca.reproduce table5.2  --checkpoints <archive> [--workers 4]
    python -m chapter5_geomaca.reproduce table5.3  --checkpoints <archive>
    python -m chapter5_geomaca.reproduce figure5.3 --checkpoints <archive>
    python -m chapter5_geomaca.reproduce table5.4  --checkpoints <archive>
    python -m chapter5_geomaca.reproduce table5.5  [--workers 4]
    python -m chapter5_geomaca.reproduce table5.6  --lbf-results <folder>
    python -m chapter5_geomaca.reproduce table5.7  --lbf-results <folder>
    python -m chapter5_geomaca.reproduce all       --checkpoints <archive> --lbf-results <folder>

Tables 5.2-5.4 and Figure 5.3 evaluate the trained checkpoints (CPU) and
share one evaluation record, ``<results>/evaluation/citybuilder_evaluation.json``;
replicates already evaluated are reused, so after the first of these commands
the others only evaluate what is missing. Table 5.5 runs the no-government
evaluation and takes its Heuristic column from the same record (evaluating the
Heuristic if it is not there yet). Tables 5.6 and 5.7 are computed from the
per-seed result files of the Level-Based Foraging runs.

``--from-records`` makes every output from the records in
``chapter5_geomaca/records/`` instead (the evaluation behind the printed
tables), with no checkpoint and no evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
RECORDS = PACKAGE_ROOT / "records"
DEFAULT_RESULTS = PACKAGE_ROOT / "results"

TABLE_52_METHODS = ("heuristic", "qmix", "coma", "geomaca")
TABLE_54_METHODS = ("geomaca", "abl_uniform", "abl_temporal", "abl_temporal_agent")
TARGETS = ("table5.2", "table5.3", "figure5.3", "table5.4", "table5.5", "table5.6", "table5.7")


def _evaluation(args, methods: Sequence[str]) -> Path:
    """Path of the CityBuilder evaluation record covering ``methods``."""
    if args.from_records:
        return RECORDS / "citybuilder_evaluation.json"
    from .evaluation.evaluate import EVALUATION_FILE, run_evaluation

    out_dir = Path(args.results) / "evaluation"
    learned = [m for m in methods if m != "heuristic"]
    geomaca_runs, baseline_runs = _checkpoint_roots(args)
    if learned and (geomaca_runs is None or baseline_runs is None):
        raise SystemExit("give --checkpoints (the root of the checkpoint archive), "
                         "or --geomaca-runs and --baseline-runs")
    run_evaluation(geomaca_runs or Path("."), baseline_runs or Path("."), out_dir, methods,
                   None, args.episodes, args.workers, args.seed_all, False)
    return out_dir / EVALUATION_FILE


def _checkpoint_roots(args):
    root = Path(args.checkpoints) if args.checkpoints else None
    geomaca_runs = Path(args.geomaca_runs) if args.geomaca_runs else (root / "cb_runs_v4" if root else None)
    baseline_runs = Path(args.baseline_runs) if args.baseline_runs else (root / "cb_runs_v3" if root else None)
    return geomaca_runs, baseline_runs


def make(target: str, args) -> List[Path]:
    tables = Path(args.results) / "tables"
    if target in ("table5.2", "table5.3"):
        from .evaluation.tables import write_tables

        path = _evaluation(args, TABLE_52_METHODS)
        return write_tables(path, tables, which=(target[5:],))
    if target == "figure5.3":
        from .evaluation.figures import cb_pillar_aggregate_bars

        path = _evaluation(args, TABLE_52_METHODS)
        out = Path(args.results) / "figures"
        cb_pillar_aggregate_bars(json.loads(path.read_text(encoding="utf-8")), out)
        return [out / "cb_pillar_aggregate_bars.pdf", out / "cb_pillar_aggregate_bars.png",
                out / "cb_pillar_aggregate_bars.json"]
    if target == "table5.4":
        from .evaluation import ablation_table

        path = _evaluation(args, TABLE_54_METHODS)
        geomaca_runs = None if args.from_records else _checkpoint_roots(args)[0]
        return ablation_table.write(path, tables, geomaca_runs)
    if target == "table5.5":
        from .evaluation import no_government, no_government_table

        if args.from_records:
            nogov = RECORDS / "no_government_evaluation.json"
        else:
            nogov = Path(args.results) / "evaluation" / "no_government_evaluation.json"
            if args.rerun or not nogov.exists():
                out = no_government.run(max(1, args.workers), args.episodes)
                nogov.parent.mkdir(parents=True, exist_ok=True)
                nogov.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        path = _evaluation(args, ("heuristic",))
        return [no_government_table.write(path, nogov, tables)]
    if target in ("table5.6", "table5.7"):
        from .lbf import tables as lbf_tables

        if args.from_records or not args.lbf_results:
            if not args.from_records:
                print("No --lbf-results given: using records/lbf_per_seed.csv")
            runs = lbf_tables.load_per_seed_csv(RECORDS / "lbf_per_seed.csv")
        else:
            runs = lbf_tables.load_runs(Path(args.lbf_results))
        written = lbf_tables.build_tables(runs, tables)
        keep = "lbf_train_window" if target == "table5.6" else "lbf_eval"
        return [p for p in written if p.name.startswith(keep) or p.name == "lbf_per_seed.csv"]
    raise ValueError(f"unknown target {target!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m chapter5_geomaca.reproduce", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=TARGETS + ("all",))
    ap.add_argument("--checkpoints", type=str, default=os.environ.get("CHAPTER5_CHECKPOINTS"),
                    help="root of the checkpoint archive (cb_runs_v4/, cb_runs_v3/); "
                         "default $CHAPTER5_CHECKPOINTS")
    ap.add_argument("--geomaca-runs", type=str, default=None)
    ap.add_argument("--baseline-runs", type=str, default=None)
    ap.add_argument("--lbf-results", type=str, default=os.environ.get("CHAPTER5_LBF_RESULTS"),
                    help="folder with the 60 Level-Based Foraging result files; "
                         "default $CHAPTER5_LBF_RESULTS")
    ap.add_argument("--results", type=str, default=str(DEFAULT_RESULTS),
                    help="output folder (evaluation/, tables/, figures/)")
    ap.add_argument("--workers", type=int, default=1, help="evaluation processes (CPU)")
    ap.add_argument("--episodes", type=int, default=50, help="episodes per replicate")
    ap.add_argument("--seed-all", action="store_true",
                    help="seed Python's random module as well (repeatable evaluation)")
    ap.add_argument("--rerun", action="store_true", help="rerun the no-government evaluation")
    ap.add_argument("--from-records", action="store_true",
                    help="use the evaluation records in chapter5_geomaca/records/")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    targets = TARGETS if args.target == "all" else (args.target,)
    for target in targets:
        print(f"== {target}")
        for p in make(target, args):
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
