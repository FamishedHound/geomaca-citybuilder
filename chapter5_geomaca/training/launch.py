"""Train every CityBuilder run behind Chapter 5, several runs at a time.

Groups (seeds as in the chapter):

* ``geomaca``   - GeoMACA, seeds 42 123 456 789 1024 1337 2048 3141 4096 5555
                  (Table 5.2), written to ``<out>/cb_runs_v4/geomaca_seed<k>/``;
* ``ablations`` - Uniform credit, Temporal only and Temporal + Agent, seeds
                  789 1024 1337 2048 3141 4096 5555 (Table 5.4), written to
                  ``<out>/cb_runs_v4/abl_<variant>_seed<k>/``;
* ``baselines`` - QMIX and COMA, the ten seeds of Table 5.2, written to
                  ``<out>/cb_runs_v3/<method>_seed<k>/``.

Every GeoMACA-family run is

    python -u -m chapter5_geomaca.training.train_with_geomaca --chapter5 --seed S
        --num_episodes 2000 --save_path <dir> --patience 0 --patience_min_episodes 0
        --use_temporal T --use_geographic G --use_agent A --warmup 500

(early stopping after 500 episodes without an improvement of more than 0.01 in
the episode's summed developer reward, counted from the first episode; the
windowed patience rule is off). Every baseline run is

    python -u -m chapter5_geomaca.training.baselines.<method>_baseline --seed S
        --num_episodes 2000 --save_path <dir> --patience 0 --patience_min_episodes 0
        --use_uniform_developer True

Usage (from the folder that contains the package)::

    python -m chapter5_geomaca.training.launch geomaca --out runs --max-concurrent 10
    python -m chapter5_geomaca.training.launch ablations --out runs
    python -m chapter5_geomaca.training.launch baselines --out runs --max-concurrent 12

Runs whose ``checkpoints/final.pt`` exists are skipped, so a group can be
restarted. Each run's log is ``<out>/<cb_runs_v4|cb_runs_v3>/logs/<run>.log``.
The evaluation reads the same layout: ``--checkpoints <out>``.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

PACKAGE_PARENT = Path(__file__).resolve().parents[2]

ALL_SEEDS = (42, 123, 456, 789, 1024, 1337, 2048, 3141, 4096, 5555)
ABLATION_SEEDS = (789, 1024, 1337, 2048, 3141, 4096, 5555)

ABLATIONS = [
    # (label-prefix, use_temporal, use_geographic, use_agent)
    ("abl_uniform",         False, False, False),
    ("abl_temporal",        True,  False, False),
    ("abl_temporal_agent",  True,  False, True),
]
GEOMACA_MODULE = "chapter5_geomaca.training.train_with_geomaca"
BASELINE_MODULES = {
    "qmix": "chapter5_geomaca.training.baselines.qmix_baseline",
    "coma": "chapter5_geomaca.training.baselines.coma_baseline",
}


@dataclass
class Job:
    label: str
    save_path: Path = field(default_factory=Path)
    log_path: Path = field(default_factory=Path)
    cmd: list = field(default_factory=list)


def _flag(value: bool) -> str:
    return "True" if value else "False"


def geomaca_command(seed: int, save_path: Path, use_temporal: bool = True,
                    use_geographic: bool = True, use_agent: bool = True,
                    num_episodes: int = 2000, python: str = sys.executable) -> List[str]:
    """Command of one GeoMACA-family run (Chapter 5 configuration and protocol)."""
    return [python, "-u", "-m", GEOMACA_MODULE, "--chapter5",
            "--seed", str(seed),
            "--num_episodes", str(num_episodes),
            "--save_path", str(save_path),
            "--patience", "0",
            "--patience_min_episodes", "0",
            "--use_temporal", _flag(use_temporal),
            "--use_geographic", _flag(use_geographic),
            "--use_agent", _flag(use_agent),
            "--warmup", "500"]


def baseline_command(method: str, seed: int, save_path: Path, num_episodes: int = 2000,
                     python: str = sys.executable) -> List[str]:
    """Command of one QMIX or COMA run."""
    return [python, "-u", "-m", BASELINE_MODULES[method],
            "--seed", str(seed),
            "--num_episodes", str(num_episodes),
            "--save_path", str(save_path),
            "--patience", "0", "--patience_min_episodes", "0",
            "--use_uniform_developer", "True"]


def build_jobs(group: str, out: Path, num_episodes: int,
               seeds: Optional[Sequence[int]] = None) -> List[Job]:
    jobs: List[Job] = []
    if group in ("geomaca", "ablations"):
        root = Path(out) / "cb_runs_v4"
        variants = ([("geomaca", True, True, True)] if group == "geomaca" else ABLATIONS)
        default = ALL_SEEDS if group == "geomaca" else ABLATION_SEEDS
        for prefix, ut, ug, ua in variants:
            for s in (seeds or default):
                label = f"{prefix}_seed{s}"
                j = Job(label=label, save_path=root / label, log_path=root / "logs" / f"{label}.log")
                j.cmd = geomaca_command(s, j.save_path, ut, ug, ua, num_episodes)
                jobs.append(j)
    elif group == "baselines":
        root = Path(out) / "cb_runs_v3"
        for method in BASELINE_MODULES:
            for s in (seeds or ALL_SEEDS):
                label = f"{method}_seed{s}"
                j = Job(label=label, save_path=root / label, log_path=root / "logs" / f"{label}.log")
                j.cmd = baseline_command(method, s, j.save_path, num_episodes)
                jobs.append(j)
    else:
        raise ValueError(f"unknown group {group!r}")
    return jobs


def run_one(job: Job, env: dict) -> tuple:
    job.save_path.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(job.log_path, "w", encoding="utf-8", errors="replace") as logfp:
        logfp.write(f"# {job.label} cmd={' '.join(job.cmd)}\n# start={time.ctime()}\n\n")
        logfp.flush()
        proc = subprocess.Popen(job.cmd, cwd=str(PACKAGE_PARENT), env=env,
                                stdout=logfp, stderr=subprocess.STDOUT)
        rc = proc.wait()
    return job, rc, time.time() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("group", choices=["geomaca", "ablations", "baselines"])
    p.add_argument("--out", type=str, required=True, help="root of the run folders")
    p.add_argument("--max-concurrent", type=int, default=10)
    p.add_argument("--num_episodes", type=int, default=2000)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    args = p.parse_args(argv)

    jobs = build_jobs(args.group, Path(args.out), args.num_episodes, args.seeds)
    pending = [j for j in jobs if not (j.save_path / "checkpoints" / "final.pt").exists()]
    progress = Path(args.out) / "PROGRESS.md"
    progress.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.group}: {len(pending)} pending of {len(jobs)} total\n")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("PYTHONUNBUFFERED", "1")

    job_q: "queue.Queue[Optional[Job]]" = queue.Queue()
    for j in pending:
        job_q.put(j)
    for _ in range(args.max_concurrent):
        job_q.put(None)

    results = []
    lock = threading.Lock()
    completed = [0]

    def log(msg: str) -> None:
        with lock:
            with open(progress, "a", encoding="utf-8") as fp:
                fp.write(msg + "\n")
            print(msg, flush=True)

    log(f"\n## {args.group} started ({time.ctime()}) jobs={len(pending)} ep={args.num_episodes}")

    def worker(wid: int) -> None:
        while True:
            j = job_q.get()
            if j is None:
                return
            log(f"[w{wid}] START {j.label}")
            try:
                _, rc, secs = run_one(j, env)
            except Exception as exc:  # noqa: BLE001
                log(f"[w{wid}] CRASH {j.label}: {exc!r}")
                results.append((j, -1, 0.0)); continue
            results.append((j, rc, secs))
            with lock:
                completed[0] += 1
            log(f"[w{wid}] DONE {j.label} "
                f"{'OK' if rc == 0 else f'FAIL rc={rc}'} {secs/60:.1f}min "
                f"({completed[0]}/{len(pending)})")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(args.max_concurrent)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    n_ok = sum(1 for _, rc, _ in results if rc == 0)
    log(f"\n## {args.group} finished ({time.ctime()}) ok={n_ok}/{len(results)} wall={wall/60:.1f}min")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
