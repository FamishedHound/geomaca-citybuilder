"""Peak memory of one update of the GeoMACA credit module as the number of zones grows.

One update is what ``GeoMACATrainer._update_geomaca`` does once per episode
with the Chapter 5 configuration (``--chapter5``): a forward pass of the
credit module on one episode, the objective 0.1 L_entropy + 0.05
L_consistency + 0.02 L_causality, a backward pass, gradient clipping at 0.5
and one Adam step (learning rate 1e-4). The module is the CityBuilder one
(width 128, four attention heads, three agents, 20-dimensional observations,
actions padded to five dimensions, second softmax, training mode). The
episode is synthetic, with one zone developed per step, so the number of
steps equals the number of zones n; its inputs are the per-step record of
the n zone states (1 x n x n x 20), the agents' actions (1 x n x 3 x 5), the
observations (1 x n x 20) and the developed zones.

Memory is the peak of the bytes held by PyTorch tensors during the update,
counting the inputs, the module, its gradients and the optimiser state. On
the CPU it is read from the allocation and free events that the PyTorch
profiler records for the CPU allocator; with ``--device cuda`` it is
``torch.cuda.max_memory_allocated``. A quadratic a n^2 + b n + c is fitted to
the sizes of 1,000 zones or more (at least three) and gives the extrapolation
to 10,000 zones and the number of zones at which the peak exceeds
``--limit-gb`` (32 GB by default, the memory of the graphics card used in the
thesis).

Usage (from the folder that contains the package)::

    python -m chapter5_geomaca.scaling.memory                       # 9, 1000, 2000, 3000 zones
    python -m chapter5_geomaca.scaling.memory --zones 1000 2000 3000 4000 --out memory.json

The 3,000-zone update takes about a minute on one CPU core and 1.6 GB of RAM.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# The profiler looks for a GPU when it starts; keep that quiet on a CPU-only run.
os.environ.setdefault("KINETO_LOG_LEVEL", "5")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..geomaca.losses import GeoMACALossComputer  # noqa: E402
from ..geomaca.model import GeoMACA, GeoMACAConfig  # noqa: E402
from ..training.train_with_geomaca import CHAPTER5_METHOD, GeoMACATrainingConfig  # noqa: E402

DEFAULT_ZONES = (9, 1000, 2000, 3000)
FIT_MIN_ZONES = 1000
CLIP_NORM = 0.5  # as in GeoMACATrainer._update_geomaca


def chapter5_config(device: str = "cpu") -> GeoMACATrainingConfig:
    """The trainer configuration of Chapter 5 (``--chapter5``)."""
    return GeoMACATrainingConfig(device=device, **CHAPTER5_METHOD)


def build(n: int, device: str = "cpu"):
    """Credit module, optimiser, loss and a synthetic n-zone episode, as the trainer builds them."""
    cfg = chapter5_config(device)
    model = GeoMACA(GeoMACAConfig(
        state_dim=cfg.base_state_dim,
        action_dim=cfg.gov_action_dim,
        hidden_dim=cfg.geomaca_hidden_dim,
        num_zones=n,
        num_turns=n,  # one zone per step
        num_agents=3,
        num_heads=cfg.geomaca_num_heads,
        use_temporal=cfg.use_temporal,
        use_geographic=cfg.use_geographic,
        use_agent=cfg.use_agent,
        combination=cfg.geomaca_combination,
        temperature=cfg.geomaca_temperature,
        attention_normalization=cfg.attention_normalization,
    )).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.geomaca_lr)
    losses = GeoMACALossComputer(
        entropy_weight=cfg.geomaca_entropy_weight,
        consistency_weight=cfg.geomaca_consistency_weight,
        causality_weight=cfg.geomaca_causality_weight,
        state_dim=cfg.base_state_dim,
        device=device,
        agent_entropy_coef=cfg.agent_entropy_coef,
    )
    trajectory = {
        "zone_states": torch.rand(1, n, n, cfg.base_state_dim, device=device),
        "agent_actions": torch.rand(1, n, 3, cfg.gov_action_dim, device=device),
        "global_state": torch.rand(1, n, cfg.base_state_dim, device=device),
        "zone_developed": torch.arange(n, device=device).unsqueeze(0),
    }
    reward = torch.tensor([-0.1], device=device)
    return model, optimizer, losses, trajectory, reward


def one_update(model, optimizer, losses, trajectory, reward) -> float:
    """Forward pass, objective, backward pass, clipping and Adam step; returns the loss."""
    credit, _ = model(trajectory, reward)
    out = losses.compute(credit, trajectory, reward)
    optimizer.zero_grad()
    out["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    optimizer.step()
    return float(out["total"].detach())


def _update(n: int, device: str) -> None:
    model, optimizer, losses, trajectory, reward = build(n, device)
    one_update(model, optimizer, losses, trajectory, reward)


def peak_from_allocator_events(events) -> Dict[str, int]:
    """Peak and final total of the allocation (+) and free (-) events, in time order."""
    current = peak = 0
    for nbytes in events:
        current += nbytes
        peak = max(peak, current)
    return {"peak_bytes": peak, "end_bytes": current}


def _cpu_peak(n: int) -> Dict[str, int]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        _update(n, "cpu")
    # Every allocation and free of the CPU allocator inside the block, with its size.
    records = [e for e in prof.profiler.kineto_results.events() if e.name() == "[memory]"]
    records.sort(key=lambda e: e.start_ns())
    out = peak_from_allocator_events([e.nbytes() for e in records])
    out["allocator_events"] = len(records)
    return out


def _cuda_peak(n: int) -> Dict[str, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _update(n, "cuda")
    torch.cuda.synchronize()
    out = {"peak_bytes": torch.cuda.max_memory_allocated() - base,
           "end_bytes": torch.cuda.memory_allocated() - base}
    torch.cuda.empty_cache()
    return out


def measure(n: int, device: str = "cpu", seed: int = 0) -> Dict:
    """Peak memory of one update with n zones (and n steps)."""
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    res = _cpu_peak(n) if device == "cpu" else _cuda_peak(n)
    return {"zones": n, "steps": n, "credit_entries": 3 * n * n,
            "zone_state_values": 20 * n * n, "device": device,
            "seconds": time.perf_counter() - t0, **res}


def fit_quadratic(zones: Sequence[int], peaks: Sequence[int]) -> Optional[Dict[str, float]]:
    """Least-squares a n^2 + b n + c (exact through three points)."""
    if len(zones) < 3:
        return None
    a, b, c = np.polyfit(np.asarray(zones, dtype=float), np.asarray(peaks, dtype=float), 2)
    return {"a": float(a), "b": float(b), "c": float(c)}


def predict(fit: Dict[str, float], n: float) -> float:
    return fit["a"] * n * n + fit["b"] * n + fit["c"]


def zones_at(fit: Dict[str, float], limit_bytes: float) -> float:
    """Number of zones at which the fitted peak reaches ``limit_bytes``."""
    a, b, c = fit["a"], fit["b"], fit["c"] - limit_bytes
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


def run(zones: Sequence[int], device: str = "cpu", limit_gb: float = 32.0,
        extrapolate_to: int = 10000) -> Dict:
    rows: List[Dict] = []
    for n in zones:
        row = measure(n, device)
        rows.append(row)
        print(f"  zones={n:6d}  credit entries={row['credit_entries']:>12,d}  "
              f"peak={row['peak_bytes'] / 1e9:8.4f} GB ({row['peak_bytes']:,d} bytes)  "
              f"{row['seconds']:.1f} s", flush=True)
    big = [r for r in rows if r["zones"] >= FIT_MIN_ZONES]
    fit = fit_quadratic([r["zones"] for r in big], [r["peak_bytes"] for r in big])
    result = {"device": device, "torch": torch.__version__, "measurements": rows, "fit": fit}
    if fit is not None:
        result["fit_zones"] = [r["zones"] for r in big]
        result["extrapolated"] = {"zones": extrapolate_to,
                                  "peak_bytes": predict(fit, extrapolate_to)}
        result["limit"] = {"limit_gb": limit_gb,
                           "zones_at_limit": zones_at(fit, limit_gb * 1e9),
                           "zones_at_limit_gib": zones_at(fit, limit_gb * 2 ** 30)}
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", type=int, nargs="+", default=list(DEFAULT_ZONES))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--limit-gb", type=float, default=32.0,
                    help="memory limit in GB (10^9 bytes) for the crossing point")
    ap.add_argument("--extrapolate-to", type=int, default=10000)
    ap.add_argument("--out", type=str, default=None, help="write the results as JSON")
    args = ap.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda needs a GPU")
    print(f"Peak memory of one credit-module update ({args.device}; one zone per step):")
    res = run(args.zones, args.device, args.limit_gb, args.extrapolate_to)
    fit = res["fit"]
    if fit is not None:
        e, lim = res["extrapolated"], res["limit"]
        print(f"Fit over {res['fit_zones']}: peak = {fit['a']:.1f} n^2 + {fit['b']:.0f} n "
              f"+ {fit['c']:.0f} bytes")
        print(f"Extrapolated to {e['zones']:,d} zones: {e['peak_bytes'] / 1e9:.1f} GB")
        print(f"Exceeds {lim['limit_gb']:g} GB at about {lim['zones_at_limit']:,.0f} zones "
              f"({lim['limit_gb']:g} GiB at about {lim['zones_at_limit_gib']:,.0f})")
    else:
        print(f"(give three or more sizes of at least {FIT_MIN_ZONES} zones for the extrapolation)")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
