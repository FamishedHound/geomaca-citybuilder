"""The two Level-Based Foraging tables of Chapter 5 (Tables 5.6 and 5.7).

Reads the per-seed result files written by ``chapter5_geomaca.lbf.train``
(``<method>_seed<N>.json``), or the per-seed values stored in
``records/lbf_per_seed.csv``, and writes:

  - lbf_train_window_table.tex    : Table 5.6, last-5000-episode training window
  - lbf_eval_table.tex            : Table 5.7, held-out evaluation
  - lbf_train_window_summary.json : machine-readable train-window summary
  - lbf_eval_summary.json         : machine-readable eval summary
  - lbf_per_seed.csv              : the per-seed values behind both tables

Statistics per method:
  mean, SD (across seeds), SE (= SD / sqrt(n)), n,
  95% CI half-width via t-distribution,
  Welch t-test p-value vs. geomaca_fixed,
  Cohen's d vs. geomaca_fixed (pooled SD).

Statistical unit:
  - eval table: one observation per seed = data["eval"]["mean_return"]
    (already an average over 100 held-out episodes, food reward only).
  - train-window table: one observation per seed =
    mean(history["episode_returns"][-5000:]) (training reward, including the
    proximity-shaping term).

Usage::

    python -m chapter5_geomaca.lbf.tables --results <folder with the 60 JSON files> --out <dir>
    python -m chapter5_geomaca.lbf.tables --out <dir>          # from records/lbf_per_seed.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PER_SEED_RECORD = PACKAGE_ROOT / "records" / "lbf_per_seed.csv"
OUT_DIR = PACKAGE_ROOT / "results" / "tables"

METHOD_ORDER = ["geomaca_fixed", "coma", "qmix", "vdn", "independent", "uniform"]
METHOD_DISPLAY = {
    "geomaca_fixed": r"\textbf{GeoMACA}",
    "coma":          "COMA",
    "qmix":          "QMIX",
    "vdn":           "VDN",
    "independent":   "IPPO",
    "uniform":       "Uniform",
}
TRAIN_WINDOW = 5000
PIVOT_METHOD = "geomaca_fixed"


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy dependency required for the basic stats; we
# implement Welch t-test and the t-distribution CDF locally so this script
# runs on a stock interpreter).
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def se(xs: list[float]) -> float:
    return sd(xs) / math.sqrt(len(xs)) if len(xs) > 0 else 0.0


def cohen_d(a: list[float], b: list[float]) -> float:
    """Pooled-SD Cohen's d. Sign convention: positive => a > b."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = sd(a), sd(b)
    pooled = math.sqrt(((len(a) - 1) * sa * sa + (len(b) - 1) * sb * sb)
                       / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return (mean(a) - mean(b)) / pooled


def _log_gamma(x: float) -> float:
    return math.lgamma(x)


def _t_cdf(t: float, df: float) -> float:
    """Two-sided p-value helper relies on this. Uses the regularised
    incomplete beta function via the continued-fraction expansion."""
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    # Regularised incomplete beta I_x(a,b):
    bt = math.exp(_log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        cf = _betacf(x, a, b)
        ix = bt * cf / a
    else:
        cf = _betacf(1.0 - x, b, a)
        ix = 1.0 - bt * cf / b
    # CDF of t with df dof at value t (t may be negative)
    if t >= 0:
        return 1.0 - 0.5 * ix
    return 0.5 * ix


def _betacf(x: float, a: float, b: float, max_iter: int = 200,
            eps: float = 3e-7) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def welch_t(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Welch's two-sample t-test. Returns (t, df, two-sided p)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se_diff = math.sqrt(va / len(a) + vb / len(b))
    if se_diff == 0:
        return 0.0, float("nan"), 1.0
    t = (ma - mb) / se_diff
    num = (va / len(a) + vb / len(b)) ** 2
    den = ((va / len(a)) ** 2) / (len(a) - 1) + ((vb / len(b)) ** 2) / (len(b) - 1)
    df = num / den if den > 0 else float("nan")
    p = 2.0 * (1.0 - _t_cdf(abs(t), df))
    return t, df, p


_T_CRIT_95 = {  # one-sided 0.975 quantile, df -> t_crit (for 95% CI)
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131,
}


def ci95_halfwidth(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    df = n - 1
    t_crit = _T_CRIT_95.get(df, 1.96)
    return t_crit * se(xs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_runs(lbf_dir: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {m: [] for m in METHOD_ORDER}
    for jf in sorted(Path(lbf_dir).glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        method = data.get("method")
        if method not in runs:
            continue
        runs[method].append({
            "seed": data["seed"],
            "file": jf.name,
            "eval_mean_return": data["eval"]["mean_return"],
            "eval_std_return": data["eval"]["std_return"],
            "eval_mean_food": data["eval"]["mean_food"],
            "eval_mean_coop": data["eval"]["mean_coop"],
            "train_window_return": mean(
                data["history"]["episode_returns"][-TRAIN_WINDOW:]
            ),
            "config": data.get("config", {}),
        })
    for m in runs:
        runs[m].sort(key=lambda r: r["seed"])
    return runs


PER_SEED_FIELDS = ("eval_mean_return", "eval_std_return", "eval_mean_food",
                   "eval_mean_coop", "train_window_return")


def write_per_seed_csv(runs: dict[str, list[dict[str, Any]]], path: Path) -> None:
    """Per-seed values at full precision (one row per method and seed)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("method", "seed", "file") + PER_SEED_FIELDS)
        for m in METHOD_ORDER:
            for r in runs[m]:
                w.writerow([m, r["seed"], r["file"]] + [repr(float(r[k])) for k in PER_SEED_FIELDS])


def load_per_seed_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {m: [] for m in METHOD_ORDER}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] not in runs:
                continue
            rec: dict[str, Any] = {"seed": int(row["seed"]), "file": row["file"]}
            rec.update({k: float(row[k]) for k in PER_SEED_FIELDS})
            runs[row["method"]].append(rec)
    for m in runs:
        runs[m].sort(key=lambda r: r["seed"])
    return runs


# ---------------------------------------------------------------------------
# Summary table builder
# ---------------------------------------------------------------------------
def summarise(runs: dict[str, list[dict[str, Any]]],
              field: str) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    pivot_vals = [r[field] for r in runs[PIVOT_METHOD]]
    for m in METHOD_ORDER:
        vals = [r[field] for r in runs[m]]
        s_mean = mean(vals)
        s_sd = sd(vals)
        s_se = se(vals)
        s_ci = ci95_halfwidth(vals)
        if m == PIVOT_METHOD:
            t, df, p = float("nan"), float("nan"), float("nan")
            d = float("nan")
        else:
            t, df, p = welch_t(pivot_vals, vals)
            d = cohen_d(pivot_vals, vals)
        summary[m] = {
            "n": len(vals),
            "mean": s_mean,
            "sd": s_sd,
            "se": s_se,
            "ci95_halfwidth": s_ci,
            "values": vals,
            "welch_t_vs_geomaca": t,
            "welch_df_vs_geomaca": df,
            "welch_p_vs_geomaca": p,
            "cohen_d_vs_geomaca": d,
        }
    return summary


# ---------------------------------------------------------------------------
# LaTeX writer
# ---------------------------------------------------------------------------
def fmt(x: float, prec: int = 3) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "--"
    return f"{x:.{prec}f}"


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "{--}"
    if p < 1e-4:
        return r"$<\!10^{-4}$"
    return fmt(p, 4)


def write_latex_table(summary: dict[str, dict[str, Any]],
                      caption: str, label: str, out_path: Path, title: str) -> None:
    lines: list[str] = []
    lines.append(f"% {title} -- generated by chapter5_geomaca.lbf.tables")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & $n$ & Mean $\pm$ SD & SE & 95\% CI"
        r" & $p$ vs.\ GeoMACA & Cohen's $d$ \\"
    )
    lines.append(r"\midrule")
    for m in METHOD_ORDER:
        s = summary[m]
        ci_lo = s["mean"] - s["ci95_halfwidth"]
        ci_hi = s["mean"] + s["ci95_halfwidth"]
        ci_str = (f"[{fmt(ci_lo, 3)},\\,{fmt(ci_hi, 3)}]"
                  if not math.isnan(s["ci95_halfwidth"]) else "{--}")
        d = s["cohen_d_vs_geomaca"]
        lines.append(
            f"{METHOD_DISPLAY[m]} & {s['n']} & "
            f"{fmt(s['mean'], 3)} $\\pm$ {fmt(s['sd'], 3)} & "
            f"{fmt(s['se'], 3)} & {ci_str} & "
            f"{fmt_p(s['welch_p_vs_geomaca'])} & "
            f"{'{--}' if math.isnan(d) else fmt(d, 2)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


TRAIN_CAPTION = (
    "LBF final-training-window return: mean of the last $5{,}000$ training-episode returns "
    "per seed. Training returns include the proximity-shaping term "
    "(Sec.~\\ref{sec:p3-domain}); IPPO denotes independent PPO learners."
)
EVAL_CAPTION = (
    "LBF held-out evaluation. Each seed contributes the mean episodic return, food reward "
    "only, over 100 held-out evaluation episodes of its trained policy; IPPO denotes "
    "independent PPO learners."
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_tables(runs: dict[str, list[dict[str, Any]]], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ---- Final-training-window table (Table 5.6) ----
    train_summary = summarise(runs, "train_window_return")
    write_latex_table(train_summary, TRAIN_CAPTION, "tab:lbf-train-window",
                      out_dir / "lbf_train_window_table.tex", "Table 5.6")
    with open(out_dir / "lbf_train_window_summary.json", "w") as f:
        json.dump({"metric": "train_window_return",
                   "window_size": TRAIN_WINDOW,
                   "description":
                       "Mean of the last 5000 episode returns from the training "
                       "trajectory; one observation per seed.",
                   "methods": train_summary,
                   "pivot": PIVOT_METHOD}, f, indent=2)
    written += [out_dir / "lbf_train_window_table.tex", out_dir / "lbf_train_window_summary.json"]

    # ---- Held-out eval table (Table 5.7) ----
    eval_summary = summarise(runs, "eval_mean_return")
    write_latex_table(eval_summary, EVAL_CAPTION, "tab:lbf-eval",
                      out_dir / "lbf_eval_table.tex", "Table 5.7")
    with open(out_dir / "lbf_eval_summary.json", "w") as f:
        json.dump({"metric": "eval_mean_return",
                   "description":
                       "Held-out evaluation mean return; one observation per seed "
                       "= eval.mean_return (mean over 100 eval episodes).",
                   "methods": eval_summary,
                   "pivot": PIVOT_METHOD}, f, indent=2)
    written += [out_dir / "lbf_eval_table.tex", out_dir / "lbf_eval_summary.json"]

    # ---- Per-seed long-form CSV ----
    write_per_seed_csv(runs, out_dir / "lbf_per_seed.csv")
    written.append(out_dir / "lbf_per_seed.csv")

    # ---- Console report ----
    for title, summ in (("held-out eval", eval_summary), (f"final-{TRAIN_WINDOW} train window", train_summary)):
        print("=" * 72)
        print(f"LBF: {title}")
        print("=" * 72)
        for m in METHOD_ORDER:
            s = summ[m]
            print(f"  {m:15s} n={s['n']:2d}  "
                  f"mean={s['mean']:+.3f}  sd={s['sd']:.3f}  se={s['se']:.3f}  "
                  f"p={s['welch_p_vs_geomaca']:.3g}  d={s['cohen_d_vs_geomaca']:+.2f}")
    return written


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=str, default=os.environ.get("CHAPTER5_LBF_RESULTS"),
                    help="folder with the per-seed result files <method>_seed<N>.json "
                         "(default $CHAPTER5_LBF_RESULTS; without it, records/lbf_per_seed.csv)")
    ap.add_argument("--per-seed", type=str, default=str(PER_SEED_RECORD),
                    help="per-seed CSV used when --results is not given")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args(argv)
    runs = load_runs(Path(args.results)) if args.results else load_per_seed_csv(Path(args.per_seed))
    for p in build_tables(runs, Path(args.out)):
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
