"""Figure 5.3: CityBuilder performance aggregated by Triple-Bottom-Line pillar.

For each metric, the z-score of each method's mean across the four methods of
Table 5.2 (ddof = 1), sign-flipped for the lower-is-better metrics (Density,
Segregation) so that higher is uniformly better; the z-scores are averaged
within each pillar, and Overall averages the three pillar scores. The bars
therefore equal the pillar columns of Table 5.3.

Usage::

    python -m chapter5_geomaca.evaluation.figures --eval <citybuilder_evaluation.json> --out <dir>

Writes ``cb_pillar_aggregate_bars.pdf`` and ``.png`` (vector PDF; colour-blind
safe Tol palette) and ``cb_pillar_aggregate_bars.json`` with the bar heights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = PACKAGE_ROOT / "results" / "evaluation" / "citybuilder_evaluation.json"
DEFAULT_OUT = PACKAGE_ROOT / "results" / "figures"

# Tol Bright palette: color-blind safe and works well in print
TOL_BRIGHT = {
    "blue":   "#4477AA",
    "red":    "#EE6677",
    "green":  "#228833",
    "yellow": "#CCBB44",
    "cyan":   "#66CCEE",
    "purple": "#AA3377",
    "grey":   "#BBBBBB",
}

PUB_RC = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,  # TrueType embedded so the text in the PDF can be extracted
}

PILLARS = {
    "Economy":     [("gdp", True), ("tax_revenue", True), ("developer_profit", True)],
    "Society":     [("population", True), ("segregation", False), ("qol", True)],
    "Environment": [("green_ratio", True), ("woodland_ratio", True), ("density", False)],
}
PILLAR_LABELS = ["Economy", "Society", "Environment", "Overall"]
METHOD_ORDER = [("Heuristic", "heuristic"), ("QMIX", "qmix"),
                ("COMA", "coma"), ("GeoMACA", "geomaca")]


def pillar_scores(data: Dict) -> Dict[str, Dict[str, float]]:
    """Bar heights: pillar -> method label -> z-score."""
    methods = data.get("methods", {})
    method_order = [(lab, k) for lab, k in METHOD_ORDER if k in methods]

    def metric_mean(method_key, key):
        s = methods[method_key]["summary"].get(key, {})
        return s.get("mean", float("nan"))

    def z_for(method_key, key, higher_better):
        vals = [metric_mean(k, key) for _, k in method_order]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 1.0
        if sd == 0:
            return 0.0
        zv = (metric_mean(method_key, key) - mu) / sd
        return zv if higher_better else -zv

    pillar_z = {lab: {} for lab in PILLAR_LABELS}
    for lab, mkey in method_order:
        ez = sum(z_for(mkey, k, hb) for k, hb in PILLARS["Economy"]) / 3.0
        sz = sum(z_for(mkey, k, hb) for k, hb in PILLARS["Society"]) / 3.0
        nz = sum(z_for(mkey, k, hb) for k, hb in PILLARS["Environment"]) / 3.0
        pillar_z["Economy"][lab] = ez
        pillar_z["Society"][lab] = sz
        pillar_z["Environment"][lab] = nz
        pillar_z["Overall"][lab] = (ez + sz + nz) / 3.0
    return pillar_z


def cb_pillar_aggregate_bars(data: Dict, out_dir: Path) -> Dict[str, Dict[str, float]]:
    """Grouped bars: 4 pillars (Economy, Society, Environment, Overall) x 4 methods."""
    pillar_z = pillar_scores(data)
    method_labels = list(pillar_z["Economy"].keys())
    palette = [TOL_BRIGHT["grey"], TOL_BRIGHT["blue"], TOL_BRIGHT["red"], TOL_BRIGHT["green"]]
    n_methods = len(method_labels)
    bar_w = 0.8 / n_methods
    xs = np.arange(len(PILLAR_LABELS))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        for m_idx, lab in enumerate(method_labels):
            heights = [pillar_z[p][lab] for p in PILLAR_LABELS]
            offset = (m_idx - (n_methods - 1) / 2) * bar_w
            ax.bar(xs + offset, heights, width=bar_w, label=lab, color=palette[m_idx])

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(PILLAR_LABELS)
        ax.set_ylabel("Pillar z-score (higher is better)")
        ax.grid(axis="y", linestyle=":", alpha=0.4, linewidth=0.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                  ncol=n_methods, frameon=False)

        plt.savefig(out_dir / "cb_pillar_aggregate_bars.pdf")
        plt.savefig(out_dir / "cb_pillar_aggregate_bars.png", dpi=300)
        plt.close(fig)
    (out_dir / "cb_pillar_aggregate_bars.json").write_text(json.dumps(pillar_z, indent=2),
                                                          encoding="utf-8")
    return pillar_z


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=str, default=str(DEFAULT_EVAL))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    data = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    pillar_z = cb_pillar_aggregate_bars(data, Path(args.out))
    print(f"Wrote {Path(args.out) / 'cb_pillar_aggregate_bars.pdf'} (and .png, .json)")
    for p in PILLAR_LABELS:
        print(f"  {p:12s} " + "  ".join(f"{lab}={v:+.4f}" for lab, v in pillar_z[p].items()))


if __name__ == "__main__":
    main()
