"""Tables 5.2 and 5.3 from a CityBuilder evaluation record.

Reads ``citybuilder_evaluation.json`` (written by
``chapter5_geomaca.evaluation.evaluate``) and writes

* ``citybuilder_main_table.tex``  - Table 5.2: every method and metric, mean
  +- SE across replicates, bold best mean, Welch's two-sample t-tests against
  GeoMACA, and the count of metrics on which each method has the best mean;
* ``citybuilder_ranks_table.tex`` - Table 5.3: Top-2 counts, average rank and
  pillar z-scores (Density and Segregation sign-flipped).

Statistical conventions: Welch's two-sample t-test across replicate means
with ddof=1 SD.

Usage::

    python -m chapter5_geomaca.evaluation.tables --eval <citybuilder_evaluation.json> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = PACKAGE_ROOT / "results" / "evaluation" / "citybuilder_evaluation.json"
DEFAULT_OUT = PACKAGE_ROOT / "results" / "tables"


# ---------- stats ------------------------------------------------------------
def mean_(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd_(xs):
    if len(xs) < 2:
        return 0.0
    m = mean_(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def se_(xs):
    return sd_(xs) / math.sqrt(len(xs)) if xs else float("nan")


def _log_gamma(x):
    return math.lgamma(x)


def _betacf(x, a, b, max_iter=200, eps=3e-7):
    qab = a + b; qap = a + 1.0; qam = a - 1.0
    c = 1.0; d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d; h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h


def _t_cdf(t, df):
    x = df / (df + t * t)
    a = df / 2.0; b = 0.5
    bt = math.exp(_log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        ix = bt * _betacf(x, a, b) / a
    else:
        ix = 1.0 - bt * _betacf(1.0 - x, b, a) / b
    return 1.0 - 0.5 * ix if t >= 0 else 0.5 * ix


def welch_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = mean_(a), mean_(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return 1.0
    t = (ma - mb) / se
    den = ((va / len(a)) ** 2) / (len(a) - 1) + ((vb / len(b)) ** 2) / (len(b) - 1)
    df = (va / len(a) + vb / len(b)) ** 2 / den if den > 0 else float("nan")
    return 2.0 * (1.0 - _t_cdf(abs(t), df))


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = sd_(a), sd_(b)
    pooled = math.sqrt(((len(a) - 1) * sa * sa + (len(b) - 1) * sb * sb)
                       / (len(a) + len(b) - 2))
    return (mean_(a) - mean_(b)) / pooled if pooled else float("nan")


# ---------- formatting ------------------------------------------------------
METRICS = [
    ("gdp",              "GDP (\\$M $\\uparrow$)",   "money_M", True),
    ("tax_revenue",      "Tax Revenue (\\$M $\\uparrow$)", "money_M", True),
    ("developer_profit", "Developer Profit (\\$M $\\uparrow$)", "money_M", True),
    ("green_ratio",      "Green ratio (\\% $\\uparrow$)", "pct", True),
    ("woodland_ratio",   "Woodland ratio (\\% $\\uparrow$)", "pct", True),
    ("density",          "Density (\\% $\\downarrow$)", "pct", False),
    ("population",       "Population $\\uparrow$",   "int", True),
    ("segregation",      "Segregation $\\downarrow$", "f3", False),
    ("qol",              "Quality of Life $\\uparrow$", "f3", True),
]

METHOD_ORDER_MAIN = [
    ("Heuristic",  "heuristic"),
    ("QMIX",       "qmix"),
    ("COMA",       "coma"),
    ("GeoMACA",    "geomaca"),
]

PILLAR_GROUPS = [
    ("Economy",   ["gdp", "tax_revenue", "developer_profit"]),
    ("Environ.",  ["green_ratio", "woodland_ratio", "density"]),
    ("Society",   ["population", "segregation", "qol"]),
]


def fmt(mean, se, sty):
    if isinstance(mean, float) and math.isnan(mean):
        return "--"
    if sty == "money_M":
        return f"{mean/1e6:.1f}{{\\scriptsize$\\pm${se/1e6:.2f}}}"
    if sty == "pct":
        return f"{mean*100:.1f}{{\\scriptsize$\\pm${se*100:.2f}}}"
    if sty == "int":
        return f"{int(round(mean))}{{\\scriptsize$\\pm${int(round(se))}}}"
    if sty == "f3":
        return f"{mean:.3f}{{\\scriptsize$\\pm${se:.3f}}}"
    return f"{mean:.3f}"


def sig(p):
    if math.isnan(p): return ""
    if p < 1e-4: return "$^{****}$"
    if p < 1e-3: return "$^{***}$"
    if p < 1e-2: return "$^{**}$"
    if p < 5e-2: return "$^{*}$"
    return ""


def replicate_means(data: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Per-method per-metric replicate means of an evaluation record."""
    method_means: Dict[str, Dict[str, List[float]]] = {}
    for name, m in data["methods"].items():
        method_means[name] = {}
        for key, _, _, _ in METRICS:
            v = m["summary"].get(key, {}).get("values", [])
            method_means[name][key] = v
    return method_means


# ---------- Table 5.2 ---------------------------------------------------------
def main_table_tex(data: Dict, source: str = "") -> str:
    method_means = replicate_means(data)
    if "geomaca" not in method_means:
        raise ValueError("the evaluation record has no GeoMACA replicates")
    n_learned = sorted({len(method_means[k]["gdp"]) for k in ("qmix", "coma", "geomaca")
                        if k in method_means})
    n_heu = len(method_means.get("heuristic", {}).get("gdp", []))
    learned_n = str(n_learned[0]) if len(n_learned) == 1 else "/".join(map(str, n_learned))
    rows = []
    rows.append("% Table 5.2 -- generated by chapter5_geomaca.evaluation.tables"
                + (f" from {source}" if source else ""))
    rows.append("\\begin{table}[t]")
    rows.append("\\centering\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{3pt}")
    rows.append(
        "\\caption{Unified-protocol CityBuilder evaluation. All methods use the same "
        "uniform-random developer at evaluation time. Heuristic has "
        f"{n_heu} evaluation replicates; learned methods have {learned_n} retrained seeds. "
        "Cells are mean $\\pm$ SE across replicates. \\textbf{Bold} marks the best mean "
        "per metric; significance markers are Welch's two-sample $t$-tests against "
        "GeoMACA: $^*p<0.05$, $^{**}p<0.01$, $^{***}p<10^{-3}$, $^{****}p<10^{-4}$.}")
    rows.append("\\label{tab:p3-main}")
    rows.append("\\begin{tabular}{ll" + "c" * len(METHOD_ORDER_MAIN) + "}")
    rows.append("\\toprule")
    rows.append("\\textbf{Pillar} & \\textbf{Metric} & " +
                " & ".join(f"\\textbf{{{lab}}}" for lab, _ in METHOD_ORDER_MAIN) + " \\\\")
    rows.append("\\midrule")

    geo_data = method_means["geomaca"]
    metric_lookup = {m[0]: m for m in METRICS}

    # Track per-method wins for footer
    win_counts = {k: 0 for _, k in METHOD_ORDER_MAIN}

    for gi, (pillar_name, keys) in enumerate(PILLAR_GROUPS):
        n_rows_in_group = len(keys)
        for ri, key in enumerate(keys):
            _, label, sty, hb = metric_lookup[key]
            ms = []
            for _, k in METHOD_ORDER_MAIN:
                if k not in method_means: continue
                xs = method_means[k][key]
                ms.append((k, mean_(xs)))
            if not ms:
                continue
            best_val = max(m for _, m in ms) if hb else min(m for _, m in ms)
            best_method = next(k for k, m in ms if abs(m - best_val) < 1e-9)
            win_counts[best_method] = win_counts.get(best_method, 0) + 1
            cells = []
            for lab, k in METHOD_ORDER_MAIN:
                if k not in method_means:
                    cells.append("--"); continue
                xs = method_means[k][key]
                mu = mean_(xs); se = se_(xs)
                cell = fmt(mu, se, sty)
                if abs(mu - best_val) < 1e-9:
                    cell = f"\\textbf{{{cell}}}"
                if k != "geomaca":
                    cell = cell + sig(welch_p(geo_data[key], xs))
                cells.append(cell)
            pillar_cell = (
                f"\\multirow{{{n_rows_in_group}}}{{*}}{{\\textbf{{{pillar_name}}}}}"
                if ri == 0 else ""
            )
            rows.append(f"{pillar_cell} & {label} & " + " & ".join(cells) + " \\\\")
        if gi < len(PILLAR_GROUPS) - 1:
            rows.append("\\midrule")

    rows.append("\\midrule")
    win_cells = [f"{win_counts.get(k, 0)}/9" for _, k in METHOD_ORDER_MAIN]
    # Bold the largest win count
    max_wins = max(win_counts.values()) if win_counts else 0
    win_cells = [
        f"\\textbf{{{c}}}" if win_counts.get(k, 0) == max_wins and max_wins > 0 else c
        for c, (_, k) in zip(win_cells, METHOD_ORDER_MAIN)
    ]
    rows.append("\\multicolumn{2}{l}{\\textbf{Metric wins}} & " + " & ".join(win_cells) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


# ---------- Table 5.3 ---------------------------------------------------------
PILLARS = {
    "Economy": ("gdp", "tax_revenue", "developer_profit"),
    "Society": ("population", "segregation", "qol"),
    "Environment": ("green_ratio", "woodland_ratio", "density"),
}


def ranks_and_pillars(data: Dict) -> Dict[str, Dict[str, float]]:
    """Top-2 count, average rank and pillar z-scores of each method of Table 5.2."""
    method_means = replicate_means(data)
    methods = [k for _, k in METHOD_ORDER_MAIN if k in method_means
               and method_means[k]["gdp"]]
    # Per-metric rank (1=best)
    ranks = {k: {} for k in methods}
    for key, _, _, hb in METRICS:
        vals = [(m, mean_(method_means[m][key])) for m in methods]
        sorted_v = sorted(vals, key=(lambda p: -p[1]) if hb else (lambda p: p[1]))
        for r, (m, _) in enumerate(sorted_v, 1):
            ranks[m][key] = r

    # z-score within methods
    def z(m, key, hb):
        vals = [mean_(method_means[mm][key]) for mm in methods]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 1.0
        if sd == 0: return 0.0
        zv = (mean_(method_means[m][key]) - mu) / sd
        return zv if hb else -zv

    higher_map = {key: hb for key, _, _, hb in METRICS}
    out = {}
    for k in methods:
        top2 = sum(1 for r in ranks[k].values() if r <= 2)
        avg = sum(ranks[k].values()) / len(ranks[k])
        ez = sum(z(k, key, higher_map[key]) for key in PILLARS["Economy"]) / 3
        sz = sum(z(k, key, higher_map[key]) for key in PILLARS["Society"]) / 3
        nz = sum(z(k, key, higher_map[key]) for key in PILLARS["Environment"]) / 3
        ov = (ez + sz + nz) / 3
        out[k] = {"top2": top2, "avg_rank": avg, "economy": ez, "society": sz,
                  "environment": nz, "overall": ov}
    return out


def ranks_table_tex(data: Dict, source: str = "") -> str:
    stats = ranks_and_pillars(data)
    rows = []
    rows.append("% Table 5.3 -- generated by chapter5_geomaca.evaluation.tables"
                + (f" from {source}" if source else ""))
    rows.append("\\begin{table}[t]\\centering\\small")
    rows.append("\\caption{Aggregated CityBuilder comparison. "
                "\\emph{Top-2} is the count of metrics on which the "
                "method ranks first or second among the four methods. "
                "\\emph{Avg.\\ rank} averages the per-metric rank "
                "(lower better). Pillar columns are z-scores within the "
                "four-method group; negative-direction "
                "metrics (Density, Segregation) are sign-flipped.}")
    rows.append("\\label{tab:p3-ranks}")
    rows.append("\\begin{tabular}{lcccccc}")
    rows.append("\\toprule")
    rows.append("\\textbf{Method} & \\textbf{Top-2} & \\textbf{Avg rank} & "
                "\\textbf{Eco $z$} & \\textbf{Soc $z$} & \\textbf{Env $z$} & \\textbf{Overall} \\\\")
    rows.append("\\midrule")
    for lab, k in METHOD_ORDER_MAIN:
        if k not in stats: continue
        s = stats[k]
        rows.append(f"{lab} & {s['top2']}/9 & {s['avg_rank']:.2f} & "
                    f"{s['economy']:+.2f} & {s['society']:+.2f} & {s['environment']:+.2f} & "
                    f"{s['overall']:+.2f} \\\\")
    rows.append("\\bottomrule\\end{tabular}\\end{table}")
    return "\n".join(rows) + "\n"


def write_tables(eval_path: Path, out_dir: Path, which=("5.2", "5.3")) -> List[Path]:
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if "5.2" in which:
        p = out_dir / "citybuilder_main_table.tex"
        p.write_text(main_table_tex(data, Path(eval_path).name), encoding="utf-8")
        written.append(p)
    if "5.3" in which:
        p = out_dir / "citybuilder_ranks_table.tex"
        p.write_text(ranks_table_tex(data, Path(eval_path).name), encoding="utf-8")
        written.append(p)
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=str, default=str(DEFAULT_EVAL))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    for p in write_tables(Path(args.eval), Path(args.out)):
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
