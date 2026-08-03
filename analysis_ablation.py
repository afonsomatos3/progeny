"""
analysis_ablation.py — compare grammar ablation results.

Reads ablation output from pipeline_results/ablation/ and generates
comparative charts showing how each removed grammar rule affects
proxy discovery quality.

Usage:
    python analysis_ablation.py pipeline_results/ablation
    python analysis_ablation.py pipeline_results/ablation --out figs/ablation --grammar ext
"""

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

# ── loading ───────────────────────────────────────────────────────────────────

def load_ablation(ablation_root: pathlib.Path, grammar_filter: str | None = None) -> dict:
    """
    Returns dict: {ablation_label: [row_dicts]}
    Each row has grammar, dataset, protected_attr, auc, nodes, etc.
    """
    results: dict[str, list[dict]] = defaultdict(list)

    for ablation_dir in sorted(ablation_root.iterdir()):
        if not ablation_dir.is_dir():
            continue
        label = ablation_dir.name
        for csv_path in sorted(ablation_dir.rglob("*.csv")):
            stem = csv_path.stem
            if any(x in stem for x in ("partials", "near-miss", "bak", "orig")):
                continue
            grammar = ("arith" if stem.startswith("arith") else
                       "ext"   if stem.startswith("ext")   else None)
            if grammar is None:
                continue
            if grammar_filter and grammar != grammar_filter:
                continue
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["_grammar"]  = grammar
                    row["_ablation"] = label
                    results[label].append(row)

    return dict(results)


def _flt(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return default


# ── charts ────────────────────────────────────────────────────────────────────

def chart_auc_comparison(data: dict, out_dir: pathlib.Path, grammar: str = "ext"):
    """
    Bar chart: mean AUC per ablation config (baseline highlighted).
    One bar per ablation label, grouped by dataset.
    """
    labels   = list(data.keys())
    if not labels:
        return
    datasets = sorted({r.get("dataset","?")
                       for rows in data.values() for r in rows
                       if r["_grammar"] == grammar})
    if not datasets:
        return

    x     = np.arange(len(labels))
    width = 0.8 / max(len(datasets), 1)
    cmap  = plt.cm.get_cmap("tab10", len(datasets))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))

    for di, ds in enumerate(datasets):
        means, errs = [], []
        for label in labels:
            vals = [_flt(r, "auc")
                    for r in data.get(label, [])
                    if r["_grammar"] == grammar and r.get("dataset") == ds]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else float("nan"))
            errs.append(np.std(vals) / len(vals)**0.5 if len(vals) > 1 else 0)
        offset = (di - len(datasets)/2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9,
               label=ds, color=cmap(di), alpha=0.8,
               yerr=errs, capsize=2, error_kw=dict(lw=0.8))

    # Vertical line at baseline
    if "baseline" in labels:
        bl_idx = labels.index("baseline")
        ax.axvline(bl_idx, color="black", lw=1, ls="--", alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Mean proxy AUC by ablation config — {grammar} grammar")
    ax.legend(title="Dataset", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fname = f"ablation_auc_{grammar}.png"
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f"  [saved] {fname}")


def chart_auc_delta(data: dict, out_dir: pathlib.Path, grammar: str = "ext"):
    """
    Delta-AUC bars: how much does each ablation reduce AUC vs baseline?
    Negative = the removed rule was helpful.
    """
    if "baseline" not in data:
        print("  [skip] No baseline found — cannot compute delta AUC")
        return

    baseline_aucs: dict[str, float] = {}
    for ds in {r.get("dataset","?") for r in data["baseline"] if r["_grammar"]==grammar}:
        vals = [_flt(r,"auc") for r in data["baseline"]
                if r["_grammar"]==grammar and r.get("dataset")==ds]
        baseline_aucs[ds] = np.mean([v for v in vals if not np.isnan(v)])

    labels   = [l for l in data.keys() if l != "baseline"]
    datasets = sorted(baseline_aucs.keys())

    x     = np.arange(len(labels))
    width = 0.8 / max(len(datasets), 1)
    cmap  = plt.cm.get_cmap("tab10", len(datasets))

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.6), 5))

    for di, ds in enumerate(datasets):
        deltas = []
        for label in labels:
            vals = [_flt(r,"auc") for r in data.get(label,[])
                    if r["_grammar"]==grammar and r.get("dataset")==ds]
            vals = [v for v in vals if not np.isnan(v)]
            mean = np.mean(vals) if vals else float("nan")
            deltas.append(mean - baseline_aucs.get(ds, float("nan")))
        offset = (di - len(datasets)/2 + 0.5) * width
        ax.bar(x + offset, deltas, width * 0.9,
               label=ds, color=cmap(di), alpha=0.8)

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("ΔAUC vs baseline")
    ax.set_title(f"AUC change from removing grammar rule — {grammar}")
    ax.legend(title="Dataset", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.3f"))
    fig.tight_layout()
    fname = f"ablation_delta_auc_{grammar}.png"
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f"  [saved] {fname}")


def chart_proxy_count_comparison(data: dict, out_dir: pathlib.Path, grammar: str = "ext"):
    """Bar chart: number of proxies found per ablation config."""
    labels = list(data.keys())
    counts = [sum(1 for r in data.get(l,[]) if r["_grammar"] == grammar) for l in labels]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4))
    colours = ["#2196F3" if l == "baseline" else "#78909C" for l in labels]
    ax.bar(labels, counts, color=colours, alpha=0.85)
    ax.set_ylabel("Proxies found")
    ax.set_title(f"Proxy count per ablation — {grammar}")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fname = f"ablation_count_{grammar}.png"
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f"  [saved] {fname}")


def chart_complexity_comparison(data: dict, out_dir: pathlib.Path, grammar: str = "ext"):
    """Box plot: node count distribution per ablation config."""
    labels = list(data.keys())
    node_data = [[_flt(r,"nodes") for r in data.get(l,[])
                  if r["_grammar"] == grammar and not np.isnan(_flt(r,"nodes"))]
                 for l in labels]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4))
    bp = ax.boxplot(
        [d if d else [0] for d in node_data],
        labels=labels, patch_artist=True,
        boxprops=dict(facecolor="#78909C", alpha=0.6),
        medianprops=dict(color="black", lw=1.5),
    )
    # Highlight baseline box in blue
    if "baseline" in labels:
        bi = labels.index("baseline")
        bp["boxes"][bi].set_facecolor("#2196F3")

    ax.set_ylabel("Expression nodes")
    ax.set_title(f"Expression complexity per ablation — {grammar}")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fname = f"ablation_complexity_{grammar}.png"
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f"  [saved] {fname}")


def write_summary(data: dict, out_dir: pathlib.Path, grammar: str = "ext"):
    """CSV summary table."""
    path = out_dir / f"ablation_summary_{grammar}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ablation","dataset","n_proxies","mean_auc","std_auc",
                    "mean_nodes","mean_precision","mean_recall","mean_coverage"])
        for label, rows in data.items():
            sub = [r for r in rows if r["_grammar"] == grammar]
            datasets = sorted({r.get("dataset","?") for r in sub})
            for ds in datasets:
                dsub = [r for r in sub if r.get("dataset") == ds]
                def _stat(key):
                    vs = [_flt(r,key) for r in dsub if not np.isnan(_flt(r,key))]
                    return (f"{np.mean(vs):.4f}", f"{np.std(vs):.4f}") if vs else ("","")
                auc_mean, auc_std = _stat("auc")
                node_mean, _ = _stat("nodes")
                prec_mean, _ = _stat("precision")
                rec_mean,  _ = _stat("recall")
                cov_mean,  _ = _stat("coverage")
                w.writerow([label, ds, len(dsub), auc_mean, auc_std,
                            node_mean, prec_mean, rec_mean, cov_mean])
    print(f"  [saved] {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Grammar ablation analysis charts")
    ap.add_argument("ablation_dir", help="Ablation results root (pipeline_results/ablation)")
    ap.add_argument("--out",     default=None)
    ap.add_argument("--grammar", default="ext", choices=["arith","ext"])
    args = ap.parse_args()

    ablation_dir = pathlib.Path(args.ablation_dir)
    if not ablation_dir.exists():
        sys.exit(f"Not found: {ablation_dir}")

    out_dir = pathlib.Path(args.out) if args.out else ablation_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ablation results from {ablation_dir} …")
    data = load_ablation(ablation_dir, grammar_filter=None)
    if not data:
        sys.exit("No ablation result CSVs found.")
    print(f"  {len(data)} ablation configs: {list(data.keys())}")

    grm = args.grammar
    print(f"Generating charts (grammar={grm}) …")
    chart_auc_comparison(data, out_dir, grammar=grm)
    chart_auc_delta(data, out_dir, grammar=grm)
    chart_proxy_count_comparison(data, out_dir, grammar=grm)
    chart_complexity_comparison(data, out_dir, grammar=grm)
    write_summary(data, out_dir, grammar=grm)

    # Also run arith if both grammars present
    if grm != "arith":
        has_arith = any(r["_grammar"] == "arith"
                        for rows in data.values() for r in rows)
        if has_arith:
            print("Generating arith charts …")
            chart_auc_comparison(data, out_dir, grammar="arith")
            chart_auc_delta(data, out_dir, grammar="arith")

    print(f"\nDone. Figures in: {out_dir}")


if __name__ == "__main__":
    main()
