"""
analysis_stats.py — proxy discovery statistics and charts.

Reads a pipeline results folder (pipeline_results/and_balance or any
pipeline_runs/TIMESTAMP layout) and produces:
  • summary_stats.csv     — per-dataset / per-grammar aggregate table
  • figures/              — PNG charts

Usage:
    python analysis_stats.py pipeline_results/and_balance
    python analysis_stats.py pipeline_results/and_balance --out figs/ --config balanced
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

# ── colour palette (accessible, print-friendly) ───────────────────────────────
C_ARITH  = "#4C72B0"   # blue
C_EXT    = "#DD8452"   # orange
C_BLINE  = "#55A868"   # green (baseline)
PALETTE  = [C_ARITH, C_EXT, C_BLINE, "#C44E52", "#8172B2", "#937860"]

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})


# ── data loading ─────────────────────────────────────────────────────────────

def _detect_grammar(stem: str) -> str:
    """arith / ext / continuous from filename stem."""
    if stem.startswith("arith"):
        return "arith"
    if stem.startswith("ext"):
        return "ext"
    if stem.startswith("cont"):
        return "continuous"
    return "unknown"


def load_run(run_dir: pathlib.Path, config_filter: str | None = None) -> list[dict]:
    """
    Walk run_dir looking for top-result CSVs (not partials / near-miss / bak).
    Returns list of row dicts augmented with {grammar, config, dataset_dir}.
    """
    rows = []
    for csv_path in sorted(run_dir.rglob("*.csv")):
        stem = csv_path.stem
        if any(x in stem for x in ("partials", "near-miss", "bak", "orig")):
            continue
        if any(x in csv_path.suffixes for x in (".bak", ".orig")):
            continue
        grammar = _detect_grammar(stem)
        if grammar == "unknown":
            continue
        # config = first sub-dir under run_dir (balanced / prec_heavy / …)
        try:
            rel = csv_path.relative_to(run_dir)
            config = rel.parts[0] if len(rel.parts) > 1 else "default"
        except ValueError:
            config = "default"
        if config_filter and config != config_filter:
            continue
        # dataset name = directory immediately under the config dir
        try:
            rel     = csv_path.relative_to(run_dir)
            # rel.parts: (config, dataset, filename)
            dataset_dir = rel.parts[1] if len(rel.parts) >= 3 else rel.parts[0]
        except ValueError:
            dataset_dir = "unknown"

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_grammar"]     = grammar
                row["_config"]      = config
                row["_dataset_dir"] = dataset_dir   # e.g. "law_school", "adult"
                row["_file"]        = str(csv_path)
                rows.append(row)
    return rows


def _flt(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return default


# ── statistics ────────────────────────────────────────────────────────────────

def compute_stats(rows: list[dict]) -> dict:
    """
    Returns nested dict:  stats[config][grammar][dataset] = {metric: [values]}
    """
    stats: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in rows:
        cfg     = row["_config"]
        grm     = row["_grammar"]
        dataset = row.get("_dataset_dir", row.get("dataset", "?"))
        for metric in ("auc", "precision", "recall", "coverage", "nodes", "cohens_d"):
            v = _flt(row, metric)
            if not np.isnan(v):
                stats[cfg][grm][dataset][metric].append(v)
    return stats


def flat_table(rows: list[dict]) -> list[dict]:
    """Return per-row dicts with numeric fields coerced, keeping meta fields."""
    out = []
    for row in rows:
        r = dict(row)   # keep everything including _grammar / _config
        for m in ("auc", "precision", "recall", "coverage", "nodes",
                  "cohens_d", "elapsed_s"):
            r[m] = _flt(row, m)
        out.append(r)
    return out


# ── charts ───────────────────────────────────────────────────────────────────

def chart_auc_by_dataset(rows: list[dict], out_dir: pathlib.Path, config: str = "balanced"):
    """Grouped bar: mean AUC per dataset, grouped by grammar."""
    subset = [r for r in rows if r["_config"] == config]
    if not subset:
        return

    datasets  = sorted({r.get("_dataset_dir", r.get("dataset","?")) for r in subset})
    grammars  = ["arith", "ext"]
    colours   = [C_ARITH, C_EXT]

    x     = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.3), 4))

    for i, (grm, col) in enumerate(zip(grammars, colours)):
        means, errs = [], []
        for ds in datasets:
            vals = [_flt(r, "auc") for r in subset
                    if r["_grammar"] == grm and r.get("_dataset_dir", r.get("dataset","?")) == ds]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else float("nan"))
            errs.append(np.std(vals) / (len(vals)**0.5) if len(vals) > 1 else 0)
        ax.bar(x + i * width, means, width, label=grm, color=col, alpha=0.85,
               yerr=errs, capsize=3, error_kw=dict(lw=1))

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title(f"Mean proxy AUC by dataset — {config} config")
    ax.set_ylim(0.5, 1.0)
    ax.legend(title="Grammar")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_dir / f"auc_by_dataset_{config}.png")
    plt.close(fig)
    print(f"  [saved] auc_by_dataset_{config}.png")


def chart_complexity_distribution(rows: list[dict], out_dir: pathlib.Path, config: str = "balanced"):
    """Box plot: expression node-count distribution per dataset."""
    subset = [r for r in rows if r["_config"] == config]
    if not subset:
        return

    datasets = sorted({r.get("_dataset_dir", r.get("dataset","?")) for r in subset})
    data_arith = [[_flt(r,"nodes") for r in subset
                   if r.get("_dataset_dir", r.get("dataset","?"))==ds and r["_grammar"]=="arith"
                   and not np.isnan(_flt(r,"nodes"))] for ds in datasets]
    data_ext   = [[_flt(r,"nodes") for r in subset
                   if r.get("_dataset_dir", r.get("dataset","?"))==ds and r["_grammar"]=="ext"
                   and not np.isnan(_flt(r,"nodes"))] for ds in datasets]

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.5), 4))
    x = np.arange(len(datasets))
    w = 0.3

    def _bp(ax, data, positions, col, label):
        bp = ax.boxplot(
            [d if d else [0] for d in data],
            positions=positions, widths=w, patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.6),
            medianprops=dict(color="black", lw=1.5),
            whiskerprops=dict(color=col), capprops=dict(color=col),
            flierprops=dict(marker="o", color=col, alpha=0.4, ms=3),
            manage_ticks=False,
        )
        bp["boxes"][0].set_label(label)
        return bp

    _bp(ax, data_arith, x - w/2, C_ARITH, "arith")
    _bp(ax, data_ext,   x + w/2, C_EXT,   "ext")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Expression nodes")
    ax.set_title(f"Expression complexity (node count) — {config} config")
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, fc=C_ARITH, alpha=0.6, label="arith"),
        plt.Rectangle((0,0),1,1, fc=C_EXT,   alpha=0.6, label="ext"),
    ])
    fig.tight_layout()
    fig.savefig(out_dir / f"complexity_distribution_{config}.png")
    plt.close(fig)
    print(f"  [saved] complexity_distribution_{config}.png")


def chart_proxy_count(rows: list[dict], out_dir: pathlib.Path):
    """Stacked bar: proxy count by config × grammar."""
    configs  = sorted({r["_config"] for r in rows})
    grammars = ["arith", "ext"]
    colours  = [C_ARITH, C_EXT]

    x     = np.arange(len(configs))
    width = 0.5
    fig, ax = plt.subplots(figsize=(max(4, len(configs) * 1.4), 4))

    bottoms = np.zeros(len(configs))
    for grm, col in zip(grammars, colours):
        counts = [sum(1 for r in rows if r["_config"]==cfg and r["_grammar"]==grm)
                  for cfg in configs]
        ax.bar(x, counts, width, bottom=bottoms, label=grm, color=col, alpha=0.85)
        bottoms += np.array(counts, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=10)
    ax.set_ylabel("Number of proxy expressions")
    ax.set_title("Total proxy count by configuration")
    ax.legend(title="Grammar")
    fig.tight_layout()
    fig.savefig(out_dir / "proxy_count_by_config.png")
    plt.close(fig)
    print("  [saved] proxy_count_by_config.png")


def chart_coverage_precision(rows: list[dict], out_dir: pathlib.Path, config: str = "balanced"):
    """Scatter: coverage vs precision, coloured by AUC, per grammar."""
    subset = [r for r in rows if r["_config"] == config]
    if not subset:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, grm in zip(axes, ["arith", "ext"]):
        sub = [r for r in subset if r["_grammar"] == grm]
        cov  = [_flt(r, "coverage") for r in sub]
        prec = [_flt(r, "precision") for r in sub]
        auc  = [_flt(r, "auc")       for r in sub]
        sc = ax.scatter(cov, prec, c=auc, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                        alpha=0.7, s=30, edgecolors="none")
        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Precision (%)")
        ax.set_title(f"{grm} grammar")
    cb = fig.colorbar(sc, ax=axes, shrink=0.8)
    cb.set_label("AUC")
    fig.suptitle(f"Coverage vs Precision — {config} config")
    fig.tight_layout()
    fig.savefig(out_dir / f"coverage_precision_{config}.png")
    plt.close(fig)
    print(f"  [saved] coverage_precision_{config}.png")


def chart_auc_heatmap(rows: list[dict], out_dir: pathlib.Path, config: str = "balanced", grammar: str = "ext"):
    """Heatmap: mean AUC per dataset × protected_attr."""
    subset = [r for r in rows if r["_config"] == config and r["_grammar"] == grammar]
    if not subset:
        return

    datasets = sorted({r.get("_dataset_dir", r.get("dataset","?")) for r in subset})
    attrs    = sorted({r.get("protected_attr","?") for r in subset})

    mat = np.full((len(attrs), len(datasets)), float("nan"))
    for i, attr in enumerate(attrs):
        for j, ds in enumerate(datasets):
            vals = [_flt(r,"auc") for r in subset
                    if r.get("protected_attr")==attr and r.get("_dataset_dir", r.get("dataset","?"))==ds]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                mat[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(5, len(datasets)*1.2), max(3, len(attrs)*0.7)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(attrs)))
    ax.set_yticklabels(attrs, fontsize=9)
    for i in range(len(attrs)):
        for j in range(len(datasets)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if mat[i,j] < 0.85 else "white")
    fig.colorbar(im, ax=ax, label="AUC")
    ax.set_title(f"AUC heatmap — {config} / {grammar}")
    fig.tight_layout()
    fig.savefig(out_dir / f"auc_heatmap_{config}_{grammar}.png")
    plt.close(fig)
    print(f"  [saved] auc_heatmap_{config}_{grammar}.png")


# ── summary CSV ───────────────────────────────────────────────────────────────

def write_summary_csv(rows: list[dict], out_path: pathlib.Path) -> None:
    datasets = sorted({r.get("_dataset_dir", r.get("dataset","?")) for r in rows})
    grammars = sorted({r["_grammar"] for r in rows})
    configs  = sorted({r["_config"]  for r in rows})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config","grammar","dataset",
                    "n_proxies","mean_auc","std_auc","max_auc",
                    "mean_nodes","mean_precision","mean_recall","mean_coverage"])
        for cfg in configs:
            for grm in grammars:
                for ds in datasets:
                    sub = [r for r in rows
                           if r["_config"]==cfg and r["_grammar"]==grm
                           and r.get("_dataset_dir", r.get("dataset","?"))==ds]
                    if not sub:
                        continue
                    aucs  = [_flt(r,"auc")       for r in sub if not np.isnan(_flt(r,"auc"))]
                    nodes = [_flt(r,"nodes")      for r in sub if not np.isnan(_flt(r,"nodes"))]
                    prec  = [_flt(r,"precision")  for r in sub if not np.isnan(_flt(r,"precision"))]
                    rec   = [_flt(r,"recall")     for r in sub if not np.isnan(_flt(r,"recall"))]
                    cov   = [_flt(r,"coverage")   for r in sub if not np.isnan(_flt(r,"coverage"))]
                    w.writerow([
                        cfg, grm, ds,
                        len(sub),
                        f"{np.mean(aucs):.4f}"  if aucs  else "",
                        f"{np.std(aucs):.4f}"   if aucs  else "",
                        f"{np.max(aucs):.4f}"   if aucs  else "",
                        f"{np.mean(nodes):.1f}" if nodes else "",
                        f"{np.mean(prec):.1f}"  if prec  else "",
                        f"{np.mean(rec):.1f}"   if rec   else "",
                        f"{np.mean(cov):.1f}"   if cov   else "",
                    ])
    print(f"  [saved] {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Proxy discovery statistics and charts")
    ap.add_argument("run_dir", help="Pipeline results folder")
    ap.add_argument("--out",    default=None, help="Output directory (default: run_dir/figures)")
    ap.add_argument("--config", default=None, help="Restrict to one config (balanced/prec_heavy/…)")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"Not found: {run_dir}")

    out_dir = pathlib.Path(args.out) if args.out else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {run_dir} …")
    rows = load_run(run_dir, config_filter=args.config)
    if not rows:
        sys.exit("No result CSVs found.")
    print(f"  {len(rows)} proxy rows across "
          f"{len({r['_config'] for r in rows})} config(s), "
          f"{len({r.get('_dataset_dir', r.get('dataset','?')) for r in rows})} dataset(s)")

    configs = sorted({r["_config"] for r in rows})
    main_cfg = args.config or ("balanced" if "balanced" in configs else configs[0])

    print("Generating charts …")
    chart_auc_by_dataset(rows, out_dir, config=main_cfg)
    chart_complexity_distribution(rows, out_dir, config=main_cfg)
    chart_proxy_count(rows, out_dir)
    chart_coverage_precision(rows, out_dir, config=main_cfg)
    chart_auc_heatmap(rows, out_dir, config=main_cfg, grammar="ext")
    chart_auc_heatmap(rows, out_dir, config=main_cfg, grammar="arith")

    write_summary_csv(flat_table(rows), out_dir / "summary_stats.csv")
    print(f"\nDone. Figures in: {out_dir}")


if __name__ == "__main__":
    main()
