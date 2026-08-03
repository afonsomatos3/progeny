"""
analysis_evolution.py — complexity vs. discovery-time charts.

Reads partials CSVs (which carry first_seen_s and nodes per proxy)
and plots how expression complexity evolves over the GP time budget.

Usage:
    python analysis_evolution.py pipeline_results/and_balance
    python analysis_evolution.py pipeline_results/and_balance --config balanced --grammar ext
    python analysis_evolution.py pipeline_results/and_balance --dataset law_school adult
"""

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
})

GRAMMAR_COLOUR = {"arith": "#4C72B0", "ext": "#DD8452"}


# ── loading ───────────────────────────────────────────────────────────────────

def load_partials(run_dir: pathlib.Path,
                  config_filter: str | None = None,
                  grammar_filter: str | None = None,
                  dataset_filter: list[str] | None = None) -> list[dict]:
    rows = []
    for path in sorted(run_dir.rglob("*-partials.csv")):
        if ".bak" in path.suffixes or ".orig" in path.suffixes:
            continue
        stem    = path.stem.replace("-partials", "")
        grammar = "arith" if stem.startswith("arith") else ("ext" if stem.startswith("ext") else None)
        if grammar is None:
            continue
        if grammar_filter and grammar != grammar_filter:
            continue
        try:
            rel        = path.relative_to(run_dir)
            config     = rel.parts[0] if len(rel.parts) > 1 else "default"
            dataset_dir = rel.parts[1] if len(rel.parts) >= 3 else rel.parts[0]
        except ValueError:
            config, dataset_dir = "default", "unknown"
        if config_filter and config != config_filter:
            continue

        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ds = dataset_dir   # use path-based name, not the split column
                if dataset_filter and ds not in dataset_filter:
                    continue
                try:
                    row["_nodes"]       = float(row.get("nodes", "nan"))
                    row["_first_seen"]  = float(row.get("first_seen_s", "nan"))
                    row["_auc"]         = float(row.get("auc", "nan"))
                except ValueError:
                    continue
                row["_grammar"]     = grammar
                row["_config"]      = config
                row["_dataset_dir"] = dataset_dir
                rows.append(row)
    return rows


# ── charts ────────────────────────────────────────────────────────────────────

def chart_evolution_scatter(rows: list[dict], out_dir: pathlib.Path,
                             config: str = "balanced", grammar: str = "ext"):
    """
    One panel per dataset. X = discovery time (s), Y = node count, colour = AUC.
    Each dot is one partial proxy.
    """
    subset = [r for r in rows if r["_config"] == config and r["_grammar"] == grammar]
    if not subset:
        print(f"  [skip] No data for config={config}, grammar={grammar}")
        return

    datasets = sorted({r.get("_dataset_dir", r.get("dataset","?")) for r in subset})
    n        = len(datasets)
    ncols    = min(3, n)
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             squeeze=False)

    vmin, vmax = 0.5, 1.0

    for idx, ds in enumerate(datasets):
        ax  = axes[idx // ncols][idx % ncols]
        sub = [r for r in subset if r.get("_dataset_dir", r.get("dataset","?")) == ds]
        t   = [r["_first_seen"] for r in sub]
        n_  = [r["_nodes"]      for r in sub]
        auc = [r["_auc"]        for r in sub]

        sc = ax.scatter(t, n_, c=auc, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                        alpha=0.7, s=20, edgecolors="none")
        ax.set_title(ds, fontsize=10, fontweight="bold")
        ax.set_xlabel("Discovery time (s)", fontsize=8)
        ax.set_ylabel("Nodes", fontsize=8)

        # trend line
        if len(t) >= 3:
            coef = np.polyfit(t, n_, 1)
            xr   = np.linspace(min(t), max(t), 50)
            ax.plot(xr, np.polyval(coef, xr), "--", lw=1, color="grey", alpha=0.6)

    # hide unused panels
    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("AUC")
    fig.suptitle(f"Expression complexity vs. discovery time — {config} / {grammar}",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    fname = f"evolution_{config}_{grammar}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {fname}")


def chart_auc_over_time(rows: list[dict], out_dir: pathlib.Path,
                        config: str = "balanced"):
    """
    Cumulative best-AUC-over-time curves, one per grammar, overlaid per dataset.
    """
    subset = [r for r in rows if r["_config"] == config]
    if not subset:
        return

    datasets = sorted({r.get("_dataset_dir", r.get("dataset","?")) for r in subset})
    grammars = ["arith", "ext"]
    n        = len(datasets)
    ncols    = min(3, n)
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.0, nrows * 3.0),
                             squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        for grm, col in zip(grammars, [GRAMMAR_COLOUR["arith"], GRAMMAR_COLOUR["ext"]]):
            sub = sorted(
                [r for r in subset if r.get("_dataset_dir", r.get("dataset","?")) == ds and r["_grammar"] == grm],
                key=lambda r: r["_first_seen"],
            )
            if not sub:
                continue
            times = [r["_first_seen"] for r in sub]
            aucs  = [r["_auc"]        for r in sub]
            best  = [max(aucs[:i+1]) for i in range(len(aucs))]
            ax.step(times, best, where="post", color=col, label=grm, lw=1.5)

        ax.set_title(ds, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Best AUC so far", fontsize=8)
        ax.set_ylim(0.5, 1.0)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Best proxy AUC over time — {config}", fontsize=11, y=1.01)
    fig.tight_layout()
    fname = f"auc_over_time_{config}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {fname}")


def chart_complexity_vs_auc(rows: list[dict], out_dir: pathlib.Path,
                             config: str = "balanced"):
    """
    Scatter: node count (X) vs AUC (Y), grouped by grammar and dataset.
    Shows whether more complex expressions achieve higher AUC.
    """
    subset = [r for r in rows if r["_config"] == config]
    if not subset:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, grm in zip(axes, ["arith", "ext"]):
        sub = [r for r in subset if r["_grammar"] == grm]
        nodes = [r["_nodes"] for r in sub]
        aucs  = [r["_auc"]   for r in sub]
        datasets = [r.get("_dataset_dir", r.get("dataset","?")) for r in sub]
        ds_list  = sorted(set(datasets))
        cmap     = cm.get_cmap("tab10", len(ds_list))
        ds_idx   = {ds: i for i, ds in enumerate(ds_list)}

        for ds in ds_list:
            nn = [n for n, d in zip(nodes, datasets) if d == ds]
            aa = [a for a, d in zip(aucs,  datasets) if d == ds]
            ax.scatter(nn, aa, label=ds, alpha=0.5, s=15, color=cmap(ds_idx[ds]))

        # quadratic trend
        valid = [(n, a) for n, a in zip(nodes, aucs)
                 if not (np.isnan(n) or np.isnan(a))]
        if len(valid) >= 3:
            nn_, aa_ = zip(*valid)
            coef = np.polyfit(nn_, aa_, 2)
            xr   = np.linspace(min(nn_), max(nn_), 100)
            ax.plot(xr, np.polyval(coef, xr), "--", color="black", lw=1, alpha=0.5)

        ax.set_title(f"{grm} grammar")
        ax.set_xlabel("Expression nodes")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle(f"Expression complexity vs. AUC — {config}", fontsize=11)
    fig.tight_layout()
    fname = f"complexity_vs_auc_{config}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {fname}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Complexity vs. discovery-time charts")
    ap.add_argument("run_dir")
    ap.add_argument("--out",     default=None)
    ap.add_argument("--config",  default=None)
    ap.add_argument("--grammar", default=None, choices=["arith","ext"])
    ap.add_argument("--dataset", nargs="*", default=None)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"Not found: {run_dir}")

    out_dir = pathlib.Path(args.out) if args.out else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading partials from {run_dir} …")
    rows = load_partials(run_dir,
                         config_filter=args.config,
                         grammar_filter=args.grammar,
                         dataset_filter=args.dataset)
    if not rows:
        sys.exit("No partial-proxy rows found.")
    print(f"  {len(rows)} partial proxies")

    configs = sorted({r["_config"] for r in rows})
    main_cfg = args.config or ("balanced" if "balanced" in configs else configs[0])

    print("Generating charts …")
    chart_evolution_scatter(rows, out_dir, config=main_cfg, grammar="ext")
    chart_evolution_scatter(rows, out_dir, config=main_cfg, grammar="arith")
    chart_auc_over_time(rows, out_dir, config=main_cfg)
    chart_complexity_vs_auc(rows, out_dir, config=main_cfg)

    print(f"\nDone. Figures in: {out_dir}")


if __name__ == "__main__":
    main()
