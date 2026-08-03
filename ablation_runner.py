"""
ablation_runner.py — grammar ablation study.

Runs the pipeline multiple times, each time disabling one grammar rule group,
then saves results to pipeline_results/ablation/.

Each ablation run uses a short time budget (default 60 s) for speed.
Run the full analysis afterwards with analysis_ablation.py.

Usage:
    python ablation_runner.py                          # all datasets, 60s budget
    python ablation_runner.py --dataset law_school     # single dataset
    python ablation_runner.py --budget 120             # longer runs
    python ablation_runner.py --configs baseline no_if # specific configs only
"""

import argparse
import subprocess
import sys
import pathlib
import time

# ── Ablation configurations ───────────────────────────────────────────────────
#
# Each entry: (label, --disable-nodes value, description)
# label is used as the subfolder name.
# disable-nodes is a comma-separated list of grammar node class names.
# An empty string means the full grammar (baseline).

ABLATIONS = [
    ("baseline",     "",                                  "Full grammar (all rules)"),
    ("no_if",        "IfThenElse,TTypedIf",               "No conditional (IF)"),
    ("no_and",       "BoolAnd,CondAnd,TCondAnd",          "No conjunction (AND)"),
    ("no_minmax",    "Max2,Min2,NumMax,NumMin,TNumMax,TNumMin", "No min/max"),
    ("no_abs",       "Abs,NumAbs,TNumAbs",                "No absolute value"),
    ("no_nonlinear", "Abs,Max2,Min2,NumAbs,NumMax,NumMin,TNumAbs,TNumMax,TNumMin",
                                                          "No nonlinear ops (abs, min, max)"),
]

DATASETS = ["law_school", "adult", "compas", "oulad", "german_credit", "folktables"]

SPLITS = {
    "law_school":    "processed",
    "adult":         "processed",
    "compas":        "raw",
    "oulad":         "raw",
    "german_credit": None,
    "folktables":    None,
}


def run_ablation(label: str, disable_nodes: str, dataset: str,
                 budget: int, out_root: pathlib.Path, seed: int = 42) -> bool:
    split = SPLITS.get(dataset)
    out_dir = out_root / label / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "pipeline.py",
        "--dataset",     dataset,
        "--time-budget", str(budget),
        "--population",  "100",
        "--seed",        str(seed),
        "--jobs",        "-1",
        "--output-dir",  str(out_dir),
        "--prec-weight", "1.0",
        "--rec-weight",  "1.0",
        "--cov-weight",  "1.0",
    ]
    if split:
        cmd += ["--splits", split]
    if disable_nodes:
        cmd += ["--disable-nodes", disable_nodes]

    print(f"  [{dataset}] ablation={label!r}  budget={budget}s", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=pathlib.Path(__file__).parent,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"    FAILED ({elapsed:.0f}s) — stderr tail:")
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
            return False
        print(f"    OK ({elapsed:.0f}s)")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Grammar ablation study runner")
    ap.add_argument("--dataset",  nargs="*", default=None,
                    help="Datasets to run (default: all six)")
    ap.add_argument("--budget",   type=int,  default=60,
                    help="Time budget per attribute per grammar (default: 60s)")
    ap.add_argument("--configs",  nargs="*", default=None,
                    help="Ablation labels to run (default: all)")
    ap.add_argument("--out",      default="pipeline_results/ablation",
                    help="Output root (default: pipeline_results/ablation)")
    ap.add_argument("--seed",     type=int,  default=42)
    args = ap.parse_args()

    datasets = args.dataset or DATASETS
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    ablations = ABLATIONS
    if args.configs:
        ablations = [(l, d, desc) for l, d, desc in ABLATIONS if l in args.configs]
        if not ablations:
            sys.exit(f"No matching configs in: {[l for l,_,_ in ABLATIONS]}")

    n_total  = len(ablations) * len(datasets)
    n_done   = 0
    n_failed = 0

    print(f"Ablation study: {len(ablations)} configs × {len(datasets)} datasets"
          f" = {n_total} runs  (budget={args.budget}s each)")
    print(f"Output: {out_root}\n")

    for label, disable_nodes, desc in ablations:
        print(f"── {label}: {desc} ──")
        if disable_nodes:
            print(f"   Disabled: {disable_nodes}")
        for ds in datasets:
            ok = run_ablation(label, disable_nodes, ds,
                              args.budget, out_root, seed=args.seed)
            n_done  += 1
            n_failed += 0 if ok else 1
        print()

    print(f"Done. {n_done - n_failed}/{n_done} runs succeeded.")
    print(f"Run  python analysis_ablation.py {out_root}  to generate charts.")


if __name__ == "__main__":
    main()
