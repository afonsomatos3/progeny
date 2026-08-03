#!/usr/bin/env python3
"""
run_pipeline_sweep.py
=====================
Runs pipeline.py for all datasets across a set of fitness hyperparameter
combinations and organises the results under a timestamped folder.
Optionally runs mlp_attribution_from_run.py automatically afterwards.

  pipeline_runs/
    <YYYYMMDD_HHMMSS>/
      law_school/
        report-logistic_regression_p1_r1_c1.html
        arith-logistic_regression_p1_r1_c1.csv
        ext-logistic_regression_p1_r1_c1.csv
        ...
      adult/  compas/  oulad/
      _sweep_log.txt      ← full console output for every run
      _sweep_summary.md   ← quick-reference table of what was run
      attribution_<model>/attribution_report.html  ← if --attribution enabled

Usage
-----
  python run_pipeline_sweep.py                          # sweep only
  python run_pipeline_sweep.py --attribution            # sweep + attribution
  python run_pipeline_sweep.py --attribution --attribution-model logistic_regression
  python run_pipeline_sweep.py --attribution --attribution-fast
  python run_pipeline_sweep.py --datasets law_school adult
  python run_pipeline_sweep.py --configs baseline prec_heavy
  python run_pipeline_sweep.py --classifier random_forest
  python run_pipeline_sweep.py --time-budget 60         # faster, for testing
  python run_pipeline_sweep.py --dry-run                # print commands, don't run
"""

import argparse
import subprocess
import sys
import textwrap
from threading import Thread
from datetime import datetime
from pathlib import Path

# ── Fitness hyperparameter configurations ─────────────────────────────────────
# Each entry:  name (used in logging), prec_weight, rec_weight, cov_weight, penalty
#
#  baseline       — equal weights, standard penalty  (default pipeline settings)
#  prec_heavy     — precision dominates; finds tight, high-confidence proxies
#  rec_heavy      — recall dominates; finds broad proxies that capture most of group
#  cov_heavy      — coverage dominates; expressions that fire on large dataset slice
#  high_penalty   — aggressive complexity penalty → shorter, simpler expressions
#  low_penalty    — relaxed penalty → allows complex, deeply nested expressions
#  prec_rec       — both precision and recall boosted; balanced discriminatory power
#  balanced_sparse— mild boost to prec+rec with moderate sparsity push

CONFIGS = [
    # name              prec  rec   cov   penalty
    ("baseline",        1.0,  1.0,  1.0,  0.03),
    ("prec_heavy",      2.0,  0.5,  0.5,  0.03),
    ("rec_heavy",       0.5,  2.0,  0.5,  0.03),
    ("cov_heavy",       0.5,  0.5,  2.0,  0.03),
    ("no_cov",          2.0,  1.0,  0.0,  0.03),  # coverage disabled — prec+rec only
    ("high_penalty",    1.0,  1.0,  1.0,  0.10),
    ("low_penalty",     1.0,  1.0,  1.0,  0.01),
    ("prec_rec",        2.0,  2.0,  1.0,  0.03),
    ("balanced_sparse", 1.5,  1.5,  1.0,  0.05),
    ("rafael",           0.85,  0.15,  0.1,  0.05),  # Rafael's config from his paper
]

DATASETS = ["law_school", "adult", "compas", "oulad",
            "german_credit", "folktables"]

# Default split to use per dataset.
# law_school and adult each have a "processed" and a "non_processed" version;
# we always prefer processed.  compas and oulad use "encoded" (domain-informed
# preprocessing) rather than "raw".
DATASET_SPLITS = {
    "law_school":    ["processed"],
    "adult":         ["processed"],
    "compas":        ["encoded"],
    "oulad":         ["encoded"],
    "german_credit": ["processed"],
    "folktables":    ["processed"],
}

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline.py"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Format float for display: 1.0→1  1.5→1.5"""
    return str(int(v)) if v == int(v) else f"{v:.2g}"


def _run_tag(classifier: str, prec: float, rec: float, cov: float) -> str:
    """Reproduce the tag that pipeline.py embeds in its output filenames."""
    def wfmt(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return f"{classifier}_p{wfmt(prec)}_r{wfmt(rec)}_c{wfmt(cov)}"


def _is_completed(out_root: Path, dataset: str, tag: str,
                  continuous: bool) -> bool:
    """Return True if this (dataset, config) run already produced valid output.

    A run is considered complete when its arith CSV **and** ext CSV exist and
    are non-empty.  If --continuous was requested we also require the cont CSV.
    """
    ds_dir = out_root / dataset
    required = [
        ds_dir / f"arith-{tag}.csv",
        ds_dir / f"ext-{tag}.csv",
    ]
    if continuous:
        required.append(ds_dir / f"cont-{tag}.csv")
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def build_cmd(dataset: str, cfg: tuple, classifier: str,
              time_budget: int, out_dir: Path,
              jobs: int = 1, continuous: bool = False,
              max_depth: int = 6, population: int = 200,
              cont_max_depth: int = 5) -> list[str]:
    name, prec, rec, cov, penalty = cfg
    splits = DATASET_SPLITS.get(dataset, [])
    cmd = [
        sys.executable, str(PIPELINE),
        "--dataset",        dataset,
        "--classifier",     classifier,
        "--prec-weight",    str(prec),
        "--rec-weight",     str(rec),
        "--cov-weight",     str(cov),
        "--penalty",        str(penalty),
        "--time-budget",    str(time_budget),
        "--output-dir",     str(out_dir),
        "--jobs",           str(jobs),
        "--max-depth",      str(max_depth),
        "--population",     str(population),
        "--cont-max-depth", str(cont_max_depth),
    ]
    if splits:
        cmd += ["--splits"] + splits
    if continuous:
        cmd.append("--continuous")
    return cmd


def header(msg: str) -> str:
    bar = "─" * 60
    return f"\n{bar}\n{msg}\n{bar}"


def _stream_pipe(pipe, sink, prefix: str = "") -> None:
    """Stream subprocess output live while mirroring it into a log buffer."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            text = f"{prefix}{line}" if prefix else line
            print(text, end="")
            sink.append(text)
    finally:
        pipe.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep pipeline.py over datasets × fitness configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Available config names:
              baseline, prec_heavy, rec_heavy, cov_heavy,
              high_penalty, low_penalty, prec_rec, balanced_sparse
        """),
    )
    p.add_argument(
        "--datasets", nargs="+", default=DATASETS, metavar="DS",
        choices=DATASETS,
        help=f"Datasets to run (default: all). Choices: {DATASETS}",
    )
    p.add_argument(
        "--configs", nargs="+", default=None, metavar="CFG",
        help="Subset of config names to run (default: all)",
    )
    p.add_argument(
        "--classifier", default="logistic_regression",
        choices=["logistic_regression", "random_forest",
                 "gradient_boosting", "decision_tree", "svm"],
        help="Stage-1 classifier (default: logistic_regression)",
    )
    p.add_argument(
        "--time-budget", type=int, default=120,
        help="GP seconds per attribute per grammar stage (default: 120)",
    )
    p.add_argument(
        "--run-name", default=None,
        help="Name for the output folder (default: YYYYMMDD_HHMMSS)",
    )
    p.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="Resume an existing run by folder name or timestamp "
             "(e.g. --resume 20260329_131511).  The script reuses that "
             "folder, skips every (dataset, config) whose output CSVs "
             "already exist, and only reruns what is missing or failed.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    p.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel GP threads inside each pipeline.py call "
             "(default: 1). -1 = use all available cores.",
    )
    p.add_argument(
        "--continuous", action="store_true", default=False,
        help="Enable Stage 4 (continuous grammar, AUC fitness) in every run.",
    )
    p.add_argument(
        "--max-depth", type=int, default=6,
        help="Max GP tree depth for binary grammars (default: 6). Use 3 for readable AND expressions.",
    )
    p.add_argument(
        "--cont-max-depth", type=int, default=5,
        help="Max GP tree depth for continuous grammar (default: 5). "
             "Deeper to allow multi-feature arithmetic like A*B+C.",
    )
    p.add_argument(
        "--population", type=int, default=200,
        help="GP population size (default: 200). Larger = more diversity.",
    )

    # ── Attribution (mlp_attribution_from_run.py) ─────────────────────────────
    attr = p.add_argument_group(
        "attribution",
        "Run mlp_attribution_from_run.py automatically after the sweep finishes.",
    )
    attr.add_argument(
        "--attribution", action="store_true", default=False,
        help="Run attribution analysis on the sweep results when all GP runs finish.",
    )
    attr.add_argument(
        "--attribution-model",
        default="mlp",
        choices=["mlp", "logistic_regression", "random_forest", "xgboost"],
        help="Model to use for attribution (default: mlp).",
    )
    attr.add_argument(
        "--attribution-config", default=None, metavar="CFG",
        help="Filter attribution to one config label, e.g. 'p1_r1_c1' for baseline. "
             "Default: use all configs.",
    )
    attr.add_argument(
        "--attribution-fast", action="store_true", default=False,
        help="Pass --fast to attribution (fewer epochs/IG steps — much quicker).",
    )
    attr.add_argument(
        "--attribution-workers", type=int, default=1,
        help="Parallel workers for attribution (default: 1). 0 = all CPU cores.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Filter configs
    selected_configs = [c for c in CONFIGS
                        if args.configs is None or c[0] in args.configs]
    if not selected_configs:
        sys.exit(f"No matching configs for: {args.configs}")

    # ── Output directory: resume existing or create new ───────────────────────
    if args.resume:
        # Accept bare timestamp ("20260329_131511") or any trailing path component
        resume_id = Path(args.resume).name
        out_root  = ROOT / "pipeline_runs" / resume_id
        if not out_root.exists():
            sys.exit(f"Resume folder not found: {out_root}\n"
                     "Pass just the folder name, e.g. --resume 20260329_131511")
        run_name = resume_id
        print(header(f"Resuming run: {run_name}"))
    else:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = ROOT / "pipeline_runs" / run_name
        if not args.dry_run:
            out_root.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets) * len(selected_configs)
    print(header(
        f"Pipeline Sweep  —  {total} runs\n"
        f"Datasets : {', '.join(args.datasets)}\n"
        f"Configs  : {', '.join(c[0] for c in selected_configs)}\n"
        f"Classifier: {args.classifier}  |  Time budget: {args.time_budget}s\n"
        f"Output   : {out_root}"
        + (f"\nResume mode: skipping completed runs" if args.resume else "")
    ))

    if args.dry_run:
        print("\n[DRY RUN — commands that would be executed]\n")

    log_lines: list[str] = []
    results:   list[dict] = []

    # Pre-load any results already recorded in the existing log so the
    # summary markdown at the end includes both skipped and new runs.
    if args.resume:
        existing_log = out_root / "_sweep_log.txt"
        if existing_log.exists():
            log_lines.append(existing_log.read_text(encoding="utf-8"))

    run_idx = 0
    for dataset in args.datasets:
        for cfg in selected_configs:
            run_idx += 1
            name, prec, rec, cov, penalty = cfg
            tag   = _run_tag(args.classifier, prec, rec, cov)
            label = (f"[{run_idx:2d}/{total}]  {dataset:<12}  {name:<16}"
                     f"  prec={_fmt(prec)}  rec={_fmt(rec)}"
                     f"  cov={_fmt(cov)}  pen={penalty}")

            # ── Resume: skip if outputs already exist ─────────────────────────
            if args.resume and _is_completed(out_root, dataset, tag,
                                             args.continuous):
                print(f"\n{label}")
                print(f"  → SKIP  (outputs already present)")
                results.append({
                    "dataset": dataset, "config": name,
                    "prec": prec, "rec": rec, "cov": cov, "penalty": penalty,
                    "status": "SKIP", "elapsed_s": 0,
                })
                continue

            cmd = build_cmd(dataset, cfg, args.classifier,
                            args.time_budget, out_root,
                            jobs=args.jobs, continuous=args.continuous,
                            max_depth=args.max_depth, population=args.population,
                            cont_max_depth=args.cont_max_depth)

            print(f"\n{label}")

            if args.dry_run:
                print("  " + " ".join(cmd))
                continue

            log_lines.append(f"\n{'='*70}\n{label}\nCMD: {' '.join(cmd)}\n{'='*70}\n")

            t0 = datetime.now()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=str(ROOT),
                )
                stdout_chunks: list[str] = []
                stderr_chunks: list[str] = []
                stdout_thread = Thread(
                    target=_stream_pipe,
                    args=(proc.stdout, stdout_chunks),
                    daemon=True,
                )
                stderr_thread = Thread(
                    target=_stream_pipe,
                    args=(proc.stderr, stderr_chunks, "[STDERR] "),
                    daemon=True,
                )
                stdout_thread.start()
                stderr_thread.start()
                returncode = proc.wait()
                stdout_thread.join()
                stderr_thread.join()
                elapsed = (datetime.now() - t0).total_seconds()
                status = "OK" if returncode == 0 else f"ERR({returncode})"
                print(f"  → {status}  ({elapsed:.0f}s)")

                if stdout_chunks:
                    log_lines.append("".join(stdout_chunks))
                if stderr_chunks:
                    log_lines.append("".join(stderr_chunks))

                results.append({
                    "dataset": dataset, "config": name,
                    "prec": prec, "rec": rec, "cov": cov, "penalty": penalty,
                    "status": status, "elapsed_s": int(elapsed),
                })

            except Exception as exc:
                elapsed = (datetime.now() - t0).total_seconds()
                print(f"  → EXCEPTION: {exc}")
                log_lines.append(f"EXCEPTION: {exc}\n")
                results.append({
                    "dataset": dataset, "config": name,
                    "prec": prec, "rec": rec, "cov": cov, "penalty": penalty,
                    "status": "EXCEPTION", "elapsed_s": int(elapsed),
                })

    if args.dry_run:
        return

    # ── Write log ─────────────────────────────────────────────────────────────
    log_path = out_root / "_sweep_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # ── Write summary markdown ────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Pipeline Sweep — {run_name}",
        f"",
        f"Generated: {now_str}  |  Classifier: `{args.classifier}`  "
        f"|  Time budget: `{args.time_budget}s`",
        f"",
        f"## Fitness Configurations",
        f"",
        f"| Config | prec_w | rec_w | cov_w | penalty | Description |",
        f"|--------|--------|-------|-------|---------|-------------|",
    ]
    descriptions = {
        "baseline":        "Equal weights, standard penalty",
        "prec_heavy":      "Precision dominates — tight, high-confidence proxies",
        "rec_heavy":       "Recall dominates — broad proxies covering most of group",
        "cov_heavy":       "Coverage dominates — expressions firing on large slice",
        "high_penalty":    "Aggressive complexity penalty → short, simple expressions",
        "low_penalty":     "Relaxed penalty → deep, complex expressions allowed",
        "prec_rec":        "Precision + recall both boosted — balanced discriminatory power",
        "balanced_sparse": "Mild prec+rec boost with moderate sparsity push",
    }
    for n, prec, rec, cov, penalty in CONFIGS:
        lines.append(f"| `{n}` | {prec} | {rec} | {cov} | {penalty} "
                     f"| {descriptions.get(n, '')} |")

    lines += [
        f"",
        f"## Run Results",
        f"",
        f"| # | Dataset | Config | Status | Time (s) |",
        f"|---|---------|--------|--------|----------|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r['dataset']} | {r['config']} "
            f"| {r['status']} | {r['elapsed_s']} |"
        )

    ok      = sum(1 for r in results if r["status"] == "OK")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    ran     = len(results) - skipped
    lines += [
        f"",
        f"**{ok}/{ran} runs completed successfully.**"
        + (f"  ({skipped} skipped — already present)" if skipped else ""),
        f"",
        f"## Output Files",
        f"",
        f"Each run produces 5 files in `<dataset>/`:",
        f"",
        f"```",
        f"report-<classifier>_p<prec>_r<rec>_c<cov>.html    ← full 3-stage report",
        f"arith-<classifier>_p<prec>_r<rec>_c<cov>.csv      ← Stage 2 GP results",
        f"arith-<classifier>_p<prec>_r<rec>_c<cov>-partials.csv",
        f"ext-<classifier>_p<prec>_r<rec>_c<cov>.csv        ← Stage 3 GP results",
        f"ext-<classifier>_p<prec>_r<rec>_c<cov>-partials.csv",
        f"```",
        f"",
        f"Full console output: `_sweep_log.txt`",
    ]

    summary_path = out_root / "_sweep_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # ── Final status ──────────────────────────────────────────────────────────
    skip_note = f"  ({skipped} skipped)" if skipped else ""
    print(header(f"Sweep complete  —  {ok}/{ran} runs OK{skip_note}"))
    print(f"  Results : {out_root}/")
    print(f"  Summary : {summary_path.name}")
    print(f"  Log     : {log_path.name}")

    # ── Attribution phase (optional) ──────────────────────────────────────────
    if args.attribution:
        attr_script = ROOT / "mlp_attribution_from_run.py"
        if not attr_script.exists():
            print(f"\n[attribution] mlp_attribution_from_run.py not found — skipping.")
        else:
            attr_cmd = [
                sys.executable, str(attr_script),
                "--run-dir", str(out_root),
                "--model",   args.attribution_model,
                "--workers", str(args.attribution_workers),
            ]
            if args.attribution_config:
                attr_cmd += ["--config", args.attribution_config]
            if args.attribution_fast:
                attr_cmd.append("--fast")

            print(header(
                f"Attribution  —  model={args.attribution_model}"
                + (f"  config={args.attribution_config}" if args.attribution_config else "  (all configs)")
                + (f"  [fast]" if args.attribution_fast else "")
            ))
            print("  CMD: " + " ".join(attr_cmd))
            try:
                t0_attr = datetime.now()
                proc = subprocess.Popen(
                    attr_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=str(ROOT),
                )
                attr_log: list[str] = []
                t_out = Thread(target=_stream_pipe, args=(proc.stdout, attr_log), daemon=True)
                t_err = Thread(target=_stream_pipe, args=(proc.stderr, attr_log, "[STDERR] "), daemon=True)
                t_out.start(); t_err.start()
                rc = proc.wait()
                t_out.join();  t_err.join()
                elapsed_attr = (datetime.now() - t0_attr).total_seconds()
                status_attr  = "OK" if rc == 0 else f"ERR({rc})"
                print(f"\n  → Attribution {status_attr}  ({elapsed_attr:.0f}s)")
                log_path.open("a", encoding="utf-8").write(
                    f"\n{'='*70}\nATTRIBUTION ({args.attribution_model})\n{'='*70}\n"
                    + "".join(attr_log)
                )
            except Exception as exc:
                print(f"  → Attribution EXCEPTION: {exc}")


if __name__ == "__main__":
    main()
