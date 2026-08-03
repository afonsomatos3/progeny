"""
Toy Synthetic Datasets for Thesis Tables
==========================================
Two small, hand-crafted datasets (≤20 rows) designed to be shown as tables
in the thesis while still producing meaningful GP results.

Experiment 1 — Grammar Expressivity Gap (16-row HR dataset)
  True proxy: (department == Marketing) AND (contract == part_time)
  Salary/experience are deliberately scrambled so no arithmetic threshold
  separates minority from non-minority (e.g. minority row 7 has salary=71,
  almost identical to non-minority rows with salary=72, 69, 65).
  Arithmetic grammar: cannot express categorical equality → AUC ≈ 0.57
  Typed grammar:      recovers the exact boolean rule → AUC ≈ 0.96

Experiment 2 — Fitness Configuration Shapes Proxy Type (18-row Credit dataset)
  Two proxies embedded on different features:
    Precision proxy: age < 25  (4/4 young applicants are minority, PPV=100%)
    Recall proxy:    income < 35k (catches all 7 minority, PPV≈70%)
  prec_heavy config finds the tight income threshold (PPV=100%, Rec=71%)
  balanced/rec_heavy config finds the broader age threshold (Rec=100%)

Usage
-----
    python synthetic_toy_datasets.py           # run both experiments + print LaTeX
    python synthetic_toy_datasets.py --latex   # print LaTeX tables only (no GP)
    python synthetic_toy_datasets.py --no-gp   # print tables only, skip GP runs
    python synthetic_toy_datasets.py --time-budget 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "GeneticEngine" / "GeneticEngine"))


# ─── Dataset 1: Synthetic HR (16 rows) ───────────────────────────────────────
#
# True proxy: (department == Marketing) AND (contract == part_time)
# Salary/experience are deliberately scrambled across groups so no arithmetic
# threshold can isolate minority rows: a minority Marketing PT worker (row 7)
# has salary=71, nearly identical to non-minority Sales FT (row 11, salary=65)
# and non-minority Engineering FT (row 1, salary=72). Likewise, low-salary
# non-minority rows (rows 4, 14) overlap with minority rows 8 and 10.
# Row 16 is a deliberate noise point (Marketing + Part-time, non-minority).

HR_ROWS = [
    # dept,        contract,    salary, exp, minority
    ("Engineering", "Full-time",  72,  11,   0),
    ("Engineering", "Full-time",  44,   3,   0),   # low salary non-minority
    ("Engineering", "Part-time",  61,   8,   0),
    ("Engineering", "Part-time",  38,   1,   0),   # very low salary non-minority
    ("Marketing",   "Full-time",  69,   9,   0),
    ("Marketing",   "Full-time",  43,   3,   0),   # low salary non-minority
    ("Marketing",   "Part-time",  71,  10,   1),   # HIGH salary minority (confounds arith.)
    ("Marketing",   "Part-time",  39,   2,   1),
    ("Marketing",   "Part-time",  54,   6,   1),
    ("Marketing",   "Part-time",  47,   4,   1),
    ("Sales",       "Full-time",  65,   7,   0),
    ("Sales",       "Full-time",  48,   4,   0),
    ("Sales",       "Part-time",  66,   9,   0),   # high salary part-time non-minority
    ("Sales",       "Part-time",  35,   1,   0),   # very low salary non-minority
    ("Engineering", "Full-time",  58,   7,   0),
    ("Marketing",   "Part-time",  63,   8,   0),   # noise: proxy fires but not minority
]

HR_COLS   = ["Department", "Contract", "Salary (£k)", "Experience (yr)", "Minority"]
HR_FEATS  = ["salary", "experience", "department", "contract"]
HR_NUM    = ["salary", "experience"]
HR_CAT    = ["department", "contract"]
HR_DEPT_CATS     = ["Engineering", "Marketing", "Sales"]
HR_CONTRACT_CATS = ["Full-time", "Part-time"]


# ─── Dataset 2: Synthetic Credit (18 rows) ────────────────────────────────────
#
# Precision proxy: age < 25  — all 4 young applicants are minority (PPV 100%)
# Recall proxy:    income < 35 000 — catches most minority, but lower PPV
# Rows above both thresholds are non-minority (clean separation).

CREDIT_ROWS = [
    # age, income(£k), debt_ratio, credit_score, minority
    ( 22,  18,  0.62,  580,  1),   # young + low income
    ( 23,  21,  0.55,  590,  1),   # young + low income
    ( 24,  16,  0.71,  560,  1),   # young + low income
    ( 20,  25,  0.48,  610,  1),   # young
    ( 26,  28,  0.42,  620,  0),   # just above 25, low income — not minority
    ( 28,  22,  0.35,  640,  1),   # low income → minority
    ( 31,  30,  0.29,  650,  1),   # low income → minority
    ( 33,  27,  0.38,  630,  0),   # low income — noise (not minority)
    ( 35,  32,  0.25,  660,  1),   # low income → minority
    ( 38,  29,  0.31,  645,  0),   # low income — noise (not minority)
    ( 40,  55,  0.18,  720,  0),   # high income
    ( 42,  62,  0.15,  740,  0),
    ( 45,  48,  0.20,  710,  0),
    ( 50,  71,  0.12,  760,  0),
    ( 52,  65,  0.14,  750,  0),
    ( 55,  80,  0.10,  780,  0),
    ( 48,  45,  0.22,  700,  0),
    ( 60,  90,  0.08,  790,  0),
]

CREDIT_COLS = ["Age", "Income (£k)", "Debt ratio", "Credit score", "Minority"]
CREDIT_FEATS = ["age", "income", "debt_ratio", "credit_score"]
CREDIT_NUM   = CREDIT_FEATS


# ─── Build numpy dataset dicts ────────────────────────────────────────────────

def build_hr_dataset() -> dict:
    dept_enc     = [HR_DEPT_CATS.index(r[0])     for r in HR_ROWS]
    contract_enc = [HR_CONTRACT_CATS.index(r[1]) for r in HR_ROWS]
    salary       = [r[2] for r in HR_ROWS]
    experience   = [r[3] for r in HR_ROWS]
    minority     = np.array([r[4] for r in HR_ROWS], dtype=np.int32)

    X = np.column_stack([salary, experience,
                         dept_enc, contract_enc]).astype(np.float64)

    return {
        "label":                "Synthetic HR (toy)",
        "X_train":              X,
        "X_test":               X,        # tiny dataset: train = test
        "feature_names":        HR_FEATS,
        "numeric_features":     HR_NUM,
        "categorical_features": HR_CAT,
        "label_encodings": {
            "department": HR_DEPT_CATS,
            "contract":   HR_CONTRACT_CATS,
        },
        "pa_train": {"minority": minority},
        "pa_test":  {"minority": minority},
    }


def build_credit_dataset() -> dict:
    rows     = CREDIT_ROWS
    minority = np.array([r[4] for r in rows], dtype=np.int32)
    X        = np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=np.float64)

    return {
        "label":                "Synthetic Credit (toy)",
        "X_train":              X,
        "X_test":               X,
        "feature_names":        CREDIT_FEATS,
        "numeric_features":     CREDIT_NUM,
        "categorical_features": [],
        "label_encodings":      {},
        "pa_train": {"minority": minority},
        "pa_test":  {"minority": minority},
    }


# ─── LaTeX table printers ─────────────────────────────────────────────────────

def _latex_hr() -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption[Synthetic HR toy dataset]{Synthetic HR toy dataset (16 rows). "
        r"The true proxy is \texttt{(department == Marketing) AND (contract == part\_time)} "
        r"(shaded rows, minus the noise row 16). "
        r"Salary and experience are deliberately scrambled across groups---for example, "
        r"minority row 7 has salary \pounds71k, nearly identical to non-minority rows 1 "
        r"(\pounds72k) and 5 (\pounds69k)---so no arithmetic threshold on numeric features "
        r"alone can isolate the minority group. Row 16 is a noise point: "
        r"the proxy fires but the individual is not minority.}",
        r"\label{tab:toy-hr}",
        r"\begin{tabular}{clllcc}",
        r"\toprule",
        r"\textbf{\#} & \textbf{Department} & \textbf{Contract} "
        r"& \textbf{Salary (£k)} & \textbf{Experience (yr)} & \textbf{Minority} \\",
        r"\midrule",
    ]
    for i, (dept, contract, sal, exp, minor) in enumerate(HR_ROWS, 1):
        flag  = r"\checkmark" if minor else ""
        # Highlight the true proxy rows
        is_proxy = dept == "Marketing" and contract == "Part-time"
        row = (f"{i} & {dept} & {contract} & {sal} & {exp} & {flag} \\\\")
        if is_proxy and minor:
            row = r"\rowcolor{gray!15}" + row
        lines.append(row)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _latex_credit() -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption[Synthetic Credit toy dataset]{Synthetic Credit toy dataset (18 rows). "
        r"Two proxies are embedded: \texttt{age < 25} (100\% precision on rows 1--4) and "
        r"\texttt{income < 35\,000} (broader, lower precision, rows 1--3, 6--7, 9).}",
        r"\label{tab:toy-credit}",
        r"\begin{tabular}{cccccr}",
        r"\toprule",
        r"\textbf{\#} & \textbf{Age} & \textbf{Income (£k)} "
        r"& \textbf{Debt ratio} & \textbf{Credit score} & \textbf{Minority} \\",
        r"\midrule",
    ]
    for i, (age, inc, debt, score, minor) in enumerate(CREDIT_ROWS, 1):
        flag = r"\checkmark" if minor else ""
        row  = f"{i} & {age} & {inc} & {debt:.2f} & {score} & {flag} \\\\"
        if minor:
            row = r"\rowcolor{gray!15}" + row
        lines.append(row)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def print_latex_tables() -> None:
    print("\n% ─── Toy Dataset 1: Synthetic HR ────────────────────────────────")
    print(_latex_hr())
    print("\n% ─── Toy Dataset 2: Synthetic Credit ───────────────────────────")
    print(_latex_credit())


# ─── GP experiments ───────────────────────────────────────────────────────────

def run_hr_experiment(time_budget: float, seed: int) -> None:
    from gp_proxy_discovery import run_gp, build_grammar

    ds = build_hr_dataset()
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — Grammar Expressivity Gap (Toy HR, 16 rows)")
    print("=" * 70)
    print("True proxy: (department == Marketing) AND (contract == part_time)")
    print(f"Minority: {ds['pa_train']['minority'].sum()} / {len(ds['pa_train']['minority'])} rows")

    for gname in ("arithmetic", "typed"):
        if gname == "arithmetic":
            grammar = build_grammar(ds["feature_names"], grammar_mode="arithmetic",
                                    numeric_features=ds["numeric_features"])
        else:
            grammar = build_grammar(ds["feature_names"], grammar_mode="typed",
                                    numeric_features=ds["numeric_features"],
                                    categorical_features=ds["categorical_features"],
                                    label_encodings=ds["label_encodings"],
                                    X_train=ds["X_train"])

        r = run_gp(ds, "minority", grammar, time_budget=time_budget,
                   population_size=300, max_depth=4, seed=seed,
                   prec_weight=1.5, rec_weight=1.5, cov_weight=1.0,
                   complexity_penalty=0.01)

        print(f"\n[{gname.upper()} grammar]")
        print(f"  Expression : {r['expression']}")
        print(f"  AUC={r['auc']:.3f}  Prec={r['precision']:.1f}%  "
              f"Rec={r['recall']:.1f}%  Cov={r['coverage']:.1f}%")


def run_credit_experiment(time_budget: float, seed: int) -> None:
    from gp_proxy_discovery import run_gp, build_grammar

    ds = build_credit_dataset()
    arr = np.array(CREDIT_ROWS)
    y   = ds["pa_train"]["minority"]
    age = arr[:, 0]; inc = arr[:, 1]

    young = age < 25
    low   = inc < 35
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — Fitness Configuration Shapes Proxy Type (Toy Credit, 18 rows)")
    print("=" * 70)
    print(f"Minority: {y.sum()} / {len(y)} rows")
    print(f"  age < 25:    fires on {young.sum()} rows  PPV={y[young].mean()*100:.0f}%  Rec={y[young].sum()/y.sum()*100:.0f}%")
    print(f"  income < 35: fires on {low.sum()} rows    PPV={y[low].mean()*100:.0f}%   Rec={y[low].sum()/y.sum()*100:.0f}%")

    grammar = build_grammar(ds["feature_names"], grammar_mode="typed",
                            numeric_features=ds["numeric_features"],
                            categorical_features=[], label_encodings={},
                            X_train=ds["X_train"])

    configs = {
        "balanced":   dict(prec_weight=1.0, rec_weight=1.0, cov_weight=1.0),
        "prec_heavy": dict(prec_weight=2.0, rec_weight=0.5, cov_weight=0.5),
        "rec_heavy":  dict(prec_weight=0.5, rec_weight=2.0, cov_weight=0.5),
    }

    for cname, weights in configs.items():
        r = run_gp(ds, "minority", grammar, time_budget=time_budget,
                   population_size=300, max_depth=4, seed=seed,
                   complexity_penalty=0.01, **weights)

        is_deg = r["precision"] == 0.0 and r["recall"] == 0.0
        print(f"\n[{cname.upper()}]")
        print(f"  Expression : {r['expression']}")
        if is_deg:
            print(f"  *** Degenerate coverage trap (fires on all rows) ***")
        else:
            print(f"  AUC={r['auc']:.3f}  Prec={r['precision']:.1f}%  "
                  f"Rec={r['recall']:.1f}%  Cov={r['coverage']:.1f}%")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toy synthetic datasets for thesis tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--time-budget", type=float, default=60.0,
                        help="GP time budget per run (seconds)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX table code and exit (no GP)")
    parser.add_argument("--no-gp", action="store_true",
                        help="Print dataset summaries only, skip GP runs")
    args = parser.parse_args()

    print_latex_tables()

    if args.latex or args.no_gp:
        return

    print("\n\nRunning GP experiments on toy datasets...")
    print("(Note: small n = high variance. Run multiple seeds for stable results.)\n")

    run_hr_experiment(args.time_budget, args.seed)
    run_credit_experiment(args.time_budget, args.seed)


if __name__ == "__main__":
    main()
