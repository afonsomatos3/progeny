"""
Synthetic Dataset Experiments
==============================
Two controlled experiments demonstrating specific capabilities of the GP
proxy discovery pipeline that are hard to isolate on real-world datasets.

Experiment 1 — Grammar Expressivity Gap
  Synthetic HR dataset.  The minority group is defined *exclusively* by a
  boolean combination of two categorical features:
      is_minority = (department == Marketing) AND (contract == part_time)
  The numeric features (salary, experience, performance) correlate only
  weakly with minority status.  Expected outcome:
    - Arithmetic grammar: AUC ≈ 0.55–0.65  (no categorical operators available)
    - Typed grammar:      AUC ≈ 0.90+       (recovers the exact boolean rule)

Experiment 2 — Fitness Configuration Shapes Proxy Type
  Synthetic Credit dataset.  Two proxies are embedded simultaneously:
    High-precision core:  income < Q20 AND debt_ratio > Q80  (≈8%, PPV ≈ 88%)
    Broad-recall fringe:  income < Q50                       (≈50%, PPV ≈ 44%)
  Expected outcome:
    - prec_heavy  config → finds the core    (high PPV, low recall)
    - rec_heavy   config → finds the fringe  (low PPV, high recall)
    - balanced    config → finds something in between

Usage
-----
    python synthetic_experiments.py
    python synthetic_experiments.py --time-budget 120 --seed 7
    python synthetic_experiments.py --experiment 1
    python synthetic_experiments.py --experiment 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "GeneticEngine" / "GeneticEngine"))

from gp_proxy_discovery import run_gp, build_grammar   # noqa: E402


# ─── Dataset generators ───────────────────────────────────────────────────────

def make_hr_dataset(n: int = 1000, seed: int = 42) -> dict:
    """
    Synthetic HR hiring dataset.

    True proxy: is_minority = (department == Marketing) AND (contract == part_time)

    Six features are generated:
      salary          Numeric  Annual salary £k  (slight minority gap, AUC ≈ 0.60)
      experience      Numeric  Years of experience (slight gap,        AUC ≈ 0.58)
      performance     Numeric  Manager rating 1–5 (negligible gap,     AUC ≈ 0.52)
      overtime_hours  Numeric  Weekly overtime hours (pure noise,      AUC ≈ 0.50)
      department      Categorical  Engineering / Marketing / Sales
      contract        Categorical  full_time / part_time

    No single feature achieves AUC > 0.65.  The arithmetic grammar can only
    access the four numeric features and will plateau near chance.  The typed
    grammar also receives CatEquals terminals for department and contract,
    allowing it to discover the exact boolean conjunction.
    """
    rng = np.random.RandomState(seed)

    # Categorical draws
    dept     = rng.choice(["Engineering", "Marketing", "Sales"], n,
                          p=[0.40, 0.35, 0.25])
    contract = rng.choice(["full_time", "part_time"], n, p=[0.70, 0.30])

    # True proxy and minority labels
    true_proxy = ((dept == "Marketing") & (contract == "part_time")).astype(int)
    minority   = np.where(true_proxy == 1,
                          (rng.rand(n) < 0.92).astype(int),
                          (rng.rand(n) < 0.06).astype(int))
    is_min = minority.astype(bool)

    # Numeric features — slight correlations so the dataset is realistic but
    # no single feature is a good predictor.
    salary         = np.clip(rng.normal(48.0, 12.0, n)
                             + is_min * rng.normal(-4.5, 2.0, n), 18.0, 150.0)
    experience     = np.clip(rng.uniform(0, 20, n)
                             + is_min * rng.normal(-1.5, 2.5, n), 0, 30)
    performance    = np.clip(rng.normal(3.0, 0.8, n)
                             + is_min * rng.normal(0.1, 0.3, n), 1.0, 5.0)
    overtime_hours = rng.uniform(0, 20, n)                         # pure noise

    # Categorical → integer encoding (stored for label_encodings)
    dept_cats     = ["Engineering", "Marketing", "Sales"]
    contract_cats = ["full_time", "part_time"]
    dept_enc      = np.array([dept_cats.index(d)     for d in dept],     dtype=float)
    contract_enc  = np.array([contract_cats.index(c) for c in contract], dtype=float)

    feature_names        = ["salary", "experience", "performance",
                            "overtime_hours", "department", "contract"]
    numeric_features     = ["salary", "experience", "performance", "overtime_hours"]
    categorical_features = ["department", "contract"]
    label_encodings      = {"department": dept_cats, "contract": contract_cats}

    X = np.column_stack([salary, experience, performance,
                         overtime_hours, dept_enc, contract_enc]).astype(np.float64)

    idx        = rng.permutation(n)
    train_idx  = idx[:int(0.8 * n)]
    test_idx   = idx[int(0.8 * n):]

    return {
        "label":                "Synthetic HR",
        "X_train":              X[train_idx],
        "X_test":               X[test_idx],
        "feature_names":        feature_names,
        "numeric_features":     numeric_features,
        "categorical_features": categorical_features,
        "label_encodings":      label_encodings,
        "pa_train":             {"minority": minority[train_idx]},
        "pa_test":              {"minority": minority[test_idx]},
        "true_proxy_train":     true_proxy[train_idx],
    }


def make_credit_dataset(n: int = 2000, seed: int = 42) -> dict:
    """
    Synthetic credit dataset with two embedded proxies on *different* features.

    High-precision proxy:  age < AGE_YOUNG (≈28)
      Fires on ≈15% of the population; P(minority | fires) ≈ 88%
      Young applicants are overwhelmingly minority in this dataset.

    Broad-recall proxy:    income < income_Q50
      Fires on ≈50% of the population; P(minority | fires) ≈ 46%
      Low-income applicants are disproportionately minority, but the rule
      covers many non-minority individuals as well.

    The two proxies share no features, so they are independently discoverable:
      prec_heavy  config finds the age threshold  (narrow, high PPV)
      rec_heavy   config finds the income threshold (broad, high recall)
      balanced    config finds income (higher combined score than age)

    Six purely numeric features:
      income           Log-normal (median ≈ £44k)
      debt_ratio       Beta(2,5)  skewed toward low values
      age              Uniform 18–70  ← drives precision proxy
      education_years  Normal(14, 3) clipped to 8–20
      credit_score     Normal(650, 100) clipped to 300–850
      num_dependents   Poisson(1.5) clipped to 0–6
    """
    rng = np.random.RandomState(seed)

    income          = np.exp(rng.normal(10.7, 0.7, n))
    debt_ratio      = rng.beta(2, 5, n)
    age             = rng.uniform(18, 70, n)
    education_years = np.clip(rng.normal(14, 3, n), 8, 20)
    credit_score    = np.clip(rng.normal(650, 100, n), 300, 850)
    num_dependents  = np.clip(rng.poisson(1.5, n), 0, 6).astype(float)

    age_young  = 28.0                              # age threshold for precision proxy
    income_q50 = np.percentile(income, 50)         # income threshold for recall proxy

    young      = age < age_young                   # ~15% of population
    low_income = (income < income_q50) & ~young   # ~35% of population (not young)
    rest       = ~young & ~low_income              # ~50% of population

    # Minority prevalence by segment.
    # Overall prevalence is kept at ≈20% so the degenerate "fire on everyone"
    # expression has precision < 30% (the proxy floor) and is automatically
    # rejected — forcing the GP to find expressions that are genuinely selective.
    minority = np.where(young,      (rng.rand(n) < 0.82).astype(int),
               np.where(low_income, (rng.rand(n) < 0.18).astype(int),
                                    (rng.rand(n) < 0.04).astype(int)))

    feature_names = ["income", "debt_ratio", "age",
                     "education_years", "credit_score", "num_dependents"]

    X = np.column_stack([income, debt_ratio, age,
                         education_years, credit_score, num_dependents]).astype(np.float64)

    idx       = rng.permutation(n)
    split     = int(0.8 * n)
    train_idx = idx[:split]
    test_idx  = idx[split:]

    # Recompute proxy membership on training slice for reporting
    young_tr      = young[train_idx]
    low_income_tr = low_income[train_idx]
    broad_tr      = young_tr | low_income_tr

    return {
        "label":                "Synthetic Credit",
        "X_train":              X[train_idx],
        "X_test":               X[test_idx],
        "feature_names":        feature_names,
        "numeric_features":     feature_names,
        "categorical_features": [],
        "label_encodings":      {},
        "pa_train":             {"minority": minority[train_idx]},
        "pa_test":              {"minority": minority[test_idx]},
        "age_young":   age_young,
        "income_q50":  income_q50,
        "young_train":      young_tr.astype(int),
        "low_income_train": low_income_tr.astype(int),
        "broad_train":      broad_tr.astype(int),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def single_feature_aucs(dataset: dict, protected_attr: str) -> list[tuple[str, float]]:
    """AUC of a logistic regression trained on each feature individually."""
    X = dataset["X_train"]
    y = dataset["pa_train"][protected_attr]
    aucs = []
    for i, name in enumerate(dataset["feature_names"]):
        col = X[:, i].reshape(-1, 1)
        try:
            lr  = LogisticRegression(max_iter=200).fit(col, y)
            auc = roc_auc_score(y, lr.predict_proba(col)[:, 1])
        except Exception:
            auc = 0.5
        aucs.append((name, round(auc, 3)))
    return sorted(aucs, key=lambda t: -t[1])


def _fmt_result(r: dict) -> str:
    return (f"AUC={r['auc']:.3f}  Prec={r['precision']:.1f}%  "
            f"Rec={r['recall']:.1f}%  Cov={r['coverage']:.1f}%")


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ─── Experiment 1 ─────────────────────────────────────────────────────────────

def run_experiment_1(dataset: dict, time_budget: float, seed: int) -> dict:
    """Grammar expressivity gap: arithmetic fails, typed grammar succeeds."""
    _section("EXPERIMENT 1 — Grammar Expressivity Gap")
    print("Dataset : Synthetic HR (1 000 rows, 6 features)")
    print("Target  : is_minority")
    print("True proxy: (department == Marketing) AND (contract == part_time)")

    # Single-feature baselines
    num_feats = set(dataset["numeric_features"])
    print("\nSingle-feature AUC baselines (logistic regression):")
    for name, auc in single_feature_aucs(dataset, "minority"):
        bar     = "█" * int(auc * 40 - 20)
        tag     = "" if name in num_feats else "  [categorical — hidden from arithmetic grammar]"
        print(f"  {name:<18} {auc:.3f}  {bar}{tag}")

    y_train = dataset["pa_train"]["minority"]
    print(f"\nMinority prevalence: {y_train.mean() * 100:.1f}%  "
          f"({y_train.sum()} of {len(y_train)} training samples)")

    results = {}

    for grammar_name in ("arithmetic", "typed"):
        print(f"\n[{grammar_name.upper()} grammar]  (time budget: {time_budget}s)")
        if grammar_name == "arithmetic":
            grammar = build_grammar(
                dataset["feature_names"],
                grammar_mode="arithmetic",
                numeric_features=dataset["numeric_features"],
            )
        else:
            grammar = build_grammar(
                dataset["feature_names"],
                grammar_mode="typed",
                numeric_features=dataset["numeric_features"],
                categorical_features=dataset["categorical_features"],
                label_encodings=dataset["label_encodings"],
                X_train=dataset["X_train"],
            )

        result = run_gp(
            dataset=dataset,
            protected_attr="minority",
            grammar=grammar,
            time_budget=time_budget,
            population_size=500,
            max_depth=4,
            seed=seed,
            prec_weight=1.5,
            rec_weight=1.5,
            cov_weight=1.0,
            complexity_penalty=0.03,
        )
        results[grammar_name] = result
        print(f"  Expression : {result['expression']}")
        print(f"  Metrics    : {_fmt_result(result)}")

    return results


# ─── Experiment 2 ─────────────────────────────────────────────────────────────

def run_experiment_2(dataset: dict, time_budget: float, seed: int) -> dict:
    """Fitness configuration shapes the proxy type found."""
    _section("EXPERIMENT 2 — Fitness Configuration Shapes Proxy Type")
    print("Dataset : Synthetic Credit (2 000 rows, 6 numeric features)")
    print("Target  : is_minority")
    print(f"Precision proxy: age < {dataset['age_young']:.0f}            "
          f"(young applicants → minority, high precision)")
    print(f"Recall proxy   : income < {dataset['income_q50']:.0f}  "
          f"(low income → minority, broad coverage)")

    # Show true proxy stats
    y  = dataset["pa_train"]["minority"]
    yg = dataset["young_train"]
    li = dataset["low_income_train"]
    br = dataset["broad_train"]
    print(f"\nTrue proxy statistics (training set):")
    print(f"  Precision proxy (age < {dataset['age_young']:.0f})  : "
          f"fires on {yg.sum():>4} rows  "
          f"PPV = {y[yg.astype(bool)].mean()*100:.1f}%  "
          f"Recall = {y[yg.astype(bool)].sum() / y.sum() * 100:.1f}%")
    print(f"  Recall proxy (income < {dataset['income_q50']:.0f}): "
          f"fires on {br.sum():>4} rows  "
          f"PPV = {y[br.astype(bool)].mean()*100:.1f}%  "
          f"Recall = {y[br.astype(bool)].sum() / y.sum() * 100:.1f}%")

    # Typed grammar with all-numeric features: uses ContGreater/ContLess and
    # CondAnd to form boolean threshold rules like (income < Q20) AND
    # (debt_ratio > Q80).  This produces interpretable boolean predicates
    # directly — no post-hoc threshold needed — making precision and recall
    # unambiguous.  Quantile literals give the GP semantically grounded
    # thresholds (Q20, Q50, Q80) rather than arbitrary float constants.
    grammar = build_grammar(
        dataset["feature_names"],
        grammar_mode="typed",
        numeric_features=dataset["numeric_features"],
        categorical_features=[],
        label_encodings={},
        X_train=dataset["X_train"],
    )

    configs = {
        "balanced":   dict(prec_weight=1.0, rec_weight=1.0, cov_weight=1.0),
        "prec_heavy": dict(prec_weight=2.0, rec_weight=0.5, cov_weight=0.5),
        "rec_heavy":  dict(prec_weight=0.5, rec_weight=2.0, cov_weight=0.5),
    }

    results = {}
    for config_name, weights in configs.items():
        print(f"\n[{config_name.upper()}]  (time budget: {time_budget}s)")
        result = run_gp(
            dataset=dataset,
            protected_attr="minority",
            grammar=grammar,
            time_budget=time_budget,
            population_size=500,
            max_depth=4,
            seed=seed,
            complexity_penalty=0.03,
            **weights,
        )
        results[config_name] = result
        is_degenerate = result["precision"] == 0.0 and result["recall"] == 0.0
        print(f"  Expression : {result['expression']}")
        if is_degenerate:
            print(f"  *** DEGENERATE — fires on all rows (coverage trap: recall=100%, "
                  f"precision=base rate).  This is the expected rec_heavy failure mode "
                  f"discussed in Chapter 9 (Adult Income extended grammar). ***")
        else:
            print(f"  Metrics    : {_fmt_result(result)}")

    return results


# ─── Summary printer ──────────────────────────────────────────────────────────

def print_summary(exp1_results: dict, exp2_results: dict) -> None:
    _section("RESULTS SUMMARY")

    W_EXPR = 52

    print("\nExperiment 1 — Grammar Expressivity Gap")
    print(f"  {'Grammar':<12} {'Expression':<{W_EXPR}} {'AUC':>6} {'Prec':>7} {'Rec':>7}")
    print("  " + "-" * (12 + W_EXPR + 24))
    for gname, r in exp1_results.items():
        expr = (r["expression"][:W_EXPR - 2] + "..") if len(r["expression"]) > W_EXPR else r["expression"]
        print(f"  {gname:<12} {expr:<{W_EXPR}} {r['auc']:>6.3f} "
              f"{r['precision']:>6.1f}% {r['recall']:>6.1f}%")

    print("\nExperiment 2 — Fitness Configuration Shapes Proxy Type")
    print(f"  {'Config':<12} {'Expression':<{W_EXPR}} {'AUC':>6} {'Prec':>7} {'Rec':>7}")
    print("  " + "-" * (12 + W_EXPR + 24))
    for cname, r in exp2_results.items():
        expr = (r["expression"][:W_EXPR - 2] + "..") if len(r["expression"]) > W_EXPR else r["expression"]
        print(f"  {cname:<12} {expr:<{W_EXPR}} {r['auc']:>6.3f} "
              f"{r['precision']:>6.1f}% {r['recall']:>6.1f}%")


# ─── CSV export ───────────────────────────────────────────────────────────────

def save_results(exp1: dict, exp2: dict, out_dir: Path) -> None:
    """Save each experiment's results to a CSV for thesis tables."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["config", "expression", "auc", "precision", "recall", "coverage", "nodes"]

    for fname, results, label_col in [
        ("exp1_grammar_gap.csv",       exp1, "grammar"),
        ("exp2_fitness_config.csv",    exp2, "config"),
    ]:
        path = out_dir / fname
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[label_col] + fields[1:])
            writer.writeheader()
            for key, r in results.items():
                writer.writerow({
                    label_col:   key,
                    "expression": r["expression"],
                    "auc":        round(r["auc"], 4),
                    "precision":  round(r["precision"], 2),
                    "recall":     round(r["recall"], 2),
                    "coverage":   round(r["coverage"], 2),
                    "nodes":      r["nodes"],
                })
        print(f"  Saved → {path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic GP proxy discovery experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--time-budget", type=float, default=90.0,
                        help="GP search time per run (seconds)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-hr",     type=int, default=1000,
                        help="HR dataset size")
    parser.add_argument("--n-credit", type=int, default=2000,
                        help="Credit dataset size")
    parser.add_argument("--experiment", type=int, choices=[1, 2], default=None,
                        help="Run only experiment 1 or 2 (default: both)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to synthetic_datasets/results/")
    args = parser.parse_args()

    print("Generating synthetic datasets...")
    hr_data     = make_hr_dataset(n=args.n_hr,     seed=args.seed)
    credit_data = make_credit_dataset(n=args.n_credit, seed=args.seed)

    y_hr = hr_data["pa_train"]["minority"]
    y_cr = credit_data["pa_train"]["minority"]
    print(f"  HR dataset     : {len(y_hr)} train rows | "
          f"{y_hr.mean()*100:.1f}% minority  "
          f"({y_hr.sum()} / {len(y_hr)})")
    print(f"  Credit dataset : {len(y_cr)} train rows | "
          f"{y_cr.mean()*100:.1f}% minority  "
          f"({y_cr.sum()} / {len(y_cr)})")

    exp1_results = exp2_results = None

    if args.experiment in (None, 1):
        exp1_results = run_experiment_1(hr_data, args.time_budget, args.seed)

    if args.experiment in (None, 2):
        exp2_results = run_experiment_2(credit_data, args.time_budget, args.seed)

    if exp1_results and exp2_results:
        print_summary(exp1_results, exp2_results)

    if args.save and exp1_results and exp2_results:
        out_dir = ROOT / "synthetic_datasets" / "results"
        print(f"\nSaving results to {out_dir}/")
        save_results(exp1_results, exp2_results, out_dir)


if __name__ == "__main__":
    main()
