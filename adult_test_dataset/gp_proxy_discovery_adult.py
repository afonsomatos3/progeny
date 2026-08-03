"""
GP Proxy Discovery -- Adult Dataset
=====================================
Evolves mathematical expressions over base features that predict
protected attributes (race, sex) using GeneticEngine.

Runs on both the preprocessed and raw (non-processed) Adult datasets,
comparing discovered multi-attribute proxies against individual
feature baselines.
"""

import sys
import time
import argparse
import warnings
import dataclasses
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add GeneticEngine to path
ROOT = Path(__file__).resolve().parent
MAPROXIES = ROOT.parent / "MA-Proxies"
sys.path.insert(0, str(MAPROXIES / "GeneticEngine" / "GeneticEngine"))

from abc import ABC, abstractmethod
from typing import Annotated

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.vars import VarRange
from geml.common import forward_dataset
from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import weight
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")


# ── Typed grammar base classes ──────────────────────────────────────────────
# Two distinct abstract types prevent illegal mixed-type expressions such as:
#   NOT(2.0)               — BoolNot requires BoolExpr, not NumExpr
#   (2.0 + (A > B))        — Plus requires NumExpr on both sides
# BoolExpr is only reachable from NumExpr via IfThenElse.cond.

class NumExpr(ABC):
    """Numeric-valued expression. This is the GP root type."""
    @abstractmethod
    def to_sympy(self) -> str: ...
    @abstractmethod
    def to_numpy(self) -> str: ...
    def __str__(self) -> str:
        return self.to_sympy()


class BoolExpr(ABC):
    """Boolean-valued expression. Only valid as IfThenElse condition."""
    @abstractmethod
    def to_sympy(self) -> str: ...
    @abstractmethod
    def to_numpy(self) -> str: ...
    def __str__(self) -> str:
        return self.to_sympy()


# ── NumExpr: arithmetic operators ──────────────────────────────────────────

@weight(100)
@dataclass
class Plus(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} + {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} + {self.r.to_numpy()})"


@weight(50)
@dataclass
class Minus(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} - {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} - {self.r.to_numpy()})"


@weight(100)
@dataclass
class Mult(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} * {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} * {self.r.to_numpy()})"


@weight(100)
@dataclass
class SafeDiv(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} / {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), "
                f"where=b!=0.0))({self.l.to_numpy()}, {self.r.to_numpy()})")


# ── NumExpr: non-linear math ────────────────────────────────────────────────

@weight(30)
@dataclass
class Abs(NumExpr):
    e: NumExpr

    def to_sympy(self) -> str:
        return f"|{self.e.to_sympy()}|"

    def to_numpy(self) -> str:
        return f"np.abs({self.e.to_numpy()})"


@weight(30)
@dataclass
class Max2(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"max({self.l.to_sympy()}, {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.maximum({self.l.to_numpy()}, {self.r.to_numpy()})"


@weight(30)
@dataclass
class Min2(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"min({self.l.to_sympy()}, {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.minimum({self.l.to_numpy()}, {self.r.to_numpy()})"


# ── NumExpr: conditional (bridges BoolExpr → NumExpr) ──────────────────────

@weight(80)
@dataclass
class IfThenElse(NumExpr):
    """Vectorised if-then-else via np.where.
    cond must be a BoolExpr; branches must be NumExpr."""
    cond: BoolExpr
    then_expr: NumExpr
    else_expr: NumExpr

    def to_sympy(self) -> str:
        return (f"IF({self.cond.to_sympy()}, "
                f"{self.then_expr.to_sympy()}, "
                f"{self.else_expr.to_sympy()})")

    def to_numpy(self) -> str:
        return (f"np.where(({self.cond.to_numpy()}) > 0, "
                f"{self.then_expr.to_numpy()}, "
                f"{self.else_expr.to_numpy()})")


# ── NumExpr: constants ──────────────────────────────────────────────────────

@weight(25)
@dataclass
class Zero(NumExpr):
    def to_sympy(self) -> str:
        return "0.0"
    def to_numpy(self) -> str:
        return "0.0"


@weight(25)
@dataclass
class One(NumExpr):
    def to_sympy(self) -> str:
        return "1.0"
    def to_numpy(self) -> str:
        return "1.0"


@weight(25)
@dataclass
class Two(NumExpr):
    def to_sympy(self) -> str:
        return "2.0"
    def to_numpy(self) -> str:
        return "2.0"


@weight(15)
@dataclass
class Three(NumExpr):
    def to_sympy(self) -> str:
        return "3.0"
    def to_numpy(self) -> str:
        return "3.0"


@weight(15)
@dataclass
class Four(NumExpr):
    def to_sympy(self) -> str:
        return "4.0"
    def to_numpy(self) -> str:
        return "4.0"


@weight(15)
@dataclass
class Five(NumExpr):
    def to_sympy(self) -> str:
        return "5.0"
    def to_numpy(self) -> str:
        return "5.0"


@weight(25)
@dataclass
class FloatLiteral(NumExpr):
    value: float

    def to_sympy(self) -> str:
        return f"{self.value}"
    def to_numpy(self) -> str:
        return f"{self.value}"


# ── BoolExpr: comparison operators (operands must be NumExpr) ───────────────

@weight(60)
@dataclass
class GreaterThan(BoolExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} > {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} > {self.r.to_numpy()}).astype(np.float64)"


@weight(60)
@dataclass
class LessThan(BoolExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} < {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} < {self.r.to_numpy()}).astype(np.float64)"


@weight(40)
@dataclass
class EqualsApprox(BoolExpr):
    """Approximate equality: |l - r| < 0.5."""
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} == {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.abs(({self.l.to_numpy()}) - ({self.r.to_numpy()})) < 0.5)"
                f".astype(np.float64)")


# ── BoolExpr: logical combinators (operands must be BoolExpr) ───────────────

@weight(40)
@dataclass
class BoolAnd(BoolExpr):
    l: BoolExpr
    r: BoolExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} AND {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.logical_and(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(40)
@dataclass
class BoolOr(BoolExpr):
    l: BoolExpr
    r: BoolExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} OR {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.logical_or(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(20)
@dataclass
class BoolNot(BoolExpr):
    e: BoolExpr

    def to_sympy(self) -> str:
        return f"NOT({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"(np.logical_not(({self.e.to_numpy()}) > 0)).astype(np.float64)"


# ── Constants ────────────────────────────────────────────────────────────────

PROTECTED_ATTRS = ["black", "white", "asian_pac_islander", "amer_indian", "other", "male"]

ARITHMETIC_COMPONENTS = [Plus, Minus, Mult, SafeDiv, Zero, One, Two, FloatLiteral]

RELATIONAL_COMPONENTS = [
    GreaterThan, LessThan, EqualsApprox,
    BoolAnd, BoolOr, BoolNot,
    IfThenElse,
]

NONLINEAR_COMPONENTS = [Abs, Max2, Min2]

CATEGORICAL_CONSTANTS = [Three, Four, Five]

EXTENDED_COMPONENTS = (
    ARITHMETIC_COMPONENTS
    + RELATIONAL_COMPONENTS
    + NONLINEAR_COMPONENTS
    + CATEGORICAL_CONSTANTS
)

CONTINUOUS_ONLY_COMPONENTS = [Plus, Minus, Mult, SafeDiv, Abs, Max2, Min2,
                              FloatLiteral, Zero, One, Two]

GRAMMAR_PRESETS = {
    "arithmetic":  ARITHMETIC_COMPONENTS,
    "extended":    EXTENDED_COMPONENTS,
    "continuous":  CONTINUOUS_ONLY_COMPONENTS,
}

DATASETS = [
    (ROOT / "processed", "processed"),
    (ROOT / "non_processed", "non_processed"),
]


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(dataset_dir: Path, dataset_label: str) -> dict:
    """Load .npz and return standardised dict with features, targets, etc."""
    data = np.load(dataset_dir / "adult_base_features.npz", allow_pickle=True)

    # Use unscaled features where available (processed has X_train_raw)
    if "X_train_raw" in data:
        X_train = data["X_train_raw"]
        X_test = data["X_test_raw"]
    else:
        X_train = data["X_train"]
        X_test = data["X_test"]

    X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64),
                            nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(np.asarray(X_test, dtype=np.float64),
                           nan=0.0, posinf=1e6, neginf=-1e6)

    feature_names = [str(n) for n in data["feature_names"]]

    pa_train = {pa: np.asarray(data[f"{pa}_train"], dtype=np.int32)
                for pa in PROTECTED_ATTRS}
    pa_test = {pa: np.asarray(data[f"{pa}_test"], dtype=np.int32)
               for pa in PROTECTED_ATTRS}

    return {
        "label": dataset_label,
        "X_train": X_train,
        "X_test": X_test,
        "feature_names": feature_names,
        "pa_train": pa_train,
        "pa_test": pa_test,
    }


# ── Grammar construction ─────────────────────────────────────────────────────

def make_num_var(feature_names: list[str], relative_weight: int = 10):
    """Create a typed Var terminal that extends NumExpr (not the library's Expression)."""
    index_of = {name: i for i, name in enumerate(feature_names)}

    @weight(relative_weight)
    @dataclass
    class Var(NumExpr):
        name: Annotated[str, VarRange(feature_names)]

        def to_sympy(self) -> str:
            return f"{self.name}"

        def to_numpy(self) -> str:
            return f"dataset[:,{index_of[self.name]}]"

    return Var


def build_grammar(feature_names: list[str], grammar_mode: str = "extended"):
    """Build GP grammar with feature-specific Var terminals.
    Root type is NumExpr; BoolExpr is a sub-grammar reachable via IfThenElse."""
    Var = make_num_var(feature_names, relative_weight=10)
    components = GRAMMAR_PRESETS[grammar_mode]
    all_components = components + [Var]
    return extract_grammar(all_components, NumExpr)


# ── Expression complexity ─────────────────────────────────────────────────────

def count_nodes(expr) -> int:
    """Recursively count the number of nodes in a typed expression tree."""
    if not dataclasses.is_dataclass(expr):
        return 1
    total = 1
    for f in dataclasses.fields(expr):
        child = getattr(expr, f.name)
        if isinstance(child, (NumExpr, BoolExpr)):
            total += count_nodes(child)
    return total


# ── Fitness / evaluation helpers ──────────────────────────────────────────────

def safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC-ROC with safety checks. Returns max(auc, 1-auc)."""
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if not np.all(np.isfinite(y_pred)):
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)

    if np.std(y_pred) < 1e-10:
        return 0.5

    y_pred = np.clip(y_pred, -1e6, 1e6)
    auc = roc_auc_score(y_true, y_pred)
    return max(auc, 1.0 - auc)


def make_fitness_fn(X_train: np.ndarray, y_train: np.ndarray,
                    auc_weight: float = 1.0, complexity_penalty: float = 0.0):
    """Return a fitness function (NumExpr -> float) for GP."""
    max_nodes = 63.0

    def fitness(expr: NumExpr) -> float:
        try:
            y_pred = forward_dataset(expr.to_numpy(), X_train)
            auc = safe_auc(y_train, y_pred)
            nodes = count_nodes(expr)
            return auc_weight * auc - complexity_penalty * (nodes / max_nodes)
        except Exception:
            return 0.5
    return fitness


def evaluate_expression(expr: NumExpr, X_data: np.ndarray,
                        y_true: np.ndarray) -> float:
    """Evaluate an expression on arbitrary data and return AUC."""
    try:
        y_pred = forward_dataset(expr.to_numpy(), X_data)
        return safe_auc(y_true, y_pred)
    except Exception:
        return 0.5


# ── Single GP run ─────────────────────────────────────────────────────────────

def run_gp(dataset: dict, protected_attr: str, grammar,
           time_budget: float, population_size: int, max_depth: int,
           seed: int, auc_weight: float = 1.0,
           complexity_penalty: float = 0.0) -> dict:
    """Run one GP search for a single dataset + protected attribute."""
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["pa_train"][protected_attr]
    y_test = dataset["pa_test"][protected_attr]

    fitness_fn = make_fitness_fn(X_train, y_train,
                                auc_weight=auc_weight,
                                complexity_penalty=complexity_penalty)

    gp = SimpleGP(
        grammar=grammar,
        fitness_function=fitness_fn,
        minimize=False,
        max_time=time_budget,
        population_size=population_size,
        max_depth=max_depth,
        seed=seed,
    )

    start = time.time()
    result = gp.search()
    elapsed = time.time() - start

    if not result:
        return {
            "dataset": dataset["label"],
            "protected_attr": protected_attr,
            "expression": "FAILED",
            "nodes": 0,
            "train_auc": 0.5,
            "test_auc": 0.5,
            "elapsed_s": round(elapsed, 1),
        }

    best_expr = result[0].get_phenotype()

    train_auc = evaluate_expression(best_expr, X_train, y_train)
    test_auc = evaluate_expression(best_expr, X_test, y_test)
    nodes = count_nodes(best_expr)

    return {
        "dataset": dataset["label"],
        "protected_attr": protected_attr,
        "expression": str(best_expr),
        "nodes": nodes,
        "train_auc": round(train_auc, 4),
        "test_auc": round(test_auc, 4),
        "elapsed_s": round(elapsed, 1),
    }


# ── Baselines ─────────────────────────────────────────────────────────────────

def load_baselines(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_dir / "individual_proxy_baselines.csv")


def extract_features_from_expr(expression_str: str,
                               all_features: list[str]) -> list[str]:
    """Return the list of feature names that appear in an expression string."""
    import re
    return [f for f in all_features if re.search(r'\b' + re.escape(f) + r'\b', expression_str)]


def best_baseline_for_expr(baselines: pd.DataFrame, pa: str,
                           expr_features: list[str]) -> tuple[float, str]:
    """Return (best_auc, best_feature) among only the features in the expression."""
    col = f"auc_{pa}"
    subset = baselines[baselines["feature"].isin(expr_features)]
    if subset.empty:
        return 0.5, "none"
    idx = subset[col].idxmax()
    return round(float(subset.loc[idx, col]), 4), str(subset.loc[idx, "feature"])


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    print(f"\n{'=' * 90}")
    print("GP PROXY DISCOVERY - ADULT DATASET - SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Dataset':<16} {'Attr':<22} {'Train AUC':>10} {'Test AUC':>10} "
          f"{'Baseline':>10} {'Delta':>8}  Expression")
    print(f"{'-' * 16} {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}  {'-' * 30}")

    for r in results:
        delta = r["train_auc"] - r["best_baseline_auc"]
        marker = " *" if delta > 0 else ""
        expr_short = r["expression"][:50] + ("..." if len(r["expression"]) > 50 else "")
        print(f"{r['dataset']:<16} {r['protected_attr']:<22} "
              f"{r['train_auc']:>10.4f} {r['test_auc']:>10.4f} "
              f"{r['best_baseline_auc']:>10.4f} {delta:>+8.4f}{marker}  {expr_short}")

    print(f"\n* = GP exceeded best single-feature baseline among features in expression")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover multi-attribute proxy expressions via GP (Adult dataset)")
    parser.add_argument("--time-budget", type=float, default=60,
                        help="Seconds per GP run (default: 60)")
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+",
                        choices=["processed", "non_processed"],
                        default=["processed", "non_processed"])
    parser.add_argument("--attributes", nargs="+",
                        choices=PROTECTED_ATTRS, default=PROTECTED_ATTRS)
    parser.add_argument("--auc-weight", type=float, default=1.0,
                        help="Weight for AUC in fitness (default: 1.0)")
    parser.add_argument("--complexity-penalty", type=float, default=0.0,
                        help="Penalty per normalised node count (default: 0.0)")
    parser.add_argument("--grammar", choices=["arithmetic", "extended"],
                        default="extended",
                        help="Grammar mode: 'arithmetic' or 'extended'. Default: extended")
    return parser.parse_args()


# ── HTML report generation ────────────────────────────────────────────────────

HTML_CSS = """\
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem auto; max-width: 1200px; background: #f8f9fa; color: #222; }
  h1 { border-bottom: 3px solid #333; padding-bottom: .4rem; }
  h2 { margin-top: 2.5rem; color: #444; }
  h3 { margin-top: 1.5rem; color: #555; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
  th, td { padding: .55rem .75rem; text-align: left; border: 1px solid #ddd; }
  th { background: #343a40; color: #fff; font-weight: 600; }
  tr:nth-child(even) { background: #f2f2f2; }
  tr:hover { background: #e8edf2; }
  .strong { background: #d4edda !important; font-weight: 600; }
  .note { font-size: .85rem; color: #666; margin-top: -.8rem; margin-bottom: 1.5rem; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
  .summary-card { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .summary-card h3 { margin: 0 0 .3rem 0; font-size: .9rem; color: #888; text-transform: uppercase; }
  .summary-card .val { font-size: 1.4rem; font-weight: 700; color: #222; }
  .expr-box { font-family: 'Fira Code', 'Consolas', monospace; font-size: .82rem; background: #f4f4f4; border: 1px solid #ddd; border-radius: 4px; padding: .6rem .8rem; margin: .3rem 0; word-break: break-all; line-height: 1.5; }
  .delta-pos { color: #28a745; font-weight: 700; }
  .delta-zero { color: #888; }
  .feat-tag { display: inline-block; background: #e9ecef; border: 1px solid #ced4da; border-radius: 3px; padding: .1rem .4rem; margin: .1rem .15rem; font-size: .8rem; font-family: 'Fira Code', 'Consolas', monospace; }
  .feat-tag.best { background: #ffc107; border-color: #e0a800; font-weight: 600; }
  .section-divider { border: none; border-top: 2px solid #dee2e6; margin: 2.5rem 0; }
  .config { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 1rem 1.5rem; margin-bottom: 2rem; font-size: .9rem; }
  .config code { background: #e9ecef; padding: .1rem .4rem; border-radius: 3px; font-size: .85rem; }
  .baseline-note { font-size: .82rem; color: #6c757d; font-style: italic; margin-bottom: 1.5rem; }
"""


def _feat_tags(features_csv: str, best_feature: str) -> str:
    """Render feature names as HTML tags, highlighting the baseline-best one."""
    tags = []
    for f in features_csv.split(", "):
        f = f.strip()
        if not f:
            continue
        cls = "feat-tag best" if f == best_feature else "feat-tag"
        tags.append(f'<span class="{cls}">{escape(f)}</span>')
    return " ".join(tags)


def _delta_span(delta: float) -> str:
    cls = "delta-pos" if delta > 0 else "delta-zero"
    return f'<span class="{cls}">{delta:+.4f}</span>'


def generate_html(results: list[dict], args) -> str:
    """Build a complete HTML report from the results list."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Group results by dataset
    datasets_seen = []
    by_dataset: dict[str, list[dict]] = {}
    for r in results:
        ds = r["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
            datasets_seen.append(ds)
        by_dataset[ds].append(r)

    # ── Header ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MA-Proxies &mdash; Adult Dataset GP Proxy Discovery Results</title>
<style>{HTML_CSS}</style>
</head>
<body>

<h1>MA-Proxies &mdash; Adult Dataset GP Proxy Discovery Results</h1>
<p>Genetic Programming was used to evolve mathematical expressions over base
features that predict each protected attribute in the UCI Adult dataset.</p>

<div class="config">
  <strong>GP Configuration:</strong>
  Population: <code>{args.population_size}</code> &middot;
  Max depth: <code>{args.max_depth}</code> &middot;
  Time budget: <code>{args.time_budget}s</code> &middot;
  Seed: <code>{args.seed}</code> &middot;
  AUC weight: <code>{args.auc_weight}</code> &middot;
  Complexity penalty: <code>{args.complexity_penalty}</code> &middot;
  Grammar: <code>{args.grammar}</code> ({', '.join(c.__name__ for c in GRAMMAR_PRESETS[args.grammar])}) + features
</div>

<p class="baseline-note">
  <strong>Baseline definition:</strong> For each evolved expression, the baseline
  is the highest single-feature AUC among only the features present in that
  expression &mdash; not the global best.
</p>
"""

    # ── Per-dataset sections ──────────────────────────────────────────────
    for ds_label in datasets_seen:
        rows = by_dataset[ds_label]
        ds_title = "Processed" if ds_label == "processed" else "Non-Processed"
        html += f'<hr class="section-divider">\n'
        html += f'<h2>{escape(ds_title)} Dataset '
        html += f'<span style="font-weight:400;color:#888">({ds_label})</span></h2>\n'

        # Summary cards
        html += '<div class="summary-grid">\n'
        for r in rows:
            delta = r["train_auc"] - r["best_baseline_auc"]
            color = "#28a745" if delta > 0 else "#222"
            html += f"""  <div class="summary-card">
    <h3>{escape(r["protected_attr"])}</h3>
    <div class="val" style="color:{color}">{r["test_auc"]:.4f} <span style="font-size:.8rem">(test)</span></div>
    <div style="font-size:.85rem;color:#666">Baseline {r["best_baseline_auc"]:.4f} ({escape(r["best_baseline_feature"])}) &middot; {_delta_span(delta)}</div>
  </div>\n"""
        html += '</div>\n\n'

        # Results table
        html += """<table>
  <thead>
    <tr>
      <th>Protected Attr</th>
      <th>Features Used</th>
      <th>Nodes</th>
      <th>Train AUC</th>
      <th>Test AUC</th>
      <th>Best Single Feature</th>
      <th>Delta</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>\n"""
        for r in rows:
            delta = r["train_auc"] - r["best_baseline_auc"]
            row_cls = ' class="strong"' if delta > 0 else ""
            feat_html = _feat_tags(r["expression_features"],
                                   r["best_baseline_feature"])
            html += f"""    <tr{row_cls}>
      <td>{escape(r["protected_attr"])}</td>
      <td>{feat_html}</td>
      <td>{r["nodes"]}</td>
      <td>{r["train_auc"]:.4f}</td>
      <td>{r["test_auc"]:.4f}</td>
      <td>{r["best_baseline_auc"]:.4f} ({escape(r["best_baseline_feature"])})</td>
      <td>{_delta_span(delta)}</td>
      <td>{r["elapsed_s"]}s</td>
    </tr>\n"""
        html += "  </tbody>\n</table>\n\n"

        # Expressions
        html += "<h3>Discovered Expressions</h3>\n\n"
        for r in rows:
            delta = r["train_auc"] - r["best_baseline_auc"]
            html += f'<p><strong>{escape(r["protected_attr"])}</strong> '
            html += f'(test AUC {r["test_auc"]:.4f} &mdash; '
            html += f'{r["nodes"]} nodes &mdash; '
            html += f'features: {escape(r["expression_features"])})</p>\n'
            html += f'<div class="expr-box">{escape(r["expression"])}</div>\n'
            html += f'<p class="note">Best single feature in expr: '
            html += f'<strong>{escape(r["best_baseline_feature"])}</strong> '
            html += f'({r["best_baseline_auc"]:.4f}). '
            html += f'Delta: {_delta_span(delta)}</p>\n\n'

    # ── Footer ────────────────────────────────────────────────────────────
    html += f"""
<p style="font-size:.82rem;color:#999;margin-top:2rem;text-align:center">
  Generated by <code>gp_proxy_discovery_adult.py</code> on {timestamp} &middot;
  Seed {args.seed} &middot; {args.time_budget}s budget &middot;
  AUC weight {args.auc_weight} &middot; Complexity penalty {args.complexity_penalty} &middot;
  Grammar: {args.grammar}
</p>

</body>
</html>
"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    dataset_map = {label: path for path, label in DATASETS}
    all_results = []

    print(f"  Grammar: {args.grammar}  |  "
          f"AUC weight: {args.auc_weight}  |  "
          f"Complexity penalty: {args.complexity_penalty}")

    for ds_label in args.datasets:
        ds_dir = dataset_map[ds_label]
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_label}")
        print(f"{'=' * 70}")

        dataset = load_dataset(ds_dir, ds_label)
        grammar = build_grammar(dataset["feature_names"], args.grammar)
        baselines = load_baselines(ds_dir)

        print(f"  Features ({len(dataset['feature_names'])}): "
              f"{', '.join(dataset['feature_names'])}")
        print(f"  Train: {dataset['X_train'].shape[0]}  |  "
              f"Test: {dataset['X_test'].shape[0]}")

        for pa in args.attributes:
            print(f"\n  --- Evolving proxy for: {pa.upper()} "
                  f"(budget {args.time_budget}s) ---")

            result = run_gp(
                dataset=dataset,
                protected_attr=pa,
                grammar=grammar,
                time_budget=args.time_budget,
                population_size=args.population_size,
                max_depth=args.max_depth,
                seed=args.seed,
                auc_weight=args.auc_weight,
                complexity_penalty=args.complexity_penalty,
            )

            # Extract features used in the expression and find the best
            # single-feature baseline among only those features
            expr_feats = extract_features_from_expr(
                result["expression"], dataset["feature_names"])
            bl_auc, bl_feat = best_baseline_for_expr(baselines, pa, expr_feats)
            result["best_baseline_auc"] = bl_auc
            result["best_baseline_feature"] = bl_feat
            result["expression_features"] = ", ".join(expr_feats)

            all_results.append(result)

            print(f"    Expression : {result['expression']}")
            print(f"    Features   : {result['expression_features']}")
            print(f"    Nodes      : {result['nodes']}")
            print(f"    Train AUC  : {result['train_auc']}")
            print(f"    Test  AUC  : {result['test_auc']}")
            print(f"    Baseline   : {bl_auc:.4f} ({bl_feat})")
            print(f"    Delta      : {result['train_auc'] - bl_auc:+.4f}")
            print(f"    Time       : {result['elapsed_s']}s")

    # Save results
    results_df = pd.DataFrame(all_results)
    out_csv = ROOT / "gp_proxy_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    out_html = ROOT / "gp_proxy_results.html"
    out_html.write_text(generate_html(all_results, args))
    print(f"HTML report saved to: {out_html}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
