"""
GP Proxy Discovery -- OULAD Dataset
=====================================
Evolves mathematical expressions over base features that predict
the protected attribute (gender) using GeneticEngine.

Dataset: Open University Learning Analytics Dataset (studentInfo.csv)
Protected attribute: gender -> binary 'female' (1 = F, 0 = M)

Two dataset versions:
  - raw:     minimal preprocessing, label-encode all categorical strings
  - encoded: domain-informed ordinal encoding for ordered categories
             (imd_band, age_band, highest_education, final_result, disability)

Usage:
    python gp_proxy_discovery_oulad.py                        # default (extended, both)
    python gp_proxy_discovery_oulad.py --grammar arithmetic
    python gp_proxy_discovery_oulad.py --datasets raw
    python gp_proxy_discovery_oulad.py --datasets encoded
    python gp_proxy_discovery_oulad.py --time-budget 120
"""

import sys
import re
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from scipy.stats import pointbiserialr

ROOT = Path(__file__).resolve().parent
MAPROXIES = ROOT.parent
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

# ── Nominal column names to one-hot encode ───────────────────────────────────
# These are unordered categoricals — arithmetic/ordering operators are
# semantically meaningless on their label-encoded integers.
NOMINAL_COLS = ["code_module", "region"]


def _add_onehot(df: pd.DataFrame, nominal_cols: list) -> pd.DataFrame:
    """Replace nominal columns with one-hot binary indicator columns."""
    for col in nominal_cols:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col).astype(np.int8)
        dummies.columns = [
            c.replace(" ", "_").replace("-", "_") for c in dummies.columns
        ]
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


# ── Typed grammar base classes ──────────────────────────────────────────────
# Two independent ABCs prevent illegal mixed-type expressions such as:
#   NOT(2.0)          — BoolNot requires BoolExpr, not NumExpr
#   (2.0 + (A > B))   — Plus requires NumExpr on both sides
# BoolExpr is only reachable from NumExpr via IfThenElse.cond.

class NumExpr(ABC):
    """Numeric-valued expression. GP root type."""
    @abstractmethod
    def to_sympy(self) -> str: ...
    @abstractmethod
    def to_numpy(self) -> str: ...
    def __str__(self) -> str: return self.to_sympy()


class BoolExpr(ABC):
    """Boolean-valued expression (0.0/1.0). Only valid as IfThenElse condition."""
    @abstractmethod
    def to_sympy(self) -> str: ...
    @abstractmethod
    def to_numpy(self) -> str: ...
    def __str__(self) -> str: return self.to_sympy()


# ── NumExpr: arithmetic ──────────────────────────────────────────────────────

@weight(100)
@dataclass
class Plus(NumExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} + {self.r.to_sympy()})"
    def to_numpy(self): return f"({self.l.to_numpy()} + {self.r.to_numpy()})"

@weight(50)
@dataclass
class Minus(NumExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} - {self.r.to_sympy()})"
    def to_numpy(self): return f"({self.l.to_numpy()} - {self.r.to_numpy()})"

@weight(100)
@dataclass
class Mult(NumExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} * {self.r.to_sympy()})"
    def to_numpy(self): return f"({self.l.to_numpy()} * {self.r.to_numpy()})"

@weight(100)
@dataclass
class SafeDiv(NumExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} / {self.r.to_sympy()})"
    def to_numpy(self):
        return (f"(lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=np.float64),"
                f" where=b!=0.0))({self.l.to_numpy()}, {self.r.to_numpy()})")

@weight(30)
@dataclass
class Abs(NumExpr):
    e: NumExpr
    def to_sympy(self): return f"|{self.e.to_sympy()}|"
    def to_numpy(self): return f"np.abs({self.e.to_numpy()})"

@weight(30)
@dataclass
class Max2(NumExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"max({self.l.to_sympy()}, {self.r.to_sympy()})"
    def to_numpy(self): return f"np.maximum({self.l.to_numpy()}, {self.r.to_numpy()})"

@weight(30)
@dataclass
class Min2(NumExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"min({self.l.to_sympy()}, {self.r.to_sympy()})"
    def to_numpy(self): return f"np.minimum({self.l.to_numpy()}, {self.r.to_numpy()})"

@weight(50)
@dataclass
class Round(NumExpr):
    """Round a continuous value to nearest integer."""
    e: NumExpr
    def to_sympy(self): return f"round({self.e.to_sympy()})"
    def to_numpy(self): return f"np.round({self.e.to_numpy()}).astype(np.float64)"

@weight(35)
@dataclass
class Sigmoid(NumExpr):
    """Logistic sigmoid: maps any NumExpr to (0, 1)."""
    e: NumExpr
    def to_sympy(self): return f"sigma({self.e.to_sympy()})"
    def to_numpy(self):
        return f"(1.0 / (1.0 + np.exp(-np.clip({self.e.to_numpy()}, -500.0, 500.0))))"

@weight(30)
@dataclass
class Square(NumExpr):
    """Square of a NumExpr."""
    e: NumExpr
    def to_sympy(self): return f"({self.e.to_sympy()})^2"
    def to_numpy(self): return f"np.square({self.e.to_numpy()})"


# ── NumExpr: conditional (bridges BoolExpr → NumExpr) ───────────────────────

@weight(10)
@dataclass
class IfThenElse(NumExpr):
    """IF(cond, then_expr, else_expr) — cond must be BoolExpr; branches NumExpr."""
    cond: BoolExpr; then_expr: NumExpr; else_expr: NumExpr
    def to_sympy(self):
        return f"IF({self.cond.to_sympy()}, {self.then_expr.to_sympy()}, {self.else_expr.to_sympy()})"
    def to_numpy(self):
        return (f"np.where(({self.cond.to_numpy()}) > 0, "
                f"{self.then_expr.to_numpy()}, {self.else_expr.to_numpy()})")


# ── NumExpr: constants ────────────────────────────────────────────────────────

@weight(25)
@dataclass
class Zero(NumExpr):
    def to_sympy(self): return "0.0"
    def to_numpy(self): return "0.0"

@weight(25)
@dataclass
class One(NumExpr):
    def to_sympy(self): return "1.0"
    def to_numpy(self): return "1.0"

@weight(25)
@dataclass
class Two(NumExpr):
    def to_sympy(self): return "2.0"
    def to_numpy(self): return "2.0"

@weight(25)
@dataclass
class FloatLiteral(NumExpr):
    value: float
    def to_sympy(self): return f"{self.value}"
    def to_numpy(self): return f"{self.value}"

# Integer constants (match label-encoded categorical levels)
for _name, _val in [("Three","3.0"),("Four","4.0"),("Five","5.0"),
                    ("Six","6.0"),("Seven","7.0"),("Eight","8.0"),("Nine","9.0"),
                    ("Ten","10.0"),("Eleven","11.0"),("Twelve","12.0")]:
    exec(f"""
@weight(12)
@dataclass
class {_name}(NumExpr):
    def to_sympy(self): return "{_val}"
    def to_numpy(self): return "{_val}"
""")


# ── BoolExpr: comparisons (operands must be NumExpr) ────────────────────────

@weight(60)
@dataclass
class GreaterThan(BoolExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} > {self.r.to_sympy()})"
    def to_numpy(self): return f"({self.l.to_numpy()} > {self.r.to_numpy()}).astype(np.float64)"

@weight(60)
@dataclass
class LessThan(BoolExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} < {self.r.to_sympy()})"
    def to_numpy(self): return f"({self.l.to_numpy()} < {self.r.to_numpy()}).astype(np.float64)"

@weight(50)
@dataclass
class GreaterEqual(BoolExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} >= {self.r.to_sympy()})"
    def to_numpy(self): return f"({self.l.to_numpy()} >= {self.r.to_numpy()}).astype(np.float64)"

@weight(50)
@dataclass
class LessEqual(BoolExpr):
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} <= {self.r.to_sympy()})"
    def to_numpy(self): return f"({self.l.to_numpy()} <= {self.r.to_numpy()}).astype(np.float64)"

@weight(45)
@dataclass
class EqualsApprox(BoolExpr):
    """Approximate equality: |l - r| < 0.5  (matches label-encoded integers)."""
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} == {self.r.to_sympy()})"
    def to_numpy(self):
        return (f"(np.abs(({self.l.to_numpy()}) - ({self.r.to_numpy()})) < 0.5)"
                f".astype(np.float64)")

@weight(35)
@dataclass
class NotEquals(BoolExpr):
    """Approximate inequality: |l - r| >= 0.5."""
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"({self.l.to_sympy()} != {self.r.to_sympy()})"
    def to_numpy(self):
        return (f"(np.abs(({self.l.to_numpy()}) - ({self.r.to_numpy()})) >= 0.5)"
                f".astype(np.float64)")

@weight(70)
@dataclass
class IsCategory(BoolExpr):
    """Exact nominal category check: round(e) == round(k)."""
    e: NumExpr; k: NumExpr
    def to_sympy(self): return f"(round({self.e.to_sympy()}) == {self.k.to_sympy()})"
    def to_numpy(self):
        return (f"(np.round({self.e.to_numpy()}).astype(np.int64) == "
                f"np.round({self.k.to_numpy()}).astype(np.int64)).astype(np.float64)")

@weight(40)
@dataclass
class IsNotCategory(BoolExpr):
    """Nominal inequality check: round(e) != round(k)."""
    e: NumExpr; k: NumExpr
    def to_sympy(self): return f"(round({self.e.to_sympy()}) != {self.k.to_sympy()})"
    def to_numpy(self):
        return (f"(np.round({self.e.to_numpy()}).astype(np.int64) != "
                f"np.round({self.k.to_numpy()}).astype(np.int64)).astype(np.float64)")

@weight(40)
@dataclass
class SameCategory(BoolExpr):
    """Check if two NumExprs round to the same integer category."""
    l: NumExpr; r: NumExpr
    def to_sympy(self): return f"(round({self.l.to_sympy()}) == round({self.r.to_sympy()}))"
    def to_numpy(self):
        return (f"(np.round({self.l.to_numpy()}).astype(np.int64) == "
                f"np.round({self.r.to_numpy()}).astype(np.int64)).astype(np.float64)")


# ── BoolExpr: logical combinators (operands must be BoolExpr) ────────────────

@weight(40)
@dataclass
class BoolAnd(BoolExpr):
    l: BoolExpr; r: BoolExpr
    def to_sympy(self): return f"({self.l.to_sympy()} AND {self.r.to_sympy()})"
    def to_numpy(self):
        return (f"(np.logical_and(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")

@weight(40)
@dataclass
class BoolOr(BoolExpr):
    l: BoolExpr; r: BoolExpr
    def to_sympy(self): return f"({self.l.to_sympy()} OR {self.r.to_sympy()})"
    def to_numpy(self):
        return (f"(np.logical_or(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")

@weight(8)
@dataclass
class BoolNot(BoolExpr):
    e: BoolExpr
    def to_sympy(self): return f"NOT({self.e.to_sympy()})"
    def to_numpy(self):
        return f"(np.logical_not(({self.e.to_numpy()}) > 0)).astype(np.float64)"


PROTECTED_ATTRS = ["female"]

ARITHMETIC_COMPONENTS = [Plus, Minus, Mult, SafeDiv, Zero, One, Two, FloatLiteral]
RELATIONAL_COMPONENTS = [GreaterThan, LessThan,
                         EqualsApprox, NotEquals, BoolAnd, BoolNot, IfThenElse]
NONLINEAR_COMPONENTS  = [Abs, Max2, Min2]
CATEGORICAL_CONSTANTS = [Three, Four, Five, Six, Seven, Eight, Nine, Ten, Eleven, Twelve]
# CATEGORICAL_COMPONENTS (IsCategory, IsNotCategory, SameCategory, Round,
# Sigmoid, Square) intentionally excluded from EXTENDED_COMPONENTS.
#
# Why:
#   • IsCategory / SameCategory  allow expression==expression comparisons which
#     produce opaque sub-trees.  EqualsApprox(Var, constant) already covers
#     legitimate categorical equality checks, cleanly.
#   • Round(constant)  folds to a constant and adds nothing.
#   • Sigmoid          maps anything to (0,1); confusing when nested.
#   • Square           doesn't contribute to proxy discovery and nests deeply.

EXTENDED_COMPONENTS = (ARITHMETIC_COMPONENTS + RELATIONAL_COMPONENTS
                       + NONLINEAR_COMPONENTS + CATEGORICAL_CONSTANTS)

CONTINUOUS_ONLY_COMPONENTS = [Plus, Minus, Mult, SafeDiv, Abs, Max2, Min2,
                              FloatLiteral, Zero, One, Two]

GRAMMAR_PRESETS = {
    "arithmetic":  ARITHMETIC_COMPONENTS,
    "extended":    EXTENDED_COMPONENTS,
    "continuous":  CONTINUOUS_ONLY_COMPONENTS,
}

# ── Protected-attribute metadata (shown in HTML) ─────────────────────────────

ATTR_META = {
    "female": {"type": "Simple", "description": "Gender == Female (F)"},
}


# ── OULAD data loading ────────────────────────────────────────────────────────

CSV_PATH = ROOT / "studentInfo.csv"

# Ordinal mappings for the encoded version
_IMD_MID = {          # "0-10%" → midpoint 5, etc.
    "0-10%": 5, "10-20%": 15, "20-30%": 25, "30-40%": 35, "40-50%": 45,
    "50-60%": 55, "60-70%": 65, "70-80%": 75, "80-90%": 85, "90-100%": 95,
}
_AGE_ORD   = {"0-35": 0, "35-55": 1, "55<=": 2}
_EDU_ORD   = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
_RESULT_ORD = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}


def _load_raw_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Binary protected attribute
    df["female"] = (df["gender"].str.strip() == "F").astype(int)
    return df


def _prepare_raw(df: pd.DataFrame):
    """Label-encode all string columns; keep numerics as-is."""
    # Drop final_result: it is the downstream prediction target, not a proxy feature.
    # Including it would give the GP trivial access to the outcome variable.
    feat_df = df.drop(columns=["id_student", "gender", "female", "final_result"],
                      errors="ignore").copy()
    # One-hot encode nominal (unordered) categoricals before label-encoding
    feat_df = _add_onehot(feat_df, NOMINAL_COLS)
    for c in feat_df.columns:
        if feat_df[c].dtype == object:
            feat_df[c] = feat_df[c].fillna("MISSING")
            feat_df[c] = LabelEncoder().fit_transform(feat_df[c].astype(str))
        else:
            feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").fillna(0)
    return feat_df


def _prepare_encoded(df: pd.DataFrame):
    """Domain-informed ordinal encoding for ordered categoricals."""
    # Drop final_result: it is the downstream prediction target, not a proxy feature.
    feat_df = df.drop(columns=["id_student", "gender", "female", "final_result"],
                      errors="ignore").copy()

    feat_df["imd_band"] = feat_df["imd_band"].map(_IMD_MID).fillna(
        feat_df["imd_band"].map(_IMD_MID).median()
    ).fillna(50)

    feat_df["age_band"] = feat_df["age_band"].map(_AGE_ORD).fillna(0)

    feat_df["highest_education"] = feat_df["highest_education"].map(_EDU_ORD).fillna(
        feat_df["highest_education"].map(_EDU_ORD).median()
    ).fillna(2)

    # final_result is dropped before this point; no encoding needed.

    feat_df["disability"] = (feat_df["disability"].str.strip() == "Y").astype(int)

    # One-hot encode nominal (unordered) categoricals before label-encoding
    feat_df = _add_onehot(feat_df, NOMINAL_COLS)

    # Remaining strings → label-encode
    for c in feat_df.columns:
        if feat_df[c].dtype == object:
            feat_df[c] = feat_df[c].fillna("MISSING")
            feat_df[c] = LabelEncoder().fit_transform(feat_df[c].astype(str))
        else:
            feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").fillna(0)

    return feat_df


def _build_dataset_dict(feat_df: pd.DataFrame, female: np.ndarray, label: str) -> dict:
    X = feat_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2,
                                           random_state=42, stratify=female)
    return {
        "label":        label,
        "X_train":      X[train_idx],
        "X_test":       X[test_idx],
        "feature_names": list(feat_df.columns),
        "pa_train":     {"female": female[train_idx]},
        "pa_test":      {"female": female[test_idx]},
    }


def load_oulad_datasets() -> dict:
    print(f"Loading OULAD data from: {CSV_PATH}")
    df = _load_raw_df()
    print(f"  Shape: {df.shape}  |  female={df['female'].sum()} "
          f"({100*df['female'].mean():.1f}%)  |  male={len(df)-df['female'].sum()}")
    female = df["female"].values.astype(int)

    raw_feat  = _prepare_raw(df)
    enc_feat  = _prepare_encoded(df)

    ds = {}
    for label, feat_df in [("raw", raw_feat), ("encoded", enc_feat)]:
        ds[label] = _build_dataset_dict(feat_df, female, label)
        print(f"  {label}: {len(feat_df.columns)} features, "
              f"train={ds[label]['X_train'].shape[0]}, "
              f"test={ds[label]['X_test'].shape[0]}")
        print(f"    features: {', '.join(feat_df.columns)}")
    return ds


# ── Individual proxy baselines ────────────────────────────────────────────────

N_MI_BINS = 20


def compute_baselines(dataset: dict) -> pd.DataFrame:
    X, y = dataset["X_train"], dataset["pa_train"]["female"]
    records = []
    for i, col in enumerate(dataset["feature_names"]):
        xi = X[:, i]
        # AUC
        try:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(xi.reshape(-1, 1), y)
            auc = roc_auc_score(y, clf.predict_proba(xi.reshape(-1, 1))[:, 1])
        except Exception:
            auc = 0.5
        # MI
        n_unique = len(np.unique(xi[np.isfinite(xi)]))
        xi_bin = (pd.cut(xi, bins=N_MI_BINS, labels=False, duplicates="drop")
                  if n_unique > N_MI_BINS else xi.astype(int))
        xi_bin = np.nan_to_num(xi_bin)
        mi = mutual_info_score(y, xi_bin)
        # Correlation
        try:
            corr, pval = pointbiserialr(y, xi)
        except Exception:
            corr, pval = 0.0, 1.0
        records.append({"feature": col, "auc_female": round(auc, 4),
                        "mi": round(mi, 4), "abs_corr": round(abs(corr), 4),
                        "pval": round(pval, 6)})
    return pd.DataFrame(records).sort_values("auc_female", ascending=False)


# ── Grammar ───────────────────────────────────────────────────────────────────

def _make_num_var(feature_names: list[str], relative_weight: int = 10):
    """Create a typed Var terminal that extends NumExpr."""
    index_of = {name: i for i, name in enumerate(feature_names)}

    @weight(relative_weight)
    @dataclass
    class Var(NumExpr):
        name: Annotated[str, VarRange(feature_names)]

        def to_sympy(self) -> str: return f"{self.name}"
        def to_numpy(self) -> str: return f"dataset[:,{index_of[self.name]}]"

    return Var


def build_grammar(feature_names: list[str], mode: str,
                  disable_nodes: frozenset = frozenset(), **_kw):
    """Build GP grammar. Root type is NumExpr for both modes."""
    Var = _make_num_var(feature_names, relative_weight=10)
    components = GRAMMAR_PRESETS[mode]
    if disable_nodes:
        components = [c for c in components if c.__name__ not in disable_nodes]
    return extract_grammar(components + [Var], NumExpr)


# ── GP helpers ────────────────────────────────────────────────────────────────

def count_nodes(expr) -> int:
    if not dataclasses.is_dataclass(expr):
        return 1
    return 1 + sum(count_nodes(getattr(expr, f.name))
                   for f in dataclasses.fields(expr)
                   if isinstance(getattr(expr, f.name), (NumExpr, BoolExpr)))


def safe_auc(y_true, y_pred) -> float:
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=np.float64).ravel(),
                           nan=0.0, posinf=1e6, neginf=-1e6)
    if np.std(y_pred) < 1e-10:
        return 0.5
    auc = roc_auc_score(y_true, np.clip(y_pred, -1e6, 1e6))
    return max(auc, 1.0 - auc)


def make_fitness(X_train, y_train, auc_weight=1.0, penalty=0.0):
    def fitness(expr):
        try:
            auc = safe_auc(y_train, forward_dataset(expr.to_numpy(), X_train))
            return auc_weight * auc - penalty * (count_nodes(expr) / 63.0)
        except Exception:
            return 0.5
    return fitness


def eval_expr(expr, X, y) -> float:
    try:
        return safe_auc(y, forward_dataset(expr.to_numpy(), X))
    except Exception:
        return 0.5


def run_gp(dataset, pa, grammar, budget, pop, depth, seed,
           auc_weight=1.0, penalty=0.0) -> dict:
    X_tr, X_te = dataset["X_train"], dataset["X_test"]
    y_tr, y_te = dataset["pa_train"][pa], dataset["pa_test"][pa]

    gp = SimpleGP(grammar=grammar, fitness_function=make_fitness(X_tr, y_tr, auc_weight, penalty),
                  minimize=False, max_time=budget, population_size=pop,
                  max_depth=depth, seed=seed)
    t0 = time.time()
    result = gp.search()
    elapsed = round(time.time() - t0, 1)

    if not result:
        return {"dataset": dataset["label"], "protected_attr": pa,
                "expression": "FAILED", "nodes": 0,
                "train_auc": 0.5, "test_auc": 0.5, "elapsed_s": elapsed}

    expr = result[0].get_phenotype()
    return {"dataset": dataset["label"], "protected_attr": pa,
            "expression": str(expr), "nodes": count_nodes(expr),
            "train_auc": round(eval_expr(expr, X_tr, y_tr), 4),
            "test_auc":  round(eval_expr(expr, X_te, y_te), 4),
            "elapsed_s": elapsed}


# ── Baseline helpers ──────────────────────────────────────────────────────────

def extract_features(expr_str, all_features):
    return [f for f in all_features
            if re.search(r'\b' + re.escape(f) + r'\b', expr_str)]


def best_baseline(baselines: pd.DataFrame, expr_feats: list) -> tuple:
    sub = baselines[baselines["feature"].isin(expr_feats)]
    if sub.empty:
        return 0.5, "none"
    row = sub.loc[sub["auc_female"].idxmax()]
    return round(float(row["auc_female"]), 4), str(row["feature"])


# ── HTML generation ───────────────────────────────────────────────────────────

_CSS = """\
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


def _feat_tags(features_csv, best_feat):
    out = []
    for f in features_csv.split(", "):
        f = f.strip()
        if f:
            cls = "feat-tag best" if f == best_feat else "feat-tag"
            out.append(f'<span class="{cls}">{escape(f)}</span>')
    return " ".join(out)


def _delta(delta):
    cls = "delta-pos" if delta > 0 else "delta-zero"
    return f'<span class="{cls}">{delta:+.4f}</span>'


def generate_html(results: list[dict], args) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Group by dataset
    seen, by_ds = [], {}
    for r in results:
        ds = r["dataset"]
        by_ds.setdefault(ds, []).append(r)
        if ds not in seen:
            seen.append(ds)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OULAD Dataset — GP Proxy Discovery Results</title>
<style>{_CSS}</style>
</head>
<body>

<h1>OULAD Dataset &mdash; GP Proxy Discovery Results</h1>
<p>Genetic Programming evolved mathematical expressions over base features
to predict the protected attribute <strong>gender</strong>
(female vs male) in the Open University Learning Analytics Dataset.</p>

<div class="config">
  <strong>Configuration:</strong>
  Population: <code>{args.population_size}</code> &middot;
  Max depth: <code>{args.max_depth}</code> &middot;
  Time budget: <code>{args.time_budget}s</code> &middot;
  Seed: <code>{args.seed}</code> &middot;
  AUC weight: <code>{args.auc_weight}</code> &middot;
  Complexity penalty: <code>{args.complexity_penalty}</code> &middot;
  Grammar: <code>{args.grammar}</code>
</div>

<h2>Protected Attributes</h2>
<table>
  <tr><th>Attribute</th><th>Type</th><th>Description</th></tr>
"""
    for pa, meta in ATTR_META.items():
        html += f'  <tr><td>{escape(pa)}</td><td>{escape(meta["type"])}</td><td>{escape(meta["description"])}</td></tr>\n'
    html += "</table>\n"

    html += """<p class="baseline-note">
  <strong>Baseline:</strong> For each expression, the baseline is the best
  single-feature AUC among only the features present in that expression.
</p>\n"""

    for ds_label in seen:
        rows = by_ds[ds_label]
        ds_title = "Raw Dataset" if ds_label == "raw" else "Encoded Dataset"
        html += f'<hr class="section-divider">\n<h2>{escape(ds_title)}</h2>\n'

        # Summary cards
        html += '<div class="summary-grid">\n'
        for r in rows:
            d = r["train_auc"] - r["best_baseline_auc"]
            color = "#28a745" if d > 0 else "#222"
            html += f"""  <div class="summary-card">
    <h3>{escape(r["protected_attr"])}</h3>
    <div class="val" style="color:{color}">{r["test_auc"]:.4f} <span style="font-size:.8rem">(test)</span></div>
    <div style="font-size:.85rem;color:#666">Baseline {r["best_baseline_auc"]:.4f}
    ({escape(r["best_baseline_feature"])}) &middot; {_delta(d)}</div>
  </div>\n"""
        html += '</div>\n\n'

        # Results table
        html += ("<table><thead><tr>"
                 "<th>Protected Attr</th><th>Features Used</th><th>Nodes</th>"
                 "<th>Train AUC</th><th>Test AUC</th>"
                 "<th>Best Single Feature</th><th>Delta</th><th>Time</th>"
                 "</tr></thead><tbody>\n")
        for r in rows:
            d = r["train_auc"] - r["best_baseline_auc"]
            cls = ' class="strong"' if d > 0 else ""
            fh = _feat_tags(r["expression_features"], r["best_baseline_feature"])
            html += (f'<tr{cls}><td>{escape(r["protected_attr"])}</td>'
                     f'<td>{fh}</td><td>{r["nodes"]}</td>'
                     f'<td>{r["train_auc"]:.4f}</td><td>{r["test_auc"]:.4f}</td>'
                     f'<td>{r["best_baseline_auc"]:.4f} ({escape(r["best_baseline_feature"])})</td>'
                     f'<td>{_delta(d)}</td><td>{r["elapsed_s"]}s</td></tr>\n')
        html += "</tbody></table>\n\n"

        # Expressions
        html += "<h3>Discovered Expressions</h3>\n"
        for r in rows:
            d = r["train_auc"] - r["best_baseline_auc"]
            html += (f'<p><strong>{escape(r["protected_attr"])}</strong> '
                     f'(test AUC {r["test_auc"]:.4f} &mdash; {r["nodes"]} nodes)</p>\n'
                     f'<div class="expr-box">{escape(r["expression"])}</div>\n'
                     f'<p class="note">Baseline: {escape(r["best_baseline_feature"])} '
                     f'({r["best_baseline_auc"]:.4f}). Delta: {_delta(d)}</p>\n\n')

    html += f"""
<p style="font-size:.82rem;color:#999;margin-top:2rem;text-align:center">
  Generated by <code>gp_proxy_discovery_oulad.py</code> on {ts}
</p>
</body>
</html>
"""
    return html


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results):
    print(f"\n{'='*80}")
    print("GP PROXY DISCOVERY - OULAD - SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} {'Attr':<10} {'Train AUC':>10} {'Test AUC':>10} "
          f"{'Baseline':>10} {'Delta':>8}  Expression")
    print("-"*80)
    for r in results:
        d = r["train_auc"] - r["best_baseline_auc"]
        short = r["expression"][:50] + ("..." if len(r["expression"]) > 50 else "")
        marker = " *" if d > 0 else ""
        print(f"{r['dataset']:<10} {r['protected_attr']:<10} "
              f"{r['train_auc']:>10.4f} {r['test_auc']:>10.4f} "
              f"{r['best_baseline_auc']:>10.4f} {d:>+8.4f}{marker}  {short}")
    print("* = GP exceeded best single-feature baseline")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OULAD GP Proxy Discovery")
    p.add_argument("--time-budget", type=float, default=60)
    p.add_argument("--population-size", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", nargs="+", choices=["raw", "encoded"],
                   default=["raw", "encoded"])
    p.add_argument("--attributes", nargs="+", choices=PROTECTED_ATTRS,
                   default=PROTECTED_ATTRS)
    p.add_argument("--auc-weight", type=float, default=1.0)
    p.add_argument("--complexity-penalty", type=float, default=0.0)
    p.add_argument("--grammar", choices=["arithmetic", "extended"],
                   default="extended")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"  Grammar: {args.grammar}  |  AUC weight: {args.auc_weight}  |  "
          f"Penalty: {args.complexity_penalty}")

    all_datasets = load_oulad_datasets()

    baselines_cache = {}
    for label, ds in all_datasets.items():
        bl = compute_baselines(ds)
        baselines_cache[label] = bl
        top3 = bl.head(3)
        print(f"  Baselines ({label}): "
              + ", ".join(f"{r.feature}({r.auc_female:.4f})"
                          for _, r in top3.iterrows()))
        bl.to_csv(ROOT / f"individual_proxy_baselines_{label}.csv", index=False)

    all_results = []

    for ds_label in args.datasets:
        ds = all_datasets[ds_label]
        bl = baselines_cache[ds_label]
        grammar = build_grammar(ds["feature_names"], args.grammar)

        print(f"\n{'='*60}\nDataset: {ds_label}\n{'='*60}")
        print(f"  Features: {', '.join(ds['feature_names'])}")

        for pa in args.attributes:
            print(f"\n  --- {pa.upper()} (budget {args.time_budget}s) ---")
            r = run_gp(ds, pa, grammar,
                       args.time_budget, args.population_size,
                       args.max_depth, args.seed,
                       args.auc_weight, args.complexity_penalty)

            feats = extract_features(r["expression"], ds["feature_names"])
            bl_auc, bl_feat = best_baseline(bl, feats)
            r["best_baseline_auc"]     = bl_auc
            r["best_baseline_feature"] = bl_feat
            r["expression_features"]   = ", ".join(feats)
            all_results.append(r)

            d = r["train_auc"] - bl_auc
            print(f"    Expression : {r['expression']}")
            print(f"    Features   : {r['expression_features']}")
            print(f"    Nodes      : {r['nodes']}")
            print(f"    Train AUC  : {r['train_auc']}")
            print(f"    Test  AUC  : {r['test_auc']}")
            print(f"    Baseline   : {bl_auc:.4f} ({bl_feat})")
            print(f"    Delta      : {d:+.4f}")
            print(f"    Time       : {r['elapsed_s']}s")

    out_csv = ROOT / "gp_proxy_results.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    out_html = ROOT / "gp_proxy_results.html"
    out_html.write_text(generate_html(all_results, args))
    print(f"HTML report saved to: {out_html}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
