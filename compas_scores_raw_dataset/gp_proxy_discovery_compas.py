"""
GP Proxy Discovery -- COMPAS Dataset
======================================
Evolves mathematical expressions over base features that predict
protected attributes (race) using GeneticEngine.

Runs on two versions of the COMPAS dataset:
  - raw: minimal preprocessing, label encoding of categoricals
  - encoded: more thoughtful numerical encoding (age derivation,
             ordinal score mapping, etc.)

The COMPAS (Correctional Offender Management Profiling for
Alternative Sanctions) dataset contains risk assessment scores
for defendants in Broward County, Florida.  The protected
attribute is race (Ethnic_Code_Text), with binary columns for:
  african_american, caucasian, hispanic, other_race

Usage:
    python gp_proxy_discovery_compas.py                        # default (extended grammar, both datasets)
    python gp_proxy_discovery_compas.py --grammar arithmetic   # arithmetic only
    python gp_proxy_discovery_compas.py --datasets raw         # raw dataset only
    python gp_proxy_discovery_compas.py --datasets encoded     # encoded dataset only
    python gp_proxy_discovery_compas.py --attributes african_american caucasian
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

# Add GeneticEngine to path
ROOT = Path(__file__).resolve().parent
MAPROXIES = ROOT.parent
sys.path.insert(0, str(MAPROXIES / "GeneticEngine" / "GeneticEngine"))

from geneticengine.grammar.grammar import extract_grammar
from geml.grammars.symbolic_regression import (
    Expression,
    Plus,
    Minus,
    Mult,
    SafeDiv,
    FloatLiteral,
    Zero,
    One,
    Two,
    make_var,
)
from geml.common import forward_dataset
from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import weight, abstract
from geneticengine.grammar.metahandlers.vars import VarRange
from dataclasses import dataclass
from typing import Annotated

warnings.filterwarnings("ignore")

# ── Nominal column names to one-hot encode ───────────────────────────────────
# These are unordered categoricals — arithmetic/ordering operators are
# semantically meaningless on their label-encoded integers.
NOMINAL_COLS = ["Agency_Text", "LegalStatus", "CustodyStatus", "MaritalStatus"]


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


# ── Extended grammar nodes ──────────────────────────────────────────────────
#
# Comparison operators produce 0.0/1.0 float arrays so they stay compatible
# with the existing Expression type and forward_dataset's eval() pipeline.
# Boolean combinators treat any value > 0 as truthy.
# IfThenElse uses np.where for vectorised branching.
#
# The extended grammar includes logic expressions (==, !=, >=, <=, >, <)
# plus boolean combinators (AND, OR, NOT) and conditionals (IF-THEN-ELSE).

# ── Comparison operators ────────────────────────────────────────────────────

@weight(60)
@dataclass
class GreaterThan(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} > {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} > {self.r.to_numpy()}).astype(np.float64)"


@weight(60)
@dataclass
class LessThan(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} < {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} < {self.r.to_numpy()}).astype(np.float64)"


@weight(50)
@dataclass
class GreaterEqual(Expression):
    """Greater-than-or-equal comparison: (l >= r) -> 0.0 or 1.0."""
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} >= {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} >= {self.r.to_numpy()}).astype(np.float64)"


@weight(50)
@dataclass
class LessEqual(Expression):
    """Less-than-or-equal comparison: (l <= r) -> 0.0 or 1.0."""
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} <= {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} <= {self.r.to_numpy()}).astype(np.float64)"


@weight(40)
@dataclass
class EqualsApprox(Expression):
    """Approximate equality: |l - r| < 0.5.

    Order-agnostic -- works correctly for label-encoded categoricals
    where values are integers (0, 1, 2, ...).  GP can combine this with
    integer constants: EqualsApprox(marital_status, 1.0) matches level 1.
    """
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} == {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.abs(({self.l.to_numpy()}) - ({self.r.to_numpy()})) < 0.5)"
                f".astype(np.float64)")


@weight(35)
@dataclass
class NotEquals(Expression):
    """Approximate inequality: |l - r| >= 0.5.

    Complementary to EqualsApprox.  Useful for excluding specific
    categorical levels: NotEquals(custody_status, 2.0) excludes level 2.
    """
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} != {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.abs(({self.l.to_numpy()}) - ({self.r.to_numpy()})) >= 0.5)"
                f".astype(np.float64)")


# ── Boolean combinators ─────────────────────────────────────────────────────

@weight(40)
@dataclass
class BoolAnd(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} AND {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.logical_and(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(40)
@dataclass
class BoolOr(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} OR {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.logical_or(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(8)
@dataclass
class BoolNot(Expression):
    e: Expression

    def to_sympy(self) -> str:
        return f"NOT({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"(np.logical_not(({self.e.to_numpy()}) > 0)).astype(np.float64)"


# ── Conditional ─────────────────────────────────────────────────────────────

@weight(10)
@dataclass
class IfThenElse(Expression):
    """Vectorised if-then-else via np.where.

    cond > 0  ->  then_expr,  otherwise  ->  else_expr.
    When cond is a comparison (0/1), this is a true conditional.
    When cond is a general expression, it thresholds at 0.
    """
    cond: Expression
    then_expr: Expression
    else_expr: Expression

    def to_sympy(self) -> str:
        return (f"IF({self.cond.to_sympy()}, "
                f"{self.then_expr.to_sympy()}, "
                f"{self.else_expr.to_sympy()})")

    def to_numpy(self) -> str:
        return (f"np.where(({self.cond.to_numpy()}) > 0, "
                f"{self.then_expr.to_numpy()}, "
                f"{self.else_expr.to_numpy()})")


# ── Non-linear math ─────────────────────────────────────────────────────────

@weight(30)
@dataclass
class Abs(Expression):
    e: Expression

    def to_sympy(self) -> str:
        return f"|{self.e.to_sympy()}|"

    def to_numpy(self) -> str:
        return f"np.abs({self.e.to_numpy()})"


@weight(30)
@dataclass
class Max2(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"max({self.l.to_sympy()}, {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.maximum({self.l.to_numpy()}, {self.r.to_numpy()})"


@weight(30)
@dataclass
class Min2(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"min({self.l.to_sympy()}, {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.minimum({self.l.to_numpy()}, {self.r.to_numpy()})"


# ── Extra integer constants (help GP match categorical levels) ──────────────

@weight(15)
@dataclass
class Three(Expression):
    def to_sympy(self) -> str:
        return "3.0"
    def to_numpy(self) -> str:
        return "3.0"


@weight(15)
@dataclass
class Four(Expression):
    def to_sympy(self) -> str:
        return "4.0"
    def to_numpy(self) -> str:
        return "4.0"


@weight(15)
@dataclass
class Five(Expression):
    def to_sympy(self) -> str:
        return "5.0"
    def to_numpy(self) -> str:
        return "5.0"


@weight(10)
@dataclass
class Six(Expression):
    def to_sympy(self) -> str:
        return "6.0"
    def to_numpy(self) -> str:
        return "6.0"


@weight(10)
@dataclass
class Seven(Expression):
    def to_sympy(self) -> str:
        return "7.0"
    def to_numpy(self) -> str:
        return "7.0"


@weight(10)
@dataclass
class Eight(Expression):
    def to_sympy(self) -> str:
        return "8.0"
    def to_numpy(self) -> str:
        return "8.0"


@weight(10)
@dataclass
class Nine(Expression):
    def to_sympy(self) -> str:
        return "9.0"
    def to_numpy(self) -> str:
        return "9.0"


@weight(10)
@dataclass
class Ten(Expression):
    def to_sympy(self) -> str:
        return "10.0"
    def to_numpy(self) -> str:
        return "10.0"


# ── Categorical grammar nodes ─────────────────────────────────────────────────

@weight(70)
@dataclass
class IsCategory(Expression):
    """Exact nominal category check: round(e) == round(k)."""
    e: Expression
    k: Expression

    def to_sympy(self) -> str:
        return f"(round({self.e.to_sympy()}) == {self.k.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.round({self.e.to_numpy()}).astype(np.int64) == "
                f"np.round({self.k.to_numpy()}).astype(np.int64)).astype(np.float64)")


@weight(40)
@dataclass
class IsNotCategory(Expression):
    """Nominal inequality check: round(e) != round(k)."""
    e: Expression
    k: Expression

    def to_sympy(self) -> str:
        return f"(round({self.e.to_sympy()}) != {self.k.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.round({self.e.to_numpy()}).astype(np.int64) != "
                f"np.round({self.k.to_numpy()}).astype(np.int64)).astype(np.float64)")


@weight(40)
@dataclass
class SameCategory(Expression):
    """Check if two expressions round to the same integer category."""
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"(round({self.l.to_sympy()}) == round({self.r.to_sympy()}))"

    def to_numpy(self) -> str:
        return (f"(np.round({self.l.to_numpy()}).astype(np.int64) == "
                f"np.round({self.r.to_numpy()}).astype(np.int64)).astype(np.float64)")


@weight(50)
@dataclass
class Round(Expression):
    """Round a continuous value to nearest integer."""
    e: Expression

    def to_sympy(self) -> str:
        return f"round({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.round({self.e.to_numpy()}).astype(np.float64)"


@weight(35)
@dataclass
class Sigmoid(Expression):
    """Logistic sigmoid: maps any value to (0, 1)."""
    e: Expression

    def to_sympy(self) -> str:
        return f"sigma({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"(1.0 / (1.0 + np.exp(-np.clip({self.e.to_numpy()}, -500.0, 500.0))))"


@weight(30)
@dataclass
class Square(Expression):
    """Square of an expression."""
    e: Expression

    def to_sympy(self) -> str:
        return f"({self.e.to_sympy()})^2"

    def to_numpy(self) -> str:
        return f"np.square({self.e.to_numpy()})"


# ── Constants ────────────────────────────────────────────────────────────────

PROTECTED_ATTRS = ["african_american", "caucasian", "hispanic", "other_race"]

ARITHMETIC_COMPONENTS = [Plus, Minus, Mult, SafeDiv, Zero, One, Two, FloatLiteral]

RELATIONAL_COMPONENTS = [
    GreaterThan, LessThan,
    EqualsApprox, NotEquals,
    BoolAnd, BoolNot,
    IfThenElse,
]

NONLINEAR_COMPONENTS = [Abs, Max2, Min2]

CATEGORICAL_CONSTANTS = [Three, Four, Five, Six, Seven, Eight, Nine, Ten]

CATEGORICAL_COMPONENTS = [IsCategory, IsNotCategory, SameCategory, Round, Sigmoid, Square]

EXTENDED_COMPONENTS = (
    ARITHMETIC_COMPONENTS
    + RELATIONAL_COMPONENTS
    + NONLINEAR_COMPONENTS
    + CATEGORICAL_CONSTANTS
    + CATEGORICAL_COMPONENTS
)

CONTINUOUS_ONLY_COMPONENTS = [Plus, Minus, Mult, SafeDiv, Abs, Max2, Min2,
                              FloatLiteral, Zero, One, Two]

GRAMMAR_PRESETS = {
    "arithmetic":  ARITHMETIC_COMPONENTS,
    "extended":    EXTENDED_COMPONENTS,
    "continuous":  CONTINUOUS_ONLY_COMPONENTS,
}


# ── Typed grammar (NumExpr / CatCond) ─────────────────────────────────────────
#
# Grammar type hierarchy:
#
#   NumExpr  — real-valued output; accepts arithmetic operators only
#   CatCond  — boolean 0/1 output; accepts logical operators and comparisons
#
# Bridges:
#   NumExpr → CatCond  :  ContGreater(l: NumExpr, r: NumExpr)
#                          ContLess   (l: NumExpr, r: NumExpr)
#   CatCond → NumExpr  :  TypedIf(cond: CatCond, then: NumExpr, else: NumExpr)
#
# Terminals:
#   NumVar       — one class for all numeric / score features
#   BinaryColCond — one class per one-hot binary feature (already 0/1)
#
# Root type: NumExpr
# ─────────────────────────────────────────────────────────────────────────────

@abstract
class TNumExpr(Expression):
    """Abstract base: numeric-valued expression (real-valued output)."""


@abstract
class TCatCond(Expression):
    """Abstract base: boolean condition expression (0.0 / 1.0 output)."""


# ── TNumExpr arithmetic nodes ─────────────────────────────────────────────────

@weight(100)
@dataclass
class TNumPlus(TNumExpr):
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} + {self.r.to_sympy()})"
    def to_numpy(self) -> str: return f"({self.l.to_numpy()} + {self.r.to_numpy()})"


@weight(50)
@dataclass
class TNumMinus(TNumExpr):
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} - {self.r.to_sympy()})"
    def to_numpy(self) -> str: return f"({self.l.to_numpy()} - {self.r.to_numpy()})"


@weight(80)
@dataclass
class TNumMult(TNumExpr):
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} * {self.r.to_sympy()})"
    def to_numpy(self) -> str: return f"({self.l.to_numpy()} * {self.r.to_numpy()})"


@weight(50)
@dataclass
class TNumSafeDiv(TNumExpr):
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} / {self.r.to_sympy()})"
    def to_numpy(self) -> str:
        return (f"(lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=np.float64),"
                f" where=b!=0.0))({self.l.to_numpy()}, {self.r.to_numpy()})")


@weight(25)
@dataclass
class TNumAbs(TNumExpr):
    e: TNumExpr
    def to_sympy(self) -> str: return f"|{self.e.to_sympy()}|"
    def to_numpy(self) -> str: return f"np.abs({self.e.to_numpy()})"


@weight(20)
@dataclass
class TNumMax(TNumExpr):
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"max({self.l.to_sympy()}, {self.r.to_sympy()})"
    def to_numpy(self) -> str: return f"np.maximum({self.l.to_numpy()}, {self.r.to_numpy()})"


@weight(20)
@dataclass
class TNumMin(TNumExpr):
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"min({self.l.to_sympy()}, {self.r.to_sympy()})"
    def to_numpy(self) -> str: return f"np.minimum({self.l.to_numpy()}, {self.r.to_numpy()})"


# ── TNumExpr constants ────────────────────────────────────────────────────────

@weight(20)
@dataclass
class TNumFloatLiteral(TNumExpr):
    value: float
    def to_sympy(self) -> str: return f"{self.value}"
    def to_numpy(self) -> str: return f"{self.value}"

@weight(20)
@dataclass
class TNumZero(TNumExpr):
    def to_sympy(self) -> str: return "0.0"
    def to_numpy(self) -> str: return "0.0"

@weight(20)
@dataclass
class TNumOne(TNumExpr):
    def to_sympy(self) -> str: return "1.0"
    def to_numpy(self) -> str: return "1.0"

@weight(20)
@dataclass
class TNumTwo(TNumExpr):
    def to_sympy(self) -> str: return "2.0"
    def to_numpy(self) -> str: return "2.0"

@weight(15)
@dataclass
class TNumThree(TNumExpr):
    def to_sympy(self) -> str: return "3.0"
    def to_numpy(self) -> str: return "3.0"

@weight(15)
@dataclass
class TNumFour(TNumExpr):
    def to_sympy(self) -> str: return "4.0"
    def to_numpy(self) -> str: return "4.0"

@weight(15)
@dataclass
class TNumFive(TNumExpr):
    def to_sympy(self) -> str: return "5.0"
    def to_numpy(self) -> str: return "5.0"

@weight(10)
@dataclass
class TNumTen(TNumExpr):
    def to_sympy(self) -> str: return "10.0"
    def to_numpy(self) -> str: return "10.0"


# ── TypedIf: bridge CatCond → NumExpr ─────────────────────────────────────────

@weight(80)
@dataclass
class TTypedIf(TNumExpr):
    """IF(condition, then_expr, else_expr).

    The ONLY way a CatCond subtree can appear inside a NumExpr.
    Vectorised via np.where.
    """
    cond:      TCatCond
    then_expr: TNumExpr
    else_expr: TNumExpr

    def to_sympy(self) -> str:
        return (f"IF({self.cond.to_sympy()}, "
                f"{self.then_expr.to_sympy()}, "
                f"{self.else_expr.to_sympy()})")

    def to_numpy(self) -> str:
        return (f"np.where(({self.cond.to_numpy()}) > 0, "
                f"{self.then_expr.to_numpy()}, "
                f"{self.else_expr.to_numpy()})")


# ── TCatCond static nodes ─────────────────────────────────────────────────────

@weight(40)
@dataclass
class TCondAnd(TCatCond):
    l: TCatCond
    r: TCatCond
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} AND {self.r.to_sympy()})"
    def to_numpy(self) -> str:
        return (f"(np.logical_and(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(40)
@dataclass
class TCondOr(TCatCond):
    l: TCatCond
    r: TCatCond
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} OR {self.r.to_sympy()})"
    def to_numpy(self) -> str:
        return (f"(np.logical_or(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(8)
@dataclass
class TCondNot(TCatCond):
    e: TCatCond
    def to_sympy(self) -> str: return f"NOT({self.e.to_sympy()})"
    def to_numpy(self) -> str:
        return f"(np.logical_not(({self.e.to_numpy()}) > 0)).astype(np.float64)"


# ── Comparison nodes: NumExpr → CatCond ──────────────────────────────────────

@weight(60)
@dataclass
class TContGreater(TCatCond):
    """l > r  (both sides must be TNumExpr)."""
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} > {self.r.to_sympy()})"
    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} > {self.r.to_numpy()}).astype(np.float64)"


@weight(60)
@dataclass
class TContLess(TCatCond):
    """l < r  (both sides must be TNumExpr)."""
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} < {self.r.to_sympy()})"
    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} < {self.r.to_numpy()}).astype(np.float64)"


@weight(40)
@dataclass
class TContGreaterEq(TCatCond):
    """l >= r  (both sides must be TNumExpr)."""
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} >= {self.r.to_sympy()})"
    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} >= {self.r.to_numpy()}).astype(np.float64)"


@weight(40)
@dataclass
class TContLessEq(TCatCond):
    """l <= r  (both sides must be TNumExpr)."""
    l: TNumExpr
    r: TNumExpr
    def to_sympy(self) -> str: return f"({self.l.to_sympy()} <= {self.r.to_sympy()})"
    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} <= {self.r.to_numpy()}).astype(np.float64)"


# ── Typed terminal factories ──────────────────────────────────────────────────

def _make_typed_num_var(numeric_features: list[str], feature_names: list[str],
                        relative_weight: float = 10):
    """TNumVar terminal: selects any numeric/score feature."""
    index_of = {name: i for i, name in enumerate(feature_names)}
    options = [f for f in numeric_features if f in index_of]

    @weight(relative_weight)
    @dataclass
    class TNumVar(TNumExpr):
        name: str

        def to_sympy(self) -> str: return f"{self.name}"
        def to_numpy(self) -> str: return f"dataset[:,{index_of[self.name]}]"

    TNumVar.__init__.__annotations__["name"] = Annotated[str, VarRange(options)]
    return TNumVar


def _make_binary_cond(feat_name: str, feat_idx: int):
    """BinaryColCond terminal: fires (=1) when the one-hot column is 1.

    One-hot columns are already 0/1 so the condition is simply col > 0.5.
    The terminal has no free parameters — it either fires or it doesn't.
    """
    @weight(15)
    @dataclass
    class BinaryColCond(TCatCond):
        def to_sympy(self) -> str: return f"{feat_name}"
        def to_numpy(self) -> str:
            return f"(dataset[:,{feat_idx}] > 0.5).astype(np.float64)"

    BinaryColCond.__name__     = f"BinCond_{feat_name}"
    BinaryColCond.__qualname__ = f"BinCond_{feat_name}"
    return BinaryColCond


# ── Static typed component list (without dynamic terminals) ───────────────────

COMPAS_TYPED_STATIC = [
    TNumPlus, TNumMinus, TNumMult, TNumSafeDiv, TNumAbs, TNumMax, TNumMin,
    TNumFloatLiteral, TNumZero, TNumOne, TNumTwo, TNumThree, TNumFour, TNumFive, TNumTen,
    TTypedIf,
    TCondAnd, TCondNot,   # TCondOr excluded — no OR in grammar
    TContGreater, TContLess, TContGreaterEq, TContLessEq,
]


def _build_typed_grammar(feature_names: list[str], disable_nodes: frozenset = frozenset()):
    """Build a strictly-typed grammar for COMPAS.

    Features are partitioned into two roles:
      • One-hot binary columns (prefix matches NOMINAL_COLS) → TCatCond terminals
        These are already 0/1 indicators — no comparison needed to make them boolean.
      • Everything else (scores, ordinal encodings, etc.) → TNumVar terminals
        GP can compare these with TContGreater / TContLess to produce TCatCond.

    Root type: TNumExpr  (the final output is always a real-valued proxy score).
    """
    binary_prefixes = tuple(c + "_" for c in NOMINAL_COLS)
    binary_feats  = [f for f in feature_names if f.startswith(binary_prefixes)]
    numeric_feats = [f for f in feature_names if f not in binary_feats]

    index_of  = {name: i for i, name in enumerate(feature_names)}
    terminals: list = []

    if numeric_feats:
        TNumVar = _make_typed_num_var(numeric_feats, feature_names, relative_weight=10)
        terminals.append(TNumVar)

    for feat in binary_feats:
        terminals.append(_make_binary_cond(feat, index_of[feat]))

    # Root is TNumExpr — consistent with all other datasets.
    # Comparisons (TContGreater etc.) and conditionals (TTypedIf) are the
    # natural way to produce boolean-like scores within a numeric tree.
    static = ([c for c in COMPAS_TYPED_STATIC if c.__name__ not in disable_nodes]
              if disable_nodes else COMPAS_TYPED_STATIC)
    return extract_grammar(static + terminals, TNumExpr)


# ── COMPAS data loading and preparation ─────────────────────────────────────

# Columns to drop (PII, metadata, redundant text versions)
PII_COLS = ["LastName", "FirstName", "MiddleName"]
META_COLS = ["Person_ID", "AssessmentID", "Case_ID"]
PROTECTED_COL = "Ethnic_Code_Text"

# Columns used for pivoting (removed after pivot)
PIVOT_COLS = ["Scale_ID", "DisplayText", "RawScore", "DecileScore",
              "ScoreText", "ScaleSet"]

# Demographic columns kept per-assessment (before pivot)
DEMO_COLS = [
    "Agency_Text", "Sex_Code_Text", "DateOfBirth",
    "ScaleSet_ID", "AssessmentReason", "Language",
    "LegalStatus", "CustodyStatus", "MaritalStatus",
    "Screening_Date", "RecSupervisionLevel", "RecSupervisionLevelText",
    "AssessmentType", "IsCompleted", "IsDeleted",
]

# Scale ID mapping for pivot
SCALE_MAP = {7: "Violence", 8: "Recidivism", 18: "FTA"}


def _pivot_compas(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot COMPAS from 3-rows-per-assessment to 1-row-per-assessment.

    Each assessment has rows for Scale_ID 7 (Violence), 8 (Recidivism),
    and 18 (Failure to Appear).  After pivoting, each assessment has
    separate columns for RawScore, DecileScore, and ScoreText per scale.
    """
    # Get one row per assessment for demographic data
    keep_cols = (META_COLS + [PROTECTED_COL] + DEMO_COLS)
    # Only keep columns that exist in the dataframe
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_demo = df.drop_duplicates(subset=["AssessmentID"])[keep_cols].copy()

    # Pivot score columns by Scale_ID
    for scale_id, scale_name in SCALE_MAP.items():
        scale_df = df[df["Scale_ID"] == scale_id][
            ["AssessmentID", "RawScore", "DecileScore", "ScoreText"]
        ].copy()
        scale_df = scale_df.rename(columns={
            "RawScore": f"RawScore_{scale_name}",
            "DecileScore": f"DecileScore_{scale_name}",
            "ScoreText": f"ScoreText_{scale_name}",
        })
        # Drop duplicate AssessmentIDs (keep first)
        scale_df = scale_df.drop_duplicates(subset=["AssessmentID"])
        df_demo = df_demo.merge(scale_df, on="AssessmentID", how="left")

    return df_demo


def _define_protected_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary protected attribute columns from Ethnic_Code_Text."""
    eth = df[PROTECTED_COL].fillna("Unknown").str.strip()

    df["african_american"] = (eth == "African-American").astype(int)
    df["caucasian"] = (eth == "Caucasian").astype(int)
    df["hispanic"] = (eth == "Hispanic").astype(int)
    # "other_race" groups: Other, Asian, Native American, Arabic, Unknown, etc.
    df["other_race"] = (~eth.isin(
        ["African-American", "Caucasian", "Hispanic"]
    )).astype(int)

    return df


def _prepare_raw(df: pd.DataFrame) -> tuple:
    """Prepare the 'raw' version: label-encode all categorical strings.

    Minimal transformation -- just makes strings into integers so GP
    can operate on numeric arrays.
    """
    df = df.copy()

    # Drop PII, metadata, protected column, and redundant text columns
    drop_cols = (PII_COLS + META_COLS + [PROTECTED_COL,
                 "RecSupervisionLevelText", "Screening_Date"])
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # One-hot encode nominal (unordered) categoricals before label-encoding
    df = _add_onehot(df, NOMINAL_COLS)

    # Identify string/object columns and label-encode remaining ones
    feature_cols = [c for c in df.columns if c not in PROTECTED_ATTRS]
    label_encoders = {}
    for c in feature_cols:
        if df[c].dtype == object:
            le = LabelEncoder()
            df[c] = df[c].fillna("MISSING")
            df[c] = le.fit_transform(df[c].astype(str))
            label_encoders[c] = le
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df, feature_cols, label_encoders


def _prepare_encoded(df: pd.DataFrame) -> tuple:
    """Prepare the 'encoded' version: thoughtful numerical encoding.

    - Derives age from DateOfBirth and Screening_Date
    - Maps ScoreText to ordinal (Low=0, Medium=1, High=2)
    - Label-encodes remaining categoricals
    """
    df = df.copy()

    # Derive age from DateOfBirth and Screening_Date
    try:
        dob = pd.to_datetime(df["DateOfBirth"], format="mixed", dayfirst=False)
        screen = pd.to_datetime(df["Screening_Date"], format="mixed", dayfirst=False)
        df["Age"] = ((screen - dob).dt.days / 365.25).round(1)
        df["Age"] = df["Age"].fillna(df["Age"].median())
    except Exception:
        df["Age"] = 0.0

    # Drop PII, metadata, protected column, and columns replaced by derived ones
    drop_cols = (PII_COLS + META_COLS + [PROTECTED_COL,
                 "RecSupervisionLevelText", "Screening_Date", "DateOfBirth"])
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Ordinal-encode ScoreText columns (Low=0, Medium=1, High=2)
    score_map = {"Low": 0, "Medium": 1, "High": 2}
    for scale_name in SCALE_MAP.values():
        col = f"ScoreText_{scale_name}"
        if col in df.columns:
            df[col] = df[col].map(score_map).fillna(0).astype(int)

    # Ordinal-encode RecSupervisionLevel if present as text
    # (it's already numeric in the data, but ensure it)
    if "RecSupervisionLevel" in df.columns:
        df["RecSupervisionLevel"] = pd.to_numeric(
            df["RecSupervisionLevel"], errors="coerce").fillna(0)

    # One-hot encode nominal (unordered) categoricals before label-encoding
    df = _add_onehot(df, NOMINAL_COLS)

    # Label-encode remaining string columns
    feature_cols = [c for c in df.columns if c not in PROTECTED_ATTRS]
    label_encoders = {}
    for c in feature_cols:
        if df[c].dtype == object:
            le = LabelEncoder()
            df[c] = df[c].fillna("MISSING")
            df[c] = le.fit_transform(df[c].astype(str))
            label_encoders[c] = le
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df, feature_cols, label_encoders


def load_compas_datasets() -> dict:
    """Load and prepare both versions of the COMPAS dataset.

    Returns a dict mapping dataset label -> dataset dict with
    X_train, X_test, feature_names, pa_train, pa_test.
    """
    csv_path = ROOT / "compas-scores-raw.csv"
    print(f"Loading COMPAS data from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"  Raw shape: {df_raw.shape}")

    # Show ethnicity distribution
    print(f"\n  Ethnic_Code_Text distribution:")
    for eth, count in df_raw[PROTECTED_COL].value_counts().items():
        print(f"    {eth}: {count}")

    # Pivot to one row per assessment
    print("\n  Pivoting to one row per assessment...")
    df_pivoted = _pivot_compas(df_raw)
    print(f"  Pivoted shape: {df_pivoted.shape}")

    # Define protected attributes
    df_pivoted = _define_protected_attrs(df_pivoted)

    # Print protected attribute distributions
    print(f"\n  Protected attribute distributions (pivoted):")
    for pa in PROTECTED_ATTRS:
        n1 = df_pivoted[pa].sum()
        pct = 100 * n1 / len(df_pivoted)
        print(f"    {pa:20s}: {n1:>6d} ({pct:.1f}%)")

    datasets = {}

    # ── Raw version ──────────────────────────────────────────────────
    print("\n  Preparing 'raw' version (label-encoded categoricals)...")
    df_raw_v, raw_features, raw_encoders = _prepare_raw(df_pivoted)
    datasets["raw"] = _build_dataset_dict(df_raw_v, raw_features, "raw")

    # ── Encoded version ──────────────────────────────────────────────
    print("  Preparing 'encoded' version (age derivation, ordinal encoding)...")
    df_enc_v, enc_features, enc_encoders = _prepare_encoded(df_pivoted)
    datasets["encoded"] = _build_dataset_dict(df_enc_v, enc_features, "encoded")

    return datasets


def _build_dataset_dict(df: pd.DataFrame, feature_cols: list,
                        label: str) -> dict:
    """Split data and build the standard dataset dict."""
    X = df[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Extract protected attribute arrays
    pa_arrays = {pa: df[pa].values.astype(int) for pa in PROTECTED_ATTRS}

    # Stratified split (80/20) -- stratify on african_american (largest minority)
    stratify_col = pa_arrays["african_american"]
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=stratify_col,
    )

    X_train, X_test = X[train_idx], X[test_idx]
    pa_train = {pa: arr[train_idx] for pa, arr in pa_arrays.items()}
    pa_test = {pa: arr[test_idx] for pa, arr in pa_arrays.items()}

    print(f"    {label}: {len(feature_cols)} features, "
          f"train={X_train.shape[0]}, test={X_test.shape[0]}")

    return {
        "label": label,
        "X_train": X_train,
        "X_test": X_test,
        "feature_names": feature_cols,
        "pa_train": pa_train,
        "pa_test": pa_test,
    }


# ── Individual proxy baselines ──────────────────────────────────────────────

N_MI_BINS = 20


def compute_individual_baselines(dataset: dict) -> pd.DataFrame:
    """Compute AUC, MI, and correlation for each feature vs each protected attr."""
    X_train = dataset["X_train"]
    feature_names = dataset["feature_names"]
    results = []

    for pa in PROTECTED_ATTRS:
        y = dataset["pa_train"][pa]
        for i, col in enumerate(feature_names):
            xi = X_train[:, i]

            if len(np.unique(y)) < 2:
                results.append({
                    "feature": col, "protected_attr": pa,
                    "auc": 0.5, "mi": 0.0, "abs_corr": 0.0, "corr_pval": 1.0,
                })
                continue

            # AUC (logistic regression)
            try:
                clf = LogisticRegression(max_iter=1000, solver="lbfgs")
                clf.fit(xi.reshape(-1, 1), y)
                proba = clf.predict_proba(xi.reshape(-1, 1))[:, 1]
                auc = roc_auc_score(y, proba)
            except Exception:
                auc = 0.5

            # Mutual Information
            xi_finite = xi[np.isfinite(xi)]
            if len(np.unique(xi_finite)) > N_MI_BINS:
                xi_binned = pd.cut(xi, bins=N_MI_BINS, labels=False,
                                   duplicates="drop")
                xi_binned = np.nan_to_num(xi_binned)
            else:
                xi_binned = xi.astype(int)
            mi = mutual_info_score(y, xi_binned)

            # Point-biserial correlation
            try:
                corr, pval = pointbiserialr(y, xi)
                corr = abs(corr)
            except Exception:
                corr, pval = 0.0, 1.0

            results.append({
                "feature": col,
                "protected_attr": pa,
                "auc": round(auc, 4),
                "mi": round(mi, 4),
                "abs_corr": round(corr, 4),
                "corr_pval": round(pval, 6),
            })

    df_baselines = pd.DataFrame(results)
    return df_baselines


def _baselines_to_wide(df_baselines: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form baselines to wide format for backward compat."""
    parts = []
    for pa in PROTECTED_ATTRS:
        sub = df_baselines[df_baselines["protected_attr"] == pa].copy()
        sub = sub.rename(columns={
            "auc": f"auc_{pa}", "mi": f"mi_{pa}",
            "abs_corr": f"corr_{pa}", "corr_pval": f"pval_{pa}",
        }).drop(columns=["protected_attr"])
        parts.append(sub.set_index("feature"))
    return pd.concat(parts, axis=1).reset_index()


# ── Grammar construction ─────────────────────────────────────────────────────

def build_grammar(feature_names: list[str], grammar_mode: str = "extended",
                  disable_nodes: frozenset = frozenset(), **_kw):
    """Build GP grammar with feature-specific Var terminals.

    grammar_mode:
      'arithmetic' — +,-,*,/ over all features (untyped Expression root)
      'extended'   — typed grammar: NumExpr / CatCond split with TypedIf bridge.
    disable_nodes: frozenset of class name strings to exclude (ablation study).
    """
    if grammar_mode == "extended":
        return _build_typed_grammar(feature_names, disable_nodes=disable_nodes)

    # arithmetic / continuous: untyped, all features through a single Var terminal
    index_of = {name: i for i, name in enumerate(feature_names)}
    Var = make_var(feature_names, relative_weight=10)
    Var.feature_names = feature_names
    Var.to_numpy = lambda self: f"dataset[:,{index_of[self.name]}]"
    components = GRAMMAR_PRESETS.get(grammar_mode, GRAMMAR_PRESETS["arithmetic"])
    if disable_nodes:
        components = [c for c in components if c.__name__ not in disable_nodes]
    return extract_grammar(components + [Var], Expression)


# ── Expression complexity ─────────────────────────────────────────────────────

def count_nodes(expr) -> int:
    """Recursively count the number of nodes in a typed expression tree."""
    if not dataclasses.is_dataclass(expr):
        return 1
    total = 1
    for f in dataclasses.fields(expr):
        child = getattr(expr, f.name)
        if isinstance(child, (TNumExpr, TCatCond, Expression)):
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
    """Return a fitness function (Expression -> float) for GP."""
    max_nodes = 63.0

    def fitness(expr: Expression) -> float:
        try:
            y_pred = forward_dataset(expr.to_numpy(), X_train)
            auc = safe_auc(y_train, y_pred)
            nodes = count_nodes(expr)
            return auc_weight * auc - complexity_penalty * (nodes / max_nodes)
        except Exception:
            return 0.5
    return fitness


def evaluate_expression(expr: Expression, X_data: np.ndarray,
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

def extract_features_from_expr(expression_str: str,
                               all_features: list[str]) -> list[str]:
    """Return the list of feature names that appear in an expression string."""
    return [f for f in all_features
            if re.search(r'\b' + re.escape(f) + r'\b', expression_str)]


def best_baseline_for_expr(baselines_wide: pd.DataFrame, pa: str,
                           expr_features: list[str]) -> tuple[float, str]:
    """Return (best_auc, best_feature) among only the features in the expression."""
    col = f"auc_{pa}"
    if col not in baselines_wide.columns:
        return 0.5, "none"
    subset = baselines_wide[baselines_wide["feature"].isin(expr_features)]
    if subset.empty:
        return 0.5, "none"
    idx = subset[col].idxmax()
    return round(float(subset.loc[idx, col]), 4), str(subset.loc[idx, "feature"])


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    print(f"\n{'=' * 90}")
    print("GP PROXY DISCOVERY - COMPAS DATASET - SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Dataset':<10} {'Attr':<20} {'Train AUC':>10} {'Test AUC':>10} "
          f"{'Baseline':>10} {'Delta':>8}  Expression")
    print(f"{'-' * 10} {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}  {'-' * 30}")

    for r in results:
        delta = r["train_auc"] - r["best_baseline_auc"]
        marker = " *" if delta > 0 else ""
        expr_short = r["expression"][:50] + ("..." if len(r["expression"]) > 50 else "")
        print(f"{r['dataset']:<10} {r['protected_attr']:<20} "
              f"{r['train_auc']:>10.4f} {r['test_auc']:>10.4f} "
              f"{r['best_baseline_auc']:>10.4f} {delta:>+8.4f}{marker}  {expr_short}")

    print(f"\n* = GP exceeded best single-feature baseline among features in expression")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover multi-attribute proxy expressions via GP (COMPAS dataset)")
    parser.add_argument("--time-budget", type=float, default=60,
                        help="Seconds per GP run (default: 60)")
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+",
                        choices=["raw", "encoded"],
                        default=["raw", "encoded"])
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
<title>MA-Proxies &mdash; COMPAS Dataset GP Proxy Discovery Results</title>
<style>{HTML_CSS}</style>
</head>
<body>

<h1>MA-Proxies &mdash; COMPAS Dataset GP Proxy Discovery Results</h1>
<p>Genetic Programming was used to evolve mathematical expressions over base
features that predict each race category in the COMPAS recidivism dataset.</p>

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
        ds_title = "Raw" if ds_label == "raw" else "Encoded"
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
  Generated by <code>gp_proxy_discovery_compas.py</code> on {timestamp} &middot;
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

    print(f"  Grammar: {args.grammar}  |  "
          f"AUC weight: {args.auc_weight}  |  "
          f"Complexity penalty: {args.complexity_penalty}")

    # Load and prepare both dataset versions
    all_datasets = load_compas_datasets()

    # Compute baselines for each dataset version
    baselines_cache = {}
    for ds_label, dataset in all_datasets.items():
        print(f"\n  Computing individual baselines for '{ds_label}'...")
        bl_long = compute_individual_baselines(dataset)
        baselines_cache[ds_label] = _baselines_to_wide(bl_long)

        # Print top baselines per protected attr
        for pa in PROTECTED_ATTRS:
            col = f"auc_{pa}"
            if col in baselines_cache[ds_label].columns:
                top = baselines_cache[ds_label].nlargest(3, col)[["feature", col]]
                print(f"    {pa}: top features = "
                      f"{', '.join(f'{r.feature}({r[col]:.4f})' for _, r in top.iterrows())}")

    # Save baselines
    for ds_label, bl_df in baselines_cache.items():
        bl_path = ROOT / f"individual_proxy_baselines_{ds_label}.csv"
        bl_df.to_csv(bl_path, index=False)
        print(f"  Baselines saved: {bl_path}")

    all_results = []

    for ds_label in args.datasets:
        dataset = all_datasets[ds_label]
        baselines = baselines_cache[ds_label]
        grammar = build_grammar(dataset["feature_names"], args.grammar)

        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_label}")
        print(f"{'=' * 70}")
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
