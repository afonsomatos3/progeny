"""
GP Proxy Discovery
===================
Evolves mathematical expressions over base features that predict
protected attributes (race, sex) using GeneticEngine.

Runs on both the preprocessed and raw (non-processed) datasets,
comparing discovered multi-attribute proxies against individual
feature baselines.
"""

import re
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.cluster import normalized_mutual_info_score

# Add GeneticEngine to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "GeneticEngine" / "GeneticEngine"))

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
from geneticengine.grammar.decorators import weight, abstract, get_gengy
from geneticengine.grammar.metahandlers.vars import VarRange
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.representations.tree.initializations import LocalSynthesisContext, MaxDepthDecider
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
from dataclasses import dataclass
from typing import Annotated

warnings.filterwarnings("ignore")

# Lower the library FloatLiteral weight from its default (25) so that feature
# variables (Var, weight 10) appear ~2x as often as arbitrary float constants
# in arithmetic/extended grammars.  Named constants (Zero/One/Two, weight 10)
# remain unchanged — they produce readable expressions like IF(cond, 1.0, 0.0).
get_gengy(FloatLiteral)["weight"] = 5


# ── Extended grammar nodes ──────────────────────────────────────────────────
#
# Comparison operators produce 0.0/1.0 float arrays so they stay compatible
# with the existing Expression type and forward_dataset's eval() pipeline.
# Boolean combinators treat any value > 0 as truthy.
# IfThenElse uses np.where for vectorised branching.
#
# This keeps everything as a single-type grammar (Expression) which avoids
# restructuring the GP search, while enabling threshold / conditional logic
# that handles categorical features correctly (via == instead of arithmetic).

# ── Comparison operators ────────────────────────────────────────────────────

@weight(80)
@dataclass
class GreaterThan(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} > {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} > {self.r.to_numpy()}).astype(np.float64)"


@weight(80)
@dataclass
class LessThan(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} < {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} < {self.r.to_numpy()}).astype(np.float64)"


@weight(60)
@dataclass
class EqualsApprox(Expression):
    """Approximate equality: |l - r| < 0.5.

    Order-agnostic — works correctly for label-encoded categoricals
    where values are integers (0, 1, 2, …).  GP can combine this with
    integer constants: EqualsApprox(grad, 1.0) matches grad == 1.
    """
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} == {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.abs(({self.l.to_numpy()}) - ({self.r.to_numpy()})) < 0.5)"
                f".astype(np.float64)")


# ── Boolean combinators ─────────────────────────────────────────────────────

@weight(50)
@dataclass
class BoolAnd(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"{self.l.to_sympy()} AND {self.r.to_sympy()}"

    def to_numpy(self) -> str:
        return (f"(np.logical_and(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(15)
@dataclass
class BoolOr(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} OR {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.logical_or(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(5)
@dataclass
class BoolNot(Expression):
    e: Expression

    def to_sympy(self) -> str:
        return f"NOT({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"(np.logical_not(({self.e.to_numpy()}) > 0)).astype(np.float64)"


# ── Conditional ─────────────────────────────────────────────────────────────

@weight(25)
@dataclass
class IfThenElse(Expression):
    """Vectorised if-then-else via np.where.

    cond > 0  →  then_expr,  otherwise  →  else_expr.
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

@weight(20)
@dataclass
class Abs(Expression):
    e: Expression

    def to_sympy(self) -> str:
        return f"|{self.e.to_sympy()}|"

    def to_numpy(self) -> str:
        return f"np.abs({self.e.to_numpy()})"


@weight(15)
@dataclass
class Max2(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"max({self.l.to_sympy()}, {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.maximum({self.l.to_numpy()}, {self.r.to_numpy()})"


@weight(15)
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


# ── Typed grammar: NumExpr and CatCond hierarchies ──────────────────────────
#
# "typed" mode enforces a strict semantic split:
#   NumExpr  — numeric-valued subtrees (real output); arithmetic lives here.
#   CatCond  — boolean conditions (0.0/1.0 output); comparisons live here.
#   TypedIf  — bridge: IF(CatCond, NumExpr, NumExpr) → NumExpr
#
# This prevents GP from producing semantically meaningless operations such as
# arithmetic on nominal categoricals (Country + Relationship) or treating a
# continuous feature as a boolean flag.  Feature terminals are split:
#   NumVar   — created dynamically for each continuous feature
#   CatEquals — created dynamically for each (categorical feature, category) pair
#
# The root type for the typed grammar is NumExpr.
# ─────────────────────────────────────────────────────────────────────────────

@abstract
class NumExpr(Expression):
    """Abstract base: numeric-valued expression (real-valued output)."""


@abstract
class CatCond(Expression):
    """Abstract base: boolean condition expression (0.0 / 1.0 output)."""


# ── NumExpr arithmetic nodes ─────────────────────────────────────────────────

@weight(100)
@dataclass
class NumPlus(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} + {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} + {self.r.to_numpy()})"


@weight(50)
@dataclass
class NumMinus(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} - {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} - {self.r.to_numpy()})"


@weight(100)
@dataclass
class NumMult(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} * {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} * {self.r.to_numpy()})"


@weight(80)
@dataclass
class NumSafeDiv(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} / {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=np.float64),"
                f" where=b!=0.0))({self.l.to_numpy()}, {self.r.to_numpy()})")


@weight(20)
@dataclass
class NumAbs(NumExpr):
    e: NumExpr

    def to_sympy(self) -> str:
        return f"|{self.e.to_sympy()}|"

    def to_numpy(self) -> str:
        return f"np.abs({self.e.to_numpy()})"


@weight(15)
@dataclass
class NumMax(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"max({self.l.to_sympy()}, {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.maximum({self.l.to_numpy()}, {self.r.to_numpy()})"


@weight(15)
@dataclass
class NumMin(NumExpr):
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"min({self.l.to_sympy()}, {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.minimum({self.l.to_numpy()}, {self.r.to_numpy()})"


# ── NumExpr literal constants ─────────────────────────────────────────────────

@weight(2)
@dataclass
class NumFloatLiteral(NumExpr):
    value: float

    def to_sympy(self) -> str:
        return f"{self.value}"

    def to_numpy(self) -> str:
        return f"{self.value}"


@weight(20)
@dataclass
class NumZero(NumExpr):
    def to_sympy(self) -> str: return "0.0"
    def to_numpy(self) -> str: return "0.0"


@weight(20)
@dataclass
class NumOne(NumExpr):
    def to_sympy(self) -> str: return "1.0"
    def to_numpy(self) -> str: return "1.0"


@weight(20)
@dataclass
class NumTwo(NumExpr):
    def to_sympy(self) -> str: return "2.0"
    def to_numpy(self) -> str: return "2.0"


@weight(15)
@dataclass
class NumThree(NumExpr):
    def to_sympy(self) -> str: return "3.0"
    def to_numpy(self) -> str: return "3.0"


@weight(15)
@dataclass
class NumFour(NumExpr):
    def to_sympy(self) -> str: return "4.0"
    def to_numpy(self) -> str: return "4.0"


@weight(15)
@dataclass
class NumFive(NumExpr):
    def to_sympy(self) -> str: return "5.0"
    def to_numpy(self) -> str: return "5.0"


# ── TypedIf: bridge from CatCond to NumExpr ───────────────────────────────────

@weight(10)
@dataclass
class TypedIf(NumExpr):
    """IF(condition, then_expr, else_expr) — condition must be a CatCond.

    Vectorised via np.where.  This is the primary bridge between the
    categorical condition subtree and the numeric output subtree.
    """
    cond:      CatCond
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


# ── CatCond static nodes ──────────────────────────────────────────────────────

@weight(80)
@dataclass
class CondAnd(CatCond):
    l: CatCond
    r: CatCond

    def to_sympy(self) -> str:
        return f"{self.l.to_sympy()} AND {self.r.to_sympy()}"

    def to_numpy(self) -> str:
        return (f"(np.logical_and(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(15)
@dataclass
class CondOr(CatCond):
    l: CatCond
    r: CatCond

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} OR {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return (f"(np.logical_or(({self.l.to_numpy()}) > 0, "
                f"({self.r.to_numpy()}) > 0)).astype(np.float64)")


@weight(5)
@dataclass
class CondNot(CatCond):
    e: CatCond

    def to_sympy(self) -> str:
        return f"NOT({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"(np.logical_not(({self.e.to_numpy()}) > 0)).astype(np.float64)"


@weight(80)
@dataclass
class ContGreater(CatCond):
    """Continuous comparison: l > r.  Both sides must be NumExpr."""
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} > {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} > {self.r.to_numpy()}).astype(np.float64)"


@weight(80)
@dataclass
class ContLess(CatCond):
    """Continuous comparison: l < r.  Both sides must be NumExpr."""
    l: NumExpr
    r: NumExpr

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} < {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} < {self.r.to_numpy()}).astype(np.float64)"


@weight(20)
@dataclass
class Between(CatCond):
    """Interval predicate: lo ≤ e ≤ hi.

    Handles lo > hi gracefully using min/max so the expression always
    produces a valid interval and never becomes trivially always-false
    due to inverted bounds chosen by mutation.
    """
    e:  NumExpr
    lo: NumExpr
    hi: NumExpr

    def to_sympy(self) -> str:
        return (f"({self.lo.to_sympy()} \u2264 {self.e.to_sympy()}"
                f" \u2264 {self.hi.to_sympy()})")

    def to_numpy(self) -> str:
        e_s  = self.e.to_numpy()
        lo_s = self.lo.to_numpy()
        hi_s = self.hi.to_numpy()
        return (
            f"(np.logical_and("
            f"({e_s}) >= np.minimum({lo_s}, {hi_s}), "
            f"({e_s}) <= np.maximum({lo_s}, {hi_s})"
            f")).astype(np.float64)"
        )


# ── Dynamic typed terminal factories ─────────────────────────────────────────

def make_num_var(numeric_features: list[str], feature_names: list[str],
                 relative_weight: float = 10):
    """Create a NumVar(NumExpr) terminal for continuous features."""
    index_of = {name: i for i, name in enumerate(feature_names)}
    options   = [f for f in numeric_features if f in index_of]

    @weight(relative_weight)
    @dataclass
    class NumVar(NumExpr):
        name: str  # annotation set dynamically below

        def to_sympy(self) -> str:
            return f"{self.name}"

        def to_numpy(self) -> str:
            return f"dataset[:,{index_of[self.name]}]"

    NumVar.__init__.__annotations__["name"] = Annotated[str, VarRange(options)]
    return NumVar


def make_cat_equals(feat_name: str, feat_idx: int, categories: list[str],
                    relative_weight: float = 15):
    """Create a CatEquals(CatCond) terminal for one categorical feature.

    The terminal selects one category by index (stored as a string so that
    VarRange can constrain it to the valid set for this feature).

    to_sympy() shows the actual category name; to_numpy() emits an exact
    integer-equality test using |col - cat_idx| < 0.5.
    """
    valid_indices = [str(i) for i in range(len(categories))]

    @weight(relative_weight)
    @dataclass
    class CatEquals(CatCond):
        cat_idx: str  # annotation set dynamically below

        def to_sympy(self) -> str:
            idx      = int(self.cat_idx)
            cat_name = (categories[idx]
                        if 0 <= idx < len(categories) else self.cat_idx)
            return f"({feat_name} == {cat_name})"

        def to_numpy(self) -> str:
            return (f"(np.abs(dataset[:,{feat_idx}]"
                    f" - {self.cat_idx}.0) < 0.5).astype(np.float64)")

    CatEquals.__name__     = f"CatEquals_{feat_name}"
    CatEquals.__qualname__ = f"CatEquals_{feat_name}"
    CatEquals.__init__.__annotations__["cat_idx"] = Annotated[str, VarRange(valid_indices)]
    return CatEquals


def make_cat_not_equals(feat_name: str, feat_idx: int, categories: list[str],
                        relative_weight: float = 8):
    """Create a CatNotEquals(CatCond) terminal: fires when feature ≠ category.

    Complements make_cat_equals to allow GP to express exclusion conditions:
    (LegalStatus != Pretrial), (Relationship != Husband), etc.
    Weighted lower than CatEquals (8 vs 15) because negations are less common
    in natural partial proxy expressions.
    """
    valid_indices = [str(i) for i in range(len(categories))]

    @weight(relative_weight)
    @dataclass
    class CatNotEquals(CatCond):
        cat_idx: str  # annotation set dynamically below

        def to_sympy(self) -> str:
            idx      = int(self.cat_idx)
            cat_name = (categories[idx]
                        if 0 <= idx < len(categories) else self.cat_idx)
            return f"({feat_name} != {cat_name})"

        def to_numpy(self) -> str:
            return (f"(np.abs(dataset[:,{feat_idx}]"
                    f" - {self.cat_idx}.0) >= 0.5).astype(np.float64)")

    CatNotEquals.__name__     = f"CatNotEquals_{feat_name}"
    CatNotEquals.__qualname__ = f"CatNotEquals_{feat_name}"
    CatNotEquals.__init__.__annotations__["cat_idx"] = Annotated[str, VarRange(valid_indices)]
    return CatNotEquals


def make_quantile_literals(feat_name: str, feat_idx: int,
                           X_train: np.ndarray,
                           relative_weight: float = 30) -> list:
    """Create NumExpr terminal constants anchored to feature quantiles.

    Produces one terminal class per distinct quantile value (Q10, Q25, Q50,
    Q75, Q90).  Their to_sympy() renders as "Q50(RawScore_Violence)" rather
    than "-3.83", making expressions readable aloud.  Duplicate values (common
    for binary features) are silently dropped.

    Weighted higher than NumFloatLiteral (25) so the GP preferentially picks
    meaningful thresholds over arbitrary float ephemerals.
    """
    col = X_train[:, feat_idx].astype(float)
    col = col[np.isfinite(col)]
    if len(col) == 0:
        return []

    quantile_specs = [
        ("Q10", 10), ("Q25", 25), ("Q50", 50), ("Q75", 75), ("Q90", 90),
    ]
    seen_vals: set[float] = set()
    classes: list = []

    for qname, pct in quantile_specs:
        qval = round(float(np.percentile(col, pct)), 6)
        if qval in seen_vals:
            continue
        seen_vals.add(qval)

        # Use a factory function to capture qname/qval/feat_name by value.
        # A plain _qname=qname assignment inside the loop is NOT sufficient:
        # the inner class methods close over the variable name, not its value,
        # so all classes would share the last iteration's values at call time.
        def _make_ql_class(qn: str, qv: float, fn: str, w: float):
            @weight(w)
            @dataclass
            class QuantileLiteral(NumExpr):
                def to_sympy(self) -> str:
                    return f"{qn}({fn})"

                def to_numpy(self) -> str:
                    return f"{qv}"

            QuantileLiteral.__name__     = f"QL_{fn}_{qn}"
            QuantileLiteral.__qualname__ = f"QL_{fn}_{qn}"
            return QuantileLiteral

        classes.append(_make_ql_class(qname, qval, feat_name, relative_weight))

    return classes


# Static component lists for typed grammar (without dynamic terminals)
TYPED_STATIC_COMPONENTS = [
    # NumExpr arithmetic
    NumPlus, NumMinus, NumMult, NumSafeDiv, NumAbs, NumMax, NumMin,
    # NumExpr literals
    NumFloatLiteral, NumZero, NumOne, NumTwo, NumThree, NumFour, NumFive,
    # TypedIf bridge (low weight — partial proxies are conjunctions, not branches)
    TypedIf,
    # CatCond combinator: AND only — NOT excluded (CatEquals/CatNotEquals cover both
    # directions for categorical; ContGreater/ContLess cover both directions for numeric,
    # so CondNot only creates double-negative redundancies like NOT(x != y) = x == y)
    CondAnd,
    # CatCond comparisons
    ContGreater, ContLess, Between,
]


# ── Constants ────────────────────────────────────────────────────────────────

PROTECTED_ATTRS = ["black", "asian", "hisp", "other", "male"]

ARITHMETIC_COMPONENTS = [Plus, Minus, Mult, SafeDiv, Abs, Max2, Min2, Zero, One, Two, FloatLiteral]

RELATIONAL_COMPONENTS = [
    GreaterThan, LessThan, EqualsApprox,
    BoolAnd,   # BoolNot/BoolOr excluded: NOT is redundant (< and > cover both directions),
               # OR inflates recall trivially
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

# Continuous-only components: pure arithmetic, no comparison or boolean nodes.
# Expressions in this mode output real-valued scores; fitness is AUC-based.
# Useful for discovering linear-combination or nonlinear-scoring proxies that
# are not easily expressible as threshold rules.
CONTINUOUS_ONLY_COMPONENTS = [
    Plus, Minus, Mult, SafeDiv,
    Abs, Max2, Min2,
    FloatLiteral, Zero, One, Two,
]

GRAMMAR_PRESETS = {
    "arithmetic":  ARITHMETIC_COMPONENTS,
    "extended":    EXTENDED_COMPONENTS,
    "continuous":  CONTINUOUS_ONLY_COMPONENTS,
    # "typed" is handled specially in build_grammar (requires feature type info)
}

DATASETS = [
    (ROOT / "processed", "processed"),
    (ROOT / "non_processed", "non_processed"),
]

# ── Proxy discovery criteria ──────────────────────────────────────────────────
# A GP expression qualifies as a *partial proxy* when ALL of the following
# criteria are satisfied.  The quality bar (Criterion 2) is TYPE-DEPENDENT:
# the appropriate metric differs between continuous and logical expressions.
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  TWO ROLES OF AUC — do not confuse them                                 │
#  │                                                                         │
#  │  Role A — GP fitness for the continuous/arithmetic grammar:             │
#  │    AUC drives evolution in make_auc_fitness_fn.  Every expression is    │
#  │    ranked by how well it separates protected from non-protected across   │
#  │    all thresholds.  This is purely internal to the search.              │
#  │                                                                         │
#  │  Role B — Acceptance criterion for CONTINUOUS proxies only:             │
#  │    After the search, a continuous expression (real-valued output) is    │
#  │    accepted as a proxy if its AUC ≥ PROXY_MIN_AUC.  AUC is the right   │
#  │    metric here because the expression was optimised for it and the      │
#  │    threshold τ is chosen post-hoc — so precision at τ is not a fixed   │
#  │    property of the expression but of the chosen cut-off.               │
#  │                                                                         │
#  │  AUC is NOT used to accept LOGICAL proxies (boolean expressions).       │
#  │  Logical proxies output 0/1 directly, so precision at τ = 0.5 is the  │
#  │  natural and unambiguous quality metric.                                │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  Expression type detection: a numpy expression that contains the substring
#  ".astype(np.float64)" is boolean-rooted (a comparison or logical predicate
#  cast to float).  All others are continuous (arithmetic output).
#
#  Criterion                     Default   Applies to
#  ──────────────────────────────────────────────────────────────────────────
#  1. Precision floor  ≥ 30 %    always    Hard minimum PPV.  Prevents broad,
#     (PROXY_MIN_PRECISION_FLOOR)           indiscriminate rules from being
#                                           reported regardless of type.
#
#  2. Partial-proxy quality bar:           Use the universal precision floor for
#      Precision ≥ PROXY_MIN_PRECISION_    all expression types so arithmetic
#                  FLOOR (30 %)            grammars are not filtered out by an
#                                          extra AUC gate.
#
#  3. Recall floor ≥ 5 %         always    Non-trivial group coverage.
#     (PROXY_MIN_RECALL)

PROXY_MIN_AUC             = 0.60   # continuous acceptance: minimum ROC-AUC
PROXY_MIN_PRECISION       = 60.0   # logical acceptance: minimum PPV %
PROXY_MIN_PRECISION_FLOOR = 60.0   # universal hard floor PPV % — lowered from 80% to surface
                                   # partial proxies for minority groups (<10% prevalence) where
                                   # 80% precision is structurally unreachable
PROXY_MIN_RECALL          = 10.0   # universal recall floor %
PROXY_MAX_EVAL_CANDIDATES = 2000   # wider reevaluation pool so early good proxies are not dropped


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(dataset_dir: Path, dataset_label: str) -> dict:
    """Load .npz and return standardised dict with features, targets, etc."""
    data = np.load(dataset_dir / "law_school_base_features.npz", allow_pickle=True)

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

def build_grammar(feature_names: list[str], grammar_mode: str = "typed",
                  numeric_features: list[str] | None = None,
                  categorical_features: list[str] | None = None,
                  label_encodings: dict[str, list[str]] | None = None,
                  X_train: np.ndarray | None = None,
                  disable_nodes: frozenset = frozenset(),
                  **_kw):
    """Build GP grammar with feature-specific Var terminals.

    grammar_mode:
      'arithmetic' — original +,-,*,/ over all features (Expression root)
      'extended'   — + comparisons, conditionals, abs/max/min (Expression root)
      'typed'      — strict NumExpr / CatCond split (CatCond root):
                       • NumVar terminals for numeric_features
                       • CatEquals_<feat> terminals per categorical feature category
                       • ContGreater/ContLess compare NumExprs → CatCond
                       • TypedIf(CatCond, NumExpr, NumExpr) → NumExpr, usable
                         inside comparisons for context-dependent thresholds
                     Root is CatCond so every expression is a boolean predicate —
                     it fires or it doesn't, no post-hoc threshold needed.
                     Requires numeric_features + categorical_features (+ label_encodings
                     for CatEquals).  Falls back to 'extended' when type info is missing.

    For 'arithmetic': when numeric_features is provided, Var is restricted to
    numeric features only (arithmetic on label-encoded categoricals is meaningless).
    For 'extended': uses all features via untyped Var (backward-compatible).
    """
    index_of = {name: i for i, name in enumerate(feature_names)}

    # ── Typed mode ────────────────────────────────────────────────────────────
    if grammar_mode == "typed":
        num_feats  = [f for f in (numeric_features  or []) if f in index_of]
        cat_feats  = [f for f in (categorical_features or []) if f in index_of]
        enc        = label_encodings or {}

        # Treat all unclassified features as numeric (arithmetic ops only, no
        # categorical equality).  This covers the common case where type info
        # is not explicitly provided — all features become NumVar terminals and
        # the grammar is still boolean-rooted (CatCond root via ContGreater /
        # ContLess), preventing boolean/arithmetic mixing.
        known = set(num_feats) | set(cat_feats)
        extra = [f for f in feature_names if f not in known]
        num_feats = num_feats + extra

        # Only fall back to extended if feature_names itself is empty
        if not num_feats and not cat_feats:
            return build_grammar(feature_names, "extended")

        # NumVar terminal: one class covering all numeric features
        terminals: list = []
        if num_feats:
            NumVar = make_num_var(num_feats, feature_names, relative_weight=10)
            terminals.append(NumVar)
            # Quantile literals: one set of (Q10/Q25/Q50/Q75/Q90) per numeric
            # feature when training data is available.  These give the GP
            # semantically meaningful thresholds — "above median" rather than
            # "> -3.83" — so evolved expressions read as natural language rules.
            if X_train is not None:
                for feat in num_feats:
                    if feat not in index_of:
                        continue
                    ql_classes = make_quantile_literals(
                        feat, index_of[feat], X_train, relative_weight=30)
                    terminals.extend(ql_classes)

        # CatEquals / CatNotEquals terminals: one class pair per categorical
        # feature, constrained to valid category indices via VarRange.
        for feat in cat_feats:
            cats = enc.get(feat, [])
            if not cats:
                # No encoding info → treat as numeric
                continue
            feat_idx = index_of[feat]
            CatEq    = make_cat_equals(feat, feat_idx, cats)
            CatNeq   = make_cat_not_equals(feat, feat_idx, cats)
            terminals.append(CatEq)
            terminals.append(CatNeq)

        all_components = TYPED_STATIC_COMPONENTS + terminals
        # Root is CatCond (boolean predicate), not NumExpr.
        #
        # With NumExpr as root, GP evolves continuous scores (e.g.
        # IF(Sex==Male, Hours_per_week*fnlwgt, Capital_Gain)) that require a
        # post-hoc threshold to interpret — you cannot read the expression and
        # say "this fires when …".
        #
        # With CatCond as root, every evolved expression IS a boolean predicate:
        #   (Marital_Status == Married) AND (Hours_per_week > 40.0)
        #   NOT(Relationship == Husband) AND (Capital_Gain > 5000.0)
        #   (IF(Marital_Status == Married, Hours_per_week, Capital_Gain) > 40.0)
        #
        # TypedIf(CatCond, NumExpr, NumExpr) still participates as a NumExpr
        # inside ContGreater/ContLess, enabling context-dependent thresholds.
        # The rule fires or doesn't fire — no ambiguity.
        if disable_nodes:
            all_components = [c for c in all_components
                              if c.__name__ not in disable_nodes]
        return extract_grammar(all_components, CatCond)

    # ── Arithmetic / Extended / Continuous modes (untyped, backward-compatible) ──
    #
    # For 'arithmetic': restrict Var to numeric features only when type info is
    # available.  Arithmetic on label-encoded categoricals (e.g. Workclass + 3)
    # is semantically meaningless and produces unreadable expressions.
    #
    # For 'continuous': same restriction as arithmetic, but NO comparison/boolean
    # nodes are included.  Expressions are pure continuous scores; fitness is
    # AUC-based rather than precision/recall/coverage-based.
    if grammar_mode in ("arithmetic", "continuous") and numeric_features:
        num_only = [f for f in numeric_features if f in index_of]
        var_features = num_only if num_only else feature_names
    else:
        var_features = feature_names

    Var = make_var(var_features, relative_weight=10)
    Var.feature_names = var_features
    Var.to_numpy = lambda self: f"dataset[:,{index_of[self.name]}]"

    components = list(GRAMMAR_PRESETS[grammar_mode])
    if disable_nodes:
        components = [c for c in components if c.__name__ not in disable_nodes]

    all_components = components + [Var]
    return extract_grammar(all_components, Expression)


# ── Expression complexity ─────────────────────────────────────────────────────

def count_nodes(expr) -> int:
    """Recursively count the number of nodes in an Expression tree."""
    if not dataclasses.is_dataclass(expr):
        return 1
    total = 1
    for f in dataclasses.fields(expr):
        child = getattr(expr, f.name)
        if isinstance(child, Expression):
            total += count_nodes(child)
    return total


# ── Constant folding ──────────────────────────────────────────────────────────

def is_var_free(expr: Expression) -> bool:
    """True if the expression contains no feature variable.

    Every Var node produces 'dataset[:,i]' in its to_numpy() string, so
    checking for the substring 'dataset' is sufficient and fast.
    """
    return "dataset" not in expr.to_numpy()


def _const_val(expr: Expression) -> float | None:
    """Return the numeric value of a constant (var-free) expression, or None."""
    if not is_var_free(expr):
        return None
    try:
        v = float(eval(expr.to_numpy(), {"np": np}))  # noqa: S307
        return v if np.isfinite(v) else None
    except Exception:
        return None


_PLUS_NAMES    = frozenset({"Plus", "NumPlus"})
_MINUS_NAMES   = frozenset({"Minus", "NumMinus"})
_MULT_NAMES    = frozenset({"Mult", "NumMult"})
_SAFEDIV_NAMES = frozenset({"SafeDiv", "NumSafeDiv"})
_CMP_GT_NAMES  = frozenset({"GreaterThan", "ContGreater"})
_CMP_LT_NAMES  = frozenset({"LessThan", "ContLess"})
_CMP_NAMES     = _CMP_GT_NAMES | _CMP_LT_NAMES
_BOOL_AND_NAMES = frozenset({"BoolAnd", "CondAnd"})
_BOOL_OR_NAMES  = frozenset({"BoolOr", "CondOr"})
_BOOL_NOT_NAMES = frozenset({"BoolNot", "CondNot"})
_MIN_NAMES      = frozenset({"Min2", "NumMin"})
_MAX_NAMES      = frozenset({"Max2", "NumMax"})


def _collect_additive_terms(expr: Expression) -> list[tuple[int, Expression]]:
    """Flatten a Plus/Minus (or NumPlus/NumMinus) tree into signed (sign, term) pairs.

    2.0 - (2.0 + (2.0 + x))  →  [(+1, 2.0), (-1, 2.0), (-1, 2.0), (-1, x)]

    Only descends into Plus/Minus nodes; any other node (Mult, Var, …) is
    returned as an atomic term.
    """
    _n = type(expr).__name__
    if _n in _PLUS_NAMES:
        return _collect_additive_terms(expr.l) + _collect_additive_terms(expr.r)
    if _n in _MINUS_NAMES:
        left  = _collect_additive_terms(expr.l)
        right = [(-s, t) for s, t in _collect_additive_terms(expr.r)]
        return left + right
    return [(1, expr)]


def _rebuild_additive(non_const: list[tuple[int, Expression]],
                      const_sum: float,
                      use_num: bool = False) -> Expression:
    """Reassemble signed variable terms and a constant into an expression tree.

    Puts the constant first (when non-zero) so leading negatives are avoided:
        (-2.0 - Country)  rather than  ((0.0 - Country) - 2.0)

    use_num: if True, uses NumPlus/NumMinus/NumFloatLiteral (for typed grammar).
    """
    _Lit   = NumFloatLiteral if use_num else FloatLiteral
    _Plus  = NumPlus  if use_num else Plus
    _Minus = NumMinus if use_num else Minus

    if not non_const:
        return _Lit(const_sum)

    # constant goes first so we never need a leading 0-x negation
    parts: list[tuple[int, Expression]] = []
    if const_sum != 0.0:
        parts.append((1, _Lit(const_sum)))
    parts.extend(non_const)

    first_sign, first_expr = parts[0]
    result: Expression = (_Minus(_Lit(0.0), first_expr)
                          if first_sign == -1 else first_expr)
    for sign, term in parts[1:]:
        result = _Plus(result, term) if sign == 1 else _Minus(result, term)
    return result


def _collect_multiplicative_terms(expr: Expression) -> tuple[float, list[Expression]]:
    """Flatten a Mult (or NumMult) tree into (constant_product, [variable_factors]).

    2.0 * (3.0 * x)  →  (6.0, [x])
    Only descends into Mult/NumMult nodes.
    """
    if type(expr).__name__ in _MULT_NAMES:
        lc, lf = _collect_multiplicative_terms(expr.l)
        rc, rf = _collect_multiplicative_terms(expr.r)
        return lc * rc, lf + rf
    v = _const_val(expr)
    if v is not None:
        return v, []
    return 1.0, [expr]


def _rebuild_multiplicative(factors: list[Expression],
                             const_prod: float,
                             use_num: bool = False) -> Expression:
    """Reassemble variable factors and a constant product into a Mult tree."""
    _Lit  = NumFloatLiteral if use_num else FloatLiteral
    _Mult = NumMult if use_num else Mult

    if not factors:
        return _Lit(const_prod)
    result: Expression = factors[0]
    for f in factors[1:]:
        result = _Mult(result, f)
    if const_prod != 1.0:
        result = _Mult(_Lit(const_prod), result)
    return result


def _fold_constants_once(expr: Expression) -> Expression:
    """Simplify an expression by constant folding, algebraic identities,
    and accumulation of constants across additive/multiplicative chains.

    Rules applied bottom-up (in order):
      1. Constant folding  — var-free subtree → single FloatLiteral
      2. Algebraic identities:
           x*0=0,  0*x=0,  x*1=x,  1*x=x
           x+0=x,  0+x=x,  x-0=x
           x/1=x,  0/x=0
      3. Additive accumulation (commutativity + associativity over +/-):
           c1 + (c2 + x)  →  (c1+c2) + x
           c1 - (c2 + x)  →  (c1-c2) - x
           2.0 - (2.0 + (2.0 + x))  →  -2.0 - x
      4. Multiplicative accumulation:
           c1 * (c2 * x)  →  (c1*c2) * x

    Operates on phenotype Expression copies only — never touches the TreeNode
    genotype stored inside GeneticEngine.
    """
    _node_name = type(expr).__name__

    # ── 0. Comparison nodes: handle before the is_var_free shortcut ──────────
    # GreaterThan/LessThan.to_numpy() calls .astype(np.float64) on the result,
    # which fails when both children are Python scalars (True.astype throws).
    # Intercept here so both-constant comparisons fold correctly.
    if _node_name in _CMP_NAMES and is_var_free(expr):
        lv = _const_val(expr.l)
        rv = _const_val(expr.r)
        if lv is not None and rv is not None:
            result = (lv > rv) if _node_name in _CMP_GT_NAMES else (lv < rv)
            return FloatLiteral(1.0 if result else 0.0)

    # ── 1. Constant fold: var-free subtree → single literal ───────────────────
    if is_var_free(expr):
        v = _const_val(expr)
        if v is not None:
            # Use typed literal when node looks like a NumExpr (name starts with Num
            # or is in the typed arithmetic set)
            _is_num = isinstance(expr, NumExpr) or _node_name.startswith("Num")
            return NumFloatLiteral(v) if _is_num else FloatLiteral(v)
        return expr

    if not dataclasses.is_dataclass(expr):
        return expr

    # ── Recurse into children first (bottom-up) ───────────────────────────────
    new_fields = {}
    for f in dataclasses.fields(expr):
        child = getattr(expr, f.name)
        new_fields[f.name] = fold_constants(child) if isinstance(child, Expression) else child

    node = type(expr)(**new_fields)
    _name = type(node).__name__

    # Detect whether we're operating in the typed (NumExpr / Num*) grammar
    use_num = isinstance(node, NumExpr) or _name.startswith("Num")

    # ── 2. Algebraic identities ───────────────────────────────────────────────
    _zero = NumFloatLiteral(0.0) if use_num else FloatLiteral(0.0)

    if _name in _MULT_NAMES:
        lv, rv = _const_val(node.l), _const_val(node.r)
        if lv == 0.0 or rv == 0.0:
            return _zero
        if lv == 1.0:
            return node.r
        if rv == 1.0:
            return node.l

    elif _name in _PLUS_NAMES:
        lv, rv = _const_val(node.l), _const_val(node.r)
        if lv == 0.0:
            return node.r
        if rv == 0.0:
            return node.l

    elif _name in _MINUS_NAMES:
        rv = _const_val(node.r)
        if rv == 0.0:
            return node.l

    elif _name in _SAFEDIV_NAMES:
        lv, rv = _const_val(node.l), _const_val(node.r)
        if rv == 0.0:
            return _zero                            # x / 0  →  0  (safe-div convention)
        if rv == 1.0:
            return node.l
        if lv == 0.0:
            return _zero
        if rv is not None:
            # x / c  →  (1/c) * x — lets multiplicative accumulation fold it
            _Lit  = NumFloatLiteral if use_num else FloatLiteral
            _Mult = NumMult if use_num else Mult
            return _Mult(_Lit(1.0 / rv), node.l)

    elif _name in _BOOL_AND_NAMES:
        # true AND x = x,  false AND x = false,  x AND true = x,  x AND false = false
        lv = _const_val(node.l)
        rv = _const_val(node.r)
        if lv is not None:
            return node.r if lv > 0.0 else FloatLiteral(0.0)
        if rv is not None:
            return node.l if rv > 0.0 else FloatLiteral(0.0)

    elif _name in _BOOL_OR_NAMES:
        # true OR x = true,  false OR x = x,  x OR true = true,  x OR false = x
        lv = _const_val(node.l)
        rv = _const_val(node.r)
        if lv is not None:
            return FloatLiteral(1.0) if lv > 0.0 else node.r
        if rv is not None:
            return FloatLiteral(1.0) if rv > 0.0 else node.l

    elif _name in _BOOL_NOT_NAMES:
        cv = _const_val(node.e)
        if cv is not None:
            return FloatLiteral(0.0 if cv > 0.0 else 1.0)
        # NOT(NOT(x)) → x  (double-negation elimination)
        if type(node.e).__name__ in _BOOL_NOT_NAMES:
            return node.e.e

    elif _name == "Between":
        ev  = _const_val(node.e)
        lov = _const_val(node.lo)
        hiv = _const_val(node.hi)
        if ev is not None and lov is not None and hiv is not None:
            result = min(lov, hiv) <= ev <= max(lov, hiv)
            return FloatLiteral(1.0 if result else 0.0)
        # e == lo or e == hi (as constant): min/max semantics guarantee always-True
        if ev is not None and (ev == lov or ev == hiv):
            return FloatLiteral(1.0)

    elif _name in ("TypedIf", "IfThenElse"):
        # IF(cond, a, a) → a  — condition is irrelevant when both branches identical
        if node.then_expr.to_numpy() == node.else_expr.to_numpy():
            return node.then_expr

    # ── 3. Additive accumulation across Plus/Minus chains ────────────────────
    if _name in _PLUS_NAMES | _MINUS_NAMES:
        terms      = _collect_additive_terms(node)
        const_vals = [_const_val(t) for _, t in terms]
        n_consts   = sum(1 for v in const_vals if v is not None)

        # Variable cancellation: group non-const terms by numpy repr and sum
        # their signed coefficients.  Only drop terms whose net coefficient is
        # exactly 0 (e.g. +Duration and -Duration cancel).  Terms with net ±2
        # or higher are left untouched — _rebuild_additive only handles ±1 signs.
        var_terms_raw = [(s, t) for (s, t), v in zip(terms, const_vals) if v is None]
        _key_sign: dict[str, int] = {}
        _key_term: dict[str, Expression] = {}
        for _s, _t in var_terms_raw:
            _k = _t.to_numpy()
            _key_sign[_k] = _key_sign.get(_k, 0) + _s
            _key_term[_k] = _t
        # Keep any variable whose net sign is exactly ±1; for |net|>1 fall back
        # to the original list (can't represent "2*x" cheaply in the ±1 system).
        all_net_safe = all(abs(sg) == 1 for sg in _key_sign.values() if sg != 0)
        if all_net_safe:
            var_terms = [(sg, _key_term[k]) for k, sg in _key_sign.items() if sg != 0]
        else:
            var_terms = var_terms_raw
        n_cancelled = len(var_terms_raw) - len(var_terms)

        # Restructure when: ≥2 constants to collapse, a zero constant to drop,
        # or at least one opposite-sign variable pair cancelled out.
        if n_consts >= 2 or any(v == 0.0 for v in const_vals) or n_cancelled > 0:
            const_sum = sum(s * v
                            for (s, _), v in zip(terms, const_vals)
                            if v is not None)
            return _rebuild_additive(var_terms, const_sum, use_num=use_num)

    # ── 4. Multiplicative accumulation across Mult chains ────────────────────
    if _name in _MULT_NAMES:
        const_prod, factors = _collect_multiplicative_terms(node)

        def _count_mult_consts(e: Expression) -> int:
            if type(e).__name__ in _MULT_NAMES:
                return _count_mult_consts(e.l) + _count_mult_consts(e.r)
            return 1 if _const_val(e) is not None else 0

        if _count_mult_consts(node) >= 2:
            if const_prod == 0.0:
                return _zero
            if not factors:
                return (NumFloatLiteral(const_prod) if use_num
                        else FloatLiteral(const_prod))
            return _rebuild_multiplicative(factors, const_prod, use_num=use_num)

    # ── 4b. Min/Max local constant folding ──────────────────────────────────
    if _name in _MIN_NAMES | _MAX_NAMES:
        lv, rv = _const_val(node.l), _const_val(node.r)
        if lv is not None and rv is not None:
            _Lit = NumFloatLiteral if use_num else FloatLiteral
            return _Lit(min(lv, rv) if _name in _MIN_NAMES else max(lv, rv))
        if lv is not None and rv is None:
            if _name in _MIN_NAMES and lv <= -1e12:
                return node.l
            if _name in _MAX_NAMES and lv >= 1e12:
                return node.l
        if rv is not None and lv is None:
            if _name in _MIN_NAMES and rv <= -1e12:
                return node.r
            if _name in _MAX_NAMES and rv >= 1e12:
                return node.r

    # ── 5. Comparison normalisation ──────────────────────────────────────────
    #
    # 5a. Move additive constants from LHS to RHS:
    #   (c + x) > threshold  →  x > (threshold - c)
    #
    # 5b. Flip comparison when LHS is a pure constant:
    #   c > expr  →  expr < c   (avoids confusing "constant > variable" form)
    if _name in _CMP_NAMES:
        lhs, rhs = node.l, node.r
        rhs_const = _const_val(rhs)
        lhs_const = _const_val(lhs)

        # 5b. Pure constant on LHS — flip comparison direction
        if lhs_const is not None and not is_var_free(rhs):
            _Lit = NumFloatLiteral if use_num else FloatLiteral
            # GreaterThan → LessThan, LessThan → GreaterThan
            if _name in _CMP_GT_NAMES:
                # c > expr  ≡  expr < c
                # Find the corresponding LessThan class (same module if possible)
                return LessThan(rhs, _Lit(lhs_const))
            else:
                # c < expr  ≡  expr > c
                return GreaterThan(rhs, _Lit(lhs_const))

        # 5c. Negated LHS: (0 - x) > t  →  x < -t
        if _name in _CMP_NAMES and type(lhs).__name__ in _MINUS_NAMES:
            l_lhs_const = _const_val(lhs.l)
            if l_lhs_const == 0.0 and not is_var_free(lhs.r) and rhs_const is not None:
                _Lit = NumFloatLiteral if use_num else FloatLiteral
                if _name in _CMP_GT_NAMES:
                    return LessThan(lhs.r, _Lit(-rhs_const))
                return GreaterThan(lhs.r, _Lit(-rhs_const))

        # 5a. Additive constant offset on LHS — move it to RHS
        if rhs_const is not None and not is_var_free(lhs):
            terms          = _collect_additive_terms(lhs)
            const_vals_lhs = [_const_val(t) for _, t in terms]
            n_consts_lhs   = sum(1 for v in const_vals_lhs if v is not None)
            if n_consts_lhs > 0:
                const_sum_lhs = sum(
                    s * v
                    for (s, _), v in zip(terms, const_vals_lhs)
                    if v is not None
                )
                non_const_lhs = [
                    (s, t)
                    for (s, t), v in zip(terms, const_vals_lhs)
                    if v is None
                ]
                if non_const_lhs:          # keep at least one variable term
                    _use_num = _name.startswith("Cont") or use_num
                    _Lit     = NumFloatLiteral if _use_num else FloatLiteral
                    new_lhs  = _rebuild_additive(non_const_lhs, 0.0,
                                                 use_num=_use_num)
                    new_rhs  = _Lit(round(rhs_const - const_sum_lhs, 10))
                    return type(node)(new_lhs, new_rhs)

        # 5e. Multiplicative constant on LHS: (c * x) > t  →  x > t/c  (c > 0)
        #                                                    →  x < t/c  (c < 0, flip)
        # Also catches (x * c) > t via the symmetric check.
        # After 5a strips additive offsets this fires on the residual c*x form.
        if rhs_const is not None and type(lhs).__name__ in _MULT_NAMES:
            _lv = _const_val(lhs.l)
            _rv = _const_val(lhs.r)
            _c, _var = None, None
            if _lv is not None and _lv != 0.0 and not is_var_free(lhs.r):
                _c, _var = _lv, lhs.r
            elif _rv is not None and _rv != 0.0 and not is_var_free(lhs.l):
                _c, _var = _rv, lhs.l
            if _c is not None:
                _use_num = _name.startswith("Cont") or use_num
                _Lit = NumFloatLiteral if _use_num else FloatLiteral
                _new_t = round(rhs_const / _c, 10)
                _is_gt = _name in _CMP_GT_NAMES
                if _c > 0:
                    return (GreaterThan(_var, _Lit(_new_t)) if _is_gt
                            else LessThan(_var, _Lit(_new_t)))
                else:
                    return (LessThan(_var, _Lit(_new_t)) if _is_gt
                            else GreaterThan(_var, _Lit(_new_t)))

        # 5d. min(c, x) > t  →  x > t  when c > t; otherwise always false.
        if _name in _CMP_GT_NAMES and rhs_const is not None and type(lhs).__name__ in _MIN_NAMES:
            l_const = _const_val(lhs.l)
            r_const = _const_val(lhs.r)
            if l_const is not None and r_const is None:
                if l_const <= rhs_const:
                    return FloatLiteral(0.0)
                return GreaterThan(lhs.r, rhs)
            if r_const is not None and l_const is None:
                if r_const <= rhs_const:
                    return FloatLiteral(0.0)
                return GreaterThan(lhs.l, rhs)

        # 5f. IF(cond, a, b) OP c  where a, b, c are all constants.
        # The IF just routes between two constant outputs — the comparison result
        # depends only on whether the condition is true or false.
        #   a OP c = True,  b OP c = False  →  cond
        #   a OP c = False, b OP c = True   →  NOT(cond)
        #   both True / both False           →  constant 1.0 / 0.0
        if rhs_const is not None and type(lhs).__name__ in ("TypedIf", "IfThenElse"):
            a_const = _const_val(lhs.then_expr)
            b_const = _const_val(lhs.else_expr)
            if a_const is not None and b_const is not None:
                _is_gt  = _name in _CMP_GT_NAMES
                a_fires = (a_const > rhs_const) if _is_gt else (a_const < rhs_const)
                b_fires = (b_const > rhs_const) if _is_gt else (b_const < rhs_const)
                if a_fires and not b_fires:
                    return lhs.cond
                if b_fires and not a_fires:
                    # CondNot is not in the typed grammar — only create it for
                    # atomic (leaf) CatCond terminals (CatEquals / CatNotEquals).
                    # Compound conditions (CondAnd, ContGreater, …) would produce
                    # NOT(A AND B) which is outside the grammar and displays
                    # confusingly; leave the original comparison unchanged.
                    _inner_is_compound = (
                        dataclasses.is_dataclass(lhs.cond)
                        and any(isinstance(getattr(lhs.cond, f.name), Expression)
                                for f in dataclasses.fields(lhs.cond))
                    )
                    if not _inner_is_compound:
                        return CondNot(lhs.cond)
                    # compound: fall through to return node unchanged
                else:
                    # both fire or neither fires — pure constant
                    return FloatLiteral(1.0 if (a_fires and b_fires) else 0.0)

    return node


def fold_constants(expr: Expression) -> Expression:
    """Repeatedly simplify an expression until no more rewrites apply."""
    current = expr
    for _ in range(12):
        simplified = _fold_constants_once(current)
        if type(simplified) is type(current) and simplified.to_numpy() == current.to_numpy():
            return simplified
        current = simplified
    return current


# ── Display-string prettification ────────────────────────────────────────────

_FLOAT_RE = re.compile(r'-?\d+\.\d+(?:[eE][+-]?\d+)?')


def _fmt_float(v: float) -> str:
    """Format a float for human-readable display: 4 sig figs, keep .0 suffix."""
    if v == int(v) and abs(v) < 10_000:
        return f"{int(v)}.0"
    s = f"{v:.4g}"
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


def _prettify_expr(s: str) -> str:
    """Round float literals to 4 sig figs and strip one matching outer paren layer.

    Applied to to_sympy() strings before writing to CSV / HTML so that
    expressions like '((code_module_BBB + -0.45143013166434126) > -3.6714)'
    become '(code_module_BBB - 0.4514) > -3.671'.
    Only affects display strings — numpy_expr keeps full precision.
    """
    s = _FLOAT_RE.sub(lambda m: _fmt_float(float(m.group(0))), s)
    # Normalize NOT((X == Y)) → (X != Y)  and  NOT((X != Y)) → (X == Y).
    # These appear when fold_constants creates CondNot for atomic conditions.
    # Applied iteratively since multiple such subexpressions may be present.
    _prev = None
    while _prev != s:
        _prev = s
        s = re.sub(r'NOT\(\(([^()]+) != ([^()]+)\)\)', r'(\1 == \2)', s)
        s = re.sub(r'NOT\(\(([^()]+) == ([^()]+)\)\)', r'(\1 != \2)', s)
    # Strip matching outer parenthesis layers.
    # A layer is only stripped when the opening '(' at position 0 matches
    # the closing ')' at the very last position (i.e. depth hits 0 at the end).
    # If depth hits 0 before the end, the outer parens don't wrap the whole
    # expression (e.g. "(A AND B) AND (C AND D)") — stop immediately to
    # avoid an infinite loop on such strings.
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        matched = False
        for i, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            if depth == 0:
                if i == len(s) - 1:
                    s = s[1:-1].strip()
                    matched = True
                break
        if not matched:
            break
        if not (s.startswith("(") and s.endswith(")")):
            break
    return s


# ── AND component decomposition ───────────────────────────────────────────────

_AND_NODE_NAMES = frozenset({"BoolAnd", "CondAnd"})


def decompose_and_tree(expr) -> list:
    """Flatten a top-level AND tree into its atomic condition components.

    (A AND B) AND C  →  [A, B, C]
    Non-AND expression  →  [expr]
    """
    if type(expr).__name__ in _AND_NODE_NAMES:
        return decompose_and_tree(expr.l) + decompose_and_tree(expr.r)
    return [expr]


# ── Simplifying tree representation ──────────────────────────────────────────
#
# After every mutation or crossover, GeneticEngine's tree operators may produce
# bloated expressions: identical constants collapsed into a single literal,
# subexpressions like  (-1*(5.0 - x))  that reduce to  (x - 5.0), etc.
#
# If the GP evolves these unreduced trees, mutations and crossovers operate on
# spurious internal nodes (redundant constants, identity multiplications, …)
# instead of the expression's semantically meaningful structure.  Two
# expressions that are algebraically identical get different fitness values
# because count_nodes returns different sizes, and the duplicate search_log
# entries waste evaluations.
#
# The fix: after each genetic operator, call fold_constants on the new tree and
# restore the gengy_ metadata that GeneticEngine's mutation / crossover code
# reads when it descends into children.
#
# _rewrap_folded(expr) traverses the simplified expression tree bottom-up and:
#   • sets gengy_init_values  — the constructor args, used by mutate() to
#     iterate over children when selecting a sub-node to mutate.
#   • sets gengy_synthesis_context — the LocalSynthesisContext used to build
#     a replacement node of the same type at the same depth.
# relabel_nodes_of_trees() is then called to (re-)compute the remaining
# metadata (gengy_nodes, gengy_weighted_nodes, gengy_types_this_way, etc.).
#
# FoldingTreeRepresentation is a drop-in subclass of TreeBasedRepresentation
# that wraps mutate() and crossover() with this simplification step.
# FoldingSimpleGP replaces SimpleGP's representation factory to use it.

def _rewrap_folded(expr: Expression, depth: int = 0) -> None:
    """Attach gengy_ metadata to all nodes in a fold_constants-simplified tree.

    Operates in-place (mutates expr's attributes).  Must be called before
    relabel_nodes_of_trees so that the tree is consistent for GeneticEngine.
    """
    if not dataclasses.is_dataclass(expr):
        return
    children = []
    for f in dataclasses.fields(expr):
        child = getattr(expr, f.name)
        if isinstance(child, Expression):
            _rewrap_folded(child, depth + 1)
        children.append(child)
    expr.gengy_init_values    = children
    expr.gengy_synthesis_context = LocalSynthesisContext(
        depth=depth, nodes=0, expansions=0, dependent_values={}, parent_values=[],
    )


class FoldingTreeRepresentation(TreeBasedRepresentation):
    """TreeBasedRepresentation that simplifies trees after each operator.

    Calls fold_constants on every offspring produced by mutate() or
    crossover(), then re-attaches the gengy_ metadata GeneticEngine needs.
    The result is a smaller, canonical tree that future operators can work on
    more meaningfully — semantically equivalent bloat is eliminated before the
    next generation.
    """

    def _simplify(self, genotype: Expression) -> Expression:
        simplified = fold_constants(genotype)
        _rewrap_folded(simplified)
        relabel_nodes_of_trees(simplified, self.grammar)
        return simplified

    def mutate(self, random, genotype, **kwargs):
        return self._simplify(super().mutate(random, genotype, **kwargs))

    def crossover(self, random, parent1, parent2, **kwargs):
        c1, c2 = super().crossover(random, parent1, parent2, **kwargs)
        return self._simplify(c1), self._simplify(c2)


class FoldingSimpleGP(SimpleGP):
    """SimpleGP that uses FoldingTreeRepresentation instead of the default."""

    def process_representation(self, representation, grammar, max_depth):
        if representation == "treebased":
            decider = MaxDepthDecider(self.random, grammar, max_depth)
            return FoldingTreeRepresentation(grammar=grammar, decider=decider)
        return super().process_representation(representation, grammar, max_depth)


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


def compute_ilp_metrics(numpy_expr_str: str, X: np.ndarray, y: np.ndarray,
                        min_recall_pct: float = 5.0) -> dict:
    """Compute Recall, Precision, Coverage as defined in Gonçalves et al. (TACAS 2025).

    A GP expression produces a continuous score.  We sweep percentile thresholds
    and pick the one that maximises Precision subject to Recall >= min_recall_pct,
    mirroring the ILP paper's filtering strategy.

    TP: proxy fires AND protected = 1
    FP: proxy fires AND protected = 0
    FN: proxy silent AND protected = 1

    Recall   = TP / (TP + FN)   — how much of the protected group is covered
    Precision = TP / (TP + FP)  — accuracy when the proxy fires
    Coverage  = TP / N           — share of the whole dataset that is a true positive
    """
    try:
        scores = np.asarray(forward_dataset(numpy_expr_str, X),
                            dtype=np.float64).ravel()
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

        # Orient so higher score → more likely protected
        inverted = False
        if np.std(scores) > 1e-10:
            raw_auc = roc_auc_score(y, scores)
            if raw_auc < 0.5:
                scores   = -scores
                inverted = True

        N = len(y)
        best: dict = {"recall": 0.0, "precision": 0.0, "coverage": 0.0,
                      "threshold": 0.0, "inverted": inverted}

        for pct in np.arange(5, 95, 5):
            t = float(np.percentile(scores, pct))
            fired = scores > t
            tp = int(np.sum(fired & (y == 1)))
            fp = int(np.sum(fired & (y == 0)))
            fn = int(np.sum(~fired & (y == 1)))
            recall    = tp / (tp + fn)  if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp)  if (tp + fp) > 0 else 0.0
            coverage  = tp / N
            if recall * 100 >= min_recall_pct and precision > best["precision"]:
                best = {
                    "recall":    round(recall    * 100, 1),
                    "precision": round(precision * 100, 1),
                    "coverage":  round(coverage  * 100, 1),
                    "threshold": round(t, 4),
                    "inverted":  inverted,
                }
        return best
    except Exception:
        return {"recall": 0.0, "precision": 0.0, "coverage": 0.0,
                "threshold": 0.0, "inverted": False}


def compute_proxy_metrics(numpy_expr_str: str,
                          X: np.ndarray, y: np.ndarray) -> dict:
    """Compute a full set of proxy-quality metrics for a single expression.

    Combines:
      — AUC / AP / NMI / Cohen's d : continuous ranking metrics
      — Recall / Precision / Coverage : threshold-based metrics from
        Gonçalves et al. (TACAS 2025), reflecting the paper's definition of
        proxy quality; precision is the primary indicator of a good proxy.
    """
    try:
        scores = np.asarray(forward_dataset(numpy_expr_str, X),
                            dtype=np.float64).ravel()
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        scores = np.clip(scores, -1e6, 1e6)

        if np.std(scores) < 1e-10:
            ilp = compute_ilp_metrics(numpy_expr_str, X, y)
            return {"auc": 0.5, "ap": float(y.mean()), "nmi": 0.0, "cohens_d": 0.0,
                    **ilp}

        raw_auc = roc_auc_score(y, scores)
        if raw_auc < 0.5:
            scores = -scores
            raw_auc = 1.0 - raw_auc

        ap  = average_precision_score(y, scores)

        binary = (scores > np.median(scores)).astype(int)
        nmi    = normalized_mutual_info_score(y, binary)

        m1 = scores[y == 1]
        m0 = scores[y == 0]
        if len(m1) > 0 and len(m0) > 0:
            pooled_std = np.sqrt((np.var(m1) + np.var(m0)) / 2.0)
            cohens_d   = abs(np.mean(m1) - np.mean(m0)) / (pooled_std + 1e-10)
        else:
            cohens_d = 0.0

        ilp = compute_ilp_metrics(numpy_expr_str, X, y)

        return {
            "auc":       round(raw_auc, 4),
            "ap":        round(ap, 4),
            "nmi":       round(nmi, 4),
            "cohens_d":  round(float(cohens_d), 4),
            "recall":    ilp["recall"],
            "precision": ilp["precision"],
            "coverage":  ilp["coverage"],
            "threshold": ilp["threshold"],
            "inverted":  ilp["inverted"],
        }
    except Exception:
        return {"auc": 0.5, "ap": 0.0, "nmi": 0.0, "cohens_d": 0.0,
                "recall": 0.0, "precision": 0.0, "coverage": 0.0,
                "threshold": 0.0, "inverted": False}


def evaluate_and_breakdown(
    expr,
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict] | None:
    """Per-component metrics for a top-level AND expression.

    Returns None when the expression has no AND at its root.
    For each atomic component returns a dict with keys:
      'expression', 'auc', 'precision', 'recall', 'coverage',
      'attribution_pct' — the percentage of the expression's total
      discriminatory power (excess precision above prevalence) that
      this clause is responsible for.

    Attribution formula:
        excess_i    = max(0, prec_i - prevalence)
        weight_i    = excess_i / Σⱼ excess_j
        attribution = weight_i × 100  (percentage)

    A clause whose precision equals the population prevalence (random
    firing) has excess=0 and therefore 0% attribution — it is a
    hitchhiker that contributes nothing discriminatory.
    """
    components = decompose_and_tree(expr)
    if len(components) < 2:
        return None

    prevalence = float(np.mean(y))

    raw_metrics = []
    for comp in components:
        comp_str = _prettify_expr(comp.to_sympy())
        metrics  = compute_proxy_metrics(comp.to_numpy(), X, y)
        raw_metrics.append((comp_str, metrics))

    # Attribution weights: excess precision above the random baseline
    precisions   = [m["precision"] for _, m in raw_metrics]
    excesses     = [max(0.0, p - prevalence * 100) for p in precisions]
    total_excess = sum(excesses)

    breakdown = []
    for i, (comp_str, metrics) in enumerate(raw_metrics):
        attribution = (excesses[i] / total_excess * 100.0
                       if total_excess > 0.0 else 100.0 / len(raw_metrics))
        breakdown.append({
            "expression":     comp_str,
            "auc":            metrics["auc"],
            "precision":      metrics["precision"],
            "recall":         metrics["recall"],
            "coverage":       metrics["coverage"],
            "attribution_pct": round(attribution, 1),
        })
    return breakdown


def compute_firing_profile(
    numpy_expr_str: str,
    X: np.ndarray,
    feature_names: list[str],
    threshold: float,
    label_encodings: dict[str, list[str]],
    expression_features: list[str] | None = None,
    min_lift: float = 1.1,
) -> dict[str, list[dict]]:
    """Compute how each feature in the expression behaves when it fires.

    For categorical features (in label_encodings): shows which category values
    are most over-represented in fired rows (top-3 by lift, min 1 instance).

    For continuous features (not in label_encodings): always shows high/low
    direction based on median comparison (robust to skewed distributions).

    Only analyses features listed in expression_features.

    Returns {feature_name: [{"label": str, "tooltip": str, "kind": "cat"|"cont"}, ...]}
    """
    try:
        scores = np.asarray(forward_dataset(numpy_expr_str, X),
                            dtype=np.float64).ravel()
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        fired_mask = scores > threshold
        n_fired = int(fired_mask.sum())
        if n_fired == 0:
            return {}

        n_total = len(scores)
        profile: dict[str, list[dict]] = {}

        relevant = set(expression_features) if expression_features else set(label_encodings)

        for feat in relevant:
            if feat not in feature_names:
                continue
            idx = feature_names.index(feat)

            if feat in label_encodings:
                # ── Categorical feature ──────────────────────────────────────
                col        = X[:, idx].astype(int)
                fired_col  = col[fired_mask]
                categories = label_encodings[feat]

                # Check if the encoding is ordinal (all category labels are
                # pure numbers, e.g. Age: "17","18",...,"90").  For ordinal
                # features, "high / low" is meaningful.  For nominal features
                # (Relationship, Occupation, Country, …) it is not — we must
                # always show the actual category names.
                is_ordinal = all(c.lstrip("-").isdigit() for c in categories)

                if is_ordinal:
                    # Treat like a continuous feature: show direction + example
                    overall_med = float(np.median(col))
                    fired_med   = float(np.median(fired_col))
                    direction   = "high" if fired_med >= overall_med else "low"
                    arrow       = "↑" if fired_med >= overall_med else "↓"
                    med_idx     = int(round(fired_med))
                    med_name    = (categories[med_idx]
                                   if 0 <= med_idx < len(categories)
                                   else str(med_idx))
                    ov_idx      = int(round(overall_med))
                    ov_name     = (categories[ov_idx]
                                   if 0 <= ov_idx < len(categories)
                                   else str(ov_idx))
                    profile[feat] = [{
                        "label":   f"{arrow} {direction} (e.g. {med_name})",
                        "tooltip": (f"Median when firing: {med_name} "
                                    f"vs overall median: {ov_name}"),
                        "kind":    "cont",
                    }]
                    continue

                # Nominal categorical: always show specific category names.
                # Rank all categories that appear in fired rows by lift and
                # show the top 3 — never replace with "high/low".
                entries: list[dict] = []
                for cat_idx, cat_name in enumerate(categories):
                    n_fired_cat = int((fired_col == cat_idx).sum())
                    if n_fired_cat == 0:
                        continue
                    base_rate  = float((col == cat_idx).sum()) / n_total
                    fired_rate = float(n_fired_cat) / n_fired
                    lift = fired_rate / base_rate if base_rate > 1e-9 else 0.0
                    if lift >= min_lift:
                        entries.append({
                            "label":   cat_name,
                            "tooltip": (f"{n_fired_cat} fired rows "
                                        f"({fired_rate*100:.1f}% of fires "
                                        f"vs {base_rate*100:.1f}% overall)"),
                            "kind":    "cat",
                            "_sort":   lift,
                        })

                if entries:
                    entries.sort(key=lambda e: -e["_sort"])
                    for e in entries:
                        e.pop("_sort")
                    profile[feat] = entries[:3]
                elif n_fired > 0:
                    # No category cleared lift threshold — show the single most
                    # common category in fired rows so there is always a label
                    from collections import Counter
                    top_idx, top_count = Counter(fired_col.tolist()).most_common(1)[0]
                    cat_name   = (categories[top_idx]
                                  if 0 <= top_idx < len(categories) else str(top_idx))
                    base_rate  = float((col == top_idx).sum()) / n_total
                    fired_rate = float(top_count) / n_fired
                    profile[feat] = [{
                        "label":   cat_name,
                        "tooltip": (f"{top_count} fired rows "
                                    f"({fired_rate*100:.1f}% of fires "
                                    f"vs {base_rate*100:.1f}% overall)"),
                        "kind":    "cat",
                    }]

            else:
                # ── Continuous feature ───────────────────────────────────────
                # Use median (robust to skew/outliers like Capital_Gain)
                col = X[:, idx].astype(float)
                col = np.nan_to_num(col, nan=0.0)
                overall_med = float(np.median(col))
                fired_med   = float(np.median(col[fired_mask]))
                overall_p25 = float(np.percentile(col, 25))
                overall_p75 = float(np.percentile(col, 75))
                iqr = overall_p75 - overall_p25

                direction = "high" if fired_med >= overall_med else "low"
                arrow     = "↑" if fired_med >= overall_med else "↓"

                # Only show if there is a meaningful shift (> 10% of IQR,
                # or any difference when IQR is near zero)
                shift = abs(fired_med - overall_med)
                if iqr < 1e-9 or shift / iqr >= 0.1:
                    profile[feat] = [{
                        "label":   f"{arrow} {direction}",
                        "tooltip": (f"Median {fired_med:.4g} when fires "
                                    f"vs {overall_med:.4g} overall"),
                        "kind":    "cont",
                    }]

        return profile
    except Exception:
        return {}


def make_fitness_fn(X_train: np.ndarray, y_train: np.ndarray,
                    prec_weight: float = 1.0,
                    rec_weight:  float = 1.0,
                    cov_weight:  float = 1.0,
                    complexity_penalty: float = 0.03,
                    search_log: list | None = None):
    """Return a fitness function (Expression -> float) for GP.

    fitness = prec_weight * precision
            + rec_weight  * recall
            + cov_weight  * coverage
            - complexity_penalty * (nodes / max_nodes)

    Precision, recall and coverage are computed at a fixed threshold:
      * Boolean expressions (.astype(np.float64) in numpy) -> threshold 0.5
      * Continuous expressions (arithmetic)                 -> threshold = median

    If *search_log* is provided, every unique expression evaluated is appended.

    Tabu: when a proxy-quality expression is found, its top-level AND components
    are added to a tabu set.  Future expressions whose top-level AND components
    overlap with the tabu set return -1.0, steering the search toward genuinely
    different areas.  Structural reuse that is not an AND extension (e.g. a found
    proxy appearing as an IF condition) is caught by _remove_subsumed_proxies in
    post-processing rather than here, so the search can still explore those paths.
    """
    max_nodes  = 63.0
    start_time = time.time()
    seen_exprs: set[str] = set()
    _y_pos     = y_train == 1
    _y_neg     = y_train == 0
    expr_metrics_cache: dict[str, tuple[float, float, float, bytes]] = {}
    _tabu: set[str] = set()          # sympy strings of archived AND components
    _prec_floor = PROXY_MIN_PRECISION / 100.0
    _rec_floor  = PROXY_MIN_RECALL   / 100.0
    seen_behaviors: set[bytes] = set()

    def _cached_metrics(
        numpy_str: str,
        cache: dict[str, tuple[float, float, float, bytes]],
    ) -> tuple[float, float, float, bytes]:
        """Return (precision, recall, coverage, behavior_key) with memoization.

        threshold: 0.5 for boolean expressions, 80th-percentile for continuous.
        The 80th-percentile threshold better tracks the ILP-optimal threshold used
        in final evaluation (which favours high precision over coverage).
        behavior_key is a packed bitstring of the firing vector, used to detect
        semantically identical expressions regardless of syntactic form.
        """
        cached = cache.get(numpy_str)
        if cached is not None:
            return cached

        scores = np.asarray(forward_dataset(numpy_str, X_train),
                            dtype=np.float64).ravel()
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        _is_bool = numpy_str.strip().endswith(".astype(np.float64)")
        threshold = 0.5 if _is_bool else float(np.percentile(scores, 80))
        fired = scores > threshold
        bkey = np.packbits(fired).tobytes()
        tp = float(np.sum(fired & _y_pos))
        fp = float(np.sum(fired & _y_neg))
        fn = float(np.sum(~fired & _y_pos))
        n_total = float(len(y_train))
        metrics = (
            tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            tp / n_total,
            bkey,
        )
        cache[numpy_str] = metrics
        return metrics

    def fitness(expr: Expression) -> float:
        try:
            # ── Tabu check (AND-component level) ─────────────────────────────
            # Decompose the candidate into its top-level AND components and
            # reject it if any component matches a previously found proxy part.
            # This steers the search toward genuinely new areas without blocking
            # non-AND structural reuse (e.g. IF(proxy_cond, ...)), which is
            # cleaned up by _remove_subsumed_proxies in post-processing.
            if _tabu:
                for comp in decompose_and_tree(expr):
                    s = comp.to_sympy()
                    if s in _tabu:
                        return -1.0

            nodes = count_nodes(expr)
            numpy_str = expr.to_numpy()
            precision, recall, coverage, bkey = _cached_metrics(numpy_str, expr_metrics_cache)

            # Behavioral deduplication: block any expression whose firing pattern
            # over the training set is identical to one already evaluated.
            if bkey in seen_behaviors:
                return -1.0
            seen_behaviors.add(bkey)

            fit = (prec_weight * precision
                   + rec_weight  * recall
                   + cov_weight  * coverage
                   - complexity_penalty * (nodes / max_nodes))

            # ── Archive proxy-quality expressions ────────────────────────────
            if precision >= _prec_floor and recall >= _rec_floor:
                for comp in decompose_and_tree(expr):
                    s = comp.to_sympy()
                    _tabu.add(s)
                    # Block the complementary form too so NOT(A != B) is caught
                    # when (A == B) is already tabu, and vice versa.
                    if " == " in s:
                        _tabu.add(s.replace(" == ", " != ", 1))
                    elif " != " in s:
                        _tabu.add(s.replace(" != ", " == ", 1))

            if search_log is not None:
                folded   = fold_constants(expr)
                expr_str = str(folded)
                if expr_str not in seen_exprs:
                    seen_exprs.add(expr_str)
                    search_log.append({
                        "elapsed_s":  round(time.time() - start_time, 2),
                        "fitness":    round(fit,       4),
                        "precision":  round(precision, 4),
                        "recall":     round(recall,    4),
                        "coverage":   round(coverage,  4),
                        "nodes":      count_nodes(folded),
                        "expression": expr_str,
                        "numpy_expr": folded.to_numpy(),
                    })

            return fit
        except Exception:
            return 0.0
    return fitness


def make_auc_fitness_fn(X_train: np.ndarray, y_train: np.ndarray,
                        complexity_penalty: float = 0.03,
                        search_log: list | None = None):
    """Return an AUC-based fitness function for the continuous grammar.

    fitness = AUC(expr(X_train), y_train) - complexity_penalty * (nodes / max_nodes)

    Unlike make_fitness_fn, no binarization is applied: the raw continuous
    output of the expression is treated directly as a discriminant score.
    AUC < 0.5 is automatically reflected (score inversion), so GP can discover
    both positive and negative correlations.

    Logs the same fields as make_fitness_fn so that extract_partial_proxies
    can process results uniformly (precision/recall/coverage are left at 0).
    """
    max_nodes  = 63.0
    start_time = time.time()
    seen_exprs: set[str] = set()
    seen_behaviors: set[bytes] = set()

    def fitness(expr: Expression) -> float:
        try:
            numpy_str = expr.to_numpy()
            scores    = np.asarray(forward_dataset(numpy_str, X_train),
                                   dtype=np.float64).ravel()
            scores    = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

            if len(np.unique(scores)) < 2:
                return 0.0

            # Behavioral deduplication on the raw score vector (median threshold).
            bkey = np.packbits(scores > float(np.median(scores))).tobytes()
            if bkey in seen_behaviors:
                return -1.0
            seen_behaviors.add(bkey)

            auc = safe_auc(y_train, scores)
            if auc < 0.5:          # reflect: GP can find negatively-correlated exprs
                auc = 1.0 - auc

            nodes = count_nodes(expr)
            fit   = auc - complexity_penalty * (nodes / max_nodes)

            if search_log is not None:
                folded   = fold_constants(expr)
                expr_str = str(folded)
                if expr_str not in seen_exprs:
                    seen_exprs.add(expr_str)
                    search_log.append({
                        "elapsed_s":  round(time.time() - start_time, 2),
                        "fitness":    round(fit,  4),
                        "auc":        round(auc,  4),
                        "nodes":      count_nodes(folded),
                        "expression": expr_str,
                        "numpy_expr": folded.to_numpy(),
                        # prec/recall/coverage not applicable in AUC mode
                        "precision":  0.0,
                        "recall":     0.0,
                        "coverage":   0.0,
                    })

            return fit
        except Exception:
            return 0.0

    return fitness


def evaluate_expression(expr: Expression, X_data: np.ndarray,
                        y_true: np.ndarray) -> float:
    """Evaluate an expression on arbitrary data and return AUC."""
    try:
        y_pred = forward_dataset(expr.to_numpy(), X_data)
        return safe_auc(y_true, y_pred)
    except Exception:
        return 0.5


# ── Partial proxy extraction ──────────────────────────────────────────────────

_SIMPLE_NEQ_RE = re.compile(r'^NOT\(([^()]+) != ([^()]+)\)$')
_SIMPLE_EQ_RE  = re.compile(r'^NOT\(([^()]+) == ([^()]+)\)$')
_PAREN_NEQ_RE  = re.compile(r'^\(([^()]+) != ([^()]+)\)$')
_PAREN_EQ_RE   = re.compile(r'^\(([^()]+) == ([^()]+)\)$')


def _negate_display(pexpr: str) -> str:
    """Return NOT(pexpr) simplified to its cleanest algebraic equivalent.

    Handles four cases, applied in order:
      1. NOT(A != B)             →  A == B          (no inner parens)
      2. NOT(A == B)             →  A != B          (no inner parens)
      3. (A != B)  / (A == B)   →  (A == B) / (A != B)   (with parens)
      4. De Morgan: (A op1 B) AND (C op2 D)  →  neg(A op1 B) OR neg(C op2 D)
         where each part reduces without NOT.
    Falls back to NOT(pexpr) when no simplification applies.
    """
    # 1 & 2: simple paren-free form
    m = _SIMPLE_NEQ_RE.match(pexpr)
    if m:
        return f"{m.group(1)} == {m.group(2)}"
    m = _SIMPLE_EQ_RE.match(pexpr)
    if m:
        return f"{m.group(1)} != {m.group(2)}"
    # 3: atomic with surrounding parens  (A == B) / (A != B)
    m = _PAREN_NEQ_RE.match(pexpr)
    if m:
        return f"({m.group(1)} == {m.group(2)})"
    m = _PAREN_EQ_RE.match(pexpr)
    if m:
        return f"({m.group(1)} != {m.group(2)})"
    # 4: De Morgan over top-level AND — only when every part negates cleanly
    parts = _split_top_level_and(pexpr)
    if len(parts) > 1:
        negated = [_negate_display(p.strip()) for p in parts]
        if all(not n.startswith("NOT(") for n in negated):
            return " OR ".join(negated)
    return f"NOT({pexpr})"

def _pareto_front(records: list[dict]) -> list[dict]:
    """Return Pareto-optimal records: maximise 'auc', minimise 'nodes'."""
    front = []
    for r in records:
        dominated = any(
            o["auc"] >= r["auc"] and o["nodes"] <= r["nodes"]
            and (o["auc"] > r["auc"] or o["nodes"] < r["nodes"])
            for o in records
        )
        if not dominated:
            front.append(r)
    return front


def _split_top_level_and(expr_str: str) -> list[str]:
    """Split `expr_str` on top-level ' AND ' tokens (depth-0 only)."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    i = 0
    while i < len(expr_str):
        ch = expr_str[i]
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
        elif ch == ")":
            depth -= 1
            buf.append(ch)
            i += 1
        elif depth == 0 and expr_str[i : i + 5] == " AND ":
            parts.append("".join(buf).strip())
            buf = []
            i += 5
        else:
            buf.append(ch)
            i += 1
    if buf:
        parts.append("".join(buf).strip())
    return parts or [expr_str]


def _remove_subsumed_proxies(proxies: list[dict]) -> list[dict]:
    """Drop proxies that reuse a sub-expression from a simpler proxy.

    Sorts by (component count, node count) so simpler expressions are
    processed first.  A proxy is subsumed if any already-found component
    string appears as a substring of the candidate expression — this catches
    both AND-extensions and structural reuse (e.g. a known proxy appearing
    as the condition of an IF expression).
    """
    sorted_p = sorted(
        proxies,
        key=lambda r: (len(_split_top_level_and(r["expression"])), r["nodes"]),
    )
    found_components: set[str] = set()
    kept: list[dict] = []
    for proxy in sorted_p:
        expr = proxy["expression"]
        if any(comp in expr for comp in found_components):
            continue
        kept.append(proxy)
        found_components.update(_split_top_level_and(expr))
    return kept


def extract_partial_proxies(
    search_log:          list[dict],
    X_train:             np.ndarray,
    y_train:             np.ndarray,
    feature_names:       list[str],
    min_auc:             float = PROXY_MIN_AUC,
    min_precision:       float = PROXY_MIN_PRECISION,
    min_precision_floor: float = PROXY_MIN_PRECISION_FLOOR,
    min_recall:          float = PROXY_MIN_RECALL,
    top_n:               int   = 25,
) -> list[dict]:
    """Extract all proxy expressions found anywhere in a GP search log.

    All expressions also require:
      1. precision >= min_precision_floor  (universal hard floor; this is the
         main quality gate for both continuous and logical expressions).
      2. recall >= min_recall             (non-trivial group coverage).

    Candidate selection scans *every* expression in the search log so that
    short, high-precision rules (e.g. `relationship == Husband`) are not lost:
      1. Pre-filter: fitness >= 0.
         For continuous (AUC-fitness) logs, precision is stored as 0.0 so
         the precision-based pre-filter is skipped for those entries;
         for logical (prec/rec/cov-fitness) logs, logged precision must be
         >= min_precision_floor / 2  (half the floor gives slack for the
         fixed-threshold approximation used during search vs. the ILP threshold
         used in full evaluation).
      2. Sort by fitness descending; keep at most PROXY_MAX_EVAL_CANDIDATES.
      3. Compute full metrics (AUC + ILP precision/recall/coverage) for each.
      4. Retain every expression satisfying the precision-floor + recall criteria.

    Returns up to *top_n* results sorted by precision descending (then AUC,
    then node count as tie-breaker).
    """
    if not search_log:
        return []

    df = pd.DataFrame(search_log)
    if "fitness" in df.columns:
        df = df[df["fitness"] >= 0].drop_duplicates(subset="expression")
    else:
        df = df.drop_duplicates(subset="expression")
    if df.empty:
        return []

    # Quick pre-filter on logged precision.
    # Continuous (AUC-fitness) entries log precision=0.0 — keep them all for
    # full evaluation.  Logical entries filter by half the floor for slack.
    if "precision" in df.columns:
        quick_floor = (min_precision_floor / 100.0) / 2.0
        auc_mode    = df["precision"] == 0.0          # continuous / AUC-fitness
        df = df[auc_mode | (df["precision"] >= quick_floor)]
    if df.empty:
        return []

    # Sort by fitness descending and cap candidate set for performance.
    if "fitness" in df.columns:
        df = df.sort_values("fitness", ascending=False)
    records = df.head(PROXY_MAX_EVAL_CANDIDATES).to_dict("records")

    # Full metric evaluation + universal partial-proxy criterion check.
    # Collapse any candidates that simplify to the same final displayed proxy,
    # keeping the strongest representative so equivalent forms do not waste
    # space in CSV/HTML outputs.
    results_by_display: dict[str, dict] = {}
    for r in records:
        numpy_str  = r["numpy_expr"]
        features   = extract_features_from_expr(r["expression"], feature_names)
        tr         = compute_proxy_metrics(numpy_str, X_train, y_train)
        # Criterion 1: universal precision floor (both types).
        if tr["precision"] < min_precision_floor:
            continue

        # Criterion 2: non-trivial recall (both types).
        if tr["recall"] < min_recall:
            continue

        _is_boolean = numpy_str.strip().endswith(".astype(np.float64)")
        _inverted   = tr["inverted"]
        if _is_boolean:
            _pexpr = _prettify_expr(r["expression"])
            _display_expr = _negate_display(_pexpr) if _inverted else _pexpr
        else:
            thr = tr["threshold"]
            _pexpr = _prettify_expr(r["expression"])
            if _inverted:
                _display_expr = f"({_pexpr}) < {(-thr) or 0.0:.3g}"
            else:
                _display_expr = f"({_pexpr}) > {thr or 0.0:.3g}"

        candidate = {
            "expression":   _display_expr,
            "numpy_expr":   numpy_str,
            "nodes":        r["nodes"],
            "first_seen_s": r.get("elapsed_s", 0),
            "features":     ", ".join(features),
            "auc":          tr["auc"],
            "ap":           tr["ap"],
            "nmi":          tr["nmi"],
            "cohens_d":     tr["cohens_d"],
            "recall":       tr["recall"],
            "precision":    tr["precision"],
            "coverage":     tr["coverage"],
            "threshold":    tr["threshold"],
        }

        existing = results_by_display.get(_display_expr)
        if existing is None:
            results_by_display[_display_expr] = candidate
            continue

        # Prefer higher precision, then higher AUC, then simpler expression,
        # then earlier discovery time.
        cand_key = (
            candidate["precision"],
            candidate["auc"],
            -candidate["nodes"],
            -candidate["first_seen_s"],
        )
        existing_key = (
            existing["precision"],
            existing["auc"],
            -existing["nodes"],
            -existing["first_seen_s"],
        )
        if cand_key > existing_key:
            results_by_display[_display_expr] = candidate

    results = list(results_by_display.values())
    results = _remove_subsumed_proxies(results)
    results.sort(key=lambda r: (-r["precision"], -r["auc"], r["nodes"]))
    return results[:top_n]


# ── Single GP run ─────────────────────────────────────────────────────────────

def run_gp(dataset: dict, protected_attr: str, grammar,
           time_budget: float, population_size: int, max_depth: int,
           seed: int, prec_weight: float = 1.0,
           rec_weight: float = 1.0,
           cov_weight: float = 1.0,
           complexity_penalty: float = 0.03,
           collect_partials: bool = True,
           min_partial_auc:      float = PROXY_MIN_AUC,
           min_partial_precision: float = PROXY_MIN_PRECISION_FLOOR,
           min_partial_prec_floor: float = PROXY_MIN_PRECISION_FLOOR,
           min_partial_recall:    float = PROXY_MIN_RECALL,
           top_n_partials: int = 500,   # keep a broader final candidate set for CSV/figures
           fitness_mode: str = "prec_rec_cov") -> dict:
    """Run one GP search for a single dataset + protected attribute.

    Returns a dict with the final best expression (with multi-metric evaluation)
    plus a 'partial_proxies' list of notable expressions found during the search.

    fitness_mode:
      "prec_rec_cov"  — standard precision/recall/coverage fitness (default)
      "auc"           — direct AUC maximisation (for continuous grammar)
    """
    X_train = dataset["X_train"]
    y_train = dataset["pa_train"][protected_attr]

    search_log: list[dict] = [] if collect_partials else None  # type: ignore[assignment]

    if fitness_mode == "auc":
        fitness_fn = make_auc_fitness_fn(X_train, y_train,
                                         complexity_penalty=complexity_penalty,
                                         search_log=search_log)
    else:
        fitness_fn = make_fitness_fn(X_train, y_train,
                                     prec_weight=prec_weight,
                                     rec_weight=rec_weight,
                                     cov_weight=cov_weight,
                                     complexity_penalty=complexity_penalty,
                                     search_log=search_log)

    elitism = min(10, max(1, population_size // 5))
    novelty = min(10, max(1, population_size // 5))
    if elitism + novelty >= population_size:
        elitism = max(1, population_size // 3)
        novelty = max(0, population_size - elitism - 1)

    gp = FoldingSimpleGP(
        grammar=grammar,
        fitness_function=fitness_fn,
        minimize=False,
        max_time=time_budget,
        population_size=population_size,
        elitism=elitism,
        novelty=novelty,
        max_depth=max_depth,
        seed=seed,
    )

    start  = time.time()
    result = gp.search()
    elapsed = time.time() - start

    if not result:
        return {
            "dataset":         dataset["label"],
            "protected_attr":  protected_attr,
            "expression":      "FAILED",
            "numpy_expr":      "",
            "nodes":           0,
            "auc":       0.5, "ap":       0.0,
            "nmi":       0.0, "cohens_d": 0.0,
            "recall":    0.0, "precision": 0.0, "coverage": 0.0,
            "elapsed_s":       round(elapsed, 1),
            "partial_proxies":    [],
            "near_miss_proxies": [],
            "search_log":        search_log or [],
        }

    best_expr  = fold_constants(result[0].get_phenotype())
    numpy_str  = best_expr.to_numpy()
    nodes      = count_nodes(best_expr)
    tr         = compute_proxy_metrics(numpy_str, X_train, y_train)

    # Extract partial proxies from the full search log and ensure the final
    # winner is included when it satisfies the same reporting criteria.
    partials: list[dict] = []
    if search_log:
        partials = extract_partial_proxies(
            search_log, X_train, y_train,
            dataset["feature_names"],
            min_auc=min_partial_auc,
            min_precision=min_partial_precision,
            min_precision_floor=min_partial_prec_floor,
            min_recall=min_partial_recall,
            top_n=top_n_partials,
        )

    # Build a display expression that makes the firing condition explicit.
    # For boolean-rooted grammars (typed/CatCond) the numpy string always
    # contains ".astype(np.float64)" because every comparison node emits it.
    # For continuous grammars (arithmetic) no such cast appears.
    # In the continuous case, append "> threshold" so readers immediately
    # see the cut-off: "(score_expr) > 2.43" is unambiguous.
    raw_expr_str = _prettify_expr(best_expr.to_sympy())
    tr_threshold = tr["threshold"]
    tr_inverted  = tr["inverted"]
    _is_boolean  = numpy_str.strip().endswith(".astype(np.float64)")
    if _is_boolean:
        # Boolean expression: inversion means the condition fires for the
        # wrong group — negate it so the display matches what actually fires.
        display_expr = f"NOT({raw_expr_str})" if tr_inverted else raw_expr_str
    else:
        # Continuous expression: no extra wrapping parens — raw_expr_str from
        # to_sympy() already has its own outer parens from the outermost operator.
        if tr_inverted:
            display_expr = f"{raw_expr_str} < {(-tr_threshold) or 0.0:.4g}"
        else:
            display_expr = f"{raw_expr_str} > {tr_threshold or 0.0:.4g}"

    and_components = evaluate_and_breakdown(best_expr, X_train, y_train)

    if collect_partials:
        winner_partial = {
            "expression":   display_expr,
            "numpy_expr":   numpy_str,
            "nodes":        nodes,
            "first_seen_s": round(elapsed, 1),
            "features":     ", ".join(extract_features_from_expr(raw_expr_str, dataset["feature_names"])),
            "auc":          tr["auc"],
            "ap":           tr["ap"],
            "nmi":          tr["nmi"],
            "cohens_d":     tr["cohens_d"],
            "recall":       tr["recall"],
            "precision":    tr["precision"],
            "coverage":     tr["coverage"],
            "threshold":    tr["threshold"],
        }
        if (
            tr["precision"] >= min_partial_prec_floor
            and tr["recall"] >= min_partial_recall
            and all(p["expression"] != display_expr for p in partials)
        ):
            partials.append(winner_partial)
            partials.sort(key=lambda r: (-r["precision"], -r["auc"], r["nodes"]))
            partials = partials[:top_n_partials]

    # ── Near-miss proxies: best sub-threshold expressions ─────────────────────
    # Expressions that came closest to qualifying (precision ∈ [20 %, threshold)
    # and recall ≥ 5 %) but didn't cross the proxy bar.  Useful for reporting
    # "best effort found" when no full proxy exists for a group.
    near_miss: list[dict] = []
    if search_log:
        _near_all = extract_partial_proxies(
            search_log, X_train, y_train,
            dataset["feature_names"],
            min_auc=min_partial_auc,
            min_precision=0.0,
            min_precision_floor=20.0,
            min_recall=5.0,
            top_n=15,
        )
        _proxy_exprs = {p["expression"] for p in partials}
        near_miss = [
            n for n in _near_all
            if n["expression"] not in _proxy_exprs
            and n["precision"] < min_partial_prec_floor
        ][:5]

    return {
        "dataset":         dataset["label"],
        "protected_attr":  protected_attr,
        "expression":      display_expr,
        "numpy_expr":      numpy_str,
        "nodes":           nodes,
        # Continuous metrics
        "auc":       tr["auc"],  "ap":       tr["ap"],
        "nmi":       tr["nmi"],  "cohens_d": tr["cohens_d"],
        # Paper metrics (Recall / Precision / Coverage at optimal threshold)
        "recall":    tr["recall"],    "precision": tr["precision"],
        "coverage":  tr["coverage"],
        "elapsed_s":       round(elapsed, 1),
        "partial_proxies": partials,
        "near_miss_proxies": near_miss,
        "search_log":      search_log or [],
        "and_components":  and_components,   # None, or list of per-component dicts
    }


# ── Baselines ─────────────────────────────────────────────────────────────────

def load_baselines(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_dir / "individual_proxy_baselines.csv")


def extract_features_from_expr(expression_str: str,
                               all_features: list[str]) -> list[str]:
    """Return the list of feature names that appear in an expression string."""
    import re
    # Match whole words only to avoid partial matches
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
    print("GP PROXY DISCOVERY - SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Dataset':<16} {'Attr':<8} {'AUC':>10} {'Baseline':>10} {'Delta':>8}  Expression")
    print(f"{'-' * 16} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}  {'-' * 30}")

    for r in results:
        delta = r["auc"] - r["best_baseline_auc"]
        marker = " *" if delta > 0 else ""
        expr_short = r["expression"][:60] + ("..." if len(r["expression"]) > 60 else "")
        print(f"{r['dataset']:<16} {r['protected_attr']:<8} "
              f"{r['auc']:>10.4f} "
              f"{r['best_baseline_auc']:>10.4f} {delta:>+8.4f}{marker}  {expr_short}")

    print(f"\n* = GP exceeded best single-feature baseline among features in expression")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover multi-attribute proxy expressions via GP")
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
    parser.add_argument("--prec-weight", type=float, default=1.0,
                        help="Precision weight in GP fitness (default: 1.0)")
    parser.add_argument("--rec-weight", type=float, default=1.0,
                        help="Recall weight in GP fitness (default: 1.0)")
    parser.add_argument("--cov-weight", type=float, default=1.0,
                        help="Coverage weight in GP fitness (default: 1.0)")
    parser.add_argument("--complexity-penalty", type=float, default=0.03,
                        help="Penalty per normalised node count (default: 0.03)")
    parser.add_argument("--grammar",
                        choices=["arithmetic", "extended", "typed", "all"],
                        default="typed",
                        help="Grammar mode: 'arithmetic' (original +,-,*,/), "
                             "'extended' (+ comparisons, conditionals, abs/max/min), "
                             "'typed' (strict NumExpr/CatCond split, boolean-rooted — "
                             "every expression is a readable predicate, recommended), or "
                             "'all' (run arithmetic then typed, combines results). "
                             "Default: typed")
    # Proxy reporting criteria (see PROXY_* constants for justification)
    parser.add_argument("--proxy-min-auc", type=float, default=PROXY_MIN_AUC,
                        help=f"Minimum AUC for partial proxy quality bar "
                             f"(default: {PROXY_MIN_AUC}; OR'd with --proxy-min-precision)")
    parser.add_argument("--proxy-min-precision", type=float, default=PROXY_MIN_PRECISION,
                        help=f"Minimum PPV %% for partial proxy quality bar "
                             f"(default: {PROXY_MIN_PRECISION}; OR'd with --proxy-min-auc)")
    parser.add_argument("--proxy-min-precision-floor", type=float,
                        default=PROXY_MIN_PRECISION_FLOOR,
                        help=f"Absolute minimum PPV %% (hard floor, no AUC escape; "
                             f"default: {PROXY_MIN_PRECISION_FLOOR})")
    parser.add_argument("--proxy-min-recall", type=float, default=PROXY_MIN_RECALL,
                        help=f"Minimum recall %% (non-trivial group coverage; "
                             f"default: {PROXY_MIN_RECALL})")
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
<title>MA-Proxies &mdash; GP Proxy Discovery Results</title>
<style>{HTML_CSS}</style>
</head>
<body>

<h1>MA-Proxies &mdash; GP Proxy Discovery Results</h1>
<p>Genetic Programming was used to evolve mathematical expressions over base
features that predict each protected attribute.</p>

<div class="config">
  <strong>GP Configuration:</strong>
  Population: <code>{args.population_size}</code> &middot;
  Max depth: <code>{args.max_depth}</code> &middot;
  Time budget: <code>{args.time_budget}s</code> &middot;
  Seed: <code>{args.seed}</code> &middot;
  Prec weight: <code>{args.prec_weight}</code> &middot;
  Rec weight: <code>{args.rec_weight}</code> &middot;
  Cov weight: <code>{args.cov_weight}</code> &middot;
  Complexity penalty: <code>{args.complexity_penalty}</code> &middot;
  Grammar: <code>{args.grammar}</code> ({', '.join(c.__name__ for c in (GRAMMAR_PRESETS.get(args.grammar) or TYPED_STATIC_COMPONENTS))}) + features
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
        n_feats = len(rows[0].get("expression_features", "").split(", "))
        ds_title = "Processed" if ds_label == "processed" else "Non-Processed"
        html += f'<hr class="section-divider">\n'
        html += f'<h2>{escape(ds_title)} Dataset '
        html += f'<span style="font-weight:400;color:#888">({ds_label})</span></h2>\n'

        # Summary cards
        html += '<div class="summary-grid">\n'
        for r in rows:
            delta = r["auc"] - r["best_baseline_auc"]
            color = "#28a745" if delta > 0 else "#222"
            html += f"""  <div class="summary-card">
    <h3>{escape(r["protected_attr"])}</h3>
    <div class="val" style="color:{color}">{r["auc"]:.4f}</div>
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
      <th>AUC</th>
      <th>Best Single Feature</th>
      <th>Delta</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>\n"""
        for r in rows:
            delta = r["auc"] - r["best_baseline_auc"]
            row_cls = ' class="strong"' if delta > 0 else ""
            feat_html = _feat_tags(r["expression_features"],
                                   r["best_baseline_feature"])
            html += f"""    <tr{row_cls}>
      <td>{escape(r["protected_attr"])}</td>
      <td>{feat_html}</td>
      <td>{r["nodes"]}</td>
      <td>{r["auc"]:.4f}</td>
      <td>{r["best_baseline_auc"]:.4f} ({escape(r["best_baseline_feature"])})</td>
      <td>{_delta_span(delta)}</td>
      <td>{r["elapsed_s"]}s</td>
    </tr>\n"""
        html += "  </tbody>\n</table>\n\n"

        # Expressions
        html += "<h3>Discovered Expressions</h3>\n\n"
        for r in rows:
            delta = r["auc"] - r["best_baseline_auc"]
            html += f'<p><strong>{escape(r["protected_attr"])}</strong> '
            html += f'(AUC {r["auc"]:.4f} &mdash; '
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
  Generated by <code>gp_proxy_discovery.py</code> on {timestamp} &middot;
  Seed {args.seed} &middot; {args.time_budget}s budget &middot;
  prec_w={args.prec_weight} rec_w={args.rec_weight} cov_w={args.cov_weight} &middot; Complexity penalty {args.complexity_penalty} &middot;
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

    # "all" expands to arithmetic then typed
    grammars_to_run = ["arithmetic", "typed"] if args.grammar == "all" else [args.grammar]

    print(f"  Grammar(s): {', '.join(grammars_to_run)}  |  "
          f"prec_w={args.prec_weight}  rec_w={args.rec_weight}  cov_w={args.cov_weight}  |  "
          f"Complexity penalty: {args.complexity_penalty}")

    for grammar_mode in grammars_to_run:
        print(f"\n{'#' * 70}")
        print(f"# Grammar: {grammar_mode.upper()}")
        print(f"{'#' * 70}")

        for ds_label in args.datasets:
            ds_dir = dataset_map[ds_label]
            print(f"\n{'=' * 70}")
            print(f"Dataset: {ds_label}  |  Grammar: {grammar_mode}")
            print(f"{'=' * 70}")

            dataset = load_dataset(ds_dir, ds_label)
            grammar = build_grammar(dataset["feature_names"], grammar_mode,
                                    X_train=dataset["X_train"])
            baselines = load_baselines(ds_dir)

            print(f"  Features ({len(dataset['feature_names'])}): "
                  f"{', '.join(dataset['feature_names'])}")
            print(f"  Train: {dataset['X_train'].shape[0]}  |  "
                  f"Test: {dataset['X_test'].shape[0]}")

            for pa in args.attributes:
                print(f"\n  --- Evolving proxy for: {pa.upper()} "
                      f"(budget {args.time_budget}s, grammar={grammar_mode}) ---")

                result = run_gp(
                    dataset=dataset,
                    protected_attr=pa,
                    grammar=grammar,
                    time_budget=args.time_budget,
                    population_size=args.population_size,
                    max_depth=args.max_depth,
                    seed=args.seed,
                    prec_weight=args.prec_weight,
                    rec_weight=args.rec_weight,
                    cov_weight=args.cov_weight,
                    complexity_penalty=args.complexity_penalty,
                    min_partial_auc=args.proxy_min_auc,
                    min_partial_precision=args.proxy_min_precision,
                    min_partial_prec_floor=args.proxy_min_precision_floor,
                    min_partial_recall=args.proxy_min_recall,
                )

                result["grammar_mode"] = grammar_mode

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
                print(f"    AUC        : {result['auc']}")
                print(f"    Baseline   : {bl_auc:.4f} ({bl_feat})")
                print(f"    Delta      : {result['auc'] - bl_auc:+.4f}")
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
