#!/usr/bin/env python3
"""
End-to-End GP Proxy Discovery Pipeline
=======================================
Runs all three stages for a chosen dataset and generates a self-contained
HTML report — no pre-computed CSVs required.

  Stage 1 — Classifier baseline  (all features → chosen classifier → AUC)
  Stage 2 — GP + Arithmetic grammar (discovered expression → same classifier)
  Stage 3 — GP + Extended grammar   (richer operators → same classifier)

Outputs are saved to:
    pipeline_results/{dataset}/
        arith.csv      ← Stage 2 GP results
        ext.csv        ← Stage 3 GP results
        report.html    ← Full three-stage report

Usage
-----
    python pipeline.py --dataset law_school
    python pipeline.py --dataset adult   --classifier random_forest
    python pipeline.py --dataset compas  --time-budget 60 --splits raw
    python pipeline.py --dataset oulad   --penalty 0.05 --max-depth 4

Supported datasets
------------------
    law_school   LSAC Bar-passage study (processed / non_processed)
    adult        US Census income dataset (processed / non_processed)
    compas       COMPAS recidivism scores (raw / encoded)
    oulad        Open University Learning Analytics (raw / encoded)
"""

import argparse
import concurrent.futures as _cf
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
COMPAS_ROOT      = ROOT / "compas_scores_raw_dataset"
OULAD_ROOT       = ROOT / "OULAD-student_dataset"
ADULT_ROOT       = ROOT / "adult_test_dataset"
GERMAN_ROOT      = ROOT / "German-Credit-Analysis-master"
FOLKTABLES_ROOT  = ROOT / "folktables-main-Dataset"
OUTPUT_ROOT      = ROOT / "pipeline_results"

# ── Import GP runner from law-school script (works for all datasets) ──────────
# This also adds GeneticEngine to sys.path as a side-effect.
from gp_proxy_discovery import (          # noqa: E402
    run_gp,
    build_grammar as _ls_build_grammar,
    extract_features_from_expr,
    compute_firing_profile,
)
from geml.common import forward_dataset as _forward_dataset

# ── Import HTML + classifier helpers from generate_3stage_report ──────────────
import generate_3stage_report as _gsr    # noqa: E402
from simplify_pipeline_results import simplify as _simplify_expr  # noqa: E402

# ── Dataset configurations ────────────────────────────────────────────────────
DATASET_CONFIGS: dict[str, dict] = {
    "law_school": {
        "label":   "Law School (LSAC)",
        "loader":  "npz",
        "splits": [
            ("processed",     ROOT / "processed",
             "Processed (Cleaned, 19 features)"),
            ("non_processed", ROOT / "non_processed",
             "Non-Processed (Raw, all features)"),
        ],
        "npz_name": "law_school_base_features.npz",
        "protected_attrs": ["black", "asian", "hisp", "other"],
        "attr_display": {
            "black": "Black", "asian": "Asian", "hisp": "Hispanic",
            "other": "Other Race",
        },
        "grammar_builder": "law_school",
        "baselines": {
            "processed":
                ROOT / "processed"     / "individual_proxy_baselines.csv",
            "non_processed":
                ROOT / "non_processed" / "individual_proxy_baselines.csv",
        },
    },

    "adult": {
        "label":  "Adult Census Income",
        "loader": "npz",
        "splits": [
            ("processed",     ADULT_ROOT / "processed",
             "Processed (Cleaned features)"),
            ("non_processed", ADULT_ROOT / "non_processed",
             "Non-Processed (Raw features)"),
        ],
        "npz_name": "adult_base_features.npz",
        "protected_attrs": ["male"],
        "attr_display": {"male": "Male"},
        "grammar_builder": "law_school",
        "baselines": {
            "processed":
                ADULT_ROOT / "processed"     / "individual_proxy_baselines.csv",
            "non_processed":
                ADULT_ROOT / "non_processed" / "individual_proxy_baselines.csv",
        },
    },

    "compas": {
        "label":  "COMPAS Recidivism",
        "loader": "compas",
        "splits": [
            ("raw",     None, "Raw (Minimal preprocessing)"),
            ("encoded", None, "Encoded (Domain-informed)"),
        ],
        "protected_attrs": [
            "african_american", "caucasian", "hispanic", "other_race",
        ],
        "attr_display": {
            "african_american": "African-American",
            "caucasian":        "Caucasian",
            "hispanic":         "Hispanic",
            "other_race":       "Other Race",
        },
        "grammar_builder": "compas",
        "baselines": {
            "raw":
                COMPAS_ROOT / "individual_proxy_baselines_raw.csv",
            "encoded":
                COMPAS_ROOT / "individual_proxy_baselines_encoded.csv",
        },
    },

    "oulad": {
        "label":  "OULAD (Open University)",
        "loader": "oulad",
        "splits": [
            ("raw",     None, "Raw (Label-encoded)"),
            ("encoded", None, "Encoded (Domain-informed)"),
        ],
        "protected_attrs": ["female"],
        "attr_display": {"female": "Female"},
        "grammar_builder": "oulad",
        "baselines": {
            "raw":
                OULAD_ROOT / "individual_proxy_baselines_raw.csv",
            "encoded":
                OULAD_ROOT / "individual_proxy_baselines_encoded.csv",
        },
    },

    "german_credit": {
        "label":  "German Credit (Statlog)",
        "loader": "npz",
        "splits": [
            ("processed", GERMAN_ROOT / "processed",
             "Processed (7 features, synthetic risk target)"),
        ],
        "npz_name": "german_credit_base_features.npz",
        "protected_attrs": ["male", "age_young"],
        "attr_display": {
            "male":      "Male",
            "age_young": "Young (<30)",
        },
        "grammar_builder": "law_school",
        "baselines": {
            "processed": GERMAN_ROOT / "processed" / "individual_proxy_baselines.csv",
        },
    },

    "folktables": {
        "label":  "Folktables ACS Adult (2018)",
        "loader": "npz",
        "splits": [
            ("processed", FOLKTABLES_ROOT / "processed",
             "Processed (11 features, income >= 50K target)"),
        ],
        "npz_name": "folktables_base_features.npz",
        "protected_attrs": ["white", "black", "asian", "amer_indian",
                            "other_race", "gender_male"],
        "attr_display": {
            "white":       "White",
            "black":       "Black",
            "asian":       "Asian/Pacific Islander",
            "amer_indian": "American Indian/Eskimo",
            "other_race":  "Other Race",
            "gender_male": "Male",
        },
        "grammar_builder": "law_school",
        "baselines": {
            "processed": FOLKTABLES_ROOT / "processed" / "individual_proxy_baselines.csv",
        },
    },
}


# ── Grammar builder dispatcher ────────────────────────────────────────────────

def _get_grammar_builder(cfg: dict):
    """Return the build_grammar function for this dataset."""
    kind = cfg["grammar_builder"]
    if kind == "law_school":
        return _ls_build_grammar

    if kind == "compas":
        sys.path.insert(0, str(COMPAS_ROOT))
        try:
            from gp_proxy_discovery_compas import build_grammar as _bg
        finally:
            sys.path.pop(0)
        return _bg

    if kind == "oulad":
        sys.path.insert(0, str(OULAD_ROOT))
        try:
            from gp_proxy_discovery_oulad import build_grammar as _bg
        finally:
            sys.path.pop(0)
        return _bg

    raise ValueError(f"Unknown grammar_builder: {kind!r}")


# ── Data loaders ──────────────────────────────────────────────────────────────

def _attach_baselines(ds: dict, cfg: dict, split_label: str) -> None:
    """Add 'baselines' DataFrame and n_train/n_test to a dataset dict."""
    bl_path = cfg["baselines"].get(split_label)
    ds["baselines"] = (pd.read_csv(bl_path)
                       if bl_path and Path(bl_path).exists()
                       else pd.DataFrame())
    ds.setdefault("n_train", ds["X_train"].shape[0])
    ds.setdefault("n_test",  ds["X_test"].shape[0])


def _merge_to_single_phase(ds: dict) -> None:
    """Concatenate train and test splits into a single training dataset.

    After this call X_train == X_test and pa_train == pa_test so the GP
    search and all metric evaluations use the full available data.
    """
    X_all = np.concatenate([ds["X_train"], ds["X_test"]], axis=0)
    ds["X_train"] = X_all
    ds["X_test"]  = X_all
    for pa in list(ds.get("pa_train", {}).keys()):
        pa_all = np.concatenate([ds["pa_train"][pa], ds["pa_test"][pa]])
        ds["pa_train"][pa] = pa_all
        ds["pa_test"][pa]  = pa_all
    ds["n_train"] = len(X_all)
    ds["n_test"]  = len(X_all)


def _load_metadata(split_dir: Path) -> dict:
    """Load metadata.json if it exists, else return {}."""
    import json
    meta = split_dir / "metadata.json"
    if not meta.exists():
        return {}
    try:
        return json.loads(meta.read_text())
    except Exception:
        return {}


def _load_label_encodings(split_dir: Path) -> dict[str, list[str]]:
    """Load label_encodings from metadata.json if it exists, else return {}."""
    return _load_metadata(split_dir).get("label_encodings", {})


def _load_npz_datasets(cfg: dict, selected_splits: list[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for split_label, split_dir, split_desc in cfg["splits"]:
        if split_label not in selected_splits:
            continue
        npz_path = split_dir / cfg["npz_name"]
        if not npz_path.exists():
            print(f"  [SKIP] {split_label}: {npz_path} not found")
            continue

        data     = np.load(npz_path, allow_pickle=True)
        X_train  = data["X_train_raw"] if "X_train_raw" in data else data["X_train"]
        X_test   = data["X_test_raw"]  if "X_test_raw"  in data else data["X_test"]
        X_train  = np.nan_to_num(np.asarray(X_train, dtype=np.float64),
                                 nan=0.0, posinf=1e6, neginf=-1e6)
        X_test   = np.nan_to_num(np.asarray(X_test,  dtype=np.float64),
                                 nan=0.0, posinf=1e6, neginf=-1e6)

        feature_names    = [str(n) for n in data["feature_names"]]
        meta             = _load_metadata(split_dir)
        label_encodings  = meta.get("label_encodings", {})
        numeric_features    = meta.get("numeric_features", [])
        categorical_features = meta.get("categorical_features", [])
        pa_train, pa_test = {}, {}
        for pa in cfg["protected_attrs"]:
            if f"{pa}_train" in data:
                pa_train[pa] = np.asarray(data[f"{pa}_train"], dtype=np.int32)
                pa_test[pa]  = np.asarray(data[f"{pa}_test"],  dtype=np.int32)

        ds = {
            "label":                split_label,
            "display":              split_desc,
            "X_train":              X_train,
            "X_test":               X_test,
            "feature_names":        feature_names,
            "label_encodings":      label_encodings,
            "numeric_features":     numeric_features,
            "categorical_features": categorical_features,
            "pa_train":             pa_train,
            "pa_test":              pa_test,
        }
        _attach_baselines(ds, cfg, split_label)
        _merge_to_single_phase(ds)
        result[split_label] = ds
    return result


def _load_compas_datasets(cfg: dict, selected_splits: list[str]) -> dict[str, dict]:
    sys.path.insert(0, str(COMPAS_ROOT))
    try:
        from gp_proxy_discovery_compas import load_compas_datasets
        all_ds = load_compas_datasets()
    finally:
        sys.path.pop(0)

    result: dict[str, dict] = {}
    for split_label, _, split_desc in cfg["splits"]:
        if split_label not in selected_splits or split_label not in all_ds:
            continue
        ds = all_ds[split_label]
        ds["display"] = split_desc
        _attach_baselines(ds, cfg, split_label)
        _merge_to_single_phase(ds)
        result[split_label] = ds
    return result


def _load_oulad_datasets(cfg: dict, selected_splits: list[str]) -> dict[str, dict]:
    sys.path.insert(0, str(OULAD_ROOT))
    try:
        from gp_proxy_discovery_oulad import load_oulad_datasets
        all_ds = load_oulad_datasets()
    finally:
        sys.path.pop(0)

    result: dict[str, dict] = {}
    for split_label, _, split_desc in cfg["splits"]:
        if split_label not in selected_splits or split_label not in all_ds:
            continue
        ds = all_ds[split_label]
        ds["display"] = split_desc
        _attach_baselines(ds, cfg, split_label)
        _merge_to_single_phase(ds)
        result[split_label] = ds
    return result


def load_data(cfg: dict, selected_splits: list[str]) -> dict[str, dict]:
    loader = cfg["loader"]
    if loader == "npz":
        return _load_npz_datasets(cfg, selected_splits)
    if loader == "compas":
        return _load_compas_datasets(cfg, selected_splits)
    if loader == "oulad":
        return _load_oulad_datasets(cfg, selected_splits)
    raise ValueError(f"Unknown loader: {loader!r}")


# ── GP stage runner ───────────────────────────────────────────────────────────

_CSV_COLS = [
    "dataset", "protected_attr", "expression", "numpy_expr", "nodes",
    "auc", "ap", "nmi", "cohens_d",
    "precision", "recall", "coverage",
    "elapsed_s",
    "best_baseline_auc", "best_baseline_feature", "expression_features",
    "and_components",   # JSON string: per-component metrics for AND expressions
]

_PARTIAL_CSV_COLS = [
    "dataset", "protected_attr", "expression", "numpy_expr", "nodes", "features",
    "first_seen_s",
    "auc", "ap", "nmi", "cohens_d",
    "precision", "recall", "coverage",
]

# ── Expression simplification helpers ────────────────────────────────────────


def _constant_fold_numpy(numpy_expr: str) -> str:
    """Constant-fold a numpy expression string using Python's ast module.

    Sub-expressions whose inputs are all numeric constants are evaluated and
    replaced with their computed value.  For example:
      np.where(5.0 > np.minimum(5.0, 9.0), 8.0, 3.0)  →  3.0

    Falls back silently to the original string on any failure.
    """
    if not numpy_expr:
        return numpy_expr
    try:
        import ast as _ast

        _SAFE = {
            "__builtins__": {},
            "np": type("_np", (), {
                "minimum":  min,
                "maximum":  max,
                "abs":      abs,
                "absolute": abs,
                "where":    lambda c, a, b: a if c else b,
                "sign":     lambda x: (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)),
            })(),
        }

        class _Folder(_ast.NodeTransformer):
            def _is_const(self, node):
                return isinstance(node, _ast.Constant) and isinstance(
                    node.value, (int, float, bool)
                )

            def _eval(self, node):
                try:
                    code = compile(
                        _ast.fix_missing_locations(_ast.Expression(body=node)),
                        "<cf>", "eval",
                    )
                    result = eval(code, _SAFE)  # noqa: S307
                    if isinstance(result, (int, float, bool)):
                        return _ast.Constant(value=float(result))
                except Exception:
                    pass
                return None

            def visit_BinOp(self, node):
                self.generic_visit(node)
                if self._is_const(node.left) and self._is_const(node.right):
                    return self._eval(node) or node
                return node

            def visit_Compare(self, node):
                self.generic_visit(node)
                if (self._is_const(node.left) and len(node.comparators) == 1
                        and self._is_const(node.comparators[0])):
                    return self._eval(node) or node
                return node

            def visit_UnaryOp(self, node):
                self.generic_visit(node)
                if self._is_const(node.operand):
                    return self._eval(node) or node
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                # np.where(CONST_COND, a, b) → pick the live branch
                if (isinstance(node.func, _ast.Attribute)
                        and node.func.attr == "where"
                        and len(node.args) == 3
                        and self._is_const(node.args[0])):
                    return node.args[1] if node.args[0].value else node.args[2]
                # All-constant call → evaluate directly
                if all(self._is_const(a) for a in node.args) and not node.keywords:
                    return self._eval(node) or node
                return node

        tree = _ast.parse(numpy_expr, mode="eval")
        new_tree = _Folder().visit(tree)
        _ast.fix_missing_locations(new_tree)
        return _ast.unparse(new_tree)
    except Exception:
        return numpy_expr


# ── Dataset calibration ───────────────────────────────────────────────────────

def _calibrate_dataset(ds: dict, verbose: bool = True) -> dict:
    """Analyse a dataset split once, before GP runs.

    Returns a calibration dict that downstream simplification passes use to
    apply dataset-specific folding rules:

      feature_ranges    – {name: (lo, hi)} from the training data
      binary_features   – frozenset of features whose values are all in {0, 1}
      nonneg_features   – frozenset of features where min >= 0
      constant_features – {name: value} for features that never change
    """
    X = ds.get("X_train")
    feature_names = ds.get("feature_names", [])

    feature_ranges: dict   = {}
    binary_features: set   = set()
    nonneg_features: set   = set()
    constant_features: dict = {}

    if X is not None and len(feature_names):
        X_arr = np.asarray(X, dtype=float)
        for j, name in enumerate(feature_names):
            col = X_arr[:, j]
            col = col[np.isfinite(col)]
            if len(col) == 0:
                continue
            lo, hi = float(col.min()), float(col.max())
            feature_ranges[name] = (lo, hi)
            if lo >= 0.0:
                nonneg_features.add(name)
            if lo == hi:
                constant_features[name] = lo
            unique = np.unique(col)
            if len(unique) <= 2 and set(unique).issubset({0.0, 1.0}):
                binary_features.add(name)

    calibration = {
        "feature_ranges":    feature_ranges,
        "binary_features":   frozenset(binary_features),
        "nonneg_features":   frozenset(nonneg_features),
        "constant_features": constant_features,
    }

    if verbose:
        n_feat  = len(feature_names)
        n_bin   = len(binary_features)
        n_cont  = len([f for f in feature_names
                       if f in feature_ranges and f not in binary_features
                       and f not in constant_features])
        n_const = len(constant_features)
        print(f"    features: {n_feat} total  "
              f"({n_bin} binary, {n_cont} continuous, {n_const} constant)")
        # Show range of each continuous feature
        for name in feature_names:
            if name in feature_ranges and name not in binary_features \
                    and name not in constant_features:
                lo, hi = feature_ranges[name]
                print(f"      {name}: [{lo:g}, {hi:g}]")

    return calibration


# ── Interval-arithmetic helpers ───────────────────────────────────────────────

def _split_comma_args(s: str) -> list:
    """Split s on commas at paren depth 0."""
    depth, args, start = 0, [], 0
    for i, c in enumerate(s):
        if c == '(':   depth += 1
        elif c == ')': depth -= 1
        elif c == ',' and depth == 0:
            args.append(s[start:i].strip())
            start = i + 1
    args.append(s[start:].strip())
    return args


def _find_top_binop(s: str):
    """Find the first top-level binary operator in s (space-delimited, depth 0).

    Returns (lhs, op, rhs) or None.  Two-char operators are tried before one-char
    to avoid splitting '<=', '>=' on their first character.
    """
    depth = 0
    n = len(s)
    for i in range(n):
        c = s[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ' ' and depth == 0 and i + 2 < n:
            for op in ('<=', '>=', '==', '!='):
                j, end = i + 1, i + 1 + len(op)
                if end < n and s[j:end] == op and s[end] == ' ':
                    return s[:i].strip(), op, s[end + 1:].strip()
            for op in ('<', '>', '+', '*', '/', '-'):
                if s[i + 1] == op and i + 2 < n and s[i + 2] == ' ':
                    return s[:i].strip(), op, s[i + 3:].strip()
    return None


def _interval_eval(expr: str, feature_ranges: dict):
    """Interval arithmetic: return (lo, hi) guaranteed range for expr, or None.

    feature_ranges maps feature names to (lo, hi) pairs derived from training data.
    Returns None when the range cannot be determined (unknown feature, division
    by an interval that crosses zero, etc.).
    """
    expr = expr.strip()
    if not expr or expr == 'nan':
        return None
    try:
        v = float(expr)
        return (v, v)
    except ValueError:
        pass
    if expr in feature_ranges:
        return feature_ranges[expr]

    # |X|
    if expr.startswith('|') and expr.endswith('|') and len(expr) > 2:
        iv = _interval_eval(expr[1:-1], feature_ranges)
        if iv is not None:
            lo, hi = iv
            if lo >= 0:  return (lo, hi)
            if hi <= 0:  return (-hi, -lo)
            return (0.0, max(-lo, hi))
        return None

    # NOT(X)
    if expr.startswith('NOT(') and expr.endswith(')'):
        iv = _interval_eval(expr[4:-1], feature_ranges)
        if iv == (0.0, 0.0): return (1.0, 1.0)
        if iv == (1.0, 1.0): return (0.0, 0.0)
        return (0.0, 1.0)

    # min(X,Y) / max(X,Y)
    for fn_name in ('min', 'max'):
        prefix = fn_name + '('
        if expr.startswith(prefix) and expr.endswith(')'):
            args = _split_comma_args(expr[len(prefix):-1])
            if len(args) == 2:
                a = _interval_eval(args[0], feature_ranges)
                b = _interval_eval(args[1], feature_ranges)
                if a is not None and b is not None:
                    if fn_name == 'min': return (min(a[0], b[0]), min(a[1], b[1]))
                    else:                return (max(a[0], b[0]), max(a[1], b[1]))
            return None

    # IF(cond, X, Y) → conservative union of both branches
    if expr.startswith('IF(') and expr.endswith(')'):
        args = _split_comma_args(expr[3:-1])
        if len(args) == 3:
            a = _interval_eval(args[1], feature_ranges)
            b = _interval_eval(args[2], feature_ranges)
            if a is not None and b is not None:
                return (min(a[0], b[0]), max(a[1], b[1]))
        return None

    # (X op Y) — strip outer parens if present, then find top-level operator
    inner = expr[1:-1] if (expr.startswith('(') and expr.endswith(')')) else expr
    parsed = _find_top_binop(inner)
    if parsed is None:
        return None
    lhs_s, op, rhs_s = parsed

    if op in ('+', '-', '*', '/'):
        l = _interval_eval(lhs_s, feature_ranges)
        r = _interval_eval(rhs_s, feature_ranges)
        if l is None or r is None:
            return None
        if op == '+': return (l[0] + r[0], l[1] + r[1])
        if op == '-': return (l[0] - r[1], l[1] - r[0])
        if op == '*':
            ps = [l[0]*r[0], l[0]*r[1], l[1]*r[0], l[1]*r[1]]
            return (min(ps), max(ps))
        if op == '/':
            if r[0] <= 0 <= r[1]: return None
            ps = [l[0]/r[0], l[0]/r[1], l[1]/r[0], l[1]/r[1]]
            return (min(ps), max(ps))

    if op in ('<', '>', '<=', '>=', '==', '!='):
        l = _interval_eval(lhs_s, feature_ranges)
        r = _interval_eval(rhs_s, feature_ranges)
        if l is None or r is None:
            return (0.0, 1.0)
        if op == '<':
            if l[1] < r[0]:  return (1.0, 1.0)
            if l[0] >= r[1]: return (0.0, 0.0)
        elif op == '>':
            if l[0] > r[1]:  return (1.0, 1.0)
            if l[1] <= r[0]: return (0.0, 0.0)
        elif op == '<=':
            if l[1] <= r[0]: return (1.0, 1.0)
            if l[0] > r[1]:  return (0.0, 0.0)
        elif op == '>=':
            if l[0] >= r[1]: return (1.0, 1.0)
            if l[1] < r[0]:  return (0.0, 0.0)
        elif op == '==':
            if l[0] == l[1] == r[0] == r[1]:
                return (1.0, 1.0) if l[0] == r[0] else (0.0, 0.0)
            if l[1] < r[0] or l[0] > r[1]: return (0.0, 0.0)
        elif op == '!=':
            if l[1] < r[0] or l[0] > r[1]: return (1.0, 1.0)
            if l[0] == l[1] == r[0] == r[1]: return (0.0, 0.0)
        return (0.0, 1.0)

    return None


def _constant_fold_readable(expr: str, calibration: dict | None = None) -> str:
    """Simplify constant sub-expressions in a human-readable GP expression.

    Handles min/max of constants, |constant|, comparisons between constants,
    and IF(constant_cond, A, B) → A or B.  Iterates until convergence so that
    folding one level unlocks folding at the next (e.g. |IF(0.0, 8.0, 3.0)|
    first collapses the IF to 3.0, then the |3.0| to 3.0).
    """
    import re

    # Unpack calibration
    feature_ranges  = (calibration or {}).get("feature_ranges",  {})
    binary_features = (calibration or {}).get("binary_features",  frozenset())

    # NUM also matches the literal "nan" so NaN can propagate through folds
    NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan"

    def _fmt(v: float) -> str:
        """Format a float, always keeping the decimal point (3.0 not 3)."""
        import math as _math
        if _math.isnan(v):
            return "nan"
        s = f"{v:.6g}"
        if "." not in s and "e" not in s:
            s += ".0"
        return s

    def _find_if_args(s: str, pos: int):
        """Parse comma-separated args inside IF(...) starting at pos (after '(')."""
        depth, args, start, i = 0, [], pos, pos
        while i < len(s):
            c = s[i]
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    args.append(s[start:i].strip())
                    return args, i + 1
                depth -= 1
            elif c == "," and depth == 0:
                args.append(s[start:i].strip())
                start = i + 1
            i += 1
        return None, len(s)

    # ── Balanced-paren helpers ────────────────────────────────────────────────
    # These scan right-to-left (innermost-first) to safely simplify (X op C)
    # and (C op X) patterns without being confused by C appearing inside X.

    def _fold_right(op_re, const_re, fn):
        """Simplify (X op C) for any balanced X; return (new_expr, changed)."""
        nonlocal expr
        chg = False
        while True:
            last = None
            for last in re.finditer(rf'\s+{op_re}\s+(?:{const_re})\)', expr):
                pass
            if last is None:
                break
            op_start, close_idx = last.start(), last.end() - 1
            depth, i = 1, op_start - 1
            while i >= 0 and depth:
                if expr[i] == ')':   depth += 1
                elif expr[i] == '(': depth -= 1
                i -= 1
            if depth:
                break
            open_idx = i + 1
            x = expr[open_idx + 1 : op_start].strip()
            expr = expr[:open_idx] + fn(x) + expr[close_idx + 1:]
            chg = True
        return chg

    def _fold_left(const_re, op_re, fn):
        """Simplify (C op X) for any balanced X; return (new_expr, changed)."""
        nonlocal expr
        chg = False
        while True:
            last = None
            for last in re.finditer(rf'\((?:{const_re})\s+{op_re}\s+', expr):
                pass
            if last is None:
                break
            open_idx, prefix_end = last.start(), last.end()
            depth, j = 1, prefix_end
            while j < len(expr) and depth:
                if expr[j] == '(':   depth += 1
                elif expr[j] == ')': depth -= 1
                j += 1
            if depth:
                break
            close_idx = j - 1
            x = expr[prefix_end : close_idx].strip()
            expr = expr[:open_idx] + fn(x) + expr[close_idx + 1:]
            chg = True
        return chg

    changed = True
    while changed:
        changed = False

        # min(N, N)
        def _rmin(m):
            return _fmt(min(float(m.group(1)), float(m.group(2))))
        new = re.sub(rf"min\(({NUM}),\s*({NUM})\)", _rmin, expr)
        if new != expr:
            expr, changed = new, True; continue

        # max(N, N)
        def _rmax(m):
            return _fmt(max(float(m.group(1)), float(m.group(2))))
        new = re.sub(rf"max\(({NUM}),\s*({NUM})\)", _rmax, expr)
        if new != expr:
            expr, changed = new, True; continue

        # |N| — absolute value of a constant
        def _rabs(m):
            return _fmt(abs(float(m.group(1))))
        new = re.sub(rf"\|({NUM})\|", _rabs, expr)
        if new != expr:
            expr, changed = new, True; continue

        # (N op N) — general constant arithmetic; cascades on repeated passes.
        # Division by zero yields nan (matching numpy runtime), which propagates
        # through subsequent folds until it reaches a comparison and resolves to 0.0.
        def _rconst_arith(m):
            a_s, op, b_s = m.group(1), m.group(2), m.group(3)
            try:
                a, b = float(a_s), float(b_s)
                import math as _math
                if   op == '+': v = a + b
                elif op == '-': v = a - b
                elif op == '*': v = a * b
                elif op == '/': v = float('nan') if b == 0.0 else a / b
                else: return m.group(0)
                return _fmt(v)
            except Exception:
                return m.group(0)
        new = re.sub(rf'\(({NUM})\s*([+\-*/])\s*({NUM})\)', _rconst_arith, expr)
        if new != expr:
            expr, changed = new, True; continue

        # ── Arithmetic identity / annihilator rules ──────────────────────────
        # (X + 0.0) → X,  (0.0 + X) → X,  (X - 0.0) → X
        if _fold_right(r'\+', r'0\.0', lambda x: x):  changed = True; continue
        if _fold_left(r'0\.0', r'\+', lambda x: x):   changed = True; continue
        if _fold_right(r'-',  r'0\.0', lambda x: x):  changed = True; continue
        # (X * 1.0) → X,  (1.0 * X) → X,  (X / 1.0) → X
        if _fold_right(r'\*', r'1\.0', lambda x: x):  changed = True; continue
        if _fold_left(r'1\.0', r'\*', lambda x: x):   changed = True; continue
        if _fold_right(r'/',  r'1\.0', lambda x: x):  changed = True; continue
        # (X * 0.0) → 0.0,  (0.0 * X) → 0.0
        if _fold_right(r'\*', r'0\.0', lambda _: '0.0'): changed = True; continue
        if _fold_left(r'0\.0', r'\*', lambda _: '0.0'):  changed = True; continue
        # (1.0 AND X) → X,  (X AND 1.0) → X  (True is identity for AND)
        # (0.0 AND X) → 0.0,  (X AND 0.0) → 0.0  (False annihilates AND)
        if _fold_left(r'1\.0', r'AND', lambda x: x):          changed = True; continue
        if _fold_right(r'AND', r'1\.0', lambda x: x):         changed = True; continue
        if _fold_left(r'0\.0', r'AND', lambda _: '0.0'):      changed = True; continue
        if _fold_right(r'AND', r'0\.0', lambda _: '0.0'):     changed = True; continue
        # NOT(NOT(X)) → X  (double-negation elimination)
        new = re.sub(r'NOT\(NOT\(([^()]*)\)\)', r'\1', expr)
        if new != expr:
            expr, changed = new, True; continue

        # (C * X) > 0  →  X > 0   when C > 0   (positive scalar preserves sign)
        # (C * X) < 0  →  X < 0   when C > 0
        # (C * X) > 0  →  X < 0   when C < 0   (negative scalar flips direction)
        # (C * X) < 0  →  X > 0   when C < 0
        # Applies only when the WHOLE expression is (C * balanced_X) op 0.0,
        # since non-zero RHS is better handled by _sympy_canon.
        _NONZERO_NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
        _cm_mult = re.match(rf'^\(({_NONZERO_NUM})\s*\*\s*', expr)
        if _cm_mult:
            try:
                _c_val = float(_cm_mult.group(1))
            except ValueError:
                _c_val = None
            if _c_val is not None and _c_val != 0.0:
                _depth, _j = 1, _cm_mult.end()
                while _j < len(expr) and _depth > 0:
                    if expr[_j] == '(':   _depth += 1
                    elif expr[_j] == ')': _depth -= 1
                    _j += 1
                _rest_after = expr[_j:].strip()
                _opm = re.match(r'^([><])\s*0(?:\.0)?\s*$', _rest_after)
                if _opm:
                    _op = _opm.group(1)
                    _inner = expr[_cm_mult.end() : _j - 1].strip()
                    if _c_val < 0:
                        _op = '>' if _op == '<' else '<'
                    new = f"({_inner}) {_op} 0.0"
                    if new != expr:
                        expr, changed = new, True; continue

        # IF(cond, X, X) → X — same result regardless of condition
        _if_pat = re.compile(r'IF\(')
        m = _if_pat.search(expr)
        while m:
            args, end = _find_if_args(expr, m.end())
            if args and len(args) == 3 and args[1].strip() == args[2].strip():
                expr = expr[:m.start()] + args[1].strip() + expr[end:]
                changed = True
                break
            m = _if_pat.search(expr, m.end())
        if changed: continue

        # NOT(N) → constant
        def _rnot_const(m):
            try:
                return '0.0' if float(m.group(1)) else '1.0'
            except Exception:
                return m.group(0)
        new = re.sub(rf'NOT\(({NUM})\)', _rnot_const, expr)
        if new != expr:
            expr, changed = new, True; continue

        # (N > N) / (N < N) / (N == N) / (N != N) — constant comparisons
        def _rcmp_gt(m):
            return "1.0" if float(m.group(1)) > float(m.group(2)) else "0.0"
        new = re.sub(rf"\(({NUM})\s*>\s*({NUM})\)", _rcmp_gt, expr)
        if new != expr:
            expr, changed = new, True; continue

        def _rcmp_lt(m):
            return "1.0" if float(m.group(1)) < float(m.group(2)) else "0.0"
        new = re.sub(rf"\(({NUM})\s*<\s*({NUM})\)", _rcmp_lt, expr)
        if new != expr:
            expr, changed = new, True; continue

        def _rcmp_eq(m):
            return "1.0" if float(m.group(1)) == float(m.group(2)) else "0.0"
        new = re.sub(rf"\(({NUM})\s*==\s*({NUM})\)", _rcmp_eq, expr)
        if new != expr:
            expr, changed = new, True; continue

        def _rcmp_ne(m):
            return "0.0" if float(m.group(1)) == float(m.group(2)) else "1.0"
        new = re.sub(rf"\(({NUM})\s*!=\s*({NUM})\)", _rcmp_ne, expr)
        if new != expr:
            expr, changed = new, True; continue

        def _rcmp_le(m):
            try: return "1.0" if float(m.group(1)) <= float(m.group(2)) else "0.0"
            except Exception: return m.group(0)
        new = re.sub(rf"\(({NUM})\s*<=\s*({NUM})\)", _rcmp_le, expr)
        if new != expr:
            expr, changed = new, True; continue

        def _rcmp_ge(m):
            try: return "1.0" if float(m.group(1)) >= float(m.group(2)) else "0.0"
            except Exception: return m.group(0)
        new = re.sub(rf"\(({NUM})\s*>=\s*({NUM})\)", _rcmp_ge, expr)
        if new != expr:
            expr, changed = new, True; continue

        # X / X → 1.0 for simple identifier self-division
        new = re.sub(r'\((\w+)\s*/\s*\1\)', "1.0", expr)
        if new != expr:
            expr, changed = new, True; continue

        # max(C, X) op T / max(X, C) op T — eliminate redundant floor
        #   max(C, X) > T  where C <  T  →  X > T   (C can't win at threshold T)
        #   max(C, X) < T  where C <  T  →  X < T   (max = X when X drives it past C)
        #   max(C, X) > T  where C >= T  →  True     (floor already exceeds threshold)
        #   max(C, X) < T  where C >= T  →  False    (floor already meets threshold)
        # Same rules apply when argument order is swapped.
        def _rmax_op(m, c_idx, x_idx):
            c, x, op, t = float(m.group(c_idx)), m.group(x_idx), m.group(3), float(m.group(4))
            if op == ">":
                if c < t:  return f"{x} > {m.group(4)}"
                if c > t:  return "1.0"   # always true
            else:  # "<"
                if c < t:  return f"{x} < {m.group(4)}"
                if c >= t: return "0.0"   # always false
            return m.group(0)
        new = re.sub(rf"max\(({NUM}),\s*(\w+)\)\s*([><])\s*({NUM})",
                     lambda m: _rmax_op(m, 1, 2), expr)
        if new != expr:
            expr, changed = new, True; continue
        new = re.sub(rf"max\((\w+),\s*({NUM})\)\s*([><])\s*({NUM})",
                     lambda m: _rmax_op(m, 2, 1), expr)
        if new != expr:
            expr, changed = new, True; continue

        # min(C, X) op T / min(X, C) op T — eliminate redundant ceiling
        #   min(C, X) < T  where C >  T  →  X < T   (ceiling can't win below T)
        #   min(C, X) > T  where C >  T  →  X > T
        #   min(C, X) < T  where C <= T  →  True
        #   min(C, X) > T  where C <= T  →  False
        def _rmin_op(m, c_idx, x_idx):
            c, x, op, t = float(m.group(c_idx)), m.group(x_idx), m.group(3), float(m.group(4))
            if op == "<":
                if c > t:  return f"{x} < {m.group(4)}"
                if c <= t: return "1.0"   # always true
            else:  # ">"
                if c > t:  return f"{x} > {m.group(4)}"
                if c <= t: return "0.0"   # always false
            return m.group(0)
        new = re.sub(rf"min\(({NUM}),\s*(\w+)\)\s*([><])\s*({NUM})",
                     lambda m: _rmin_op(m, 1, 2), expr)
        if new != expr:
            expr, changed = new, True; continue
        new = re.sub(rf"min\((\w+),\s*({NUM})\)\s*([><])\s*({NUM})",
                     lambda m: _rmin_op(m, 2, 1), expr)
        if new != expr:
            expr, changed = new, True; continue

        # Between expression redundant-bound elimination  (Unicode ≤ = ≤)
        # Patterns:
        #   (C1 ≤ C2 ≤ expr) where C1 ≤ C2  →  (C2 ≤ expr)    left trivially True
        #   (C1 ≤ C2 ≤ expr) where C1 >  C2  →  0.0             always False
        #   (expr ≤ C1 ≤ C2) where C1 ≤ C2  →  (expr ≤ C1)    right trivially True
        #   (expr ≤ C1 ≤ C2) where C1 >  C2  →  0.0             always False
        # expr may contain nested parentheses so we use balanced-paren scanning.

        # ── Left-constant case: (C1 ≤ C2 ≤ expr) ────────────────────────────
        m = re.search(rf'\(({NUM})\s*≤\s*({NUM})\s*≤\s*', expr)
        if m:
            c1, c2 = float(m.group(1)), float(m.group(2))
            depth, i = 1, m.end()
            while i < len(expr) and depth > 0:
                if expr[i] == '(':   depth += 1
                elif expr[i] == ')': depth -= 1
                i += 1
            rest = expr[m.end():i - 1]
            if c1 <= c2:
                new = expr[:m.start()] + f"({_fmt(c2)} ≤ {rest})" + expr[i:]
            else:
                new = expr[:m.start()] + "0.0" + expr[i:]
            if new != expr:
                expr, changed = new, True; continue

        # ── Right-constant case: (expr ≤ C1 ≤ C2) ───────────────────────────
        m = re.search(rf'≤\s*({NUM})\s*≤\s*({NUM})\s*\)', expr)
        if m:
            c1, c2 = float(m.group(1)), float(m.group(2))
            close_pos = m.end() - 1   # position of the closing )
            depth, i = 1, close_pos - 1
            while i >= 0 and depth > 0:
                if expr[i] == ')':   depth += 1
                elif expr[i] == '(': depth -= 1
                i -= 1
            open_pos = i + 1          # position of the matching (
            lhs = expr[open_pos + 1 : m.start()]
            if c1 <= c2:
                new = expr[:open_pos] + f"({lhs} ≤ {_fmt(c1)})" + expr[close_pos + 1:]
            else:
                new = expr[:open_pos] + "0.0" + expr[close_pos + 1:]
            if new != expr:
                expr, changed = new, True; continue

        # ── Mid-constant chained ≤: (A ≤ C ≤ B) → AND((A ≤ C), (C ≤ B)) ──────
        # Fires when the middle term is a bare constant and neither adjacent-
        # constant case above matched.  The split lets subsequent rules simplify
        # each half independently (including algebraic inversion below).
        # Works for both parenthesised forms `(A ≤ C ≤ B)` and bare top-level
        # forms `A ≤ C ≤ B` (no enclosing parens).
        _cm = re.search(rf'≤\s*({NUM})\s*≤', expr)
        if _cm:
            _cv = float(_cm.group(1)); _cf = _fmt(_cv)
            # Scan backwards to find the ( that opened this Between expression.
            _i, _depth, _open = _cm.start() - 1, 0, -1
            while _i >= 0 and expr[_i] in ' \t': _i -= 1
            while _i >= 0:
                if expr[_i] == ')':   _depth += 1
                elif expr[_i] == '(':
                    if _depth == 0:   _open = _i; break
                    _depth -= 1
                _i -= 1
            if _open >= 0:
                # Enclosed form: (A ≤ C ≤ B)
                _depth, _j = 1, _open + 1
                while _j < len(expr) and _depth > 0:
                    if expr[_j] == '(':   _depth += 1
                    elif expr[_j] == ')': _depth -= 1
                    _j += 1
                _close = _j - 1
                _lhs_p = expr[_open + 1 : _cm.start()].strip()
                _rhs_p = expr[_cm.end()  : _close].strip()
                _bc = True
                for _p in (_lhs_p, _rhs_p):
                    try: float(_p)
                    except ValueError: _bc = False; break
                if not _bc:
                    _piece = f"AND(({_lhs_p} ≤ {_cf}), ({_cf} ≤ {_rhs_p}))"
                    new = expr[:_open] + _piece + expr[_close + 1:]
                    if new != expr: expr, changed = new, True; continue
            else:
                # Top-level form: entire expr is A ≤ C ≤ B with no outer parens
                _lhs_p = expr[:_cm.start()].strip()
                _rhs_p = expr[_cm.end():].strip()
                _bc = True
                for _p in (_lhs_p, _rhs_p):
                    try: float(_p)
                    except ValueError: _bc = False; break
                if not _bc:
                    _piece = f"AND(({_lhs_p} ≤ {_cf}), ({_cf} ≤ {_rhs_p}))"
                    if _piece != expr: expr, changed = _piece, True; continue

        # ── General chained inequality split: A op1 M op2 B → AND((A op1 M), (M op2 B)) ──
        # Handles any middle term (variable, expression, or constant not already
        # caught by the mid-constant rule above).  Scans only top-level operators
        # (depth == 0) so nested inequalities inside IF/AND/parens are ignored.
        _ci_ops = []
        _ci_depth, _ci_i = 0, 0
        while _ci_i < len(expr):
            if expr[_ci_i] == '(':
                _ci_depth += 1
            elif expr[_ci_i] == ')':
                _ci_depth -= 1
            elif _ci_depth == 0:
                for _iop in ('<=', '>=', '≤', '≥', '<', '>'):
                    if expr[_ci_i:_ci_i + len(_iop)] == _iop:
                        _ci_ops.append((_ci_i, _iop))
                        _ci_i += len(_iop) - 1
                        break
            _ci_i += 1
        if len(_ci_ops) == 2:
            (_p1, _op1), (_p2, _op2) = _ci_ops
            _lhs_ci = expr[:_p1].strip()
            _mid_ci = expr[_p1 + len(_op1):_p2].strip()
            _rhs_ci = expr[_p2 + len(_op2):].strip()
            _piece_ci = f"AND(({_lhs_ci} {_op1} {_mid_ci}), ({_mid_ci} {_op2} {_rhs_ci}))"
            if _piece_ci != expr:
                expr, changed = _piece_ci, True; continue

        # ── Algebraic inversion: (C1 - inner) op C2 → (inner flip_op C1-C2) ───
        # Example: (6.0 - X) ≤ 5.0  →  (X ≥ 1.0)
        # Negating both sides of (C1 - inner) op C2 flips the operator.
        _FLIP = {'≤': '≥', '≥': '≤', '<': '>', '>': '<',
                 '<=': '>=', '>=': '<='}
        _ai1_re = re.compile(rf'\(({NUM})\s*-\s*')
        _ai1_done = False
        for _aim in reversed(list(_ai1_re.finditer(expr))):
            _c1 = float(_aim.group(1))
            _depth, _j = 1, _aim.end()
            while _j < len(expr) and _depth > 0:
                if expr[_j] == '(':   _depth += 1
                elif expr[_j] == ')': _depth -= 1
                _j += 1
            _cj = _j - 1
            _inner = expr[_aim.end() : _cj]
            _after = expr[_cj + 1:].lstrip()
            # Require a bare constant immediately after the closing )
            _opm = re.match(r'(≤|≥|<=|>=|<|>)\s*(-?[\d.eE+\-]+)(?![\d.])', _after)
            if _opm:
                _op2, _c2 = _opm.group(1), float(_opm.group(2))
                _skip = (len(expr[_cj + 1:]) - len(_after)) + _opm.end()
                _end_pos = _cj + 1 + _skip
                _rep = f"({_inner} {_FLIP[_op2]} {_fmt(_c1 - _c2)})"
                new = expr[:_aim.start()] + _rep + expr[_end_pos:]
                if new != expr: expr, changed = new, True; _ai1_done = True; break
        if _ai1_done: continue

        # ── Algebraic inversion: C2 op (C1 - inner) → (inner op C1-C2) ────────
        # Example: 5.0 ≤ (16.67 - IF(...))  →  (IF(...) ≤ 11.67)
        # Here the op direction is preserved (only rhs shifts).
        _ai2_re = re.compile(rf'({NUM})\s*(≤|≥|<=|>=|<|>)\s*\(({NUM})\s*-\s*')
        _ai2_done = False
        for _aim in reversed(list(_ai2_re.finditer(expr))):
            _c2, _op2, _c1 = float(_aim.group(1)), _aim.group(2), float(_aim.group(3))
            _depth, _j = 1, _aim.end()
            while _j < len(expr) and _depth > 0:
                if expr[_j] == '(':   _depth += 1
                elif expr[_j] == ')': _depth -= 1
                _j += 1
            _cj = _j - 1
            _inner = expr[_aim.end() : _cj]
            _rep = f"({_inner} {_op2} {_fmt(_c1 - _c2)})"
            new = expr[:_aim.start()] + _rep + expr[_cj + 1:]
            if new != expr: expr, changed = new, True; _ai2_done = True; break
        if _ai2_done: continue

        # IF(N, A, B) — condition is a constant; A and B may be complex
        m = re.search(r"IF\(", expr)
        if m:
            args, end = _find_if_args(expr, m.end())
            if args and len(args) == 3:
                try:
                    cond_val = float(args[0])
                    replacement = args[1] if cond_val else args[2]
                    expr = expr[: m.start()] + replacement + expr[end:]
                    changed = True
                    continue
                except ValueError:
                    pass  # condition is not a bare constant yet

        # IF(cond, C1, C2) op C3  where C1, C2, C3 are all numeric constants.
        # The IF just routes between two constant outputs; the comparison result
        # depends only on whether the condition is true or false.
        #   C1 op C3 = True  and  C2 op C3 = False  →  cond
        #   C1 op C3 = False and  C2 op C3 = True   →  NOT(cond)
        #   both True / both False                   →  1.0 / 0.0
        # This cleans up IF(condition, score_a, score_b) op threshold patterns
        # that arise when the _is_boolean fix correctly detects IfThenElse as
        # a continuous expression and appends a threshold.
        _if_cmp_m = re.match(r'^IF\(', expr)
        if _if_cmp_m:
            _ic_args, _ic_end = _find_if_args(expr, _if_cmp_m.end())
            if _ic_args and len(_ic_args) == 3 and _ic_end < len(expr):
                _ic_rest = expr[_ic_end:].strip()
                _ic_op_m = re.match(
                    r'^([><])\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$',
                    _ic_rest
                )
                if _ic_op_m:
                    try:
                        _ic_c1  = float(_ic_args[1])
                        _ic_c2  = float(_ic_args[2])
                        _ic_c3  = float(_ic_op_m.group(2))
                        _ic_op  = _ic_op_m.group(1)
                        _ic_cond = _ic_args[0].strip()
                        _c1_fires = (_ic_c1 > _ic_c3) if _ic_op == '>' else (_ic_c1 < _ic_c3)
                        _c2_fires = (_ic_c2 > _ic_c3) if _ic_op == '>' else (_ic_c2 < _ic_c3)
                        if _c1_fires and not _c2_fires:
                            new = _ic_cond
                        elif _c2_fires and not _c1_fires:
                            new = f"NOT({_ic_cond})"
                        elif _c1_fires and _c2_fires:
                            new = '1.0'
                        else:
                            new = '0.0'
                        if new != expr:
                            expr, changed = new, True; continue
                    except (ValueError, IndexError):
                        pass

        # ── Calibration-based rules ───────────────────────────────────────────
        if calibration:
            # |X| → X when X is provably non-negative (interval lo >= 0)
            for m in reversed(list(re.finditer(r'\|([^|]+)\|', expr))):
                inner_x = m.group(1)
                iv = _interval_eval(inner_x, feature_ranges)
                if iv is not None and iv[0] >= 0:
                    expr = expr[:m.start()] + inner_x + expr[m.end():]
                    changed = True
                    break
            if changed: continue

            # Binary feature comparison simplifications.
            # A binary feature can only be 0 or 1, so thresholds that split at
            # 0.5 reduce to simple boolean identities.
            if binary_features:
                bin_re = '|'.join(
                    re.escape(f) for f in sorted(binary_features, key=len, reverse=True)
                )
                # (f > 0.5) | (f >= 1.0) | (f == 1.0) | (f != 0.0) → f
                new = re.sub(
                    rf'\(({bin_re})\s*'
                    rf'(?:>\s*0\.5|>=\s*1\.0|==\s*1\.0|!=\s*0\.0)\)',
                    r'\1', expr
                )
                if new != expr:
                    expr, changed = new, True; continue
                # (f < 0.5) | (f <= 0.0) | (f == 0.0) | (f != 1.0) → NOT(f)
                new = re.sub(
                    rf'\(({bin_re})\s*'
                    rf'(?:<\s*0\.5|<=\s*0\.0|==\s*0\.0|!=\s*1\.0)\)',
                    r'NOT(\1)', expr
                )
                if new != expr:
                    expr, changed = new, True; continue

        # ── Interval-based comparison folding ────────────────────────────────
        # When feature_ranges are provided, fold (X OP C) comparisons whose
        # result is guaranteed by the feature's data range.
        # e.g. hours_per_week ∈ [1,99]: (hours_per_week < 101) → 1.0 (always True)
        if feature_ranges:
            def _fold_range_cmp(op_re, decide):
                nonlocal expr
                chg = False
                for m in reversed(list(re.finditer(rf'\s+{op_re}\s+({NUM})\)', expr))):
                    try:
                        const_val = float(m.group(1))
                    except ValueError:
                        continue
                    op_start, close_idx = m.start(), m.end() - 1
                    depth, i = 1, op_start - 1
                    while i >= 0 and depth:
                        if expr[i] == ')':   depth += 1
                        elif expr[i] == '(': depth -= 1
                        i -= 1
                    if depth:
                        continue
                    open_idx = i + 1
                    x = expr[open_idx + 1 : op_start].strip()
                    iv = _interval_eval(x, feature_ranges)
                    if iv is None:
                        continue
                    result = decide(iv[0], iv[1], const_val)
                    if result is not None:
                        expr = expr[:open_idx] + result + expr[close_idx + 1:]
                        chg = True
                        break
                return chg

            def _dt_lt(lo, hi, c): return '1.0' if hi < c  else ('0.0' if lo >= c else None)
            def _dt_gt(lo, hi, c): return '1.0' if lo > c  else ('0.0' if hi <= c else None)
            def _dt_le(lo, hi, c): return '1.0' if hi <= c else ('0.0' if lo > c  else None)
            def _dt_ge(lo, hi, c): return '1.0' if lo >= c else ('0.0' if hi < c  else None)

            if _fold_range_cmp(r'<',  _dt_lt): changed = True; continue
            if _fold_range_cmp(r'>',  _dt_gt): changed = True; continue
            if _fold_range_cmp(r'<=', _dt_le): changed = True; continue
            if _fold_range_cmp(r'>=', _dt_ge): changed = True; continue

    # Post-loop: root-level nan comparison always evaluates to False
    # (covers expressions like "nan < 0" produced by SymPy canonicalization)
    if re.match(r'^nan\s*(?:>=|<=|>|<|==|!=)\s*[-+]?[\d.]', expr.strip()):
        expr = '0.0'

    return expr


# ── Deduplication helpers ──────────────────────────────────────────────────────

_SYMPY_SKIP = ("AND", "NOT", "IF", "Between", "CatEq", "==", "abs(", "max(", "min(", "|")


_FLIP_OP_SYMPY = {'>': '<', '<': '>', '>=': '<=', '<=': '>='}


def _sympy_canon(expr_str: str) -> str:
    """Canonicalise a pure-arithmetic comparison.

    Transforms  LHS op RHS  (op ∈ {>, <, >=, <=}, RHS a bare constant) into a
    canonical form by applying the following steps in order:

    1. Move all additive constants from LHS to RHS (no product expansion).
    2. Extract a leading multiplicative scalar and divide both sides by it
       (flipping the operator if the scalar is negative).  Handles C*expr and
       expr/C uniformly via SymPy's as_coeff_Mul().
    3. If only one free variable remains after steps 1-2, use SymPy's
       reduce_inequalities to fully solve for that variable.
    4. Evaluate pure-numeric comparisons to '1.0' / '0.0'.
    5. Fallback: if none of the above changed anything, try simplify(lhs) and
       accept only if strictly fewer ops.

    Skips silently for expressions containing GP keywords (AND, NOT, IF, …).
    """
    if any(tok in expr_str for tok in _SYMPY_SKIP):
        return expr_str
    import re as _re
    m = _re.match(r'^(.*)\s*(>=|<=|>|<)\s*(-?[\d.eE+\-]+)\s*$', expr_str.strip())
    if m is None:
        return expr_str
    lhs_str, op, rhs_str = m.group(1).strip(), m.group(2), m.group(3)
    if _re.search(r'\b0\.0\s*/\s*0\.0\b', lhs_str) or _re.search(r'/\s*0\.0\b', lhs_str):
        return expr_str
    try:
        import sympy as _sy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations

        rhs_val = float(rhs_str)
        lhs = parse_expr(lhs_str, transformations=standard_transformations)
        rhs_sym = _sy.sympify(rhs_val)

        # ── Step 1: move additive constants to RHS (no expand) ───────────────
        const_part = _sy.S.Zero
        if lhs.is_Add:
            const_args = [t for t in lhs.args if not t.free_symbols]
            if const_args:
                const_part = _sy.Add(*const_args, evaluate=True)
        var_part = lhs - const_part
        new_rhs  = rhs_sym - const_part

        # ── Step 2: pure-numeric resolution ──────────────────────────────────
        if not var_part.free_symbols:
            val = float((var_part - new_rhs).evalf())
            is_true = {'>': val > 0, '>=': val >= 0, '<': val < 0, '<=': val <= 0}.get(op)
            if is_true is not None:
                return '1.0' if is_true else '0.0'
            return expr_str

        # ── Step 3: extract leading scalar coefficient and divide both sides ─
        # Handles: C*expr, expr/C, and (C1*X + C2) after step 1 stripped C2.
        # as_coeff_Mul() returns (scalar, rest) even for expr/C (scalar=1/C).
        coeff, rest = var_part.as_coeff_Mul()
        if coeff != _sy.S.One and coeff != _sy.S.Zero and coeff.is_number:
            c_float = float(coeff.evalf())
            new_rhs2 = new_rhs / coeff
            op2 = _FLIP_OP_SYMPY.get(op, op) if c_float < 0 else op
            try:
                new_rhs2_val = float(new_rhs2.evalf())
            except Exception:
                pass
            else:
                var_part, new_rhs, op = rest, _sy.sympify(new_rhs2_val), op2

        # ── Step 4: single-variable solve (isolate the one free symbol) ──────
        free = var_part.free_symbols
        if len(free) == 1:
            [x] = free
            try:
                soln = _sy.reduce_inequalities(
                    _sy.StrictLessThan(var_part, new_rhs) if op == '<' else
                    _sy.StrictGreaterThan(var_part, new_rhs) if op == '>' else
                    _sy.LessThan(var_part, new_rhs) if op == '<=' else
                    _sy.GreaterThan(var_part, new_rhs),
                    [x]
                )
                # reduce_inequalities returns an And/Or/Relational; extract if simple
                if hasattr(soln, 'lhs') and hasattr(soln, 'rhs') and not soln.rhs.free_symbols:
                    s_op = {
                        _sy.StrictLessThan:    '<',
                        _sy.StrictGreaterThan: '>',
                        _sy.LessThan:          '<=',
                        _sy.GreaterThan:       '>=',
                    }.get(type(soln))
                    if s_op:
                        bound = float(soln.rhs.evalf())
                        return f"{soln.lhs} {s_op} {bound:.6g}"
            except Exception:
                pass

        # ── Step 5: emit if something changed in steps 1-3 ───────────────────
        changed_something = (const_part != _sy.S.Zero) or (coeff != _sy.S.One and coeff != _sy.S.Zero and coeff.is_number)
        if changed_something:
            try:
                rhs_out = float(new_rhs.evalf())
            except Exception:
                return expr_str
            return f"{var_part} {op} {rhs_out:.6g}"

        # ── Step 6: fallback — try simplify(lhs), accept only if strictly simpler
        simplified = _sy.simplify(lhs)
        if _sy.count_ops(simplified) < _sy.count_ops(lhs):
            return f"{simplified} {op} {rhs_str}"
    except Exception:
        pass
    return expr_str


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    return intersection / union if union > 0 else 1.0


def _dedup_partials(
    partial_rows: list[dict],
    datasets: dict,
    jaccard_threshold: float = 0.90,
) -> list[dict]:
    """Remove redundant partial proxies before saving.

    Per (dataset, protected_attr) group:
      1. Canonicalize arithmetic expressions with SymPy.
      2. Drop exact string duplicates, keeping the highest-precision copy.
      3. Greedy Jaccard dedup: skip any candidate whose binary firing vector
         overlaps ≥ jaccard_threshold with an already-kept candidate.
         Candidates are processed best-first (precision desc, then auc desc).

    Rows without numpy_expr (single-categorical features) pass through step 3
    unchanged since their firing vector cannot be recomputed here.
    """
    from collections import defaultdict
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(partial_rows):
        groups[(row["dataset"], row["protected_attr"])].append(i)

    keep_indices: set[int] = set()

    for (ds_label, pa), indices in groups.items():
        ds            = datasets.get(ds_label, {})
        X_test        = ds.get("X_test")
        feature_names = ds.get("feature_names", [])

        # Use pre-computed calibration (from _calibrate_dataset) if available;
        # fall back to computing feature ranges from X_test on the fly.
        calibration: dict = ds.get("calibration") or {}
        if not calibration and X_test is not None and feature_names:
            X_arr = np.asarray(X_test, dtype=float)
            fr: dict = {}
            for j, name in enumerate(feature_names):
                col = X_arr[:, j]
                col = col[np.isfinite(col)]
                if len(col) > 0:
                    fr[name] = (float(col.min()), float(col.max()))
            calibration = {"feature_ranges": fr}
        feature_ranges = calibration.get("feature_ranges", {})

        # ── Step 1: Simplify expressions (constant-fold → SymPy) ──────────────
        for i in indices:
            row = partial_rows[i]
            row["numpy_expr"] = _constant_fold_numpy(row.get("numpy_expr", ""))
            row["expression"] = _constant_fold_readable(row["expression"], calibration or None)
            row["expression"] = _sympy_canon(row["expression"])
            row["expression"] = _constant_fold_readable(row["expression"], calibration or None)
            row["expression"] = _simplify_expr(row["expression"])

        # ── Step 1b: Drop trivially constant expressions ───────────────────────
        # An expression whose value is the same for every possible feature
        # combination provides no discriminatory information and is discarded.
        if feature_ranges:
            indices = [
                i for i in indices
                if _interval_eval(partial_rows[i]["expression"], feature_ranges)
                   not in ((0.0, 0.0), (1.0, 1.0))
            ]

        # ── Step 2: String dedup after canonicalization ────────────────────────
        best_for_expr: dict[str, int] = {}
        for i in indices:
            expr = partial_rows[i]["expression"]
            prev = best_for_expr.get(expr)
            if prev is None:
                best_for_expr[expr] = i
            else:
                # Keep the candidate with higher precision; break ties by AUC
                curr_key = (partial_rows[i]["precision"], partial_rows[i]["auc"])
                prev_key = (partial_rows[prev]["precision"], partial_rows[prev]["auc"])
                if curr_key > prev_key:
                    best_for_expr[expr] = i

        deduped = sorted(best_for_expr.values())

        # ── Step 3: Jaccard behavioral dedup ──────────────────────────────────
        if X_test is None:
            keep_indices.update(deduped)
            continue

        # Process simpler (shorter) expressions first so that when two expressions
        # fire on the same individuals the simpler one is kept and the more complex
        # redundant one is discarded.  Quality is a secondary tiebreaker so that
        # among expressions of equal length the higher-precision one wins.
        deduped.sort(
            key=lambda i: (
                len(partial_rows[i]["expression"]),   # shorter first (asc)
                -partial_rows[i]["precision"],         # then higher precision (desc)
                -partial_rows[i]["auc"],               # then higher AUC (desc)
            ),
        )

        kept_vectors: list[np.ndarray] = []
        for i in deduped:
            row       = partial_rows[i]
            numpy_str = row.get("numpy_expr", "")
            threshold = row.get("_threshold", 0.0)

            if not numpy_str:
                # No numpy_expr → can't compute vector; keep unconditionally
                keep_indices.add(i)
                continue

            try:
                scores = np.asarray(
                    _forward_dataset(numpy_str, X_test), dtype=np.float64
                ).ravel()
                scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
                fired  = scores > threshold
            except Exception:
                keep_indices.add(i)
                continue

            if all(_jaccard(fired, kv) < jaccard_threshold for kv in kept_vectors):
                kept_vectors.append(fired)
                keep_indices.add(i)

    return [partial_rows[i] for i in sorted(keep_indices)]


def run_gp_stage(
    datasets:      dict[str, dict],
    grammar_mode:  str,
    cfg:           dict,
    time_budget:   float = 120,
    population:    int   = 200,
    max_depth:     int   = 6,
    seed:          int   = 42,
    prec_weight:   float = 1.0,
    rec_weight:    float = 1.0,
    cov_weight:    float = 1.0,
    penalty:       float = 0.03,
    n_jobs:        int   = 1,
    disable_nodes: frozenset = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run GP with *grammar_mode* for every dataset-split × protected-attribute pair.

    n_jobs controls parallelism:
      1   — sequential (default, safest)
      N>1 — N threads, one per (dataset, protected_attr) pair
      -1  — use all CPU cores

    Returns (main_df, partials_df, near_miss_df).
    """
    import inspect as _inspect

    grammar_fn     = _get_grammar_builder(cfg)
    _grammar_params = set(_inspect.signature(grammar_fn).parameters)

    # "continuous" grammar uses AUC fitness, not prec/rec/cov
    fitness_mode = "auc" if grammar_mode == "continuous" else "prec_rec_cov"

    # ── Build a list of tasks (ds_label, ds, pa, grammar) ─────────────────────
    tasks: list[tuple] = []
    grammars: dict[str, object] = {}   # shared per ds_label (read-only after build)

    for ds_label, ds in datasets.items():
        _extra = (
            dict(
                numeric_features=ds.get("numeric_features", []),
                categorical_features=ds.get("categorical_features", []),
                label_encodings=ds.get("label_encodings", {}),
            )
            if "numeric_features" in _grammar_params else {}
        )

        # Upgrade "extended" → "typed" when dataset has categorical features
        eff_mode = grammar_mode
        if (
            grammar_mode == "extended"
            and "numeric_features" in _grammar_params
            and ds.get("categorical_features")
            and ds.get("label_encodings")
        ):
            eff_mode = "typed"

        if disable_nodes:
            _extra["disable_nodes"] = disable_nodes
        print(f"  [{ds_label}]  building {eff_mode} grammar …", flush=True)
        grammar = grammar_fn(ds["feature_names"], eff_mode, **_extra)
        grammars[ds_label] = grammar

        for pa in cfg["protected_attrs"]:
            if pa not in ds["pa_train"]:
                print(f"    {pa:<25} [SKIP] not in dataset")
                continue
            if len(np.unique(ds["pa_train"][pa])) < 2:
                print(f"    {pa:<25} [SKIP] only one class")
                continue
            tasks.append((ds_label, ds, pa, grammar))

    # ── Worker function ────────────────────────────────────────────────────────
    def _run_one(task):
        ds_label, ds, pa, grammar = task
        print(f"    {pa:<25} [START]", flush=True)
        # Derive a per-task seed so parallel runs don't share state
        task_seed = seed ^ (hash(ds_label + pa) & 0xFFFF)
        result = run_gp(
            ds, pa, grammar,
            time_budget, population, max_depth,
            task_seed, prec_weight, rec_weight, cov_weight, penalty,
            fitness_mode=fitness_mode,
        )
        return ds_label, pa, result

    # ── Execute (sequential or threaded) ──────────────────────────────────────
    actual_jobs = (
        len(tasks) if n_jobs < 0 else min(n_jobs, len(tasks))
    ) if len(tasks) > 0 else 1

    if actual_jobs <= 1:
        task_results = [_run_one(t) for t in tasks]
    else:
        print(f"  Running {len(tasks)} GP tasks on {actual_jobs} threads …",
              flush=True)
        with _cf.ThreadPoolExecutor(max_workers=actual_jobs) as executor:
            futures = {executor.submit(_run_one, t): t for t in tasks}
            task_results = []
            for fut in _cf.as_completed(futures):
                task_results.append(fut.result())

    # ── Collect results in original order ─────────────────────────────────────
    # Re-sort by the order tasks were submitted so CSV rows are deterministic
    task_order = {(t[0], t[2]): i for i, t in enumerate(tasks)}
    task_results.sort(key=lambda r: task_order.get((r[0], r[1]), 0))

    rows:          list[dict] = []
    partial_rows:  list[dict] = []
    near_miss_rows: list[dict] = []

    for ds_label, pa, result in task_results:
        ds           = datasets[ds_label]
        baselines    = ds.get("baselines", pd.DataFrame())
        expr_str     = result["expression"]

        # Simplify the main-result expression the same way partials are
        # simplified in _dedup_partials: constant-fold → SymPy canon →
        # constant-fold again.  This moves additive offsets from the LHS into
        # the threshold (e.g. "(1.985 + X) - Y < 1.985" → "X - Y < 0"),
        # isolates single-variable expressions, and removes abs() from
        # provably non-negative features.
        cal = ds.get("calibration")
        expr_str = _constant_fold_readable(expr_str, cal)
        expr_str = _sympy_canon(expr_str)
        expr_str = _constant_fold_readable(expr_str, cal)
        expr_str = _simplify_expr(expr_str)

        print(
            f"    {pa:<25} "
            f"auc={result['auc']:.4f}"
            f"  prec={result['precision']:.4f}  rec={result['recall']:.4f}"
            f"  cov={result['coverage']:.4f}"
            f"  ({result['elapsed_s']:.0f}s)  {expr_str[:45]}"
        )

        bl_auc, bl_feat = _gsr._best_single(baselines, pa)
        expr_feats = extract_features_from_expr(expr_str, ds["feature_names"])

        import json as _json
        _and_comp = result.get("and_components")
        rows.append({
            "dataset":               ds_label,
            "protected_attr":        pa,
            "expression":            expr_str,
            "numpy_expr":            result.get("numpy_expr", ""),
            "nodes":                 result["nodes"],
            "auc":                   result["auc"],
            "ap":                    result["ap"],
            "nmi":                   result["nmi"],
            "cohens_d":              result["cohens_d"],
            "precision":             result["precision"],
            "recall":                result["recall"],
            "coverage":              result["coverage"],
            "elapsed_s":             result["elapsed_s"],
            "best_baseline_auc":     bl_auc,
            "best_baseline_feature": bl_feat,
            "expression_features":   ", ".join(expr_feats),
            "and_components":        _json.dumps(_and_comp) if _and_comp else "",
        })

        # Partial proxies
        n_partials = len(result.get("partial_proxies", []))
        if n_partials:
            print(f"      → {n_partials} partial proxies captured")
        label_encodings = ds.get("label_encodings", {})
        feature_names   = ds["feature_names"]
        X_test          = ds["X_test"]
        for p in result.get("partial_proxies", []):
            numpy_str  = p.get("numpy_expr", "")
            threshold  = p.get("threshold", 0.0)
            expr_feats = [f.strip() for f in p.get("features", "").split(",") if f.strip()]
            _single_cat = (
                len(expr_feats) == 1 and expr_feats[0] in label_encodings
            )
            if numpy_str and not _single_cat:
                profile = compute_firing_profile(
                    numpy_str, X_test, feature_names,
                    threshold, label_encodings,
                    expression_features=expr_feats or None,
                )
            else:
                profile = {}
            partial_rows.append({
                "dataset":        ds_label,
                "protected_attr": pa,
                **{k: p[k] for k in _PARTIAL_CSV_COLS
                   if k not in ("dataset", "protected_attr")},
                "firing_profile": profile,
                "_threshold":     threshold,   # used by _dedup_partials; dropped on CSV save
            })

        # Near-miss proxies (sub-threshold best efforts)
        for n in result.get("near_miss_proxies", []):
            nm_expr = n.get("expression", "")
            cal = ds.get("calibration")
            nm_expr = _constant_fold_readable(nm_expr, cal)
            nm_expr = _sympy_canon(nm_expr)
            nm_expr = _constant_fold_readable(nm_expr, cal)
            nm_expr = _simplify_expr(nm_expr)
            near_miss_rows.append({
                "dataset":        ds_label,
                "protected_attr": pa,
                **{k: n.get(k, "") for k in _PARTIAL_CSV_COLS
                   if k not in ("dataset", "protected_attr")},
                "expression": nm_expr,
            })

    # ── Deduplicate partial proxies before converting to DataFrame ────────────
    if partial_rows:
        n_before = len(partial_rows)
        partial_rows = _dedup_partials(partial_rows, datasets)
        n_after = len(partial_rows)
        if n_after < n_before:
            print(f"  [dedup] {n_before} → {n_after} partial proxies "
                  f"({n_before - n_after} removed as redundant)")

    main_df = (pd.DataFrame(rows, columns=_CSV_COLS)
               if rows else pd.DataFrame(columns=_CSV_COLS))
    # _threshold is a temporary key not in _PARTIAL_CSV_COLS; columns= silently drops it
    partials_df = (pd.DataFrame(partial_rows, columns=_PARTIAL_CSV_COLS + ["firing_profile"])
                   if partial_rows else pd.DataFrame(columns=_PARTIAL_CSV_COLS + ["firing_profile"]))
    near_miss_df = (pd.DataFrame(near_miss_rows, columns=_PARTIAL_CSV_COLS)
                    if near_miss_rows else pd.DataFrame(columns=_PARTIAL_CSV_COLS))
    return main_df, partials_df, near_miss_df


# ── AND component breakdown HTML ──────────────────────────────────────────────

def _and_breakdown_html(and_components: list[dict]) -> str:
    """Render per-component metrics + attribution for an AND expression.

    Attribution % = each clause's share of the total excess precision above
    the population baseline.  A clause firing randomly (prec ≈ prevalence)
    gets 0% attribution — it is a hitchhiker.
    """
    if not and_components or len(and_components) < 2:
        return ""
    from html import escape as _esc

    def _prec_color(v):
        if v >= 85: return "#4ade80"
        if v >= 60: return "#fde68a"
        return "#f87171"

    def _attr_color(v):
        if v >= 60: return "#4ade80"
        if v >= 30: return "#fde68a"
        return "#f87171"

    header = (
        '<div style="margin-top:.6rem;background:#1c2333;border:1px solid #30363d;'
        'border-radius:6px;overflow:hidden;font-size:.76rem">'
        '<div style="background:#21262d;color:#8b949e;padding:.3rem .7rem;'
        'font-weight:600;letter-spacing:.03em">AND component breakdown</div>'
        '<table style="width:100%;border-collapse:collapse;color:#c9d1d9">'
        '<thead><tr>'
        '<th style="padding:.3rem .6rem;color:#8b949e;text-align:left;'
        'border-bottom:1px solid #30363d">Component</th>'
        '<th style="padding:.3rem .6rem;color:#8b949e;text-align:right;'
        'border-bottom:1px solid #30363d">Attribution</th>'
        '<th style="padding:.3rem .6rem;color:#8b949e;text-align:right;'
        'border-bottom:1px solid #30363d">Prec</th>'
        '<th style="padding:.3rem .6rem;color:#8b949e;text-align:right;'
        'border-bottom:1px solid #30363d">Rec</th>'
        '<th style="padding:.3rem .6rem;color:#8b949e;text-align:right;'
        'border-bottom:1px solid #30363d">Cov</th>'
        '<th style="padding:.3rem .6rem;color:#8b949e;text-align:right;'
        'border-bottom:1px solid #30363d">AUC</th>'
        '</tr></thead><tbody>'
    )
    rows_html = ""
    for i, comp in enumerate(and_components):
        bg  = "background:#161b22" if i % 2 == 0 else "background:#1c2333"
        td  = f"padding:.3rem .6rem;{bg};border-bottom:1px solid #21262d"
        prec = comp.get("precision",       0)   or 0
        rec  = comp.get("recall",          0)   or 0
        cov  = comp.get("coverage",        0)   or 0
        auc  = comp.get("auc",             0.5) or 0.5
        attr = comp.get("attribution_pct", None)
        attr_cell = (
            f'<td style="{td};text-align:right;color:{_attr_color(attr)};font-weight:700">'
            f'{attr:.0f}%</td>'
            if attr is not None else
            f'<td style="{td};text-align:right;color:#8b949e">—</td>'
        )
        rows_html += (
            f'<tr>'
            f'<td style="{td};font-family:monospace;color:#79c0ff">'
            f'{_esc(str(comp.get("expression","?")))}</td>'
            + attr_cell +
            f'<td style="{td};text-align:right;color:{_prec_color(prec)};font-weight:600">'
            f'{prec:.1f}%</td>'
            f'<td style="{td};text-align:right">{rec:.1f}%</td>'
            f'<td style="{td};text-align:right">{cov:.1f}%</td>'
            f'<td style="{td};text-align:right">{auc:.3f}</td>'
            f'</tr>'
        )
    return header + rows_html + "</tbody></table></div>"


# ── Near-miss HTML section ────────────────────────────────────────────────────

def _build_near_miss_html(
    s2_near_miss: "pd.DataFrame",
    s3_near_miss: "pd.DataFrame",
    attr_display: dict[str, str],
) -> str:
    """Build an HTML section showing the best sub-threshold expressions.

    These are expressions that came closest to qualifying as proxies but
    fell short of the 60 % precision floor or 10 % recall floor.  Useful
    for thesis discussion of groups where no strong proxy was found.
    """
    from html import escape as _esc

    all_rows = pd.concat(
        [s2_near_miss.assign(_stage="Arith"),
         s3_near_miss.assign(_stage="Typed")],
        ignore_index=True,
    )
    if all_rows.empty:
        return ""

    # Sort: by group (dataset, protected_attr), then by precision desc
    all_rows = all_rows.sort_values(
        ["dataset", "protected_attr", "precision"],
        ascending=[True, True, False],
    )

    def _bg_prec(v: float) -> str:
        t = max(0.0, min(1.0, v / 60.0))
        r = int(220 - t * 20)
        g = int(220 - t * 0 + t * 40)
        b = int(220 - t * 60)
        return f"background:rgb({r},{g},{b});color:#333"

    rows_html = ""
    prev_group = None
    for _, row in all_rows.iterrows():
        group = (row.get("dataset", ""), row.get("protected_attr", ""))
        if group != prev_group:
            prev_group = group
            ds_lbl  = str(group[0])
            pa_lbl  = attr_display.get(str(group[1]), str(group[1]))
            rows_html += (
                f'<tr style="background:#f0f4f8">'
                f'<td colspan="6" style="padding:.4rem .8rem;font-weight:700;'
                f'color:#444;border-bottom:1px solid #ccc">'
                f'{_esc(ds_lbl)} — {_esc(pa_lbl)}</td></tr>\n'
            )

        expr   = _esc(str(row.get("expression", "")))
        stage  = str(row.get("_stage", ""))
        prec   = float(row.get("precision",  0) or 0)
        rec    = float(row.get("recall",     0) or 0)
        cov    = float(row.get("coverage",   0) or 0)
        nodes  = int(  row.get("nodes",      0) or 0)

        td = "padding:.35rem .6rem;border-bottom:1px solid #e0e0e0;font-size:.85rem"
        rows_html += (
            f'<tr>'
            f'<td style="{td};font-family:monospace;color:#1a1a1a;max-width:500px;'
            f'word-break:break-word">{expr}</td>'
            f'<td style="{td};text-align:center;color:#666">{stage}</td>'
            f'<td style="{td};text-align:right;{_bg_prec(prec)}">{prec:.1f}%</td>'
            f'<td style="{td};text-align:right">{rec:.1f}%</td>'
            f'<td style="{td};text-align:right">{cov:.1f}%</td>'
            f'<td style="{td};text-align:right;color:#999">{nodes}</td>'
            f'</tr>\n'
        )

    if not rows_html:
        return ""

    return f"""
<section style="margin:2rem auto;max-width:1200px;padding:0 1rem">
  <h2 style="font-size:1.15rem;color:#444;border-bottom:2px solid #ccc;
             padding-bottom:.4rem;margin-bottom:.6rem">
    Best Expressions Below Proxy Threshold
  </h2>
  <p style="font-size:.85rem;color:#666;margin-bottom:1rem">
    These expressions were found during the GP search but did not reach the
    proxy quality bar (precision &ge; 60 % and recall &ge; 10 %).
    They represent the tool&rsquo;s best effort for groups where no strong
    proxy was identified — useful for documenting partial or inconclusive
    findings in the thesis.
  </p>
  <table style="width:100%;border-collapse:collapse;background:#fff;
                box-shadow:0 1px 4px rgba(0,0,0,.08)">
    <thead>
      <tr style="background:#5a6470;color:#fff">
        <th style="padding:.45rem .8rem;text-align:left">Expression</th>
        <th style="padding:.45rem .6rem;text-align:center">Stage</th>
        <th style="padding:.45rem .6rem;text-align:right">Precision</th>
        <th style="padding:.45rem .6rem;text-align:right">Recall</th>
        <th style="padding:.45rem .6rem;text-align:right">Coverage</th>
        <th style="padding:.45rem .6rem;text-align:right">Nodes</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</section>
"""


# ── Partial-proxy HTML section ────────────────────────────────────────────────

def _build_partials_html(
    s2_partials: pd.DataFrame,
    s3_partials: pd.DataFrame,
    attr_display: dict[str, str],
) -> str:
    """Build an HTML section summarising partial proxies from both GP stages."""
    from html import escape as _esc

    # ── colour helpers — always dark text ─────────────────────────────────────
    def _bg(val: float, lo: float, hi: float) -> str:
        """Grey (low) → green (high) background, always #222 text."""
        t   = max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9)))
        r_c = int(220 - t * 60)
        g_c = int(220 - t * 20 + t * 60)
        b_c = int(220 - t * 80)
        return f"background:rgb({r_c},{g_c},{b_c});color:#222"

    def _pct_cell(val: float, lo: float = 0.0, hi: float = 100.0) -> str:
        bold = "600" if val >= 70 else "400"
        return (f'<td style="{_bg(val,lo,hi)};font-weight:{bold};'
                f'text-align:right;padding:.35rem .6rem">{val:.1f}%</td>')

    def _auc_cell(val: float) -> str:
        bold = "600" if val >= 0.70 else "400"
        return (f'<td style="{_bg(val,0.5,1.0)};font-weight:{bold};'
                f'text-align:right;padding:.35rem .6rem">{val:.4f}</td>')

    # ── natural-language interpretation of what a partial proxy discovers ──────
    def _interpretation(row: "pd.Series", attr: str) -> str:
        prec = row["precision"]
        rec  = row["recall"]
        cov  = row["coverage"]
        feats = str(row.get("features", "")).strip()
        feat_str = f" (using {_esc(feats)})" if feats else ""

        # Classify quality tier
        if prec >= 85:
            prec_label = "very reliably"
        elif prec >= 70:
            prec_label = "reliably"
        elif prec >= 55:
            prec_label = "with moderate confidence"
        else:
            prec_label = "with limited confidence"

        if rec >= 50:
            rec_label = "majority"
        elif rec >= 20:
            rec_label = "a sizeable portion"
        elif rec >= 10:
            rec_label = "a notable fraction"
        else:
            rec_label = "a small but specific subset"

        return (
            f"When this expression fires{feat_str}, it <strong>{prec_label}</strong> "
            f"predicts <em>{_esc(attr)}</em> ({prec:.0f}% precision). "
            f"It covers {rec_label} of all <em>{_esc(attr)}</em> individuals "
            f"({rec:.0f}% recall), representing {cov:.1f}% of the full dataset. "
            f"This is a <strong>partial proxy</strong>: it captures one discriminatory "
            f"pattern but does not cover the whole group."
        )

    # ── firing profile renderer ────────────────────────────────────────────────
    def _profile_html(profile) -> str:
        """Render the firing profile dict as compact inline badges.

        profile = {feature: [{"label": str, "tooltip": str, "kind": "cat"|"cont"}, ...]}
        Categorical entries (specific values like "Husband") get green badges.
        Continuous entries (direction like "↑ high") get blue badges.
        Returns empty string if no profile data.
        """
        if not profile or not isinstance(profile, dict):
            return ""
        parts: list[str] = []
        for feat, entries in profile.items():
            if not entries:
                continue
            badges = []
            for entry in entries[:3]:  # top 3 per feature
                if not isinstance(entry, dict):
                    continue
                label   = entry.get("label", "?")
                tooltip = entry.get("tooltip", "")
                kind    = entry.get("kind", "cat")
                if kind == "cont":
                    style = ("display:inline-block;background:#cce5ff;color:#004085;"
                             "border-radius:3px;padding:1px 5px;margin:1px;"
                             "font-size:.72rem;font-weight:600")
                else:
                    style = ("display:inline-block;background:#d4edda;color:#155724;"
                             "border-radius:3px;padding:1px 5px;margin:1px;"
                             "font-size:.72rem;font-weight:600")
                badges.append(
                    f'<span title="{_esc(tooltip)}" style="{style}">'
                    f'{_esc(label)}</span>'
                )
            if badges:
                parts.append(
                    f'<span style="font-size:.72rem;color:#555">{_esc(feat)}:</span> '
                    + " ".join(badges)
                )
        if not parts:
            return ""
        return (
            '<div style="margin-top:.4rem;line-height:1.8" '
            'title="Independent per-feature tendencies — not a required combination">'
            '<span style="font-size:.72rem;color:#666;font-weight:600">'
            'Typical when firing &nbsp;</span>'
            '<span style="font-size:.68rem;color:#999">(per feature, independently)</span>'
            '<span style="font-size:.72rem;color:#666">: </span>'
            + " &nbsp; ".join(parts)
            + "</div>"
        )

    # ── per-stage table ────────────────────────────────────────────────────────
    DS_ORDER = ["processed", "non_processed", "raw", "encoded"]

    def _section(df: pd.DataFrame, stage_label: str) -> str:
        if df.empty:
            return (f"<p style='color:#666'><em>No partial proxies meeting quality "
                    f"thresholds were found for {_esc(stage_label)}.</em></p>\n")

        html = f"<h3 style='margin-top:1.8rem'>{_esc(stage_label)}</h3>\n"

        # Sort by dataset in canonical order (processed before non_processed)
        ds_present  = df["dataset"].unique().tolist()
        ordered_ds  = [d for d in DS_ORDER if d in ds_present]
        ordered_ds += [d for d in ds_present if d not in ordered_ds]
        df = df.copy()
        df["_ds_order"] = df["dataset"].map({d: i for i, d in enumerate(ordered_ds)})
        df = df.sort_values(["_ds_order", "protected_attr"]).drop(columns="_ds_order")

        for (ds, pa), grp in df.groupby(["dataset", "protected_attr"], sort=False):
            attr = attr_display.get(str(pa), str(pa))
            n    = len(grp)
            html += (
                f'<p style="margin:.8rem 0 .3rem;font-size:.95rem">'
                f'<strong>{_esc(str(ds))}</strong> &mdash; '
                f'<em>{_esc(attr)}</em> &nbsp;'
                f'<span style="color:#888;font-size:.82rem">'
                f'({n} partial {"proxy" if n==1 else "proxies"})</span></p>\n'
            )

            th = ("background:#495057;color:#fff;padding:.4rem .6rem;"
                  "text-align:left;white-space:nowrap")
            th_r = th + ";text-align:right"
            html += (
                '<div style="overflow-x:auto;margin-bottom:1.8rem">'
                '<table style="border-collapse:collapse;width:100%;font-size:.82rem">'
                f'<thead><tr>'
                f'<th style="{th}">What this proxy discovers</th>'
                f'<th style="{th_r}" title="Nodes in the expression tree">Nodes</th>'
                f'<th style="{th_r}" title="Seconds elapsed when first seen during search">Found at</th>'
                f'<th style="{th_r}" title="Precision — primary quality indicator">Precision</th>'
                f'<th style="{th_r}" title="Recall">Recall</th>'
                f'<th style="{th_r}" title="Coverage">Coverage</th>'
                f'<th style="{th_r}" title="AUC">AUC</th>'
                f'<th style="{th}">Expression</th>'
                f'</tr></thead><tbody>\n'
            )

            for i, (_, row) in enumerate(
                grp.sort_values("precision", ascending=False).iterrows()
            ):
                bg_row = "background:#f8f9fa" if i % 2 == 0 else "background:#fff"
                td = f"style='{bg_row};padding:.4rem .6rem;border-bottom:1px solid #dee2e6;color:#222'"
                html += f"<tr>\n"
                # Interpretation column
                interp = _interpretation(row, attr)
                html += (f'  <td style="{bg_row};padding:.4rem .6rem;'
                         f'border-bottom:1px solid #dee2e6;color:#222;'
                         f'max-width:320px;white-space:normal;'
                         f'font-size:.8rem;line-height:1.45">{interp}</td>\n')
                # Nodes / found-at
                html += f'  <td {td} style="{bg_row};padding:.4rem .6rem;border-bottom:1px solid #dee2e6;color:#222;text-align:right">{int(row["nodes"])}</td>\n'
                html += f'  <td style="{bg_row};padding:.4rem .6rem;border-bottom:1px solid #dee2e6;color:#222;text-align:right">{row["first_seen_s"]:.1f}s</td>\n'
                # R/P/C
                html += _pct_cell(row["precision"])
                html += _pct_cell(row["recall"])
                html += _pct_cell(row["coverage"])
                # AUC
                html += _auc_cell(row["auc"])
                # Expression + firing profile
                expr    = _esc(str(row["expression"]))
                profile = row.get("firing_profile") if "firing_profile" in row.index else {}
                prof_html = _profile_html(profile)
                html += (f'  <td style="{bg_row};padding:.4rem .6rem;'
                         f'border-bottom:1px solid #dee2e6;color:#222;'
                         f'max-width:320px;white-space:normal;vertical-align:top">'
                         f'<code style="font-size:.72rem;word-break:break-all">{expr}</code>'
                         f'{prof_html}</td>\n')
                html += "</tr>\n"

            html += "</tbody></table></div>\n"
        return html

    # ── assemble full section ──────────────────────────────────────────────────
    body = """
<hr style="border-top:2px solid #dee2e6;margin:2.5rem 0">
<h2>Partial Proxies Discovered During Search</h2>
<div style="background:#e8f4f8;border-left:4px solid #17a2b8;border-radius:0 6px 6px 0;
            padding:1rem 1.2rem;margin-bottom:1.5rem;font-size:.88rem;color:#222;line-height:1.6">
  <strong>What is a partial proxy?</strong> (Gonçalves et al., TACAS 2025)<br>
  A proxy relation is <em>partial</em> when it only holds for a <em>subset</em> of
  the data. For example, <code>Marital&nbsp;=&nbsp;husband&nbsp;&rarr;&nbsp;Sex&nbsp;=&nbsp;male</code>
  is a partial proxy: it correctly predicts Sex for rows where Marital is
  <em>husband</em> or <em>wife</em>, but not for <em>single</em>.
  This is the norm in real-world data. Each entry below captures one specific
  pattern that the GP discovered to be predictive of the protected attribute for
  part of the population. Together, multiple partial proxies explain the full
  discriminatory structure that no single expression covers completely.
  <br><br>
  <strong>How to read the table:</strong>
  <strong>Precision</strong> (primary) — when the expression fires, how often is it predicting the right group?
  <strong>Recall</strong> — what fraction of that group does it cover?
  <strong>Coverage</strong> — what fraction of the whole dataset is a true positive?
  Applied thresholds: Prec &gt; 60%, Rec &gt; 10% (no coverage floor — coverage is
  naturally bounded by group size for minority attributes).
</div>
"""
    body += _section(s2_partials, "Stage 2 — Arithmetic Grammar")
    body += _section(s3_partials, "Stage 3 — Extended Grammar")
    return body


# ── Continuous-grammar HTML section ──────────────────────────────────────────

def _build_continuous_section(
    s4_df:        pd.DataFrame,
    s4_partials:  pd.DataFrame,
    attr_display: dict[str, str],
) -> str:
    """Render Stage 4 (continuous grammar, AUC fitness) results as an HTML section.

    Shows a table of best expressions per (dataset, protected_attr) with their
    AUC and Cohen's d.  No precision/recall columns since continuous mode does
    not binarise the output — the expression is a continuous score.
    """
    from html import escape as _esc

    def _auc_color(v: float) -> str:
        if v >= 0.75: return "#4ade80"
        if v >= 0.65: return "#86efac"
        if v >= 0.55: return "#fde68a"
        return "#f87171"

    STYLE = (
        "background:#0f1117;color:#c9d1d9;"
        "font-family:'Segoe UI',system-ui,sans-serif;font-size:13px"
    )
    TH = ("background:#21262d;color:#8b949e;font-weight:600;"
          "padding:5px 8px;border-bottom:1px solid #30363d;text-align:left")
    TD = "padding:4px 8px;border-bottom:1px solid #1c2333;vertical-align:middle"

    body = (
        f'<div style="border-top:3px solid #a855f7;margin:32px 0">'
        f'<div style="background:#161b22;border-bottom:1px solid #30363d;'
        f'padding:16px 40px">'
        f'<h2 style="font-size:18px;font-weight:700;color:#e6edf3">'
        f'Stage 4 — Continuous Grammar (AUC Fitness)</h2>'
        f'<p style="color:#8b949e;font-size:12px;margin-top:4px">'
        f'Pure-arithmetic expressions (no comparisons, no booleans). '
        f'Fitness = AUC. Outputs continuous scores — no binarisation required.</p>'
        f'</div>'
        f'<div style="padding:16px 40px 32px">'
    )

    if s4_df.empty:
        body += '<p style="color:#8b949e">No results.</p>'
    else:
        body += (
            f'<table style="width:100%;border-collapse:collapse;{STYLE}">'
            f'<thead><tr>'
            f'<th style="{TH}">Dataset</th>'
            f'<th style="{TH}">Protected attr</th>'
            f'<th style="{TH}">Expression</th>'
            f'<th style="{TH}">Nodes</th>'
            f'<th style="{TH}">AUC</th>'
            f'<th style="{TH}">Cohen\'s d</th>'
            f'<th style="{TH}">Best baseline AUC</th>'
            f'</tr></thead><tbody>'
        )
        for _, row in s4_df.iterrows():
            pa_disp = attr_display.get(str(row.get("protected_attr", "")),
                                       str(row.get("protected_attr", "")))
            auc     = float(row.get("auc", 0.5))
            cd      = float(row.get("cohens_d", 0.0))
            bl_auc  = float(row.get("best_baseline_auc", 0.5))
            expr    = _esc(str(row.get("expression", ""))[:90])
            body += (
                f'<tr>'
                f'<td style="{TD}">{_esc(str(row.get("dataset","")))}</td>'
                f'<td style="{TD}">{_esc(pa_disp)}</td>'
                f'<td style="{TD};font-family:monospace;color:#79c0ff">{expr}</td>'
                f'<td style="{TD};text-align:right">{int(row.get("nodes",0))}</td>'
                f'<td style="{TD};text-align:right;color:{_auc_color(auc)}">'
                f'<b>{auc:.4f}</b></td>'
                f'<td style="{TD};text-align:right">{cd:.3f}</td>'
                f'<td style="{TD};text-align:right;color:#8b949e">{bl_auc:.4f}</td>'
                f'</tr>'
            )
        body += '</tbody></table>'

    body += '</div></div>'
    return body


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end GP proxy discovery pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        required=True,
        help="Dataset to analyse",
    )
    p.add_argument(
        "--splits", nargs="+",
        help="Which data splits to include (default: all for dataset)",
    )
    p.add_argument(
        "--classifier",
        choices=list(_gsr.CLASSIFIER_OPTIONS.keys()),
        default="logistic_regression",
        help="Classifier for ALL three stages (default: logistic_regression)",
    )
    p.add_argument(
        "--top-features", type=int, default=6,
        help="Top N features shown in Stage 1 chart (default: 6)",
    )
    # GP hyper-parameters
    p.add_argument(
        "--time-budget", type=float, default=120,
        help="GP seconds per attribute per grammar (default: 120)",
    )
    p.add_argument(
        "--population", type=int, default=200,
        help="GP population size (default: 200)",
    )
    p.add_argument(
        "--max-depth", type=int, default=6,
        help="GP max expression tree depth for binary grammars (default: 6)",
    )
    p.add_argument(
        "--cont-max-depth", type=int, default=5,
        help="GP max tree depth for continuous grammar Stage 4 (default: 5). "
             "Deeper than binary to allow multi-feature arithmetic combinations.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument("--prec-weight", type=float, default=1.0,
                   help="Precision weight in GP fitness (default: 1.0)")
    p.add_argument("--rec-weight",  type=float, default=1.0,
                   help="Recall weight in GP fitness (default: 1.0)")
    p.add_argument("--cov-weight",  type=float, default=1.0,
                   help="Coverage weight in GP fitness (default: 1.0)")
    p.add_argument(
        "--penalty", type=float, default=0.03,
        help="Complexity penalty in GP fitness (default: 0.03)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output HTML path (default: pipeline_results/{dataset}/report.html)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output root directory (default: pipeline_results/). "
             "Results land in <output-dir>/{dataset}/",
    )
    p.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel GP threads per stage: 1=sequential, -1=all cores "
             "(default: 1). Each (dataset-split × protected-attr) pair is "
             "one thread, so --jobs 4 runs 4 GP searches simultaneously.",
    )
    p.add_argument(
        "--continuous", action="store_true", default=False,
        help="Add Stage 4: GP with pure-arithmetic continuous grammar. "
             "Fitness = AUC (no binarisation). Outputs cont-<tag>.csv.",
    )
    p.add_argument(
        "--disable-nodes", default="",
        help="Comma-separated grammar node class names to exclude from the "
             "grammar (ablation study). E.g. --disable-nodes IfThenElse,Max2,Min2",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = DATASET_CONFIGS[args.dataset]

    # Resolve splits
    all_split_labels = [s[0] for s in cfg["splits"]]
    selected_splits  = args.splits if args.splits else all_split_labels

    # Output paths — classifier + hyperparameters embedded so each run stays separate.
    # Weights formatted as integers when whole (1.0→1), else one decimal (1.5→1.5).
    def _wfmt(v: float) -> str:
        return str(int(v)) if v == int(v) else f"{v:.1f}"

    _run_tag = (f"{args.classifier}"
                f"_p{_wfmt(args.prec_weight)}"
                f"_r{_wfmt(args.rec_weight)}"
                f"_c{_wfmt(args.cov_weight)}")

    out_root  = args.output_dir if args.output_dir else OUTPUT_ROOT
    out_dir   = out_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output or (out_dir / f"report-{_run_tag}.html")
    arith_csv = out_dir / f"arith-{_run_tag}.csv"
    ext_csv   = out_dir / f"ext-{_run_tag}.csv"

    clf_name, clf_spec, _, _ = _gsr.CLASSIFIER_OPTIONS[args.classifier]

    print("=" * 60)
    print(f"Dataset    : {cfg['label']}")
    print(f"Splits     : {', '.join(selected_splits)}")
    print(f"Classifier : {clf_name}  ({clf_spec})")
    print(f"GP budget  : {args.time_budget}s/attr  pop={args.population}"
          f"  depth={args.max_depth}  seed={args.seed}"
          f"  prec_w={args.prec_weight}  rec_w={args.rec_weight}  cov_w={args.cov_weight}")
    print(f"Output dir : {out_dir}")
    print("=" * 60)
    print()

    # Propagate dataset-specific globals to generate_3stage_report
    _gsr.PROTECTED_ATTRS = cfg["protected_attrs"]
    _gsr.ATTR_DISPLAY    = cfg["attr_display"]

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    datasets = load_data(cfg, selected_splits)
    if not datasets:
        print("ERROR: No datasets loaded. Check data paths and --splits.")
        sys.exit(1)

    for ds_label, ds in datasets.items():
        print(f"  {ds_label}: {len(ds['feature_names'])} features  "
              f"train={ds['n_train']}  test={ds['n_test']}")
    print()

    # ── Dataset calibration ────────────────────────────────────────────────────
    # Analyse each split once before GP runs: compute feature ranges, detect
    # binary / constant / non-negative features.  Results are cached in ds and
    # reused by constant-folding and deduplication throughout the pipeline.
    print("=== Dataset calibration ===")
    for ds_label, ds in datasets.items():
        print(f"  [{ds_label}]")
        ds["calibration"] = _calibrate_dataset(ds, verbose=True)
    print()

    # ── Stage 1: Classifier on all features ──────────────────────────────────
    print(f"=== Stage 1: {clf_name} on all base features ===")
    s1_all: dict[str, list[dict]] = {}
    for ds_label, ds in datasets.items():
        print(f"  [{ds_label}]")
        results = _gsr.run_stage1(ds, args.classifier, args.top_features)
        s1_all[ds_label] = results
        for r in results:
            attr = cfg["attr_display"].get(r["protected_attr"], r["protected_attr"])
            print(f"    {attr:<25} auc={r['auc']:.4f}"
                  f"  Δ_single={r['auc'] - r['best_single_auc']:+.4f}")
    print()

    _disabled = frozenset(n.strip() for n in args.disable_nodes.split(",") if n.strip())
    if _disabled:
        print(f"Grammar ablation: disabling nodes {sorted(_disabled)}")

    # ── Stage 2: GP + Arithmetic Grammar ─────────────────────────────────────
    print("=== Stage 2: GP + Arithmetic Grammar ===")
    s2_df, s2_partials, s2_near_miss = run_gp_stage(
        datasets, "arithmetic", cfg,
        args.time_budget, args.population, args.max_depth,
        args.seed, args.prec_weight, args.rec_weight, args.cov_weight,
        args.penalty, n_jobs=args.jobs, disable_nodes=_disabled,
    )
    s2_df.to_csv(arith_csv, index=False)
    arith_partials_csv = out_dir / f"arith-{_run_tag}-partials.csv"
    s2_partials.drop(columns=["firing_profile"], errors="ignore").to_csv(arith_partials_csv, index=False)
    arith_near_miss_csv = out_dir / f"arith-{_run_tag}-near-miss.csv"
    s2_near_miss.to_csv(arith_near_miss_csv, index=False)
    print(f"  Saved → {arith_csv}  ({len(s2_partials)} partial proxies → {arith_partials_csv.name})")
    print()

    # Evaluate Stage 2 expressions with classifier
    s2_clf: dict[str, dict] = {}
    for ds_label, ds in datasets.items():
        sub = s2_df[s2_df["dataset"] == ds_label]
        if sub.empty:
            continue
        print(f"  Evaluating Stage 2 on '{ds_label}' with {clf_name} …")
        s2_clf[ds_label] = _gsr.run_stage_gp(s2_df, ds, args.classifier)
        for pa, res in s2_clf[ds_label].items():
            attr = cfg["attr_display"].get(pa, pa)
            print(f"    {attr:<25} clf_train={res['clf_train_auc']:.4f}"
                  f"  clf_test={res['clf_test_auc']:.4f}")
    print()

    # ── Stage 3: GP + Extended Grammar ───────────────────────────────────────
    print("=== Stage 3: GP + Extended Grammar ===")
    s3_df, s3_partials, s3_near_miss = run_gp_stage(
        datasets, "extended", cfg,
        args.time_budget, args.population, args.max_depth,
        args.seed, args.prec_weight, args.rec_weight, args.cov_weight,
        args.penalty, n_jobs=args.jobs, disable_nodes=_disabled,
    )
    s3_df.to_csv(ext_csv, index=False)
    ext_partials_csv = out_dir / f"ext-{_run_tag}-partials.csv"
    s3_partials.drop(columns=["firing_profile"], errors="ignore").to_csv(ext_partials_csv, index=False)
    ext_near_miss_csv = out_dir / f"ext-{_run_tag}-near-miss.csv"
    s3_near_miss.to_csv(ext_near_miss_csv, index=False)
    print(f"  Saved → {ext_csv}  ({len(s3_partials)} partial proxies → {ext_partials_csv.name})")
    print()

    # Evaluate Stage 3 expressions with classifier
    s3_clf: dict[str, dict] = {}
    for ds_label, ds in datasets.items():
        sub = s3_df[s3_df["dataset"] == ds_label]
        if sub.empty:
            continue
        print(f"  Evaluating Stage 3 on '{ds_label}' with {clf_name} …")
        s3_clf[ds_label] = _gsr.run_stage_gp(s3_df, ds, args.classifier)
        for pa, res in s3_clf[ds_label].items():
            attr = cfg["attr_display"].get(pa, pa)
            print(f"    {attr:<25} clf_train={res['clf_train_auc']:.4f}"
                  f"  clf_test={res['clf_test_auc']:.4f}")
    print()

    # ── Stage 4 (optional): GP + Continuous Grammar ───────────────────────────
    s4_df        = pd.DataFrame(columns=_CSV_COLS)
    s4_partials  = pd.DataFrame(columns=_PARTIAL_CSV_COLS + ["firing_profile"])
    s4_near_miss = pd.DataFrame(columns=_PARTIAL_CSV_COLS)
    cont_csv     = out_dir / f"cont-{_run_tag}.csv"

    if args.continuous:
        print("=== Stage 4: GP + Continuous Grammar (AUC fitness) ===")
        print("    Pure arithmetic expressions, no comparisons or booleans.")
        print("    Fitness = AUC(continuous score, protected attr).\n")
        s4_df, s4_partials, s4_near_miss = run_gp_stage(
            datasets, "continuous", cfg,
            args.time_budget, args.population, args.cont_max_depth,
            args.seed, args.prec_weight, args.rec_weight, args.cov_weight,
            args.penalty, n_jobs=args.jobs,
        )
        s4_df.to_csv(cont_csv, index=False)
        cont_partials_csv = out_dir / f"cont-{_run_tag}-partials.csv"
        s4_partials.drop(columns=["firing_profile"], errors="ignore").to_csv(
            cont_partials_csv, index=False)
        print(f"  Saved → {cont_csv}  "
              f"({len(s4_partials)} partial proxies → {cont_partials_csv.name})")
        print()

    # ── Generate HTML report ──────────────────────────────────────────────────
    print("Generating HTML report …")
    html = _gsr.build_html(
        clf_key   = args.classifier,
        s1_all    = s1_all,
        s2_df     = s2_df,    s2_clf = s2_clf,    s2_path = arith_csv,
        s3_df     = s3_df,    s3_clf = s3_clf,    s3_path = ext_csv,
        ds_labels = selected_splits,
        top_n     = args.top_features,
    )

    # Inject continuous-grammar section into the report when Stage 4 ran
    cont_section = ""
    if args.continuous and not s4_df.empty:
        cont_section = _build_continuous_section(s4_df, s4_partials, cfg["attr_display"])

    html = html.replace("</body>", _build_partials_html(
        s2_partials, s3_partials, cfg["attr_display"]
    ) + _build_near_miss_html(
        s2_near_miss, s3_near_miss, cfg["attr_display"]
    ) + cont_section + "\n</body>")
    html_path.write_text(html, encoding="utf-8")
    print(f"Report  → {html_path}  ({html_path.stat().st_size / 1024:.0f} KB)")
    print()
    print("Done!  All outputs in:", out_dir)


if __name__ == "__main__":
    main()
