#!/usr/bin/env python3
"""
Three-Stage Proxy Discovery Report Generator
=============================================
Generates a self-contained HTML report comparing three approaches to finding
multi-attribute proxy features for protected attributes.

All three stages are evaluated using the SAME classifier, so AUC numbers
are directly comparable across stages:

  Stage 1 – Classifier Baseline
      Classifier trained on ALL base features → AUC.
      "What can a practitioner do with a black-box model?"

  Stage 2 – GP + Arithmetic Grammar
      GP finds an interpretable expression (using +, −, ×, ÷).
      The expression is evaluated on the data producing a single derived
      feature.  The SAME classifier is then trained on that ONE feature → AUC.

  Stage 3 – GP + Extended Grammar
      Same pipeline as Stage 2 but the GP used the extended operator set
      (comparisons, conditionals, abs/max/min).

Usage
-----
    python generate_3stage_report.py
    python generate_3stage_report.py --classifier random_forest
    python generate_3stage_report.py \\
        --arith-csv results/gp_proxy_grammar_results/G7_arith_baseline.csv \\
        --ext-csv   results/gp_proxy_grammar_results/G7_ext_read_d3_pen0.03_aucw2.0.csv \\
        --output    3stage_report.html

Available classifiers
---------------------
    logistic_regression   Logistic Regression (L2, C=1, max_iter=1000)
    random_forest         Random Forest (100 trees)
    gradient_boosting     Gradient Boosting (100 estimators)
    decision_tree         Decision Tree (max_depth=6)
    svm                   Linear SVM
"""

import argparse
import re
import warnings
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

DATASET_CONFIGS = {
    "law_school": {
        "label": "Law School (LSAC)",
        "splits": [
            (ROOT / "processed",     "processed",     "Processed (Cleaned, 19 features)"),
            (ROOT / "non_processed", "non_processed", "Non-Processed (Raw, all features)"),
        ],
        "npz_name": "law_school_base_features.npz",
        "default_arith_csv": (
            ROOT / "results" / "gp_proxy_grammar_results" / "G7_arith_baseline.csv"
        ),
        "default_ext_csv": (
            ROOT / "results" / "gp_proxy_grammar_results"
            / "G7_ext_read_d3_pen0.03_aucw2.0.csv"
        ),
        "protected_attrs": ["black", "asian", "hisp", "other", "male"],
        "attr_display": {
            "black": "Black", "asian": "Asian", "hisp": "Hispanic",
            "other": "Other Race", "male": "Male",
        },
        "gp_attr_map": {},
    },
    "adult": {
        "label": "Adult Census Income",
        "splits": [
            (ROOT / "adult_test_dataset" / "processed",
             "processed", "Processed (Cleaned features)"),
            (ROOT / "adult_test_dataset" / "non_processed",
             "non_processed", "Non-Processed (Raw features)"),
        ],
        "npz_name": "adult_base_features.npz",
        "default_arith_csv": None,
        "default_ext_csv": (
            ROOT / "adult_test_dataset" / "results" / "adult_extended_test.csv"
        ),
        "protected_attrs": ["black", "white", "asian_pac_islander",
                            "amer_indian", "other", "male"],
        "attr_display": {
            "black": "Black", "white": "White",
            "asian_pac_islander": "Asian/Pac. Islander",
            "amer_indian": "Amer. Indian", "other": "Other Race", "male": "Male",
        },
        # GP result CSVs may use different names than the NPZ keys.
        # These aliases are added to pa_train/pa_test at load time.
        "gp_attr_map": {"asian": "asian_pac_islander"},
    },
}

# Populated at runtime by main() based on --dataset
PROTECTED_ATTRS: list[str] = []
ATTR_DISPLAY:    dict[str, str] = {}
DATASETS:        list[tuple] = []

EXTENDED_OPS = [
    (r" > ",   "GT",  "#34d399"),
    (r" < ",   "LT",  "#34d399"),
    (r" == ",  "EQ",  "#6c8cff"),
    (r" AND ", "AND", "#fbbf24"),
    (r" OR ",  "OR",  "#fbbf24"),
    (r"NOT\(", "NOT", "#f87171"),
    (r"IF\(",  "IF",  "#fb923c"),
    (r"max\(", "max", "#22d3ee"),
    (r"min\(", "min", "#22d3ee"),
    (r"\|",    "abs", "#e879f9"),
]

CLASSIFIER_OPTIONS = {
    "logistic_regression": (
        "Logistic Regression",
        "LogisticRegression(C=1, max_iter=1000)",
        lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        True,
    ),
    "random_forest": (
        "Random Forest",
        "RandomForestClassifier(n_estimators=100)",
        lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        False,
    ),
    "gradient_boosting": (
        "Gradient Boosting",
        "GradientBoostingClassifier(n_estimators=100)",
        lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
        False,
    ),
    "decision_tree": (
        "Decision Tree",
        "DecisionTreeClassifier(max_depth=6)",
        lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
        False,
    ),
    "svm": (
        "Linear SVM",
        "LinearSVC(max_iter=2000)",
        lambda: LinearSVC(max_iter=2000, random_state=42),
        True,
    ),
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate three-stage proxy discovery HTML report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        default="law_school",
        help="Dataset to analyse (default: law_school)",
    )
    p.add_argument(
        "--classifier",
        choices=list(CLASSIFIER_OPTIONS.keys()),
        default="logistic_regression",
        help="Classifier used in ALL three stages (default: logistic_regression)",
    )
    p.add_argument(
        "--arith-csv", type=Path, default=None,
        help="CSV with GP arithmetic grammar results (Stage 2); "
             "defaults to dataset's built-in arith CSV",
    )
    p.add_argument(
        "--ext-csv", type=Path, default=None,
        help="CSV with GP extended grammar results (Stage 3); "
             "defaults to dataset's built-in ext CSV",
    )
    p.add_argument(
        "--splits", nargs="+",
        choices=["processed", "non_processed"],
        default=["processed", "non_processed"],
        dest="splits",
        help="Which dataset splits to include (default: both)",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "3stage_report.html",
    )
    p.add_argument(
        "--top-features", type=int, default=6,
        help="Top N features to show in Stage 1 importance chart",
    )
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(ds_dir: Path, ds_label: str,
                 npz_name: str, protected_attrs: list[str],
                 gp_attr_map: dict[str, str] | None = None) -> dict | None:
    npz_path = ds_dir / npz_name
    if not npz_path.exists():
        print(f"  [SKIP] {ds_label}: {npz_path} not found")
        return None

    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X_train_raw"] if "X_train_raw" in data else data["X_train"]
    X_test  = data["X_test_raw"]  if "X_test_raw"  in data else data["X_test"]
    X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64),
                            nan=0.0, posinf=1e6, neginf=-1e6)
    X_test  = np.nan_to_num(np.asarray(X_test,  dtype=np.float64),
                            nan=0.0, posinf=1e6, neginf=-1e6)

    feature_names = [str(n) for n in data["feature_names"]]
    pa_train = {}
    pa_test  = {}
    for pa in protected_attrs:
        key_tr = f"{pa}_train"
        key_te = f"{pa}_test"
        if key_tr in data and key_te in data:
            pa_train[pa] = np.asarray(data[key_tr], dtype=np.int32)
            pa_test[pa]  = np.asarray(data[key_te], dtype=np.int32)

    # Add aliases so GP CSV names (e.g. "asian") map to the same arrays
    for alias, npz_key in (gp_attr_map or {}).items():
        if npz_key in pa_train and alias not in pa_train:
            pa_train[alias] = pa_train[npz_key]
            pa_test[alias]  = pa_test[npz_key]

    bl_path   = ds_dir / "individual_proxy_baselines.csv"
    baselines = pd.read_csv(bl_path) if bl_path.exists() else pd.DataFrame()

    return {
        "label": ds_label,
        "X_train": X_train, "X_test": X_test,
        "feature_names": feature_names,
        "pa_train": pa_train, "pa_test": pa_test,
        "baselines": baselines,
        "n_train": X_train.shape[0], "n_test": X_test.shape[0],
    }


def safe_auc(y_true, y_pred) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if np.std(y_pred) < 1e-10:
        return 0.5
    auc = roc_auc_score(y_true, y_pred)
    return max(auc, 1.0 - auc)


def _best_single(baselines: pd.DataFrame, pa: str) -> tuple[float, str]:
    if baselines.empty:
        return 0.5, "none"
    col = f"auc_{pa}"
    if col not in baselines.columns:
        return 0.5, "none"
    idx = baselines[col].idxmax()
    return round(float(baselines.loc[idx, col]), 4), str(baselines.loc[idx, "feature"])


# ── Classifier helper ─────────────────────────────────────────────────────────

def _fit_and_score(clf_factory, needs_scaling: bool,
                   X_tr, X_te, y_train, y_test) -> tuple[float, float]:
    """Fit classifier and return (train_auc, test_auc)."""
    if needs_scaling:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    clf = clf_factory()
    clf.fit(X_tr, y_train)

    if hasattr(clf, "predict_proba"):
        tr_score = clf.predict_proba(X_tr)[:, 1]
        te_score = clf.predict_proba(X_te)[:, 1]
    else:
        tr_score = clf.decision_function(X_tr)
        te_score = clf.decision_function(X_te)

    return round(safe_auc(y_train, tr_score), 4), round(safe_auc(y_test, te_score), 4)


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def run_stage1(dataset: dict, clf_key: str, top_n: int) -> list[dict]:
    """Train classifier on ALL features → AUC + importances."""
    _, _, clf_factory, needs_scaling = CLASSIFIER_OPTIONS[clf_key]
    X_train       = dataset["X_train"]
    feature_names = dataset["feature_names"]

    # Always scale for the scaler-fit-transform (needed for importance extraction)
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)

    results = []
    for pa in PROTECTED_ATTRS:
        y_train = dataset["pa_train"][pa]
        if len(np.unique(y_train)) < 2:
            continue

        Xtr = X_tr_s if needs_scaling else X_train

        clf = CLASSIFIER_OPTIONS[clf_key][2]()
        clf.fit(Xtr, y_train)

        if hasattr(clf, "predict_proba"):
            tr_sc = clf.predict_proba(Xtr)[:, 1]
        else:
            tr_sc = clf.decision_function(Xtr)

        auc = round(safe_auc(y_train, tr_sc), 4)

        if hasattr(clf, "coef_"):
            coef     = clf.coef_[0]
            raw_imp  = np.abs(coef)
            # Build full linear expression sorted by |coefficient|
            order    = np.argsort(raw_imp)[::-1]
            terms    = [f"{coef[i]:+.4f}*{feature_names[i]}" for i in order
                        if abs(coef[i]) > 1e-6]
            bias     = getattr(clf, "intercept_", [0.0])[0]
            bias_str = f"{bias:+.4f}" if abs(bias) > 1e-6 else ""
            classifier_expr = "  ".join(terms) + (f"  {bias_str}" if bias_str else "")
        elif hasattr(clf, "feature_importances_"):
            raw_imp  = clf.feature_importances_
            order    = np.argsort(raw_imp)[::-1]
            terms    = [f"{raw_imp[i]:.4f}×{feature_names[i]}" for i in order
                        if raw_imp[i] > 1e-6]
            classifier_expr = "weighted sum: " + "  +  ".join(terms)
        else:
            raw_imp         = np.zeros(len(feature_names))
            classifier_expr = "(expression not available for this classifier)"

        top_idx   = np.argsort(raw_imp)[::-1][:top_n]
        top_feats = [(feature_names[i], float(raw_imp[i])) for i in top_idx]

        best_single_auc, best_single_feat = _best_single(dataset["baselines"], pa)

        results.append({
            "dataset": dataset["label"],
            "protected_attr": pa,
            "auc": auc,
            "top_features": top_feats,
            "imp_max": float(raw_imp.max()) if raw_imp.max() > 0 else 1.0,
            "best_single_auc":  best_single_auc,
            "best_single_feat": best_single_feat,
            "n_features": len(feature_names),
            "classifier_expr": classifier_expr,
        })

    return results


# ── GP expression evaluator ───────────────────────────────────────────────────

def _top_level_op(inner: str, op: str) -> int | None:
    """Return the char-index of `op` at bracket-depth 0 in `inner`, or None."""
    depth = 0
    i = 0
    op_len = len(op)
    while i < len(inner):
        c = inner[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0 and inner[i:i + op_len] == op:
            return i
        i += 1
    return None


def _top_level_div(inner: str) -> int | None:
    return _top_level_op(inner, ' / ')


def _replace_infix(s: str, op: str, func: str) -> str:
    """
    Walk `s` with balanced-paren tracking.  Every `(X op Y)` group whose
    top-level operator is `op` is converted to `func(X, Y)`.  Both X and Y
    are processed recursively so nested occurrences are handled correctly.
    """
    result = []
    i, n = 0, len(s)
    op_len = len(op)
    while i < n:
        if s[i] != '(':
            result.append(s[i])
            i += 1
            continue
        depth, j = 1, i + 1
        while j < n and depth:
            if s[j] == '(':   depth += 1
            elif s[j] == ')': depth -= 1
            j += 1
        inner = s[i + 1: j - 1]
        pos = _top_level_op(inner, op)
        if pos is not None:
            left  = _replace_infix(inner[:pos], op, func)
            right = _replace_infix(inner[pos + op_len:], op, func)
            result.append(f'{func}({left}, {right})')
        else:
            result.append(f'({_replace_infix(inner, op, func)})')
        i = j
    return ''.join(result)


def _replace_div_safe(s: str) -> str:
    """
    Walk the expression string character by character.  Every time a
    parenthesised group ( … ) is found, check whether its top-level
    operator is ' / '.  If so, convert it to _sdiv_(left, right) which
    implements SafeDiv (returns 0 when |denominator| < 1e-10).  Both
    operands are processed recursively so nested divisions are handled.

    This correctly reproduces the GP's SafeDiv semantics for ALL divisions,
    including the constant `(0.0 / 0.0)` patterns that would otherwise
    produce nan and poison the rest of the expression.
    """
    result = []
    i, n = 0, len(s)
    while i < n:
        if s[i] != '(':
            result.append(s[i])
            i += 1
            continue
        # Find the matching closing paren
        depth, j = 1, i + 1
        while j < n and depth:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
            j += 1
        inner = s[i + 1: j - 1]
        div_pos = _top_level_div(inner)
        if div_pos is not None:
            left  = _replace_div_safe(inner[:div_pos])
            right = _replace_div_safe(inner[div_pos + 3:])
            result.append(f'_sdiv_({left}, {right})')
        else:
            result.append(f'({_replace_div_safe(inner)})')
        i = j
    return ''.join(result)


def _preprocess_expr(s: str) -> str:
    """
    Convert a GP expression string (sympy-style) to numpy-evaluatable Python.

    Handles:
      max(a, b)   → np.maximum(a, b)
      min(a, b)   → np.minimum(a, b)
      IF(c,t,e)   → _IF_(c, t, e)         [defined in eval namespace]
      NOT(x)      → _NOT_(x)
      (a AND b)   → _AND_(a, b)            [iterative innermost-first]
      (a OR  b)   → _OR_(a, b)
      |x|         → np.abs(x)
      round(x)    → np.round(x)            [Python round() is not vectorised]
      sigma(x)    → _sigma_(x)             [logistic sigmoid; defined in eval ns]
      (x)^2       → (x)**2                 [^ is XOR in Python, not power]
      (X / Y)     → _sdiv_(X, Y)           [SafeDiv: 0 when |Y|<1e-10]
      literals    → np.float64(x)          [avoids Python ZeroDivisionError]
    """
    s = re.sub(r'\bmax\(', 'np.maximum(', s)
    s = re.sub(r'\bmin\(', 'np.minimum(', s)
    s = re.sub(r'\bIF\(',  '_IF_(',  s)
    s = re.sub(r'\bNOT\(', '_NOT_(', s)
    # round() — Python's built-in is not vectorised; np.round() is
    s = re.sub(r'\bround\(', 'np.round(', s)
    # sigma() — logistic sigmoid from Sigmoid grammar node
    s = re.sub(r'\bsigma\(', '_sigma_(', s)
    # ^2  —  caret is bitwise-XOR in Python, must become ** before eval
    s = re.sub(r'\^(\d+)', r'**\1', s)

    # Replace (X AND Y) / (X OR Y) using balanced-paren parser (handles nesting)
    s = _replace_infix(s, ' AND ', '_AND_')
    s = _replace_infix(s, ' OR ',  '_OR_')

    # |x| absolute value
    s = re.sub(r'\|([^|]+)\|', r'np.abs(\1)', s)

    # Replace all (X / Y) with _sdiv_(X, Y) — correctly handles nesting
    s = _replace_div_safe(s)

    # Wrap numeric literals as np.float64 so remaining float arithmetic
    # (after SafeDiv replacement) uses numpy semantics.
    s = re.sub(
        r'(?<![a-zA-Z_\d.])(-?\d+\.?\d*(?:[eE][+-]?\d+)?)(?![a-zA-Z_\d.])',
        lambda m: f'np.float64({m.group(0)})',
        s,
    )
    return s


def evaluate_gp_expr(expr_str: str,
                     X: np.ndarray,
                     feature_names: list[str]) -> np.ndarray:
    """
    Evaluate a GP expression string on dataset X.
    Returns a 1-D float64 array of length X.shape[0].
    Falls back to zeros on any error.
    """
    n = X.shape[0]
    try:
        s  = _preprocess_expr(expr_str)
        ns = {name: X[:, i] for i, name in enumerate(feature_names)}
        ns["dataset"] = X  # numpy_expr format uses dataset[:,i] indexing
        ns["np"]   = np
        ns["_IF_"] = lambda c, t, e: np.where(
            np.asarray(c, dtype=np.float64) > 0, t, e)
        ns["_NOT_"] = lambda x: (
            ~(np.asarray(x, dtype=np.float64) > 0)).astype(np.float64)
        ns["_AND_"] = lambda a, b: np.logical_and(
            np.asarray(a, dtype=np.float64) > 0,
            np.asarray(b, dtype=np.float64) > 0).astype(np.float64)
        ns["_OR_"]  = lambda a, b: np.logical_or(
            np.asarray(a, dtype=np.float64) > 0,
            np.asarray(b, dtype=np.float64) > 0).astype(np.float64)
        ns["_sigma_"] = lambda x: (
            1.0 / (1.0 + np.exp(-np.clip(
                np.asarray(x, dtype=np.float64), -500.0, 500.0))))
        def _sdiv_(a, b):
            _b = np.asarray(b, dtype=np.float64)
            _a = np.asarray(a, dtype=np.float64)
            safe_b = np.where(np.abs(_b) > 1e-10, _b, np.float64(1.0))
            return np.where(np.abs(_b) > 1e-10, _a / safe_b, np.float64(0.0))
        ns["_sdiv_"] = _sdiv_

        result = eval(s, {"__builtins__": {}}, ns)   # noqa: S307
        arr = np.asarray(result, dtype=np.float64).ravel()
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        if arr.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
        return arr
    except Exception as exc:
        print(f"      [WARN] expr eval failed: {exc}")
        return np.zeros(n, dtype=np.float64)


# ── Stages 2 & 3 ─────────────────────────────────────────────────────────────

def load_gp_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if "auc" in df.columns else None


def run_stage_gp(gp_df: pd.DataFrame, dataset: dict,
                 clf_key: str) -> dict[str, dict]:
    """
    For each GP expression in gp_df (filtered to dataset["label"]):
      1. Evaluate the expression on X_train → 1-D derived feature
      2. Train the chosen classifier on that single feature
      3. Return classifier AUC (train == test since single-phase)

    Returns {protected_attr: {"clf_train_auc": float, "clf_test_auc": float, "clf_auc": float}}
    """
    _, _, clf_factory, needs_scaling = CLASSIFIER_OPTIONS[clf_key]
    X_train       = dataset["X_train"]
    feature_names = dataset["feature_names"]
    ds_label      = dataset["label"]

    out = {}
    sub = gp_df[gp_df["dataset"] == ds_label]

    for _, row in sub.iterrows():
        pa = row["protected_attr"]

        # Prefer numpy_expr only for boolean-rooted expressions where it already
        # contains the full firing rule (e.g. typed/extended grammars with
        # explicit comparisons cast via .astype(np.float64)).  For arithmetic
        # grammar winners, numpy_expr is just the raw continuous score while
        # expression includes the thresholded firing rule (e.g. "... > 0.15"),
        # so evaluating numpy_expr would feed the classifier the wrong feature.
        numpy_expr = str(row.get("numpy_expr", "")).strip()
        expr_str   = str(row["expression"])
        use_numpy_expr = (
            numpy_expr
            and numpy_expr not in ("", "nan", "FAILED")
            and ".astype(np.float64)" in numpy_expr
        )
        eval_str = numpy_expr if use_numpy_expr else expr_str

        if pa not in dataset["pa_train"]:
            continue  # compound or unknown attr not in NPZ

        y_train = dataset["pa_train"][pa]

        if len(np.unique(y_train)) < 2:
            out[pa] = {"clf_train_auc": 0.5, "clf_test_auc": 0.5, "clf_auc": 0.5}
            continue

        # Evaluate expression → derived feature (shape: n × 1)
        feat_tr = evaluate_gp_expr(eval_str, X_train, feature_names).reshape(-1, 1)

        try:
            tr_auc, _ = _fit_and_score(
                clf_factory, needs_scaling, feat_tr, feat_tr, y_train, y_train)
        except Exception as exc:
            print(f"      [WARN] classifier failed for {pa}: {exc}")
            tr_auc = 0.5

        out[pa] = {"clf_train_auc": tr_auc, "clf_test_auc": tr_auc, "clf_auc": tr_auc}

    return out


def is_leaky(row) -> bool:
    leaky = {"race1", "race2", "race", "gender", "sex", "dnn_bar_pass_prediction"}
    feats = set(str(row.get("expression_features", "")).split(", "))
    return bool(feats & leaky)


def detect_ext_ops(expr: str) -> list[tuple[str, str]]:
    return [(lbl, col) for pat, lbl, col in EXTENDED_OPS if re.search(pat, expr)]


# ── HTML helpers ──────────────────────────────────────────────────────────────

def auc_color(auc: float) -> str:
    if auc >= 0.80: return "#34d399"
    if auc >= 0.70: return "#86efac"
    if auc >= 0.60: return "#fbbf24"
    return "#f87171"


def auc_badge(auc: float, width: int = 100) -> str:
    pct   = max(0.0, min(100.0, (auc - 0.5) / 0.5 * 100))
    color = auc_color(auc)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px">'
        f'<span style="font-variant-numeric:tabular-nums;font-weight:700;'
        f'color:{color};min-width:3.5rem;text-align:right">{auc:.4f}</span>'
        f'<span style="display:inline-block;width:{width}px;height:8px;'
        f'background:#2a2f40;border-radius:4px">'
        f'<span style="display:block;width:{pct:.1f}%;height:100%;'
        f'background:{color};border-radius:4px"></span>'
        f'</span></span>'
    )


def delta_badge(delta: float) -> str:
    if delta > 0.005:   color, sign = "#34d399", "+"
    elif delta > 0:     color, sign = "#86efac", "+"
    else:               color, sign = "#f87171", ""
    return (f'<span style="color:{color};font-weight:700;'
            f'font-variant-numeric:tabular-nums">{sign}{delta:.4f}</span>')


def feat_bar(name: str, importance: float, imp_max: float) -> str:
    pct = max(0.0, min(100.0, (importance / imp_max) * 100)) if imp_max > 0 else 0
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<span style="font-family:monospace;font-size:.78rem;color:#93c5fd;'
        f'min-width:8rem;flex-shrink:0">{escape(name)}</span>'
        f'<span style="display:inline-block;width:{int(pct * 1.2)}px;height:7px;'
        f'background:#6c8cff;border-radius:3px;opacity:.8"></span>'
        f'<span style="font-size:.75rem;color:#64748b">{importance:.4f}</span>'
        f'</div>'
    )


def op_tags(ops: list[tuple[str, str]]) -> str:
    return " ".join(
        f'<span style="background:{c}22;color:{c};border:1px solid {c}55;'
        f'border-radius:3px;padding:1px 5px;font-size:.72rem;font-weight:700;'
        f'font-family:monospace">{escape(l)}</span>'
        for l, c in ops
    )


def expr_box(expr: str, extended: bool = False) -> str:
    safe = escape(str(expr))
    if extended:
        for pat, lbl, color in EXTENDED_OPS:
            safe = re.sub(
                re.escape(escape(pat.strip())),
                f'<span style="color:{color};font-weight:700">'
                f'{escape(pat.strip())}</span>',
                safe,
            )
    return (
        f'<div style="font-family:\'JetBrains Mono\',\'Fira Code\',monospace;'
        f'font-size:.78rem;background:#0d1117;border:1px solid #2a2f40;'
        f'border-radius:6px;padding:.75rem 1rem;margin:.4rem 0;'
        f'word-break:break-all;line-height:1.6;color:#e2e8f0;'
        f'overflow-x:auto;white-space:pre-wrap">{safe}</div>'
    )


def leakage_banner() -> str:
    return (
        '<div style="background:rgba(248,113,113,.12);border-left:3px solid #f87171;'
        'border-radius:0 6px 6px 0;padding:.6rem 1rem;margin:.5rem 0;font-size:.82rem">'
        '<strong style="color:#f87171">⚠ Leakage</strong> — expression uses direct '
        'race/sex encodings; AUC ≈ 1.0 is expected.'
        '</div>'
    )


def _and_breakdown_html(and_components: list[dict]) -> str:
    """Render per-component metrics + attribution for an AND expression."""
    if not and_components or len(and_components) < 2:
        return ""

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
        bg   = "background:#161b22" if i % 2 == 0 else "background:#1c2333"
        td   = f"padding:.3rem .6rem;{bg};border-bottom:1px solid #21262d"
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
            f'{escape(str(comp.get("expression", "?")))}</td>'
            + attr_cell +
            f'<td style="{td};text-align:right;color:{_prec_color(prec)};font-weight:600">'
            f'{prec:.1f}%</td>'
            f'<td style="{td};text-align:right">{rec:.1f}%</td>'
            f'<td style="{td};text-align:right">{cov:.1f}%</td>'
            f'<td style="{td};text-align:right">{auc:.3f}</td>'
            f'</tr>'
        )
    return header + rows_html + "</tbody></table></div>"


def raw_ds_warning() -> str:
    return (
        '<div style="background:rgba(251,191,36,.1);border-left:3px solid #fbbf24;'
        'border-radius:0 6px 6px 0;padding:.75rem 1rem;margin-bottom:1rem;'
        'font-size:.85rem">'
        '<strong style="color:#fbbf24">⚠ Raw Dataset</strong> — contains columns '
        'that directly encode protected attributes '
        '(<code style="color:#93c5fd">race1</code>, '
        '<code style="color:#93c5fd">race2</code>, '
        '<code style="color:#93c5fd">gender</code>, …). '
        'Near-perfect AUC is leakage, not a real proxy discovery.'
        '</div>'
    )


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg:#0f1117; --card:#181b24; --card2:#1e2130;
  --border:#2a2f40; --text:#e2e8f0; --muted:#94a3b8;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
     font-size:14px;line-height:1.6}
.container{max-width:1280px;margin:0 auto;padding:0 2rem 5rem}
.hero{background:linear-gradient(135deg,#0f1117 0%,#1a1f35 50%,#0f1117 100%);
      border-bottom:1px solid var(--border);padding:3rem 2rem;text-align:center}
.hero h1{font-size:2.2rem;font-weight:800;letter-spacing:-.03em;
          background:linear-gradient(90deg,#60a5fa,#c084fc,#4ade80);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text}
.hero-sub{color:var(--muted);margin-top:.5rem;font-size:.95rem}
.pipeline{display:flex;align-items:center;justify-content:center;gap:0;
           margin:2rem auto 0;max-width:700px;flex-wrap:wrap}
.pipe-box{padding:.6rem 1.2rem;border-radius:8px;font-weight:700;
           font-size:.85rem;text-align:center;min-width:160px}
.pipe-s1{background:rgba(96,165,250,.15);border:1px solid rgba(96,165,250,.4);color:#60a5fa}
.pipe-s2{background:rgba(74,222,128,.15);border:1px solid rgba(74,222,128,.4);color:#4ade80}
.pipe-s3{background:rgba(192,132,252,.15);border:1px solid rgba(192,132,252,.4);color:#c084fc}
.pipe-arrow{color:#4a5568;font-size:1.4rem;padding:0 .75rem}
.stage{background:var(--card);border:1px solid var(--border);border-radius:12px;
        padding:2rem;margin-top:2.5rem}
.stage-header{display:flex;align-items:flex-start;gap:1.2rem;margin-bottom:1.5rem;
               padding-bottom:1.25rem;border-bottom:2px solid var(--border)}
.stage-num{width:2.6rem;height:2.6rem;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:1.2rem;font-weight:900;flex-shrink:0;
            margin-top:.2rem}
.s1-num{background:rgba(96,165,250,.2);color:#60a5fa;border:2px solid rgba(96,165,250,.4)}
.s2-num{background:rgba(74,222,128,.2);color:#4ade80;border:2px solid rgba(74,222,128,.4)}
.s3-num{background:rgba(192,132,252,.2);color:#c084fc;border:2px solid rgba(192,132,252,.4)}
.stage-title{font-size:1.3rem;font-weight:800}
.stage-desc{color:var(--muted);font-size:.88rem;margin-top:.25rem}
.ds-block{background:var(--card2);border:1px solid var(--border);border-radius:8px;
           padding:1.5rem;margin-top:1.5rem}
.ds-title{font-size:1rem;font-weight:700;margin-bottom:1rem;
           padding-bottom:.75rem;border-bottom:1px solid var(--border);
           display:flex;align-items:center;gap:.75rem}
.ds-tag{padding:.2rem .6rem;border-radius:4px;font-size:.72rem;font-weight:700;
         letter-spacing:.05em}
.tag-proc{background:rgba(96,165,250,.15);color:#60a5fa}
.tag-raw{background:rgba(251,191,36,.15);color:#fbbf24}
h3{font-size:.8rem;font-weight:700;color:var(--muted);text-transform:uppercase;
    letter-spacing:.08em;margin:1.25rem 0 .6rem}
table{width:100%;border-collapse:collapse;font-size:.8rem;margin-bottom:.5rem}
th{background:#12151e;color:var(--muted);font-weight:700;text-transform:uppercase;
    letter-spacing:.06em;font-size:.7rem;padding:.6rem .8rem;text-align:left;
    border-bottom:1px solid var(--border)}
td{padding:.55rem .8rem;border-bottom:1px solid rgba(42,47,64,.5);vertical-align:middle}
tr:hover td{background:rgba(108,140,255,.04)}
tr:last-child td{border-bottom:none}
.attr-chip{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.75rem;
            font-weight:700;background:rgba(147,197,253,.12);color:#93c5fd}
.note-dim{font-size:.75rem;color:#4a5568;font-variant-numeric:tabular-nums}
.gp-auc-ref{font-size:.72rem;color:#4a5568;font-style:italic}
.summary-section{background:var(--card);border:1px solid var(--border);
                  border-radius:12px;padding:2rem;margin-top:2.5rem}
.eval-note{background:rgba(108,140,255,.08);border:1px solid rgba(108,140,255,.25);
            border-radius:8px;padding:1rem 1.25rem;font-size:.85rem;margin-bottom:1.5rem;
            color:var(--muted)}
.eval-note strong{color:#93c5fd}
@media(max-width:768px){
  .container{padding:0 1rem 3rem}
  .hero h1{font-size:1.5rem}
  .pipeline{flex-direction:column}
  .pipe-arrow{transform:rotate(90deg)}
}
"""


# ── Section builders ──────────────────────────────────────────────────────────

def section_stage1(clf_key: str, all_results: dict[str, list[dict]],
                   top_n: int) -> str:
    clf_name, clf_spec, _, _ = CLASSIFIER_OPTIONS[clf_key]

    _DS_LONG = {
        "processed":     "Processed Dataset — 19 base features, no direct group identifiers",
        "non_processed": "Non-Processed Dataset — raw features including direct group identifiers",
        "raw":           "Raw Split — minimal preprocessing",
        "encoded":       "Encoded Split — domain-informed encoding",
    }
    # Iterate over splits actually present, in a stable order
    _DS_ORDER = ["processed", "non_processed", "raw", "encoded"]
    present = list(all_results.keys())
    ordered = [d for d in _DS_ORDER if d in present] + [d for d in present if d not in _DS_ORDER]

    blocks = ""
    for ds_label in ordered:
        rows = all_results.get(ds_label, [])
        if not rows:
            continue

        ds_long = _DS_LONG.get(ds_label, ds_label)
        tag_cls = "tag-proc" if ds_label in ("processed", "raw") else "tag-raw"
        tag_txt = ds_label.replace("_", " ").title()

        table_rows = ""
        feat_sections = ""
        for r in rows:
            delta = r["auc"] - r["best_single_auc"]
            attr  = ATTR_DISPLAY.get(r["protected_attr"], r["protected_attr"])
            table_rows += (
                f'<tr>'
                f'<td><span class="attr-chip">{escape(attr)}</span></td>'
                f'<td>{auc_badge(r["auc"])}</td>'
                f'<td>{auc_badge(r["best_single_auc"], 80)}'
                f'<span class="note-dim" style="margin-left:.4rem">'
                f'({escape(r["best_single_feat"])})</span></td>'
                f'<td>{delta_badge(delta)}</td>'
                f'<td class="note-dim">{r["n_features"]}</td>'
                f'</tr>'
            )
            bars = "".join(
                feat_bar(fn, fi, r["imp_max"]) for fn, fi in r["top_features"])
            expr_html = ""
            if r.get("classifier_expr"):
                expr_html = (
                    f'<div style="margin-top:.5rem;font-size:.72rem;color:#94a3b8">'
                    f'Proxy expression:</div>'
                    f'<pre style="margin:.3rem 0 0;padding:.6rem .8rem;'
                    f'background:#0f172a;border:1px solid #334155;border-radius:6px;'
                    f'font-size:.7rem;color:#e2e8f0;white-space:pre-wrap;'
                    f'word-break:break-all;max-height:120px;overflow-y:auto;'
                    f'line-height:1.5">{escape(r["classifier_expr"])}</pre>'
                )
            feat_sections += (
                f'<div style="margin:.75rem 0">'
                f'<div style="font-size:.8rem;font-weight:700;color:#93c5fd;'
                f'margin-bottom:.4rem">{escape(attr)}</div>{bars}{expr_html}</div>'
            )

        imp_label = ("Coefficients |coef|"
                     if clf_key in ("logistic_regression", "svm")
                     else "Feature importances (Gini/gain)")
        blocks += (
            f'<div class="ds-block">'
            f'<div class="ds-title">'
            f'<span class="ds-tag {tag_cls}">{tag_txt}</span>'
            f'{escape(ds_long)}'
            f'</div>'
            f'{"" if ds_label == "processed" else raw_ds_warning()}'
            f'<h3>Classifier AUC — all {rows[0]["n_features"] if rows else "N"} features</h3>'
            f'<table><thead><tr>'
            f'<th>Attribute</th><th>Training AUC</th>'
            f'<th>Best Single Feature</th><th>Δ vs Single</th><th>N Features</th>'
            f'</tr></thead><tbody>{table_rows}</tbody></table>'
            f'<h3>Top feature importances — {imp_label}</h3>'
            f'<p class="note-dim" style="margin:.4rem 0 .75rem">Top {top_n} shown.</p>'
            f'<div style="display:grid;grid-template-columns:'
            f'repeat(auto-fit,minmax(240px,1fr));gap:1.5rem">'
            f'{feat_sections}</div>'
            f'</div>'
        )

    return (
        f'<section class="stage" id="stage1">'
        f'<div class="stage-header">'
        f'<span class="stage-num s1-num">1</span>'
        f'<div>'
        f'<div class="stage-title" style="color:#60a5fa">Classifier Baseline</div>'
        f'<div class="stage-desc">'
        f'<strong>{escape(clf_name)}</strong> trained on all available base features. '
        f'High AUC means the feature set encodes group membership — but the model '
        f'is a black box: you know <em>that</em> a proxy exists, not <em>what it is</em>.'
        f'<br><span class="note-dim" style="margin-top:.3rem;display:block">'
        f'Spec: <code style="color:#60a5fa;font-family:monospace">'
        f'{escape(clf_spec)}</code></span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'{blocks}'
        f'</section>'
    )


def section_stage_gp(stage_num: int,
                     gp_df: pd.DataFrame | None,
                     clf_results: dict[str, dict[str, dict]],
                     csv_path: Path,
                     is_extended: bool,
                     ds_labels: list[str],
                     clf_key: str) -> str:

    clf_name, clf_spec, _, _ = CLASSIFIER_OPTIONS[clf_key]
    color   = "#4ade80" if stage_num == 2 else "#c084fc"
    num_cls = f"s{stage_num}-num"
    grammar = "Extended" if is_extended else "Arithmetic"
    ops_str = ("+, −, ×, ÷" if not is_extended
               else "+, −, ×, ÷  +  >, <, ≈, AND, OR, NOT, IF-THEN-ELSE, abs, max, min")

    if gp_df is None:
        return (
            f'<section class="stage" id="stage{stage_num}">'
            f'<div class="stage-header">'
            f'<span class="stage-num {num_cls}">{stage_num}</span>'
            f'<div><div class="stage-title" style="color:{color}">'
            f'GP + {grammar} Grammar</div></div></div>'
            f'<div style="color:#f87171;padding:1.5rem;text-align:center">'
            f'⚠ CSV not found: <code>{escape(str(csv_path))}</code></div>'
            f'</section>'
        )

    blocks = ""
    for ds_label in ds_labels:
        ds_df = gp_df[gp_df["dataset"] == ds_label]
        if ds_df.empty:
            continue

        tag_cls  = "tag-proc" if ds_label == "processed" else "tag-raw"
        tag_txt  = "Processed" if ds_label == "processed" else "Raw"
        ds_title = ("Processed Dataset — 19 base features"
                    if ds_label == "processed"
                    else "Non-Processed Dataset — raw features")

        ds_clf = clf_results.get(ds_label, {})

        table_rows  = ""
        expr_blocks = ""
        extra_th    = "<th>Ext. Ops</th>" if is_extended else ""

        for pa in PROTECTED_ATTRS:
            pa_df = ds_df[ds_df["protected_attr"] == pa]
            if pa_df.empty:
                continue
            row = pa_df.iloc[0]

            attr     = ATTR_DISPLAY.get(pa, pa)
            gp_auc   = float(row["auc"])             # GP's own score (reference)
            bl_auc   = float(row["best_baseline_auc"])
            bl_feat  = str(row["best_baseline_feature"])
            leaky    = is_leaky(row)
            ext_ops  = detect_ext_ops(str(row["expression"])) if is_extended else []

            # ── Classifier AUC on GP expression (the main metric) ──────
            clf_info = ds_clf.get(pa, {})
            clf_auc  = clf_info.get("clf_auc", clf_info.get("clf_test_auc", None))
            if clf_auc is not None:
                delta_clf     = clf_auc - bl_auc
                main_auc_cell = auc_badge(clf_auc)
                delta_cell    = delta_badge(delta_clf)
            else:
                main_auc_cell = '<span class="note-dim">—</span>'
                delta_cell    = '<span class="note-dim">—</span>'

            # Feature tags
            feat_str  = str(row.get("expression_features", "")).strip()
            feat_html = " ".join(
                f'<code style="background:rgba(147,197,253,.1);color:#93c5fd;'
                f'padding:1px 5px;border-radius:3px;font-size:.74rem;'
                f'font-family:monospace">{escape(f)}</code>'
                for f in feat_str.split(", ") if f
            )
            ops_html = op_tags(ext_ops) if ext_ops else ""

            table_rows += (
                f'<tr>'
                f'<td><span class="attr-chip">{escape(attr)}</span></td>'
                f'<td class="note-dim">{row["nodes"]}</td>'
                f'<td>{main_auc_cell}</td>'
                f'<td style="color:#94a3b8">{auc_badge(bl_auc, 80)}'
                f'<span class="note-dim" style="margin-left:.4rem">'
                f'({escape(bl_feat)})</span></td>'
                f'<td>{delta_cell}</td>'
                f'<td class="gp-auc-ref">{gp_auc:.4f}</td>'
                f'{"<td>" + ops_html + "</td>" if is_extended else ""}'
                f'</tr>'
            )

            # AND component breakdown (if present in CSV)
            import json as _json
            _and_raw = str(row.get("and_components", "") or "")
            try:
                _and_comp = _json.loads(_and_raw) if _and_raw.startswith("[") else None
            except Exception:
                _and_comp = None
            _and_html = _and_breakdown_html(_and_comp) if _and_comp else ""

            expr_blocks += (
                f'<div style="margin:1.1rem 0">'
                f'<div style="display:flex;align-items:center;gap:.75rem;'
                f'margin-bottom:.35rem;flex-wrap:wrap">'
                f'<span class="attr-chip">{escape(attr)}</span>'
                + (f'<span style="font-size:.78rem;color:{auc_color(clf_auc)};'
                   f'font-weight:700">CLF AUC {clf_auc:.4f}</span>'
                   if clf_auc is not None else '')
                + f'<span class="gp-auc-ref">(GP AUC: {gp_auc:.4f})</span>'
                f'<span class="note-dim">{row["nodes"]} nodes</span>'
                f'{"&nbsp;" + ops_html if ops_html else ""}'
                f'</div>'
                f'<div style="font-size:.78rem;color:#64748b;margin-bottom:.3rem">'
                f'Features: {feat_html}</div>'
                + (leakage_banner() if leaky else "")
                + expr_box(str(row["expression"]), extended=is_extended)
                + _and_html
                + f'</div>'
            )

        extra_th_gp = f'<th class="gp-auc-ref">GP AUC (ref)</th>'
        table_html = (
            f'<table><thead><tr>'
            f'<th>Attribute</th><th>Nodes</th>'
            f'<th>CLF Training AUC</th>'
            f'<th>Best Single Feature</th><th>Δ vs Single</th>'
            f'{extra_th_gp}'
            f'{extra_th}'
            f'</tr></thead><tbody>{table_rows}</tbody></table>'
        )

        blocks += (
            f'<div class="ds-block">'
            f'<div class="ds-title">'
            f'<span class="ds-tag {tag_cls}">{tag_txt}</span>'
            f'{escape(ds_title)}'
            f'</div>'
            f'{"" if ds_label == "processed" else raw_ds_warning()}'
            f'<h3>Classifier AUC — using GP expression as single derived feature</h3>'
            f'{table_html}'
            f'<h3>Discovered Expressions</h3>'
            f'{expr_blocks}'
            f'</div>'
        )

    # Config note
    cfg = (
        f'<div style="background:#12151e;border:1px solid var(--border);'
        f'border-radius:6px;padding:.9rem 1.2rem;margin-bottom:1.5rem;'
        f'font-size:.82rem;color:var(--muted)">'
        f'<strong style="color:var(--text)">Source:</strong> '
        f'<code style="color:#60a5fa;font-family:monospace">'
        f'{escape(str(csv_path.name))}</code>'
        f'&nbsp;·&nbsp; Grammar: <strong>{grammar}</strong>'
        f'&nbsp;·&nbsp; Operators: <code style="color:#93c5fd">{escape(ops_str)}</code>'
        f'</div>'
    )

    desc_extra = (
        "Only arithmetic operators are available (+, −, ×, ÷). "
        "The expression is compact but cannot represent thresholds or conditionals."
        if not is_extended
        else
        "The extended operator set adds comparisons, conditionals (IF-THEN-ELSE), "
        "and non-linear ops (abs, max, min), enabling richer proxy patterns."
    )

    return (
        f'<section class="stage" id="stage{stage_num}">'
        f'<div class="stage-header">'
        f'<span class="stage-num {num_cls}">{stage_num}</span>'
        f'<div>'
        f'<div class="stage-title" style="color:{color}">GP + {grammar} Grammar</div>'
        f'<div class="stage-desc">'
        f'GP evolves an interpretable expression over the base features. '
        f'The expression is then evaluated to produce a <strong>single derived feature</strong>, '
        f'on which <strong>{escape(clf_name)}</strong> is re-trained and evaluated — '
        f'making this directly comparable to Stage 1. '
        f'{escape(desc_extra)}'
        f'<br><span class="note-dim" style="margin-top:.3rem;display:block">'
        f'Classifier: <code style="color:{color};font-family:monospace">'
        f'{escape(clf_spec)}</code> &nbsp;·&nbsp; '
        f'<em>GP AUC (ref)</em> = the AUC GP optimised during search (shown for reference).'
        f'</span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'{cfg}'
        f'{blocks}'
        f'</section>'
    )


def section_summary(s1_all: dict, s2_df, s2_clf, s3_df, s3_clf,
                    clf_name: str) -> str:
    """Cross-stage comparison table using classifier AUC throughout."""
    _DS_SHORT = {
        "processed": "Processed", "non_processed": "Raw",
        "raw": "Raw", "encoded": "Encoded",
    }
    _DS_ORDER = ["processed", "non_processed", "raw", "encoded"]
    all_labels = set(s1_all) | set(s2_clf or {}) | set(s3_clf or {})
    ordered_labels = [d for d in _DS_ORDER if d in all_labels] + [d for d in all_labels if d not in _DS_ORDER]

    rows_html = ""
    for ds_label in ordered_labels:
        ds_short = _DS_SHORT.get(ds_label, ds_label.replace("_", " ").title())
        s1_map = {r["protected_attr"]: r for r in s1_all.get(ds_label, [])}

        for pa in PROTECTED_ATTRS:
            attr = ATTR_DISPLAY.get(pa, pa)

            s1_auc = s1_map[pa]["auc"] if pa in s1_map else None

            s2_auc = (s2_clf.get(ds_label, {}).get(pa, {}).get("clf_auc")
                      if s2_clf else None)
            s3_auc = (s3_clf.get(ds_label, {}).get(pa, {}).get("clf_auc")
                      if s3_clf else None)

            if all(v is None for v in [s1_auc, s2_auc, s3_auc]):
                continue

            def cell(auc):
                return (f'<td>{auc_badge(auc, 80)}</td>'
                        if auc is not None
                        else '<td class="note-dim" style="text-align:center">—</td>')

            rows_html += (
                f'<tr>'
                f'<td><span class="note-dim">{escape(ds_short)}</span></td>'
                f'<td><span class="attr-chip">{escape(attr)}</span></td>'
                f'{cell(s1_auc)}{cell(s2_auc)}{cell(s3_auc)}'
                f'</tr>'
            )

    return (
        f'<section class="summary-section" id="summary">'
        f'<div style="font-size:1.2rem;font-weight:800;margin-bottom:.75rem">'
        f'📊 Cross-Stage Summary</div>'
        f'<div class="eval-note">'
        f'All AUC values use <strong>{escape(clf_name)}</strong> as the evaluation '
        f'tool.  Stage 1: classifier on all features.  '
        f'Stages 2 &amp; 3: classifier re-trained on the single GP-derived feature. '
        f'Numbers are directly comparable.'
        f'</div>'
        f'<table><thead><tr>'
        f'<th>Dataset</th><th>Attribute</th>'
        f'<th style="color:#60a5fa">① Classifier — all features</th>'
        f'<th style="color:#4ade80">② GP Arithmetic — 1 derived feature</th>'
        f'<th style="color:#c084fc">③ GP Extended — 1 derived feature</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
        f'</section>'
    )


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html(clf_key: str,
               s1_all: dict,
               s2_df, s2_clf, s2_path: Path,
               s3_df, s3_clf, s3_path: Path,
               ds_labels: list[str],
               top_n: int) -> str:

    clf_name, clf_spec, _, _ = CLASSIFIER_OPTIONS[clf_key]
    date_str = datetime.now().strftime("%B %d, %Y %H:%M")

    s_stage1  = section_stage1(clf_key, s1_all, top_n)
    s_stage2  = section_stage_gp(2, s2_df, s2_clf, s2_path,
                                  is_extended=False, ds_labels=ds_labels,
                                  clf_key=clf_key)
    s_stage3  = section_stage_gp(3, s3_df, s3_clf, s3_path,
                                  is_extended=True,  ds_labels=ds_labels,
                                  clf_key=clf_key)
    s_summary = section_summary(s1_all, s2_df, s2_clf, s3_df, s3_clf, clf_name)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Three-Stage Proxy Discovery — {escape(clf_name)}</title>
<style>{CSS}</style>
</head>
<body>

<div class="hero">
  <h1>Three-Stage Proxy Discovery Framework</h1>
  <div class="hero-sub">
    LSAC Law School Dataset &nbsp;·&nbsp;
    Protected attributes: Black, Asian, Hispanic, Other Race, Male
  </div>
  <div class="hero-sub" style="margin-top:.3rem">
    All stages evaluated with: <strong style="color:#60a5fa">{escape(clf_name)}</strong>
    &nbsp;·&nbsp;
    <span style="color:#4a5568;font-size:.85rem">{escape(clf_spec)}</span>
  </div>
  <div class="pipeline">
    <div class="pipe-box pipe-s1">
      <div style="font-size:1.3rem">①</div>
      Classifier Baseline<br>
      <span style="font-size:.75rem;opacity:.8">All features → CLF → AUC</span>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-box pipe-s2">
      <div style="font-size:1.3rem">②</div>
      GP + Arithmetic<br>
      <span style="font-size:.75rem;opacity:.8">Expr → 1 feature → CLF → AUC</span>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-box pipe-s3">
      <div style="font-size:1.3rem">③</div>
      GP + Extended<br>
      <span style="font-size:.75rem;opacity:.8">Expr → 1 feature → CLF → AUC</span>
    </div>
  </div>
  <div style="color:#4a5568;font-size:.78rem;margin-top:1.25rem">
    Generated {date_str}
  </div>
</div>

<div class="container">

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:2rem">
  <div style="background:rgba(96,165,250,.08);border:1px solid rgba(96,165,250,.25);
               border-radius:8px;padding:1.25rem;font-size:.85rem">
    <div style="color:#60a5fa;font-weight:700;margin-bottom:.5rem">① Classifier Baseline</div>
    {escape(clf_name)} trained on all N base features. Black-box: high AUC confirms
    a proxy exists, but the model cannot explain <em>which combination</em> of features
    encodes group membership.
  </div>
  <div style="background:rgba(74,222,128,.08);border:1px solid rgba(74,222,128,.25);
               border-radius:8px;padding:1.25rem;font-size:.85rem">
    <div style="color:#4ade80;font-weight:700;margin-bottom:.5rem">② GP + Arithmetic Grammar</div>
    GP discovers a symbolic arithmetic expression.  That expression is evaluated
    on the data to produce <strong>one derived feature</strong>, then the same
    classifier is re-trained on it.  Interpretable and comparable to Stage 1.
  </div>
  <div style="background:rgba(192,132,252,.08);border:1px solid rgba(192,132,252,.25);
               border-radius:8px;padding:1.25rem;font-size:.85rem">
    <div style="color:#c084fc;font-weight:700;margin-bottom:.5rem">③ GP + Extended Grammar</div>
    Same pipeline as Stage 2 but GP had access to richer operators: comparisons,
    boolean logic, conditionals, abs/max/min.  Enables threshold and categorical
    patterns that arithmetic alone cannot express compactly.
  </div>
</div>

{s_stage1}
{s_stage2}
{s_stage3}
{s_summary}

<div style="text-align:center;color:#2a2f40;font-size:.78rem;margin-top:3rem">
  <code style="color:#4a5568">generate_3stage_report.py</code>
  &nbsp;·&nbsp; {date_str}
  &nbsp;·&nbsp; Classifier: {escape(clf_name)}
</div>

</div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Apply dataset config ──────────────────────────────────────────────
    global PROTECTED_ATTRS, ATTR_DISPLAY, DATASETS
    cfg = DATASET_CONFIGS[args.dataset]
    PROTECTED_ATTRS = cfg["protected_attrs"]
    ATTR_DISPLAY    = cfg["attr_display"]
    DATASETS        = cfg["splits"]

    # Fill in CSV defaults from config when not user-specified
    if args.arith_csv is None:
        args.arith_csv = cfg["default_arith_csv"]
    if args.ext_csv is None:
        args.ext_csv = cfg["default_ext_csv"]

    clf_name, clf_spec, _, _ = CLASSIFIER_OPTIONS[args.classifier]

    print(f"Dataset    : {cfg['label']}")
    print(f"Classifier : {clf_name}  ({clf_spec})")
    print(f"Arith CSV  : {args.arith_csv}")
    print(f"Ext CSV    : {args.ext_csv}")
    print(f"Output     : {args.output}")
    print()

    ds_map = {label: path for path, label, _ in DATASETS}

    # ── Load datasets + Stage 1 ──────────────────────────────────────────
    s1_all: dict[str, list[dict]] = {}
    datasets: dict[str, dict] = {}

    for ds_label in args.splits:
        if ds_label not in ds_map:
            print(f"  [SKIP] split '{ds_label}' not available for {args.dataset}")
            continue
        print(f"Loading dataset: {ds_label}")
        ds = load_dataset(ds_map[ds_label], ds_label,
                          cfg["npz_name"], PROTECTED_ATTRS,
                          cfg.get("gp_attr_map", {}))
        if ds is None:
            continue
        datasets[ds_label] = ds

        print(f"  Features: {len(ds['feature_names'])}  "
              f"Train: {ds['n_train']}  Test: {ds['n_test']}")
        print(f"  Running Stage 1 ({clf_name}) …")

        s1_results = run_stage1(ds, args.classifier, args.top_features)
        s1_all[ds_label] = s1_results

        for r in s1_results:
            attr = ATTR_DISPLAY.get(r["protected_attr"], r["protected_attr"])
            print(f"    {attr:<14} auc={r['auc']:.4f}  "
                  f"Δ_single={r['auc'] - r['best_single_auc']:+.4f}")
        print()

    # ── Load GP CSVs ─────────────────────────────────────────────────────
    if args.arith_csv is not None:
        print(f"Loading Stage 2 GP results: {args.arith_csv.name}")
        s2_df = load_gp_csv(args.arith_csv)
        if s2_df is None:
            print("  [NOT FOUND] Stage 2 will show a warning.")
    else:
        print("Stage 2 GP results: none configured for this dataset — Stage 2 will be skipped.")
        s2_df = None

    if args.ext_csv is not None:
        print(f"Loading Stage 3 GP results: {args.ext_csv.name}")
        s3_df = load_gp_csv(args.ext_csv)
        if s3_df is None:
            print("  [NOT FOUND] Stage 3 will show a warning.")
    else:
        print("Stage 3 GP results: none configured for this dataset — Stage 3 will be skipped.")
        s3_df = None

    # ── Evaluate GP expressions with the classifier ───────────────────────
    s2_clf: dict[str, dict] = {}
    s3_clf: dict[str, dict] = {}

    for ds_label, ds in datasets.items():
        if s2_df is not None and not s2_df[s2_df["dataset"] == ds_label].empty:
            print(f"  Stage 2 — evaluating GP expressions on '{ds_label}' "
                  f"with {clf_name} …")
            s2_clf[ds_label] = run_stage_gp(s2_df, ds, args.classifier)
            for pa, res in s2_clf[ds_label].items():
                attr = ATTR_DISPLAY.get(pa, pa)
                print(f"    {attr:<14} clf_auc={res['clf_auc']:.4f}")

        if s3_df is not None and not s3_df[s3_df["dataset"] == ds_label].empty:
            print(f"  Stage 3 — evaluating GP expressions on '{ds_label}' "
                  f"with {clf_name} …")
            s3_clf[ds_label] = run_stage_gp(s3_df, ds, args.classifier)
            for pa, res in s3_clf[ds_label].items():
                attr = ATTR_DISPLAY.get(pa, pa)
                print(f"    {attr:<14} clf_auc={res['clf_auc']:.4f}")

    print()
    print("Generating HTML …")
    html = build_html(
        clf_key  = args.classifier,
        s1_all   = s1_all,
        s2_df    = s2_df,   s2_clf = s2_clf,   s2_path = args.arith_csv,
        s3_df    = s3_df,   s3_clf = s3_clf,   s3_path = args.ext_csv,
        ds_labels = args.splits,
        top_n    = args.top_features,
    )

    args.output.write_text(html, encoding="utf-8")
    print(f"Report written: {args.output}  ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
