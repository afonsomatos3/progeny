"""
mlp_attribution_from_run.py
============================
Reads GP-discovered proxy expressions from a pipeline_runs/ folder,
injects each proxy as a binary column into its dataset, trains a fresh
3-layer MLP on the downstream task, and measures each feature's contribution
via Captum Integrated Gradients.

Produces a single dark-themed HTML report showing, for every dataset, which
proxy expressions receive the highest attribution weight in the downstream model.

Usage
-----
    # auto-detects the most recent run in pipeline_runs/
    python mlp_attribution_from_run.py

    # explicit run folder
    python mlp_attribution_from_run.py --run-dir pipeline_runs/20260316_124443

    # restrict to specific datasets
    python mlp_attribution_from_run.py --datasets adult law_school

    # only use best-config CSVs (faster)
    python mlp_attribution_from_run.py --config p1_r1_c1
"""

import argparse
import math as _math
import multiprocessing as _mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Active model type — set by --model CLI flag before any workers are spawned.
_ACTIVE_MODEL_TYPE: str = "mlp"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent
RUNS_DIR        = ROOT / "pipeline_runs"
ADULT_ROOT      = ROOT / "adult_test_dataset"
COMPAS_ROOT     = ROOT / "compas_scores_raw_dataset"
OULAD_ROOT      = ROOT / "OULAD-student_dataset"
GERMAN_ROOT     = ROOT / "German-Credit-Analysis-master"
FOLKTABLES_ROOT = ROOT / "folktables-main-Dataset"

sys.path.insert(0, str(ROOT / "GeneticEngine" / "GeneticEngine"))
from geml.common import forward_dataset          # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  MLP CONFIGURATION — change everything here
# ══════════════════════════════════════════════════════════════════════════════
MLP_CONFIG = {
    # Architecture
    "hidden_dims":  [64, 32],       # list of hidden layer sizes; add more for deeper net
    "activation":   "relu",          # "relu" | "tanh" | "gelu" | "leaky_relu"

    # Training
    "epochs":       60,
    "batch_size":   512,
    "lr":           1e-3,
    "weight_decay": 1e-4,

    # Optimizer
    "optimizer":    "adam",          # "adam" | "adamw" | "sgd"
    "momentum":     0.9,             # only used when optimizer == "sgd"

    # LR Scheduler
    "scheduler":    "step",          # "step" | "cosine" | "none"
    "step_size":    20,              # StepLR: decay every N epochs
    "gamma":        0.5,             # StepLR: multiply LR by this factor

    # Captum Integrated Gradients
    "ig_n_steps":   50,              # Riemann approximation steps (50–300)

    # Warm-start fine-tuning (set warmstart=False to disable)
    # The base MLP (no proxy) is trained once per dataset for `epochs`.
    # Each proxy model copies base weights and fine-tunes for `finetune_epochs`
    # at a reduced LR — dramatically faster than full retraining.
    "warmstart":         True,
    "finetune_epochs":   10,         # fine-tune steps per proxy
    "finetune_lr_factor": 0.3,       # LR multiplier during fine-tuning
}
# ══════════════════════════════════════════════════════════════════════════════


# ── Activation factory ────────────────────────────────────────────────────────

def _make_activation(name: str) -> nn.Module:
    return {
        "relu":        nn.ReLU(),
        "tanh":        nn.Tanh(),
        "gelu":        nn.GELU(),
        "leaky_relu":  nn.LeakyReLU(0.01),
    }[name]


# ── Model ─────────────────────────────────────────────────────────────────────

class ProxyNet(nn.Module):
    """Configurable MLP for proxy attribution.

    Architecture is driven entirely by MLP_CONFIG — change that dict,
    not this class.

    Layer sizes:   [n_features] → hidden_dims[0] → hidden_dims[1] → ... → 1
    Activations:   set by MLP_CONFIG["activation"]
    Output:        raw logit (sigmoid applied externally for attribution)
    """
    def __init__(self, n_features: int):
        super().__init__()
        dims   = [n_features] + MLP_CONFIG["hidden_dims"]
        act    = MLP_CONFIG["activation"]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_make_activation(act))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Optimizer factory ─────────────────────────────────────────────────────────

def _make_optimizer(name: str, params, lr: float, wd: float, momentum: float):
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name!r}")


def _make_scheduler(name: str, opt, step_size: int, gamma: float, epochs: int):
    if name == "step":
        return optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name!r}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_tr, y_tr, X_val, y_val, device):
    cfg   = MLP_CONFIG
    model = ProxyNet(X_tr.shape[1]).to(device)
    opt   = _make_optimizer(cfg["optimizer"], model.parameters(),
                            cfg["lr"], cfg["weight_decay"], cfg["momentum"])
    sched = _make_scheduler(cfg["scheduler"], opt,
                            cfg["step_size"], cfg["gamma"], cfg["epochs"])
    crit  = nn.BCEWithLogitsLoss()

    ds     = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    log = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if sched is not None:
            sched.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(
                    model(torch.tensor(X_val, dtype=torch.float32).to(device))
                ).cpu().numpy()
            auc = roc_auc_score(y_val, probs)
            log.append((epoch, total_loss / len(X_tr), auc))

    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(
            model(torch.tensor(X_val, dtype=torch.float32).to(device))
        ).cpu().numpy()
    return model, roc_auc_score(y_val, final_probs), log


# ── Warm-start fine-tuning ────────────────────────────────────────────────────

def _build_proxy_model_warmstart(base_state: dict, n_base: int,
                                  device) -> "ProxyNet":
    """Create a ProxyNet(n_base+1) initialised from the trained base model.

    The first layer weight tensor grows by one column (the proxy input).
    All hidden/output weights are copied verbatim; only the new proxy-column
    weight in layer-0 is randomly initialised (via default kaiming_uniform_).
    """
    import copy
    model = ProxyNet(n_base + 1).to(device)
    new_sd = model.state_dict()
    for k, v in base_state.items():
        if k == "net.0.weight":
            # base: (H, n_base) → proxy: (H, n_base+1)
            new_sd[k][:, :n_base] = v.clone()
            # proxy column already randomly initialised; leave as-is
        else:
            new_sd[k] = v.clone()
    model.load_state_dict(new_sd)
    return model


def finetune_model(model, X_tr, y_tr, X_val, y_val, device):
    """Fine-tune a warm-started model for a small number of epochs."""
    cfg    = MLP_CONFIG
    n_ep   = cfg["finetune_epochs"]
    lr     = cfg["lr"] * cfg["finetune_lr_factor"]
    opt    = _make_optimizer(cfg["optimizer"], model.parameters(),
                             lr, cfg["weight_decay"], cfg["momentum"])
    crit   = nn.BCEWithLogitsLoss()
    ds     = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    for _ in range(n_ep):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            nn.BCEWithLogitsLoss()(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.tensor(X_val, dtype=torch.float32).to(device))
        ).cpu().numpy()
    return model, roc_auc_score(y_val, probs)


# ── Integrated Gradients ──────────────────────────────────────────────────────

def run_ig(model, X_test, device):
    """Returns (n_samples, n_features) numpy array of IG attributions."""
    from captum.attr import IntegratedGradients

    class _Wrapped(nn.Module):
        def forward(self, x):
            return torch.sigmoid(model(x))

    wrapped  = _Wrapped().to(device)
    wrapped.eval()
    ig       = IntegratedGradients(wrapped)
    X_t      = torch.tensor(X_test, dtype=torch.float32).to(device)
    baseline = torch.zeros_like(X_t)
    attrs    = ig.attribute(X_t, baselines=baseline,
                            n_steps=MLP_CONFIG["ig_n_steps"])
    return attrs.detach().cpu().numpy()


# ── DeepLIFT ─────────────────────────────────────────────────────────────────

def run_deeplift(model, X_test, device):
    """Returns (n_samples, n_features) attribution array via DeepLIFT."""
    try:
        from captum.attr import DeepLift

        class _Wrapped(nn.Module):
            def forward(self, x):
                return torch.sigmoid(model(x))

        wrapped = _Wrapped().to(device)
        wrapped.eval()
        dl      = DeepLift(wrapped)
        X_t     = torch.tensor(X_test, dtype=torch.float32).to(device)
        baseline = torch.zeros_like(X_t)
        attrs   = dl.attribute(X_t, baselines=baseline)
        return attrs.detach().cpu().numpy()
    except Exception as e:
        print(f"      [DeepLIFT failed: {e}]")
        return None


# ── GradientShap ──────────────────────────────────────────────────────────────

def run_gradientshap(model, X_test, device,
                     X_train_ref: np.ndarray | None = None):
    """Returns (n_samples, n_features) attribution array via GradientShap.

    Uses a subsample of the training data as the reference distribution so
    that each attribution is relative to a realistic data-manifold baseline
    rather than an arbitrary origin.  Falls back to a zero baseline when no
    training reference is provided (e.g. fast/test mode).
    """
    try:
        from captum.attr import GradientShap

        class _Wrapped(nn.Module):
            def forward(self, x):
                return torch.sigmoid(model(x))

        wrapped = _Wrapped().to(device)
        wrapped.eval()
        gs  = GradientShap(wrapped)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

        if X_train_ref is not None and len(X_train_ref) > 0:
            # Sample up to 200 training rows as the baseline distribution.
            # GradientShap randomly picks one per test sample, giving
            # attributions relative to realistic reference points rather
            # than a fixed zero or noise baseline.
            k   = min(200, len(X_train_ref))
            idx = np.random.default_rng(42).choice(len(X_train_ref), k, replace=False)
            baselines = torch.tensor(
                X_train_ref[idx].astype(np.float32), dtype=torch.float32
            ).to(device)
        else:
            baselines = torch.zeros(1, X_t.shape[1], dtype=torch.float32).to(device)

        attrs = gs.attribute(X_t, baselines=baselines, n_samples=5, stdevs=0.0)
        return attrs.detach().cpu().numpy()
    except Exception as e:
        print(f"      [GradientShap failed: {e}]")
        return None


# ── Gradient × Input ──────────────────────────────────────────────────────────

def run_gradient_x_input(model, X_test, device):
    """Returns (n_samples, n_features) attribution array via Gradient × Input."""
    try:
        from captum.attr import InputXGradient

        class _Wrapped(nn.Module):
            def forward(self, x):
                return torch.sigmoid(model(x))

        wrapped = _Wrapped().to(device)
        wrapped.eval()
        gxi     = InputXGradient(wrapped)
        X_t     = torch.tensor(X_test, dtype=torch.float32).to(device)
        attrs   = gxi.attribute(X_t)
        return attrs.detach().cpu().numpy()
    except Exception as e:
        print(f"      [GradientXInput failed: {e}]")
        return None


# ── Layer-wise Relevance Propagation (LRP) ────────────────────────────────────

def run_lrp(model, X_test, device):
    """Returns (n_samples, n_features) attribution array via LRP (epsilon rule)."""
    try:
        from captum.attr import LRP

        model.eval()
        lrp  = LRP(model)
        X_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
        attrs = lrp.attribute(X_t)
        return attrs.detach().cpu().numpy()
    except Exception as e:
        print(f"      [LRP failed: {e}]")
        return None


# ── Attribution aggregator ────────────────────────────────────────────────────

ATTRIBUTION_METHODS = ["IG", "DeepLIFT", "GradSHAP", "GxI", "LRP"]

def run_all_attribution_methods(model, X_aug_te, device,
                                X_aug_tr: np.ndarray | None = None):
    """Run all attribution methods and return dict method→attrs array.

    Each array has shape (n_samples, n_features).  None means the method failed.
    X_aug_tr is passed to GradientShap as the reference distribution baseline.
    """
    runners = {
        "IG":       lambda: run_ig(model, X_aug_te, device),
        "DeepLIFT": lambda: run_deeplift(model, X_aug_te, device),
        "GradSHAP": lambda: run_gradientshap(model, X_aug_te, device, X_aug_tr),
        "GxI":      lambda: run_gradient_x_input(model, X_aug_te, device),
        "LRP":      lambda: run_lrp(model, X_aug_te, device),
    }
    results = {}
    for name, fn in runners.items():
        print(f"      Running {name}…", end="  ", flush=True)
        arr = fn()
        if arr is not None:
            print("ok")
        else:
            print("FAILED")
        results[name] = arr
    return results


def _summarise_method_attrs(all_attrs: dict, proxy_col_idx: int):
    """For each method that succeeded, compute mean |attr|, proxy rank, proxy share,
    and pairwise Spearman rank correlations across all methods.

    Returns dict: method → {"mean_abs": array, "proxy_rank": int, "proxy_share": float,
                             "proxy_attr": float, "spearman": {other_method: rho}}
                  or None if method failed.
    The key "spearman_matrix" is also added at the top level.
    """
    from scipy.stats import spearmanr

    proxy_col_idx = int(np.squeeze(proxy_col_idx))
    summary = {}
    for method, attrs in all_attrs.items():
        if attrs is None:
            summary[method] = None
            continue
        mean_abs   = np.abs(attrs).mean(axis=0)
        total      = mean_abs.sum()
        p_attr     = float(mean_abs[proxy_col_idx])
        p_share    = p_attr / total if total > 0 else 0.0
        rank_order = np.argsort(mean_abs)[::-1]
        p_rank     = int(list(rank_order).index(proxy_col_idx)) + 1
        summary[method] = {
            "mean_abs":    mean_abs,
            "proxy_rank":  p_rank,
            "proxy_share": p_share,
            "proxy_attr":  p_attr,
            "spearman":    {},
        }

    # Pairwise Spearman correlations between mean-absolute-attribution vectors.
    # ρ close to 1.0 means both methods agree on which features matter most,
    # regardless of the scale of attributions — a sign that findings are
    # method-invariant rather than artefacts of one algorithm's formulation.
    valid = {m: s for m, s in summary.items() if s is not None}
    for m1, s1 in valid.items():
        for m2, s2 in valid.items():
            if m1 == m2:
                rho = 1.0
            else:
                rho, _ = spearmanr(s1["mean_abs"], s2["mean_abs"])
            s1["spearman"][m2] = float(rho)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  NON-MLP MODELS + SHAP ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def _to_float32(X):
    return np.asarray(X, dtype=np.float32)


def train_logistic_regression(X_tr, y_tr, X_te=None, y_te=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X_tr = _to_float32(X_tr)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             random_state=42, n_jobs=1)
    clf.fit(X_tr, y_tr)
    X_eval = _to_float32(X_te) if X_te is not None else X_tr
    y_eval = y_te if X_te is not None else y_tr
    auc = roc_auc_score(y_eval, clf.predict_proba(X_eval)[:, 1])
    return clf, auc


def train_random_forest(X_tr, y_tr, X_te=None, y_te=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    X_tr = _to_float32(X_tr)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                 random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    X_eval = _to_float32(X_te) if X_te is not None else X_tr
    y_eval = y_te if X_te is not None else y_tr
    auc = roc_auc_score(y_eval, clf.predict_proba(X_eval)[:, 1])
    return clf, auc


def train_xgboost(X_tr, y_tr, X_te=None, y_te=None):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    from sklearn.metrics import roc_auc_score
    X_tr = _to_float32(X_tr)
    clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="logloss",
                        random_state=42, n_jobs=-1, verbosity=0)
    clf.fit(X_tr, y_tr)
    X_eval = _to_float32(X_te) if X_te is not None else X_tr
    y_eval = y_te if X_te is not None else y_tr
    auc = roc_auc_score(y_eval, clf.predict_proba(X_eval)[:, 1])
    return clf, auc


MODEL_TRAINERS = {
    "mlp":                 None,   # handled separately via train_model()
    "logistic_regression": train_logistic_regression,
    "random_forest":       train_random_forest,
    "xgboost":             train_xgboost,
}


def run_shap_attribution(clf, X_te: np.ndarray,
                         X_tr: np.ndarray, model_type: str) -> np.ndarray | None:
    """Run SHAP attribution for a sklearn/xgboost model.

    Returns (n_samples, n_features) array of SHAP values (class-1 direction).
    Uses TreeExplainer for tree models, LinearExplainer for logistic regression.
    """
    try:
        import shap
        X_te = _to_float32(X_te)
        X_tr = _to_float32(X_tr)
        if model_type == "xgboost":
            # Use XGBoost's native SHAP to avoid shap-library DMatrix issues
            import xgboost as xgb
            X_te_f64 = np.asarray(X_te, dtype=np.float64)
            dmat = xgb.DMatrix(X_te_f64)
            contribs = clf.get_booster().predict(dmat, pred_contribs=True)
            vals = np.asarray(contribs[:, :-1], dtype=np.float32)  # drop bias col
        elif model_type == "random_forest":
            explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
            # subsample to max 500 rows to keep SHAP fast; attribution shares are stable
            if X_te.shape[0] > 500:
                rng = np.random.default_rng(42)
                idx = rng.choice(X_te.shape[0], 500, replace=False)
                X_shap = X_te[idx]
            else:
                X_shap = X_te
            vals = explainer.shap_values(X_shap)
            # old SHAP: list [class0, class1]  → take class-1 slice
            # new SHAP: ndarray (n_samples, n_features, n_classes) → take [..., 1]
            if isinstance(vals, list):
                vals = vals[1]
            elif isinstance(vals, np.ndarray) and vals.ndim == 3:
                vals = vals[:, :, 1]
        elif model_type == "logistic_regression":
            explainer = shap.LinearExplainer(clf, X_tr,
                                             feature_perturbation="interventional")
            vals = explainer.shap_values(X_te)
            if isinstance(vals, list):
                vals = vals[1]
            elif isinstance(vals, np.ndarray) and vals.ndim == 3:
                vals = vals[:, :, 1]
        else:
            return None
        return np.asarray(vals, dtype=np.float32)
    except Exception as e:
        print(f"      [SHAP failed: {e}]")
        return None


def run_non_mlp_attribution(clf, X_te, X_tr, model_type):
    """Run all applicable attribution methods for a non-MLP model.

    Returns same format as run_all_attribution_methods: dict method→array.
    """
    results = {}
    print(f"      Running SHAP…", end="  ", flush=True)
    arr = run_shap_attribution(clf, X_te, X_tr, model_type)
    print("ok" if arr is not None else "FAILED")
    results["SHAP"] = arr
    return results


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _merge_to_full_data(ds: dict) -> dict:
    """Concatenate train and test splits so the MLP trains and is evaluated on
    the full dataset.  This matches the GP pipeline behaviour and is appropriate
    when the goal is pattern discovery rather than out-of-sample generalisation.
    """
    X_all = np.concatenate([ds["X_train"], ds["X_test"]], axis=0)
    ds["X_train"] = X_all
    ds["X_test"]  = X_all
    if ds.get("y_train") is not None and ds.get("y_test") is not None:
        y_all = np.concatenate([ds["y_train"], ds["y_test"]], axis=0)
        ds["y_train"] = y_all
        ds["y_test"]  = y_all
    for pa in list(ds.get("pa_train", {}).keys()):
        pa_all = np.concatenate([ds["pa_train"][pa], ds["pa_test"][pa]], axis=0)
        ds["pa_train"][pa] = pa_all
        ds["pa_test"][pa]  = pa_all
    return ds


def _load_npz_dataset(npz_path: Path, task_keys: list[str]) -> dict | None:
    """Load an NPZ dataset.  Returns a unified dict or None on failure."""
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)

    X_tr = np.asarray(data.get("X_train_raw", data.get("X_train")),
                      dtype=np.float64)
    X_te = np.asarray(data.get("X_test_raw",  data.get("X_test")),
                      dtype=np.float64)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=-1e6)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=1e6, neginf=-1e6)
    feat  = [str(n) for n in data["feature_names"]]

    # Find downstream task target
    y_tr = y_te = None
    for key in task_keys:
        if f"{key}_train" in data:
            y_tr = np.asarray(data[f"{key}_train"], dtype=np.float32)
            y_te = np.asarray(data[f"{key}_test"],  dtype=np.float32)
            task_used = key
            break
        if key in data:                    # some NPZs store it without suffix
            arr  = np.asarray(data[key], dtype=np.float32)
            split = int(len(arr) * 0.8)
            y_tr  = arr[:split]
            y_te  = arr[split:]
            task_used = key
            break
    else:
        task_used = None

    # Collect protected-attribute arrays
    pa_tr, pa_te = {}, {}
    for k in data.files:
        if k.endswith("_train") and k != f"{task_used}_train":
            pa = k[:-6]
            if f"{pa}_test" in data:
                pa_tr[pa] = np.asarray(data[k],          dtype=np.int32)
                pa_te[pa] = np.asarray(data[f"{pa}_test"], dtype=np.int32)

    return {
        "X_train": X_tr, "X_test": X_te,
        "feature_names": feat,
        "y_train": y_tr, "y_test": y_te,
        "task_used": task_used,
        "pa_train": pa_tr, "pa_test": pa_te,
    }


DATASET_LOADERS = {
    "adult": lambda: _load_npz_dataset(
        ADULT_ROOT / "processed" / "adult_base_features.npz",
        task_keys=["y", "income"],
    ),
    "law_school": lambda: _load_npz_dataset(
        ROOT / "processed" / "law_school_base_features.npz",
        task_keys=["y", "pass_bar", "bar", "target"],
    ),
    "compas": lambda: _load_compas(),
    "oulad":  lambda: _load_oulad(),
    "german_credit": lambda: _load_npz_dataset(
        GERMAN_ROOT / "processed" / "german_credit_base_features.npz",
        task_keys=["y"],
    ),
    "folktables": lambda: _load_npz_dataset(
        FOLKTABLES_ROOT / "processed" / "folktables_base_features.npz",
        task_keys=["y"],
    ),
}


def _load_compas() -> dict | None:
    sys.path.insert(0, str(COMPAS_ROOT))
    try:
        from gp_proxy_discovery_compas import load_compas_datasets
        all_ds = load_compas_datasets()
    except Exception as e:
        print(f"  [COMPAS] could not load: {e}")
        return None
    finally:
        sys.path.pop(0)

    ds = all_ds.get("encoded") or next(iter(all_ds.values()), None)
    if ds is None:
        return None

    X_tr = ds["X_train"].copy()
    X_te = ds["X_test"].copy()
    feat = list(ds["feature_names"])

    # Extract recidivism scores from the feature matrix as the downstream task.
    # DecileScore_Recidivism is a 1–10 risk score: >= 5 is "high risk" (binary target).
    # Remove all three recidivism-score columns from the feature matrix so the model
    # cannot trivially predict the target from its own direct encoding.
    RECID_TASK_COL  = "DecileScore_Recidivism"
    RECID_DROP_COLS = ["RawScore_Recidivism", "DecileScore_Recidivism",
                       "ScoreText_Recidivism"]

    y_tr = y_te = None
    task_used   = None
    drop_idx    = []
    for col in RECID_DROP_COLS:
        if col in feat:
            drop_idx.append(feat.index(col))
    if RECID_TASK_COL in feat:
        task_idx = feat.index(RECID_TASK_COL)
        y_full   = np.concatenate([X_tr[:, task_idx], X_te[:, task_idx]])
        split    = X_tr.shape[0]
        y_tr     = (y_full[:split]           >= 5).astype(np.float32)
        y_te     = (y_full[split:split + X_te.shape[0]] >= 5).astype(np.float32)
        task_used = f"{RECID_TASK_COL} >= 5 (high risk)"

    if drop_idx:
        keep_idx = [i for i in range(len(feat)) if i not in drop_idx]
        X_tr     = X_tr[:, keep_idx]
        X_te     = X_te[:, keep_idx]
        feat     = [feat[i] for i in keep_idx]

    return {
        "X_train": X_tr, "X_test": X_te,
        "feature_names": feat,
        "y_train": y_tr, "y_test": y_te,
        "task_used": task_used,
        "pa_train": ds.get("pa_train", {}),
        "pa_test":  ds.get("pa_test",  {}),
    }


def _load_oulad() -> dict | None:
    sys.path.insert(0, str(OULAD_ROOT))
    try:
        from gp_proxy_discovery_oulad import load_oulad_datasets
        all_ds = load_oulad_datasets()
    except Exception as e:
        print(f"  [OULAD] could not load: {e}")
        return None
    finally:
        sys.path.pop(0)

    ds = all_ds.get("encoded") or next(iter(all_ds.values()), None)
    if ds is None:
        return None

    X_tr = ds["X_train"].copy()
    X_te = ds["X_test"].copy()
    feat = list(ds["feature_names"])

    # Belt-and-suspenders: drop final_result from the feature matrix if it
    # somehow survived the loader fix (it's the downstream prediction target).
    if "final_result" in feat:
        drop_i = feat.index("final_result")
        keep   = [i for i in range(len(feat)) if i != drop_i]
        X_tr   = X_tr[:, keep]
        X_te   = X_te[:, keep]
        feat   = [feat[i] for i in keep]

    # Load downstream task (pass vs fail) from the raw studentInfo.csv.
    y_tr = y_te = None
    task_used = None
    for csv_name in ["studentInfo.csv", "student_info.csv"]:
        p = OULAD_ROOT / csv_name
        if p.exists():
            raw = pd.read_csv(p)
            for col in ["final_result", "pass", "result"]:
                if col in raw.columns:
                    vals  = raw[col].map(
                        {"Pass": 1, "Distinction": 1, "Fail": 0, "Withdrawn": 0}
                    ).fillna(raw[col])
                    arr   = vals.values.astype(np.float32)
                    n     = min(len(arr), X_tr.shape[0] + X_te.shape[0])
                    arr   = arr[:n]
                    split = X_tr.shape[0]
                    y_tr  = arr[:split]
                    y_te  = arr[n - X_te.shape[0]:]
                    task_used = f"{col} (Pass/Distinction=1)"
                    break
            if task_used:
                break

    return {
        "X_train": X_tr, "X_test": X_te,
        "feature_names": feat,
        "y_train": y_tr, "y_test": y_te,
        "task_used": task_used,
        "pa_train": ds.get("pa_train", {}),
        "pa_test":  ds.get("pa_test",  {}),
    }


# ── Proxy collection from run CSVs ────────────────────────────────────────────

GATE_MIN_PREC     = 60.0  # C2 quality bar for logical expressions (%)
GATE_MIN_REC      =  5.0  # C3 recall floor for all expressions (%)
GATE_MIN_COV      =  5.0  # coverage soft floor (%); rows below this are shown but flagged
GATE_MIN_AUC_CONT = 0.60  # C2 quality bar for continuous expressions (AUC)


def _is_degenerate_expr(expr: str) -> bool:
    """Return True for self-comparison sub-expressions that are always constant.

    Catches patterns like (X < X), (X > X), (X == X), (X != X), (X <= X),
    (X >= X) anywhere in the expression string.
    """
    import re
    # Match token op same-token where op is a comparison
    pattern = r'\b(\w+)\s*(?:<|>|==|!=|<=|>=)\s*\1\b'
    return bool(re.search(pattern, expr))


def _count_distinct_features(row) -> int:
    """Count distinct feature names listed in expression_features or features column.

    Partials CSVs use 'features'; main best-solution CSVs use 'expression_features'.
    """
    # Try both column names — partials use 'features', main CSVs use 'expression_features'
    feats = str(row.get("expression_features", "") or row.get("features", "") or "")
    if not feats or feats == "nan":
        return 0
    return len({f.strip() for f in feats.split(",") if f.strip()})


def collect_proxies(run_dir: Path, dataset_name: str,
                    config_filter: str | None = None) -> list[dict]:
    """Read result CSVs and return unique proxies in two categories:

    BINARY proxies (arith / ext grammar, best-solution CSVs + partials):
      Gate: prec >= GATE_MIN_PREC (80%), rec >= GATE_MIN_REC (15%), auc >= 0.55
      Multi-attribute candidates (>= 2 distinct features) are also sourced from
      -partials.csv files where they may not be the best-fitness solution.

    CONTINUOUS proxies (cont grammar, best-solution CSVs):
      Gate: auc >= GATE_MIN_AUC_CONT (0.60) only — no prec/rec gate.
      numpy_expr is the raw pre-threshold score expression.
      Injected into MLP as a float column (not binarised).

    Only processed/encoded dataset splits are used.
    """
    ds_dir = run_dir / dataset_name
    if not ds_dir.exists():
        print(f"  [SKIP] {dataset_name}: no folder at {ds_dir}")
        return []

    seen: set[tuple] = set()
    proxies: list[dict] = []

    # ── Binary proxies: best-solution CSVs (arith + ext) ──────────────────────
    binary_csvs = (
        sorted(ds_dir.glob("arith-*.csv"))
        + sorted(ds_dir.glob("ext-*.csv"))
    )
    for csv_path in binary_csvs:
        if "-partials" in csv_path.name:
            continue
        if config_filter and config_filter not in csv_path.name:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "numpy_expr" not in df.columns:
            continue

        grammar = "arith" if csv_path.name.startswith("arith") else "ext"
        parts = csv_path.stem.split("_", 2)
        config_label = parts[-1] if len(parts) >= 3 else csv_path.stem

        for _, row in df.iterrows():
            ds_split = str(row.get("dataset", "")).strip()
            if ds_split not in ("processed", "encoded"):
                continue
            numpy_expr = str(row.get("numpy_expr", "")).strip()
            if not numpy_expr or numpy_expr in ("", "nan", "FAILED"):
                continue
            prec = float(row.get("precision", 0) or 0)
            rec  = float(row.get("recall",    0) or 0)
            cov  = float(row.get("coverage",  0) or 0)
            auc  = float(row.get("auc",      0.5) or 0.5)
            expr = str(row.get("expression", numpy_expr))
            if prec == 0 or rec == 0:
                continue
            if prec < GATE_MIN_PREC or rec < GATE_MIN_REC:
                continue
            if _is_degenerate_expr(expr):
                continue
            pa  = str(row.get("protected_attr", ""))
            key = (pa, numpy_expr, "binary")
            if key in seen:
                continue
            seen.add(key)
            proxies.append({
                "name": f"{pa}_{grammar}_{config_label}", "expr": expr,
                "numpy_expr": numpy_expr, "protected_attr": pa,
                "grammar": grammar, "config": config_label,
                "prec": prec, "rec": rec, "cov": cov,
                "low_cov": cov < GATE_MIN_COV,
                "auc": auc, "nodes": int(row.get("nodes", 0) or 0),
                "continuous": False,
                "n_features": _count_distinct_features(row),
            })

    # ── Multi-attribute mining: partials CSVs (arith + ext), >= 2 features ────
    partial_csvs = (
        sorted(ds_dir.glob("arith-*-partials.csv"))
        + sorted(ds_dir.glob("ext-*-partials.csv"))
    )
    for csv_path in partial_csvs:
        if config_filter and config_filter not in csv_path.name:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "numpy_expr" not in df.columns:
            continue

        stem = csv_path.stem.replace("-partials", "")
        grammar = "arith" if csv_path.name.startswith("arith") else "ext"
        parts = stem.split("_", 2)
        config_label = parts[-1] if len(parts) >= 3 else stem

        for _, row in df.iterrows():
            ds_split = str(row.get("dataset", "")).strip()
            if ds_split not in ("processed", "encoded"):
                continue
            if _count_distinct_features(row) < 2:
                continue   # only want multi-attribute partial proxies
            numpy_expr = str(row.get("numpy_expr", "")).strip()
            if not numpy_expr or numpy_expr in ("", "nan", "FAILED"):
                continue
            prec = float(row.get("precision", 0) or 0)
            rec  = float(row.get("recall",    0) or 0)
            cov  = float(row.get("coverage",  0) or 0)
            auc  = float(row.get("auc",      0.5) or 0.5)
            expr = str(row.get("expression", numpy_expr))
            if prec == 0 or rec == 0:
                continue
            if prec < GATE_MIN_PREC or rec < GATE_MIN_REC:
                continue
            if _is_degenerate_expr(expr):
                continue
            pa  = str(row.get("protected_attr", ""))
            key = (pa, numpy_expr, "binary")
            if key in seen:
                continue
            seen.add(key)
            proxies.append({
                "name": f"{pa}_{grammar}_{config_label}_multi", "expr": expr,
                "numpy_expr": numpy_expr, "protected_attr": pa,
                "grammar": grammar, "config": config_label,
                "prec": prec, "rec": rec, "cov": cov,
                "low_cov": cov < GATE_MIN_COV,
                "auc": auc, "nodes": int(row.get("nodes", 0) or 0),
                "continuous": False,
                "n_features": _count_distinct_features(row),
            })

    n_binary = len(proxies)

    # ── Continuous proxies: cont CSVs (best-solution + partials) ─────────────
    # For expressions like "(max(Hours_per_week, Education_Num)) > 16.0" we emit
    # TWO entries:
    #   • continuous  — arithmetic score only (threshold stripped), AUC ≥ 0.60
    #   • binary      — full thresholded expression (True/False), prec/rec gated,
    #                   n_features ≥ 2 — goes into the Logic MA section
    import re as _re
    _THRESHOLD_RE = _re.compile(r'\s*([><=!]+)\s*(-?\d+(?:\.\d+)?)\s*$')

    cont_csvs = sorted(
        set(ds_dir.glob("cont-*.csv")) | set(ds_dir.glob("cont-*-partials.csv"))
    )
    for csv_path in cont_csvs:
        if config_filter and config_filter not in csv_path.name:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "numpy_expr" not in df.columns:
            continue

        is_partials = "-partials" in csv_path.name
        stem = csv_path.stem.replace("-partials", "")
        parts = stem.split("_", 2)
        config_label = parts[-1] if len(parts) >= 3 else stem

        for _, row in df.iterrows():
            ds_split = str(row.get("dataset", "")).strip()
            if ds_split not in ("processed", "encoded"):
                continue
            numpy_expr = str(row.get("numpy_expr", "")).strip()
            if not numpy_expr or numpy_expr in ("", "nan", "FAILED"):
                continue
            auc      = float(row.get("auc",       0.5) or 0.5)
            prec     = float(row.get("precision",  0)  or 0)
            rec      = float(row.get("recall",     0)  or 0)
            cov      = float(row.get("coverage",   0)  or 0)
            raw_expr = str(row.get("expression", numpy_expr))
            n_feats  = _count_distinct_features(row)
            pa       = str(row.get("protected_attr", ""))

            # Detect threshold suffix  e.g. " > 16.0"
            thresh_m = _THRESHOLD_RE.search(raw_expr)
            if thresh_m:
                arith_expr = raw_expr[:thresh_m.start()].strip()
                thresh_op  = thresh_m.group(1)
                thresh_val = thresh_m.group(2)
            else:
                arith_expr = raw_expr.strip()
                thresh_op = thresh_val = None

            if _is_degenerate_expr(arith_expr):
                continue

            # ── Emit continuous version (arithmetic score, AUC gate) ──────
            if auc >= GATE_MIN_AUC_CONT:
                cont_key = (pa, numpy_expr, "continuous")
                if cont_key not in seen:
                    seen.add(cont_key)
                    proxies.append({
                        "name":           f"{pa}_cont_{config_label}",
                        "expr":           arith_expr,
                        "numpy_expr":     numpy_expr,
                        "protected_attr": pa,
                        "grammar":        "cont",
                        "config":         config_label,
                        "prec": prec, "rec": rec, "cov": cov,
                        "low_cov":    False,
                        "auc":        auc,
                        "nodes":      int(row.get("nodes", 0) or 0),
                        "continuous": True,
                        "n_features": n_feats,
                    })

            # ── Also emit binary (logic) version when threshold is present ─
            # Condition: threshold exists, multi-attribute (>=2 features),
            # and passes the standard prec/rec quality gates.
            if (thresh_m is not None
                    and n_feats >= 2
                    and prec >= GATE_MIN_PREC
                    and rec  >= GATE_MIN_REC):
                bin_numpy = f"({numpy_expr} {thresh_op} {thresh_val}).astype(np.float64)"
                bin_key = (pa, bin_numpy, "binary")
                if bin_key not in seen:
                    seen.add(bin_key)
                    proxies.append({
                        "name":           f"{pa}_cont_logic_{config_label}",
                        "expr":           raw_expr,   # full expression with threshold
                        "numpy_expr":     bin_numpy,
                        "protected_attr": pa,
                        "grammar":        "cont-logic",
                        "config":         config_label,
                        "prec": prec, "rec": rec, "cov": cov,
                        "low_cov": cov < GATE_MIN_COV,
                        "auc":     auc,
                        "nodes":   int(row.get("nodes", 0) or 0),
                        "continuous": False,
                        "n_features": n_feats,
                    })

    n_cont = len(proxies) - n_binary
    print(f"  {dataset_name}: {n_binary} binary proxies  +  {n_cont} continuous proxies")
    return proxies


# ── Attribution run ───────────────────────────────────────────────────────────

def run_attribution_for_dataset(dataset_name: str, proxies: list[dict],
                                 device: str) -> dict:
    """Load the dataset, then for each proxy: inject column, train MLP, run IG.

    Returns a dict with keys: feature_names, task_used, proxy_results.
    """
    print(f"\n{'─'*60}")
    print(f"Dataset: {dataset_name.upper()}")

    loader = DATASET_LOADERS.get(dataset_name)
    if loader is None:
        print(f"  [SKIP] no loader registered for {dataset_name!r}")
        return {}

    ds = loader()
    if ds is None:
        print(f"  [SKIP] loader returned None for {dataset_name!r}")
        return {}

    X_tr_raw = ds["X_train"].astype(np.float32)
    X_te_raw = ds["X_test"].astype(np.float32)
    feat     = ds["feature_names"]
    task_used = ds["task_used"]

    # Determine target y
    y_tr = ds.get("y_train")
    y_te = ds.get("y_test")

    if y_tr is None or y_te is None:
        # Fall back: use the first available protected attribute as task target
        pa_tr = ds.get("pa_train", {})
        pa_te = ds.get("pa_test",  {})
        if pa_tr:
            first_pa   = next(iter(pa_tr))
            y_tr       = pa_tr[first_pa].astype(np.float32)
            y_te       = pa_te[first_pa].astype(np.float32)
            task_used  = f"{first_pa} (fallback — no downstream task found)"
            print(f"  ! No downstream task found. Using protected attr '{first_pa}' as target.")
        else:
            print(f"  [SKIP] no target variable available for {dataset_name!r}")
            return {}

    y_tr = np.asarray(y_tr, dtype=np.float32)
    y_te = np.asarray(y_te, dtype=np.float32)

    # Align sizes (sometimes loaders concatenate train+test; split evenly)
    n_tr = X_tr_raw.shape[0]
    n_te = X_te_raw.shape[0]
    if len(y_tr) > n_tr:
        y_tr = y_tr[:n_tr]
    if len(y_te) > n_te:
        y_te = y_te[n_te - len(y_te):]

    print(f"  Loaded: {n_tr:,} train / {n_te:,} test  |  "
          f"{len(feat)} features  |  task: {task_used or '?'}")
    print(f"  Target balance (test): {y_te.mean():.2%} positive")

    # Scale features
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_te_sc = scaler.transform(X_te_raw).astype(np.float32)

    # ── Criterion C: train base MLP (no proxy) once for this dataset ──────────
    # base_auc is the downstream AUC using only original features.
    # For each proxy, delta_auc = aug_auc - base_auc.
    # If delta_auc > 0 the proxy added information the model exploited → active.
    use_warmstart   = MLP_CONFIG.get("warmstart", True) and _ACTIVE_MODEL_TYPE == "mlp"
    base_state_dict = None
    print(f"\n  Training BASE {_ACTIVE_MODEL_TYPE.upper()} (no proxy, {len(feat)} features)"
          f"{'  [warm-start enabled]' if use_warmstart else ''} …")
    try:
        if _ACTIVE_MODEL_TYPE == "mlp":
            base_model, base_auc, _ = train_model(X_tr_sc, y_tr, X_te_sc, y_te, device)
            if use_warmstart:
                base_state_dict = {k: v.cpu()
                                   for k, v in base_model.state_dict().items()}
        else:
            trainer = MODEL_TRAINERS[_ACTIVE_MODEL_TYPE]
            _, base_auc = trainer(X_tr_sc, y_tr.astype(int), X_te_sc, y_te.astype(int))
        print(f"  Base AUC: {base_auc:.4f}")
    except Exception as e:
        print(f"  [WARN] base model failed ({e})")
        base_auc = float("nan")

    # ── Phase 1: compute proxy columns + deduplicate (must be serial) ────────────
    print(f"\n  Phase 1 — evaluating proxy expressions and deduplicating…")
    seen_fire_patterns: dict[str, set[bytes]] = {}
    work_items: list[WorkItem] = []

    for p in proxies:
        try:
            proxy_tr = np.asarray(
                forward_dataset(p["numpy_expr"], X_tr_raw), dtype=np.float32
            ).ravel()
            proxy_te = np.asarray(
                forward_dataset(p["numpy_expr"], X_te_raw), dtype=np.float32
            ).ravel()
        except Exception:
            continue

        is_continuous = p.get("continuous", False)

        if is_continuous:
            scaler_p = StandardScaler()
            proxy_tr = scaler_p.fit_transform(proxy_tr.reshape(-1, 1)).ravel().astype(np.float32)
            proxy_te = scaler_p.transform(proxy_te.reshape(-1, 1)).ravel().astype(np.float32)
        else:
            is_bool = ".astype(np.float64)" in p["numpy_expr"]
            if not is_bool:
                thr      = float(np.median(proxy_tr))
                proxy_tr = (proxy_tr > thr).astype(np.float32)
                proxy_te = (proxy_te > thr).astype(np.float32)
            else:
                proxy_tr = (proxy_tr > 0.5).astype(np.float32)
                proxy_te = (proxy_te > 0.5).astype(np.float32)

            pa_key   = p["protected_attr"]
            fire_sig = proxy_te.astype(np.uint8).tobytes()
            if pa_key not in seen_fire_patterns:
                seen_fire_patterns[pa_key] = set()
            if fire_sig in seen_fire_patterns[pa_key]:
                continue
            seen_fire_patterns[pa_key].add(fire_sig)

        fire_pct = float(proxy_te.mean() * 100)
        aug_names = feat + [p["name"]]
        X_aug_tr  = np.concatenate([X_tr_sc, proxy_tr.reshape(-1, 1)], axis=1)
        X_aug_te  = np.concatenate([X_te_sc, proxy_te.reshape(-1, 1)], axis=1)

        work_items.append(WorkItem(
            proxy       = p,
            aug_names   = aug_names,
            X_aug_tr    = X_aug_tr,
            X_aug_te    = X_aug_te,
            y_tr        = y_tr,
            y_te        = y_te,
            fire_pct    = fire_pct,
            base_auc    = base_auc,
            feat        = feat,
            proxy_col_idx   = len(feat),
            base_state_dict = base_state_dict,
            model_type      = _ACTIVE_MODEL_TYPE,
        ))

    n_workers = MLP_CONFIG.get("n_workers", 1)
    print(f"  Phase 1 done — {len(work_items)} unique proxies to process "
          f"(workers={n_workers}, epochs={MLP_CONFIG['epochs']})")

    # ── Phase 2: train MLP + attribution (parallelised over proxies) ─────────
    proxy_results = []

    def _run_serial():
        for i, item in enumerate(work_items, 1):
            print(f"  [{i}/{len(work_items)}] {item.proxy['name']}  "
                  f"fires={item.fire_pct:.1f}%", flush=True)
            result = _proxy_worker_fn(item)
            if result is not None:
                proxy_results.append(result)
                print(f"    IG rank {result['proxy_rank']}/{len(item.aug_names)}  "
                      f"share {result['proxy_share']:.1%}  "
                      f"ΔAUC {result['delta_auc']:+.4f}  "
                      f"{'✓ ACTIVE' if result['is_active_c'] else '– inactive'}")

    def _run_parallel(n_workers):
        futures = {}
        # 'spawn' avoids fork+CUDA and fork+OpenMP deadlocks (XGBoost uses OpenMP).
        # The initializer re-applies MLP_CONFIG edits that spawn workers won't inherit.
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_process_init,
            initargs=(dict(MLP_CONFIG), _ACTIVE_MODEL_TYPE),
        ) as pool:
            for item in work_items:
                fut = pool.submit(_proxy_worker_fn, item)
                futures[fut] = item
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            item = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"  [WORKER ERROR] {item.proxy['name']}: {e}")
                result = None
            if result is not None:
                proxy_results.append(result)
            pct = 100 * completed / len(work_items)
            print(f"  [{completed}/{len(work_items)}  {pct:.0f}%] "
                  f"{item.proxy['name']}  "
                  + (f"IG share {result['proxy_share']:.1%}  "
                     f"{'✓ ACTIVE' if result['is_active_c'] else '– inactive'}"
                     if result else "FAILED"),
                  flush=True)

    if n_workers <= 1:
        _run_serial()
    else:
        print(f"  Dispatching {len(work_items)} jobs across {n_workers} workers…")
        _run_parallel(n_workers)

    # Sort by proxy attribution share (highest first)
    proxy_results.sort(key=lambda r: -r["proxy_share"])

    return {
        "dataset_name": dataset_name,
        "feature_names": feat,
        "task_used":    task_used or "unknown",
        "n_test":       n_te,
        "proxy_results": proxy_results,
    }


# ── Parallel worker infrastructure ───────────────────────────────────────────

class WorkItem(NamedTuple):
    """Everything needed to train one proxy MLP + run attribution.
    All numpy arrays; no dataset-level state needed at worker time."""
    proxy:      dict        # proxy metadata dict
    aug_names:  list        # feature names + proxy name
    X_aug_tr:   np.ndarray  # (n_train, n_feats+1)
    X_aug_te:   np.ndarray  # (n_test,  n_feats+1)
    y_tr:       np.ndarray
    y_te:       np.ndarray
    fire_pct:   float
    base_auc:   float
    feat:       list        # base feature names (for _guess_components)
    proxy_col_idx: int
    base_state_dict: dict | None  # warm-start: trained base model weights
    model_type: str               # "mlp" | "logistic_regression" | "random_forest" | "xgboost"


def _worker_process_init(mlp_config_snapshot: dict, active_model_type: str) -> None:
    """Initialiser for spawn-based worker processes.

    spawn doesn't inherit modified globals from the parent, so we push
    MLP_CONFIG and _ACTIVE_MODEL_TYPE explicitly via the ProcessPoolExecutor
    initializer mechanism.
    """
    global MLP_CONFIG, _ACTIVE_MODEL_TYPE
    MLP_CONFIG.update(mlp_config_snapshot)
    _ACTIVE_MODEL_TYPE = active_model_type


def _proxy_worker_fn(item: WorkItem) -> dict | None:
    """Process a single proxy. Runs in a subprocess (no shared state needed)."""
    import warnings, math
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = item.model_type
    train_log  = []   # only populated for MLP full training

    try:
        if model_type == "mlp":
            use_warmstart = (MLP_CONFIG.get("warmstart", True)
                             and item.base_state_dict is not None)
            if use_warmstart:
                n_base = item.proxy_col_idx
                model  = _build_proxy_model_warmstart(
                    item.base_state_dict, n_base, device)
                model, final_auc = finetune_model(
                    model, item.X_aug_tr, item.y_tr,
                    item.X_aug_te, item.y_te, device)
            else:
                model, final_auc, _ = train_model(
                    item.X_aug_tr, item.y_tr, item.X_aug_te, item.y_te, device)
            all_method_attrs = run_all_attribution_methods(
                model, item.X_aug_te, device, X_aug_tr=item.X_aug_tr)
            primary_method = "IG"
        else:
            trainer = MODEL_TRAINERS[model_type]
            model, final_auc = trainer(item.X_aug_tr, item.y_tr.astype(int),
                                       item.X_aug_te, item.y_te.astype(int))
            all_method_attrs = run_non_mlp_attribution(
                model, item.X_aug_te, item.X_aug_tr, model_type)
            primary_method = "SHAP"
    except Exception as e:
        print(f"      [model failed: {e}]")
        return None

    if all_method_attrs.get(primary_method) is None:
        return None

    method_summary = _summarise_method_attrs(all_method_attrs, item.proxy_col_idx)
    primary_stats = method_summary[primary_method]
    mean_abs    = primary_stats["mean_abs"]
    proxy_rank  = primary_stats["proxy_rank"]
    proxy_share = primary_stats["proxy_share"]
    proxy_attr  = primary_stats["proxy_attr"]

    valid_ranks    = [s["proxy_rank"] for s in method_summary.values() if s is not None]
    consensus_rank = float(np.mean(valid_ranks)) if valid_ranks else float(proxy_rank)

    comp_attrs: dict[str, float] = {}
    for comp_name in _guess_components(item.proxy["expr"], item.feat):
        if comp_name in item.feat:
            idx = item.feat.index(comp_name)
            comp_attrs[comp_name] = float(mean_abs[idx])

    delta_auc   = (final_auc - item.base_auc) if not math.isnan(item.base_auc) else float("nan")
    is_active_c = (not math.isnan(delta_auc)) and (delta_auc > 0.001)

    return {
        "proxy":           item.proxy,
        "aug_names":       item.aug_names,
        "mean_abs":        mean_abs,
        "proxy_rank":      proxy_rank,
        "proxy_share":     proxy_share,
        "proxy_attr":      proxy_attr,
        "method_summary":  method_summary,
        "consensus_rank":  consensus_rank,
        "comp_attrs":      comp_attrs,
        "fire_pct":        item.fire_pct,
        "final_auc":       final_auc,
        "base_auc":        item.base_auc,
        "delta_auc":       delta_auc,
        "is_active_c":     is_active_c,
        "train_log":       train_log,
    }


def _guess_components(expr: str, feat_names: list[str]) -> list[str]:
    """Return feature names that appear in the expression string."""
    import re
    found = []
    for name in sorted(feat_names, key=len, reverse=True):
        if re.search(r'\b' + re.escape(name) + r'\b', expr):
            found.append(name)
    return found


# ── HTML helpers ──────────────────────────────────────────────────────────────

CSS = """
:root{--bg:#0f1117;--bg2:#161b22;--bg3:#1c2333;--bg4:#21262d;
      --border:#30363d;--text:#c9d1d9;--muted:#8b949e}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);
     font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.6}
.hdr{background:var(--bg2);border-bottom:1px solid var(--border);padding:24px 40px}
.hdr h1{font-size:22px;font-weight:700;color:#e6edf3;margin-bottom:5px}
.meta{color:var(--muted);font-size:12px}.meta span{margin-right:18px}
.ds-section{border-top:3px solid #3b82f6;margin:32px 0}
.ds-header{background:var(--bg2);padding:16px 40px;
           display:flex;align-items:baseline;gap:16px}
.ds-header h2{font-size:18px;font-weight:700;color:#e6edf3}
.ds-task{font-size:11px;color:#58a6ff;background:#0d1f3c;
         border:1px solid #1e40af;border-radius:10px;padding:1px 8px}
.ds-body{padding:16px 40px 32px}
.summary-table{width:100%;border-collapse:collapse;margin-bottom:24px;font-size:12px}
.summary-table th{background:var(--bg4);color:var(--muted);font-weight:600;
                  padding:5px 8px;border-bottom:1px solid var(--border);
                  text-align:left;white-space:nowrap}
.summary-table td{padding:4px 8px;border-bottom:1px solid #1c2333;
                  vertical-align:middle}
.summary-table tr:hover td{background:var(--bg4)}
.summary-table td.num{text-align:right;font-family:monospace}
.proxy-card{background:var(--bg2);border:1px solid var(--border);
            border-radius:8px;margin-bottom:20px;overflow:hidden}
.proxy-card-hdr{background:var(--bg3);border-bottom:1px solid var(--border);
                padding:12px 18px}
.proxy-card-hdr h3{font-size:14px;font-weight:700;color:#e6edf3}
.proxy-expr{font-family:monospace;font-size:12px;color:#79c0ff;margin-top:3px;
            word-break:break-all}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.pill{font-size:10px;background:var(--bg4);border:1px solid var(--border);
      border-radius:10px;padding:1px 8px;color:var(--muted)}
.pill b{color:#e6edf3}
.proxy-card-body{padding:14px 18px;display:grid;grid-template-columns:1fr;gap:14px}
.card{background:var(--bg);border:1px solid var(--border);border-radius:6px;
      padding:12px;overflow-x:auto}
.card-title{font-size:11px;font-weight:700;color:#8b949e;
            text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
table.t{width:100%;border-collapse:collapse;font-size:11px}
.t th{background:var(--bg4);color:var(--muted);padding:4px 6px;
      border-bottom:1px solid var(--border);text-align:left}
.t td{padding:3px 6px;border-bottom:1px solid #1c2333}
.t tr:hover td{background:var(--bg4)}
.t td.num{text-align:right;font-family:monospace}
.hl-proxy td{background:#0d1f3c!important}
.hl-comp  td{background:#0d1a0d!important}
.tag{display:inline-block;font-size:8px;font-weight:700;padding:1px 4px;
     border-radius:5px;margin-left:3px;vertical-align:middle}
.tag-p{background:#0d1f3c;color:#93c5fd;border:1px solid #1e40af}
.tag-c{background:#0d1a0d;color:#86efac;border:1px solid #166534}
.bar-wrap{display:inline-block;vertical-align:middle}
code{background:rgba(255,255,255,.07);padding:1px 4px;border-radius:3px;font-size:10px}
.badge-active{display:inline-block;font-size:9px;font-weight:700;padding:2px 6px;
  border-radius:4px;background:#0d2e12;color:#4ade80;border:1px solid #166534}
.badge-inactive{display:inline-block;font-size:9px;font-weight:700;padding:2px 6px;
  border-radius:4px;background:#1a1a2e;color:#6b7280;border:1px solid #374151}
"""


def _esc(s): return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def _bar(v, vmax, w=110):
    if vmax == 0: return ""
    r = min(v / vmax, 1.0)
    px = max(1, int(w * r))
    c = "#1d4ed8" if r>.75 else "#2563eb" if r>.5 else "#3b82f6" if r>.25 else "#93c5fd"
    return (f'<span class="bar-wrap"><span style="display:inline-block;'
            f'width:{px}px;height:7px;border-radius:2px;background:{c};'
            f'margin-right:5px;vertical-align:middle"></span></span>')

def _pct_color(v):
    if v >= 80: return "#4ade80"
    if v >= 60: return "#86efac"
    if v >= 40: return "#fde68a"
    return "#f87171"

def _share_color(v):
    if v >= .15: return "#4ade80"
    if v >= .08: return "#86efac"
    if v >= .04: return "#fde68a"
    return "#f87171"


def _build_attr_table(mean_abs, feat_names, proxy_name, comp_names, title, n):
    order = np.argsort(mean_abs)[::-1]
    vmax  = float(mean_abs.max()) if mean_abs.max() > 0 else 1.0
    total = float(mean_abs.sum()) or 1.0
    rows  = []
    for rank, i in enumerate(order, 1):
        name     = feat_names[i]
        val      = float(mean_abs[i])
        share    = val / total
        is_proxy = (name == proxy_name)
        is_comp  = (name in comp_names)
        cls      = ' class="hl-proxy"' if is_proxy else (' class="hl-comp"' if is_comp else '')
        tag      = ('<span class="tag tag-p">proxy</span>' if is_proxy else
                    '<span class="tag tag-c">component</span>' if is_comp else '')
        rows.append(
            f'<tr{cls}>'
            f'<td class="num" style="color:#6b7280">{rank}</td>'
            f'<td><code>{_esc(name)}</code>{tag}</td>'
            f'<td class="num">{val:.5f}</td>'
            f'<td>{_bar(val, vmax, 90)}</td>'
            f'<td class="num" style="color:{_share_color(share)}">{share:.1%}</td>'
            f'</tr>'
        )
    return (
        f'<div class="card"><div class="card-title">{title} '
        f'<span style="color:#6b7280;font-weight:400">(n={n:,})</span></div>'
        f'<table class="t"><thead><tr>'
        f'<th>Rank</th><th>Feature</th><th>Mean |IG|</th><th></th><th>Share</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _build_methods_table(method_summary: dict, n_features: int) -> str:
    """Build a comparison table and Spearman correlation matrix for all methods."""
    from html import escape as _esc2

    # ── Per-method proxy attribution row ─────────────────────────────────────
    rows = []
    for method in ATTRIBUTION_METHODS:
        s = method_summary.get(method)
        if s is None:
            rows.append(
                f'<tr><td><b style="color:#f87171">{_esc(method)}</b></td>'
                f'<td class="num" style="color:#6b7280" colspan="3">failed</td></tr>'
            )
        else:
            rank     = s["proxy_rank"]
            share    = s["proxy_share"]
            sc       = _share_color(share)
            rank_pct = rank / n_features
            rank_c   = ("#4ade80" if rank_pct <= 0.25 else
                        "#fde68a" if rank_pct <= 0.5  else "#f87171")
            rows.append(
                f'<tr>'
                f'<td><b style="color:#c9d1d9">{_esc(method)}</b></td>'
                f'<td class="num" style="color:{rank_c}">{rank}/{n_features}</td>'
                f'<td class="num" style="color:{sc}">{share:.1%}</td>'
                f'<td class="num" style="color:#8b949e">{s["proxy_attr"]:.5f}</td>'
                f'</tr>'
            )

    methods_table = (
        '<div class="card-title">Method comparison — proxy attribution rank &amp; share</div>'
        '<table class="t"><thead><tr>'
        '<th>Method</th><th>Proxy rank</th><th>Share</th><th>Mean |attr|</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )

    # ── Spearman correlation matrix ───────────────────────────────────────────
    # ρ measures agreement between two methods on the full feature ranking.
    # ρ > 0.9 = findings are method-invariant; ρ < 0.7 = notable divergence.
    valid_methods = [m for m in ATTRIBUTION_METHODS
                     if method_summary.get(m) is not None]

    spearman_html = ""
    if len(valid_methods) >= 2:
        def _rho_color(rho: float) -> str:
            if rho >= 0.90: return "#4ade80"
            if rho >= 0.70: return "#fde68a"
            return "#f87171"

        header_cells = "".join(
            f'<th style="text-align:center;font-size:10px">{_esc(m)}</th>'
            for m in valid_methods
        )
        matrix_rows = []
        for m1 in valid_methods:
            s1 = method_summary[m1]
            cells = f'<td><b style="color:#c9d1d9;font-size:10px">{_esc(m1)}</b></td>'
            for m2 in valid_methods:
                rho = s1["spearman"].get(m2, float("nan"))
                if m1 == m2:
                    cells += '<td class="num" style="color:#4b5563">1.00</td>'
                elif _math.isnan(rho):
                    cells += '<td class="num" style="color:#6b7280">n/a</td>'
                else:
                    cells += (f'<td class="num" style="color:{_rho_color(rho)}">'
                              f'{rho:.2f}</td>')
            matrix_rows.append(f"<tr>{cells}</tr>")

        spearman_html = (
            '<div style="margin-top:10px">'
            '<div class="card-title" style="margin-bottom:4px">'
            'Spearman ρ between methods — feature ranking agreement '
            '<span style="color:#8b949e;font-weight:normal">'
            '(ρ ≥ 0.90 = method-invariant &nbsp;|&nbsp; ρ &lt; 0.70 = divergent)</span>'
            '</div>'
            '<table class="t"><thead><tr><th></th>'
            f'{header_cells}</tr></thead>'
            f'<tbody>{"".join(matrix_rows)}</tbody></table>'
            '</div>'
        )

    return (
        f'<div class="card">{methods_table}{spearman_html}</div>'
    )


def _build_train_log(log, auc):
    rows = "".join(
        f'<tr><td class="num">{e}</td>'
        f'<td class="num">{l:.4f}</td>'
        f'<td class="num" style="color:#58a6ff">{a:.4f}</td></tr>'
        for e, l, a in log
    )
    return (
        f'<div class="card"><div class="card-title">Training log</div>'
        f'<table class="t"><thead><tr><th>Epoch</th><th>Loss</th><th>AUC</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        f'<div style="margin-top:6px;font-size:11px;color:#4ade80">'
        f'Final AUC: <b>{auc:.4f}</b></div></div>'
    )


def build_html(all_dataset_results: list[dict], run_dir: Path) -> str:
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">',
        '<title>Proxy Attribution Report</title>',
        f'<style>{CSS}</style></head><body>',
        f'<div class="hdr"><h1>Proxy Attribution Report — All Datasets</h1>',
        f'<div class="meta">',
        f'<span>Run: <b>{_esc(run_dir.name)}</b></span>',
        f'<span>Generated: {now}</span>',
        f'<span>Model: {"ProxyNet " + str(MLP_CONFIG["hidden_dims"]) + " · " + MLP_CONFIG["activation"].upper() + " · " + MLP_CONFIG["optimizer"].upper() + " · " + str(MLP_CONFIG["epochs"]) + " epochs" if _ACTIVE_MODEL_TYPE == "mlp" else _ACTIVE_MODEL_TYPE.replace("_", " ").title()}</span>',
        f'<span>Attribution: {"IG · DeepLIFT · GradSHAP · GxI · LRP (IG n_steps=" + str(MLP_CONFIG["ig_n_steps"]) + ")" if _ACTIVE_MODEL_TYPE == "mlp" else "SHAP (" + ("LinearExplainer" if _ACTIVE_MODEL_TYPE == "logistic_regression" else "TreeExplainer") + ")"}</span>',
        '</div></div>',
    ]

    for ds_result in all_dataset_results:
        if not ds_result or not ds_result.get("proxy_results"):
            ds_name = ds_result.get("dataset_name", "?") if ds_result else "?"
            parts.append(
                f'<div class="ds-section">'
                f'<div class="ds-header"><h2>{_esc(ds_name.replace("_"," ").title())}</h2>'
                f'<span class="ds-task">No results</span></div></div>'
            )
            continue

        ds_name    = ds_result["dataset_name"]
        feat_names = ds_result["feature_names"]
        task_used  = ds_result["task_used"]
        n_test     = ds_result["n_test"]
        p_results  = ds_result["proxy_results"]

        # Section 1: Logic multi-attribute — binary output, >= 2 features
        logic_ma_results = [r for r in p_results
                            if not r["proxy"].get("continuous")
                            and r["proxy"].get("n_features", 0) >= 2]
        # Section 2: Continuous multi-attribute — real number output, >= 2 features
        cont_ma_results  = [r for r in p_results
                            if r["proxy"].get("continuous")
                            and r["proxy"].get("n_features", 0) >= 2]

        parts.append(
            f'<div class="ds-section">'
            f'<div class="ds-header">'
            f'<h2>{_esc(ds_name.replace("_"," ").title())}</h2>'
            f'<span class="ds-task">task: {_esc(task_used)}</span>'
            f'<span style="color:#6b7280;font-size:11px">'
            f'{len(logic_ma_results)} logic MA · {len(cont_ma_results)} continuous MA proxies</span>'
            f'</div>'
            f'<div class="ds-body">'
        )

        # ── Section renderer (reused for binary and continuous) ───────────
        import math as _math

        def _render_proxy_section(section_results, section_title, section_note):
            if not section_results:
                parts.append(
                    f'<div style="color:#6b7280;font-style:italic;margin:12px 0">'
                    f'No proxies found for this section.</div>'
                )
                return
            max_share = max(x["proxy_share"] for x in section_results)
            parts.append(
                f'<div style="font-size:13px;font-weight:700;color:#e6edf3;margin:18px 0 4px">'
                f'{_esc(section_title)}</div>'
                f'<div style="font-size:11px;color:#8b949e;margin-bottom:10px">'
                f'{_esc(section_note)}</div>'
                '<table class="summary-table"><thead><tr>'
                '<th>Rank</th><th>Proxy expression</th><th>Protected</th>'
                '<th>Grammar</th>'
                f'<th title="Proxy column attribution share ({("SHAP" if _ACTIVE_MODEL_TYPE != "mlp" else "IG")})">Attr share ↓</th>'
                f'<th title="{("SHAP" if _ACTIVE_MODEL_TYPE != "mlp" else "IG")} attr rank / total features">{("SHAP" if _ACTIVE_MODEL_TYPE != "mlp" else "IG")} rank</th>'
                '<th title="Average proxy rank across all methods (lower=more agreed)">Consensus↓</th>'
                '<th>Feats</th><th>Proxy AUC</th><th>Prec</th><th>Rec</th>'
                '<th title="Coverage: fraction of dataset where proxy fires. ⚠ = below 5%">Cov</th>'
                '<th>Fires%</th>'
                f'<th>{_ACTIVE_MODEL_TYPE.replace("_"," ").title() if _ACTIVE_MODEL_TYPE != "mlp" else "MLP"} AUC</th>'
                '<th title="AUC(augmented) − AUC(base): positive = proxy adds predictive value; near-zero = redundant but still used">ΔAUC</th>'
                '</tr></thead><tbody>'
            )
            for i, r in enumerate(section_results, 1):
                p            = r["proxy"]
                share        = r["proxy_share"]
                sc           = _share_color(share)
                nf           = len(r["aug_names"])
                delta        = r.get("delta_auc", float("nan"))
                is_active    = r.get("is_active_c", False)
                delta_str    = f'{delta:+.4f}' if not _math.isnan(delta) else "n/a"
                delta_col    = ("#4ade80" if delta > 0.001 else
                                "#f87171" if not _math.isnan(delta) else "#6b7280")
                active_badge = (
                    '<span class="badge-active">✓ active</span>' if is_active
                    else '<span class="badge-inactive">–</span>'
                )
                is_cont      = p.get("continuous", False)
                n_feats      = p.get("n_features", 0)
                feats_cell   = (f'<b style="color:#f59e0b">{n_feats}</b>'
                                if n_feats >= 2 else str(n_feats))
                cons_rank    = r.get("consensus_rank", r["proxy_rank"])
                cons_pct     = cons_rank / nf
                cons_c       = "#4ade80" if cons_pct <= 0.25 else "#fde68a" if cons_pct <= 0.5 else "#f87171"
                if is_cont:
                    prec_cell = '<td class="num" style="color:#6b7280">—</td>'
                    rec_cell  = '<td class="num" style="color:#6b7280">—</td>'
                    cov_cell  = '<td class="num" style="color:#6b7280">—</td>'
                    fire_cell = '<td class="num" style="color:#6b7280">float</td>'
                else:
                    prec_cell = f'<td class="num" style="color:{_pct_color(p["prec"])}">{p["prec"]:.0f}%</td>'
                    rec_cell  = f'<td class="num">{p["rec"]:.0f}%</td>'
                    cov_cell  = (('<td class="num" style="color:#f97316" title="Low coverage (<5%)">⚠ '
                                  if p.get("low_cov") else '<td class="num">')
                                 + f'{p["cov"]:.0f}%</td>')
                    fire_cell = f'<td class="num">{r["fire_pct"]:.0f}%</td>'
                parts.append(
                    f'<tr>'
                    f'<td class="num" style="color:#6b7280">{i}</td>'
                    f'<td><code style="color:#79c0ff">{_esc(p["expr"][:70])}</code></td>'
                    f'<td><code>{_esc(p["protected_attr"])}</code></td>'
                    f'<td style="color:#8b949e;font-size:11px">{_esc(p["grammar"])}</td>'
                    f'<td class="num"><b style="color:{sc}">{share:.1%}</b>'
                    f'{_bar(share, max_share, 60)}</td>'
                    f'<td class="num" style="color:#93c5fd">{r["proxy_rank"]}/{nf}</td>'
                    f'<td class="num" style="color:{cons_c}">{cons_rank:.1f}</td>'
                    f'<td class="num">{feats_cell}</td>'
                    f'<td class="num">{p["auc"]:.3f}</td>'
                    + prec_cell + rec_cell + cov_cell + fire_cell
                    + f'<td class="num" style="color:#4ade80">{r["final_auc"]:.4f}</td>'
                    f'<td class="num" style="color:{delta_col}">{delta_str}</td>'
                    f'</tr>'
                )
            parts.append('</tbody></table>')

            # Per-proxy detail cards
            for r in section_results:
                p          = r["proxy"]
                aug_names  = r["aug_names"]
                mean_abs   = r["mean_abs"]
                comp_names = list(r["comp_attrs"].keys())
                is_cont    = p.get("continuous", False)
                _delta     = r.get("delta_auc", float("nan"))
                _base      = r.get("base_auc",  float("nan"))
                _delta_str = f'{_delta:+.4f}' if not _math.isnan(_delta) else "n/a"
                _delta_c   = ("#4ade80" if _delta > 0.001 else
                              "#f87171" if not _math.isnan(_delta) else "#8b949e")
                n_feats = p.get("n_features", 0)
                if is_cont:
                    metric_pills = (
                        f'<div class="pill">GP AUC: <b style="color:#58a6ff">{p["auc"]:.3f}</b></div>'
                        f'<div class="pill" title="Injected as standardised float — no binary threshold">'
                        f'Type: <b style="color:#4ade80">continuous score</b></div>'
                    )
                else:
                    low_cov_style = ' style="color:#f97316" title="Low coverage (<5%)"' if p.get("low_cov") else ""
                    metric_pills = (
                        f'<div class="pill">Prec: <b style="color:{_pct_color(p["prec"])}">'
                        f'{p["prec"]:.0f}%</b></div>'
                        f'<div class="pill">Rec: <b>{p["rec"]:.0f}%</b></div>'
                        f'<div class="pill">Cov: <b{low_cov_style}>'
                        f'{"⚠ " if p.get("low_cov") else ""}{p["cov"]:.0f}%</b></div>'
                        f'<div class="pill">Proxy AUC: <b style="color:#58a6ff">{p["auc"]:.3f}</b></div>'
                    )
                parts.append(
                    f'<div class="proxy-card">'
                    f'<div class="proxy-card-hdr">'
                    f'<h3>{_esc(p["name"])}'
                    + (f' <span style="font-size:10px;color:#f59e0b">({n_feats} features)</span>'
                       if n_feats >= 2 else '')
                    + f'</h3>'
                    f'<div class="proxy-expr">{_esc(p["expr"])}</div>'
                    f'<div class="pills">'
                    f'<div class="pill">Protected: <b>{_esc(p["protected_attr"])}</b></div>'
                    + metric_pills
                    + f'<div class="pill">Base AUC: <b style="color:#8b949e">'
                    f'{"n/a" if _math.isnan(_base) else f"{_base:.4f}"}</b></div>'
                    f'<div class="pill">MLP AUC: <b style="color:#4ade80">{r["final_auc"]:.4f}</b></div>'
                    f'<div class="pill" title="Positive = proxy adds predictive value; near-zero = redundant but still used by model">'
                    f'ΔAUC: <b style="color:{_delta_c}">{_delta_str}</b></div>'
                    f'<div class="pill">Attr rank: <b style="color:#93c5fd">'
                    f'{r["proxy_rank"]}/{len(aug_names)}</b></div>'
                    f'<div class="pill">Attr share: <b style="color:{_share_color(r["proxy_share"])}">'
                    f'{r["proxy_share"]:.1%}</b></div>'
                    f'</div>'
                    f'</div>'
                    f'<div class="proxy-card-body">'
                )
                parts.append(_build_attr_table(
                    mean_abs, aug_names, p["name"], comp_names,
                    "IG attribution ranking (all test)", n_test
                ))
                method_summary = r.get("method_summary", {})
                if method_summary:
                    parts.append(_build_methods_table(method_summary, len(aug_names)))
                parts.append(_build_train_log(r["train_log"], r["final_auc"]))
                parts.append('</div></div>')  # proxy-card-body, proxy-card

        # ── Section 1: Logic Multi-Attribute Partial Proxies ──────────────
        parts.append(
            '<div style="border-top:2px solid #3b82f6;margin:20px 0 0">'
            '<div style="background:#0d1f3c;padding:10px 16px;font-size:13px;'
            'font-weight:700;color:#93c5fd;letter-spacing:.04em">'
            'LOGIC MULTI-ATTRIBUTE PARTIAL PROXIES — Expressions with &gt; and AND · True/False output'
            '</div></div>'
        )
        _render_proxy_section(
            logic_ma_results,
            f"{len(logic_ma_results)} logic multi-attribute proxies ranked by attribution share",
            "Combines ≥2 features using comparisons (>) and logical AND/NOT. "
            "Each expression evaluates to True or False (injected as 0/1 column). "
            "Gated: prec ≥ 80%, rec ≥ 15%, AUC ≥ 0.55."
        )

        # ── Section 2: Continuous Multi-Attribute Partial Proxies ──────────
        parts.append(
            '<div style="border-top:2px solid #4ade80;margin:32px 0 0">'
            '<div style="background:#0d2e12;padding:10px 16px;font-size:13px;'
            'font-weight:700;color:#4ade80;letter-spacing:.04em">'
            'CONTINUOUS MULTI-ATTRIBUTE PARTIAL PROXIES — Pure arithmetic · Real number output'
            '</div></div>'
        )
        _render_proxy_section(
            cont_ma_results,
            f"{len(cont_ma_results)} continuous multi-attribute proxies ranked by attribution share",
            "Combines ≥2 features using pure arithmetic (+, −, ×, ÷). "
            "No comparisons, no True/False — produces a real number score "
            "(injected as standardised float column). Gated: GP AUC ≥ 0.60."
        )

        parts.append('</div></div>')   # ds-body, ds-section

    parts.append('</body></html>')
    return '\n'.join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def _latest_run_dir() -> Path | None:
    candidates = sorted(
        (d for d in RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")),
        key=lambda d: d.name, reverse=True
    )
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(
        description="Run MLP+Captum attribution on GP-discovered proxies."
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Path to a pipeline_runs/ timestamp folder. "
             "Defaults to the most recent run.",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["adult", "law_school", "compas", "oulad", "folktables", "german_credit"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Filter CSVs by config label substring (e.g. 'p1_r1_c1').",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output HTML path. Defaults to <run_dir>/attribution_report.html",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes for MLP+attribution (0 = all CPU cores). "
             "Default: 1 (serial).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override MLP training epochs (default: use MLP_CONFIG value).",
    )
    parser.add_argument(
        "--ig-steps", type=int, default=None,
        help="Override Integrated Gradients n_steps (default: use MLP_CONFIG value).",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: --workers 0 --epochs 10 --ig-steps 20. "
             "Cuts runtime to ~15 min on a typical CPU.",
    )
    parser.add_argument(
        "--no-warmstart", action="store_true",
        help="Disable warm-start fine-tuning: train each proxy MLP from scratch "
             "(slower but fully independent models). Default: warm-start enabled.",
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=None,
        help="Override warm-start fine-tune epochs (default: 10).",
    )
    parser.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "logistic_regression", "random_forest", "xgboost"],
        help="Model to train and attribute. "
             "mlp → ProxyNet + Captum (IG/DeepLIFT/GradSHAP/GxI/LRP). "
             "logistic_regression / random_forest / xgboost → SHAP attribution. "
             "Default: mlp",
    )
    parser.add_argument(
        "--continue", dest="resume", action="store_true",
        help="Skip datasets that already have a per-dataset attribution report "
             "for the selected model. Useful to resume a partial run.",
    )
    parser.add_argument(
        "--top-n", type=int, default=0,
        help="Keep only the top-N proxies per protected attribute per dataset, "
             "ranked by precision (binary) or AUC (continuous). "
             "Default: 0 (no cap — process all qualifying proxies).",
    )
    args = parser.parse_args()

    # ── Apply --fast / override flags ─────────────────────────────────────────
    if args.fast:
        args.epochs   = args.epochs   or 10
        args.ig_steps = args.ig_steps or 20
        # Don't auto-parallelise if CUDA is available — multiprocessing + CUDA
        # causes deadlocks; serial on GPU is faster anyway.
        if args.workers == 1:
            cuda_available = False
            try:
                import torch as _t
                cuda_available = _t.cuda.is_available()
            except Exception:
                pass
            args.workers = 1 if cuda_available else 0

    n_workers = os.cpu_count() if args.workers == 0 else args.workers
    # fork() + CUDA = deadlock. Force serial for MLP when CUDA is available;
    # GPU makes per-proxy training fast enough that parallelism isn't needed.
    if args.model == "mlp" and n_workers > 1:
        try:
            import torch as _t
            if _t.cuda.is_available():
                print("[note] CUDA detected — MLP forced to serial (fork+CUDA incompatible).")
                n_workers = 1
        except Exception:
            pass
    if args.epochs:
        MLP_CONFIG["epochs"]    = args.epochs
    if args.ig_steps:
        MLP_CONFIG["ig_n_steps"] = args.ig_steps
    if args.no_warmstart:
        MLP_CONFIG["warmstart"] = False
    if args.finetune_epochs:
        MLP_CONFIG["finetune_epochs"] = args.finetune_epochs
    MLP_CONFIG["n_workers"] = n_workers

    # Set global model type so worker functions can read it
    global _ACTIVE_MODEL_TYPE
    _ACTIVE_MODEL_TYPE = args.model

    run_dir = args.run_dir or _latest_run_dir()
    if run_dir is None:
        print(f"No run folder found in {RUNS_DIR}. "
              "Run the pipeline first or pass --run-dir.")
        sys.exit(1)

    print(f"Run folder : {run_dir}")
    print(f"Model      : {_ACTIVE_MODEL_TYPE}")
    print(f"Datasets   : {args.datasets}")
    if _ACTIVE_MODEL_TYPE == "mlp":
        print(f"Workers    : {n_workers}  |  Epochs: {MLP_CONFIG['epochs']}  |  "
              f"IG steps: {MLP_CONFIG['ig_n_steps']}")
    if args.config:
        print(f"Config filter: {args.config}")

    # Only initialize CUDA for MLP. SHAP models run on CPU, and calling
    # torch.cuda.is_available() initializes the CUDA context in the parent
    # process — any subsequent fork() then breaks with "Cannot re-initialize
    # CUDA in forked subprocess".
    device = "cuda" if (args.model == "mlp" and torch.cuda.is_available()) else "cpu"
    print(f"Device     : {device}\n")

    model_dir = run_dir / f"attribution_{_ACTIVE_MODEL_TYPE}"
    model_dir.mkdir(exist_ok=True)
    out_path = args.out or (model_dir / "attribution_report.html")
    all_results = []
    skipped_datasets = []

    for ds_name in args.datasets:
        if args.resume:
            ds_path = model_dir / f"attribution_report_{ds_name}.html"
            if ds_path.exists() and ds_path.stat().st_size > 10_000:
                print(f"  [{ds_name}] already done — skipping")
                skipped_datasets.append(ds_name)
                continue
        proxies = collect_proxies(run_dir, ds_name, args.config)
        if not proxies:
            all_results.append({"dataset_name": ds_name, "proxy_results": []})
            continue

        # ── Top-N cap: keep best N per (protected_attr, continuous) bucket ──
        if args.top_n is not None and args.top_n > 0:
            from collections import defaultdict
            buckets: dict[tuple, list] = defaultdict(list)
            for p in proxies:
                buckets[(p["protected_attr"], p["continuous"])].append(p)
            proxies = []
            for (pa, is_cont), grp in buckets.items():
                sort_key = "auc" if is_cont else "prec"
                grp_sorted = sorted(grp, key=lambda x: x.get(sort_key, 0), reverse=True)
                proxies.extend(grp_sorted[:args.top_n])
            print(f"  → top-{args.top_n} cap applied: {len(proxies)} proxies kept")

        result = run_attribution_for_dataset(ds_name, proxies, device)
        all_results.append(result)

    print("\nBuilding HTML report…")
    if skipped_datasets:
        print(f"  [note] Combined report only includes newly processed datasets. "
              f"Skipped: {', '.join(skipped_datasets)}. "
              f"Per-dataset reports for skipped datasets are unchanged.")
    if all_results:
        html = build_html(all_results, run_dir)
        out_path.write_text(html, encoding="utf-8")
        size_kb = out_path.stat().st_size // 1024
        print(f"Report written → {model_dir.name}/attribution_report.html  ({size_kb} KB)")
    else:
        print("  No new datasets processed — combined report not updated.")

    # ── Per-dataset reports ───────────────────────────────────────────────────
    for result in all_results:
        ds_name = result.get("dataset_name", "unknown")
        ds_html = build_html([result], run_dir)
        ds_path = model_dir / f"attribution_report_{ds_name}.html"
        ds_path.write_text(ds_html, encoding="utf-8")
        ds_kb = ds_path.stat().st_size // 1024
        print(f"  {ds_name} → {ds_path.name}  ({ds_kb} KB)")


if __name__ == "__main__":
    main()
