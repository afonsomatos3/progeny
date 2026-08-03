#!/usr/bin/env python3
"""
setup_german_credit.py
======================
Preprocesses the German Credit dataset (Statlog / South German Credit)
and saves a pipeline-ready NPZ + metadata.json under processed/.

Source: German-Credit-Analysis-master/german_credit_data.xlsx
        1000 rows, 10 columns including Risk (good/bad).

Target:
    bad_risk = (Risk == "bad")   ← 30% positive

Protected attributes:
    male      — Sex == "male"   (69%)
    age_young — Age < 30        (37%)

Run once:
    cd German-Credit-Analysis-master
    python setup_german_credit.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT         = Path(__file__).resolve().parent
RAW_XLSX     = ROOT / "German-Credit-Analysis-master" / "german_credit_data.xlsx"
OUT_DIR      = ROOT / "processed"
NPZ_NAME     = "german_credit_base_features.npz"
RANDOM_STATE = 42

CATEGORICAL_COLS = ["Housing", "Saving_accounts", "Checking_account", "Purpose"]
NUMERIC_COLS     = ["Age", "Job", "Credit_amount", "Duration"]

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel(RAW_XLSX)
df.columns = [c.replace(" ", "_") for c in df.columns]
print(f"Loaded {len(df)} rows.")

# ── Target: bad_risk = 1 ──────────────────────────────────────────────────────
y = (df["Risk"] == "bad").astype(np.int32).values
print(f"Target (bad_risk): {y.sum()} positive ({100*y.mean():.1f}%)")

# ── Protected attributes ──────────────────────────────────────────────────────
male      = (df["Sex"] == "male").astype(np.int32).values
age_young = (df["Age"] < 30).astype(np.int32).values
print(f"male: {male.sum()} ({100*male.mean():.1f}%)  "
      f"age_young(<30): {age_young.sum()} ({100*age_young.mean():.1f}%)")

# ── Feature matrix (drop protected + target columns) ──────────────────────────
feat_df = df.drop(columns=["Sex", "Risk"]).copy()

# Fill NaN in categoricals with "missing" sentinel
for col in CATEGORICAL_COLS:
    if col in feat_df.columns:
        feat_df[col] = feat_df[col].fillna("missing")

feature_names = list(feat_df.columns)
print(f"Features ({len(feature_names)}): {feature_names}")

# ── Integer-encode categoricals for numpy storage ──────────────────────────────
# NumPy requires a uniform numeric dtype, so categorical string columns are
# mapped to integer codes solely for storage. This does NOT imply numeric
# relationships between categories. The GP grammar enforces correct semantics:
#   • Arithmetic grammar: categorical features are EXCLUDED from terminals.
#   • Extended (typed) grammar: categories are accessed via equality operators
#     (CatEquals / CatNotEquals) using the original category names.
label_encodings: dict[str, list] = {}
for col in CATEGORICAL_COLS:
    if col not in feat_df.columns:
        continue
    le = LabelEncoder()
    feat_df[col] = le.fit_transform(feat_df[col].astype(str))
    label_encodings[col] = le.classes_.tolist()

X_raw = feat_df.values.astype(np.float64)

# ── Train / test split (80/20, stratified) ────────────────────────────────────
(X_tr_raw, X_te_raw,
 y_tr,     y_te,
 male_tr,  male_te,
 agy_tr,   agy_te) = train_test_split(
    X_raw, y, male, age_young,
    test_size=0.20, random_state=RANDOM_STATE, stratify=y,
)

# ── Scale (fit on train only) ─────────────────────────────────────────────────
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr_raw)
X_te   = scaler.transform(X_te_raw)

# ── Save NPZ ──────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)
npz_path = OUT_DIR / NPZ_NAME
np.savez_compressed(
    npz_path,
    X_train=X_tr,     X_test=X_te,
    X_train_raw=X_tr_raw, X_test_raw=X_te_raw,
    y_train=y_tr,     y_test=y_te,
    feature_names=np.array(feature_names, dtype=object),
    male_train=male_tr,       male_test=male_te,
    age_young_train=agy_tr,   age_young_test=agy_te,
)
print(f"Saved NPZ → {npz_path}  "
      f"(train={len(X_tr)}, test={len(X_te)}, features={len(feature_names)})")

# ── Per-feature individual proxy baselines ────────────────────────────────────
X_all = X_raw
protected_attrs = [("male", male, "sex"), ("age_young", age_young, "age")]
baselines: dict[str, dict] = {}

for attr_name, attr_vec, group in protected_attrs:
    per_feat: dict[str, dict] = {}
    mi_vals = mutual_info_classif(X_all, attr_vec, discrete_features=False,
                                  random_state=RANDOM_STATE)
    for i, fn in enumerate(feature_names):
        col = X_all[:, i]
        try:
            auc = roc_auc_score(attr_vec, col)
            auc = max(auc, 1 - auc)          # flip if below 0.5
        except Exception:
            auc = 0.5
        corr = float(np.corrcoef(col, attr_vec)[0, 1])
        per_feat[fn] = {
            "auc":      round(float(auc), 4),
            "mi":       round(float(mi_vals[i]), 4),
            "abs_corr": round(abs(corr), 4),
        }
    best_fn = max(per_feat, key=lambda k: per_feat[k]["auc"])
    baselines[attr_name] = {
        "group": group,
        "per_feature":      per_feat,
        "best_feature_auc": best_fn,
        "best_auc":         per_feat[best_fn]["auc"],
    }
    print(f"  {attr_name}: best={best_fn} AUC={per_feat[best_fn]['auc']:.4f}")

bl_path = OUT_DIR / "individual_proxy_baselines.csv"
rows = [{"protected_attr": attr, "feature": fn, **stats}
        for attr, bd in baselines.items()
        for fn, stats in bd["per_feature"].items()]
pd.DataFrame(rows).to_csv(bl_path, index=False)

# ── metadata.json ─────────────────────────────────────────────────────────────
meta = {
    "task_target": "bad_risk",
    "source":      "South German Credit (Statlog), Kaggle/UCI variant with Risk column",
    "preprocessing": {
        "imputation": "Saving_accounts / Checking_account NaN → 'missing' category",
        "encoding": (
            "categorical features stored as integer codes for numpy compatibility; "
            "arithmetic grammar excludes them; "
            "extended grammar uses equality operators only (CatEquals/CatNotEquals)"
        ),
        "scaling":    "StandardScaler (z-score), fit on train only",
    },
    "protected_attributes": {
        "male":      {"group": "sex", "type": "binary", "definition": "Sex == 'male'"},
        "age_young": {"group": "age", "type": "binary", "definition": "Age < 30"},
    },
    "base_feature_columns":  feature_names,
    "numeric_features":      [c for c in NUMERIC_COLS     if c in feature_names],
    "categorical_features":  [c for c in CATEGORICAL_COLS if c in feature_names],
    "label_encodings":       label_encodings,
    "scaler_mean":  scaler.mean_.tolist(),
    "scaler_std":   scaler.scale_.tolist(),
    "train_size":   int(len(X_tr)),
    "test_size":    int(len(X_te)),
    "random_state": RANDOM_STATE,
}
(OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
print(f"Saved metadata → {OUT_DIR}/metadata.json")
print("Done.")
