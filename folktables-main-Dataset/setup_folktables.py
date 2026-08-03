#!/usr/bin/env python3
"""
setup_folktables.py
===================
Preprocesses the Folktables ACS Adult Reconstruction dataset and saves a
pipeline-ready NPZ + metadata.json under folktables-main-Dataset/processed/.

Dataset: adult_reconstruction.csv  (Ding et al., 2021 — ACS 2018 data)
         49 531 rows, structurally similar to UCI Adult but more modern.

Target:
    high_income = (income >= 50 000)   ← 26.8 % positive
    (mirrors the original Adult ">50K" threshold)

Protected attributes:
    white            — race == "White"
    black            — race == "Black"
    asian            — race == "Asian-Pac-Islander"
    amer_indian      — race == "Amer-Indian-Eskimo"
    other_race       — race == "Other"
    gender_male      — gender == "Male"

Run once:
    cd folktables-main-Dataset
    python setup_folktables.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT      = Path(__file__).resolve().parent
RAW_CSV   = ROOT / "folktables-main" / "adult_reconstruction.csv"
OUT_DIR   = ROOT / "processed"
NPZ_NAME  = "folktables_base_features.npz"
RANDOM_STATE = 42

INCOME_THRESHOLD = 50_000

PROTECTED_COLS = ["race", "gender"]
TARGET_COL     = "income"

CATEGORICAL_COLS = ["workclass", "education", "marital-status",
                    "relationship", "native-country", "occupation"]
NUMERIC_COLS     = ["hours-per-week", "age", "capital-gain",
                    "capital-loss", "education-num"]

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")

# ── Build target ──────────────────────────────────────────────────────────────
y = (df[TARGET_COL] >= INCOME_THRESHOLD).astype(np.int32).values
print(f"Target (income >= {INCOME_THRESHOLD}): {y.sum()} positive "
      f"({100*y.mean():.1f}%)")

# ── Build protected attribute vectors ─────────────────────────────────────────
def _race_flag(val: str) -> np.ndarray:
    return (df["race"] == val).astype(np.int32).values

white       = _race_flag("White")
black       = _race_flag("Black")
asian       = _race_flag("Asian-Pac-Islander")
amer_indian = _race_flag("Amer-Indian-Eskimo")
other_race  = _race_flag("Other")
gender_male = (df["gender"] == "Male").astype(np.int32).values

for name, vec in [("white", white), ("black", black), ("asian", asian),
                  ("amer_indian", amer_indian), ("other_race", other_race),
                  ("gender_male", gender_male)]:
    print(f"  {name}: {vec.sum()} ({100*vec.mean():.1f}%)")

# ── Feature matrix ────────────────────────────────────────────────────────────
feat_df = df.drop(columns=["race", "gender", "income"], errors="ignore").copy()

# Normalise column names (replace hyphens with underscores for grammar)
col_rename = {c: c.replace("-", "_") for c in feat_df.columns}
feat_df = feat_df.rename(columns=col_rename)

# Update column lists with renamed names
CATEGORICAL_COLS_R = [c.replace("-", "_") for c in CATEGORICAL_COLS]
NUMERIC_COLS_R     = [c.replace("-", "_") for c in NUMERIC_COLS]

# Fill NaN
for col in CATEGORICAL_COLS_R:
    if col in feat_df.columns:
        feat_df[col] = feat_df[col].fillna("missing")
for col in NUMERIC_COLS_R:
    if col in feat_df.columns:
        feat_df[col] = feat_df[col].fillna(feat_df[col].median())

feature_names = list(feat_df.columns)
print(f"Features ({len(feature_names)}): {feature_names}")

# ── Label-encode categoricals ─────────────────────────────────────────────────
label_encodings: dict[str, list] = {}
for col in CATEGORICAL_COLS_R:
    if col not in feat_df.columns:
        continue
    le = LabelEncoder()
    feat_df[col] = le.fit_transform(feat_df[col].astype(str))
    label_encodings[col] = le.classes_.tolist()

X_raw = feat_df.values.astype(np.float64)

# ── Train/test split ───────────────────────────────────────────────────────────
all_arrays = [X_raw, y, white, black, asian, amer_indian, other_race, gender_male]
splits = train_test_split(*all_arrays, test_size=0.20,
                          random_state=RANDOM_STATE, stratify=y)

# train/test for each array
(X_tr_raw, X_te_raw,
 y_tr, y_te,
 white_tr, white_te,
 black_tr, black_te,
 asian_tr, asian_te,
 amer_indian_tr, amer_indian_te,
 other_tr, other_te,
 gm_tr, gm_te) = splits

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr_raw)
X_te   = scaler.transform(X_te_raw)

# ── Save NPZ ──────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)
npz_path = OUT_DIR / NPZ_NAME
np.savez_compressed(
    npz_path,
    X_train=X_tr,  X_test=X_te,
    X_train_raw=X_tr_raw, X_test_raw=X_te_raw,
    y_train=y_tr,  y_test=y_te,
    feature_names=np.array(feature_names, dtype=object),
    white_train=white_tr,           white_test=white_te,
    black_train=black_tr,           black_test=black_te,
    asian_train=asian_tr,           asian_test=asian_te,
    amer_indian_train=amer_indian_tr, amer_indian_test=amer_indian_te,
    other_race_train=other_tr,      other_race_test=other_te,
    gender_male_train=gm_tr,        gender_male_test=gm_te,
)
print(f"Saved NPZ → {npz_path}  "
      f"(train {len(X_tr)}, test {len(X_te)}, features {len(feature_names)})")

# ── Compute per-feature individual baselines ──────────────────────────────────
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

baselines: dict[str, dict] = {}
X_all = X_raw
protected = [
    ("white", white, "race"), ("black", black, "race"),
    ("asian", asian, "race"), ("amer_indian", amer_indian, "race"),
    ("other_race", other_race, "race"), ("gender_male", gender_male, "sex"),
]
for attr_name, attr_vec, group in protected:
    per_feat: dict[str, dict] = {}
    mi_vals = mutual_info_classif(X_all, attr_vec, discrete_features=False,
                                  random_state=RANDOM_STATE)
    for i, fn in enumerate(feature_names):
        col = X_all[:, i]
        try:
            auc = roc_auc_score(attr_vec, col)
            if auc < 0.5:
                auc = roc_auc_score(attr_vec, -col)
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
        "per_feature":    per_feat,
        "best_feature_auc": best_fn,
        "best_auc":         per_feat[best_fn]["auc"],
    }
    print(f"  {attr_name} — best single feature: {best_fn} "
          f"(AUC {per_feat[best_fn]['auc']:.4f})")

bl_path = OUT_DIR / "individual_proxy_baselines.csv"
rows = []
for attr, bd in baselines.items():
    for feat, stats in bd["per_feature"].items():
        rows.append({"protected_attr": attr, "feature": feat, **stats})
pd.DataFrame(rows).to_csv(bl_path, index=False)

# ── Save metadata.json ────────────────────────────────────────────────────────
meta = {
    "task_target": f"income_ge_{INCOME_THRESHOLD}",
    "source": "Ding et al. (2021) Folktables ACS 2018 Adult Reconstruction",
    "preprocessing": {
        "imputation": "numeric=median, categorical='missing'",
        "encoding":   "LabelEncoder for categoricals",
        "scaling":    "StandardScaler (z-score), fit on train only",
    },
    "protected_attributes": {
        "white":       {"group": "race",   "type": "binary"},
        "black":       {"group": "race",   "type": "binary"},
        "asian":       {"group": "race",   "type": "binary"},
        "amer_indian": {"group": "race",   "type": "binary"},
        "other_race":  {"group": "race",   "type": "binary"},
        "gender_male": {"group": "sex",    "type": "binary"},
    },
    "base_feature_columns": feature_names,
    "numeric_features":     [c for c in NUMERIC_COLS_R if c in feature_names],
    "categorical_features": [c for c in CATEGORICAL_COLS_R if c in feature_names],
    "label_encodings":      label_encodings,
    "scaler_mean":  scaler.mean_.tolist(),
    "scaler_std":   scaler.scale_.tolist(),
    "train_size":   int(len(X_tr)),
    "test_size":    int(len(X_te)),
    "random_state": RANDOM_STATE,
}
meta_path = OUT_DIR / "metadata.json"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"Saved metadata → {meta_path}")
print("Done.")
