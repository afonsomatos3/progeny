"""
Law School Dataset Setup and Base Feature Extraction (Processed)
================================================================

Dataset: bar_pass_prediction.csv (Law School Admissions / Bar Passage)
Source:  LSAC National Longitudinal Bar Passage Study (Wightman, 1998)

This script:
  1. Loads and inspects the raw dataset
  2. Applies column taxonomy (removes leakage, redundancy, protected cols)
  3. Cleans and preprocesses the data (imputation, encoding, scaling)
  4. Extracts base (non-protected, non-target) feature matrices (19 features)
  5. Computes individual proxy baselines for 5 protected attributes:
       Race:  black, asian, hisp, other
       Sex:   male
  6. Saves processed artefacts for downstream proxy / fairness experiments

Protected attributes match the non-processed version for direct comparison.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mutual_info_score
from scipy.stats import pointbiserialr

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "work" / "data"
RAW_CSV = DATA_DIR / "bar_pass_prediction.csv"
OUT_DIR = ROOT / "processed"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Loading raw dataset")
print("=" * 60)

df_raw = pd.read_csv(RAW_CSV)
print(f"   Shape : {df_raw.shape}")
print(f"   Columns ({len(df_raw.columns)}): {list(df_raw.columns)}")

# ---------------------------------------------------------------------------
# 2. Column taxonomy
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Column taxonomy")
print("=" * 60)

TARGET = "pass_bar"

# Protected attributes -- 5 binary columns, each its own attribute
# (matches the non-processed version for direct comparison)
PROTECTED_BINARY = {
    "black": "race",
    "asian": "race",
    "hisp":  "race",
    "other": "race",
    "male":  "sex",
}

# Other protected / sensitive columns (excluded from features)
PROTECTED_OTHER = ["race1", "sex", "gender", "race", "race2"]

# Columns that leak the target (encode bar outcome directly)
TARGET_LEAKAGE = [
    "bar1", "bar2", "bar1_yr", "bar2_yr",
    "bar", "bar_passed",
    "dnn_bar_pass_prediction",
]

# Metadata
META_COLS = ["ID"]

# Redundant features (gpa is identical to ugpa in every row)
REDUNDANT_FEATURES = ["gpa"]

# Everything left after exclusions becomes a base feature
EXCLUDE = set(
    list(PROTECTED_BINARY.keys())
    + PROTECTED_OTHER
    + [TARGET]
    + TARGET_LEAKAGE
    + META_COLS
    + REDUNDANT_FEATURES
)

BASE_FEATURE_COLS = [c for c in df_raw.columns if c not in EXCLUDE]

print(f"   Target                : {TARGET}")
print(f"   Protected attributes  : {list(PROTECTED_BINARY.keys())}")
print(f"   Other protected cols  : {PROTECTED_OTHER}")
print(f"   Target leakage        : {TARGET_LEAKAGE}")
print(f"   Redundant features    : {REDUNDANT_FEATURES}")
print(f"   Base feature cols ({len(BASE_FEATURE_COLS)}): {BASE_FEATURE_COLS}")
print(f"   Excluded cols ({len(EXCLUDE)}): {sorted(EXCLUDE)}")

# ---------------------------------------------------------------------------
# 3. Clean the data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Cleaning")
print("=" * 60)

df = df_raw.copy()

# 3a. Drop rows where the target is missing
n_before = len(df)
df = df.dropna(subset=[TARGET])
print(f"   Dropped {n_before - len(df)} rows with missing target "
      f"({len(df)} remaining)")

# 3b. Ensure target is integer 0/1
df[TARGET] = df[TARGET].astype(int)
print(f"   Target distribution:\n{df[TARGET].value_counts().to_string()}")

# 3c. Ensure protected attributes are clean integers
for pa in PROTECTED_BINARY:
    df[pa] = df[pa].fillna(0).astype(int)

print(f"\n   Protected attribute distributions:")
for pa, group in PROTECTED_BINARY.items():
    counts = df[pa].value_counts().sort_index()
    n1 = counts.get(1, 0)
    n0 = counts.get(0, 0)
    print(f"   {pa:8s} ({group}):  1={n1:>6d}  0={n0:>6d}  "
          f"(prevalence {100 * n1 / len(df):.1f}%)")

# 3d. Handle missing values in base features
print("\n   Missing values in base features (before impute):")
missing = df[BASE_FEATURE_COLS].isnull().sum()
missing_any = missing[missing > 0]
if len(missing_any) > 0:
    print(missing_any.to_string())
else:
    print("   (none)")

# Identify numeric vs categorical base features
numeric_base = df[BASE_FEATURE_COLS].select_dtypes(include=[np.number]).columns.tolist()
categorical_base = [c for c in BASE_FEATURE_COLS if c not in numeric_base]

print(f"\n   Numeric base features    ({len(numeric_base)}): {numeric_base}")
print(f"   Categorical base features({len(categorical_base)}): {categorical_base}")

# Impute numeric with median, categorical with mode
for c in numeric_base:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

for c in categorical_base:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

# 3e. Encode categorical base features for numpy storage (integer codes).
#
# NumPy arrays require a uniform numeric dtype, so categorical string columns
# must be mapped to integer codes before saving. This encoding is NOT a claim
# that the categories have numeric relationships — it is purely a storage
# compatibility step.
#
# The GP grammar enforces the correct semantics:
#   • Arithmetic grammar:  categorical features are EXCLUDED from the terminal
#     set entirely (build_grammar restricts Var to numeric_features only).
#   • Extended / typed grammar:  categorical features appear ONLY as equality
#     conditions (CatEquals / CatNotEquals), which translate back to the
#     original category names in all output expressions.
#
# The 'label_encodings' dict saved to metadata.json records the
# integer-to-category-name mapping so the grammar can reconstruct
# human-readable expressions (e.g. "grad == Y", not "grad == 2").
label_encoders = {}
for c in categorical_base:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    label_encoders[c] = le
    print(f"   Storage-encoded '{c}': {list(le.classes_)} → integers 0..{len(le.classes_)-1}")

# ---------------------------------------------------------------------------
# 4. Build feature matrix and split
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Building feature matrices & train/test split")
print("=" * 60)

X = df[BASE_FEATURE_COLS].values.astype(np.float64)
y = df[TARGET].values.astype(int)

# Extract each protected attribute array
protected_arrays = {}
for pa in PROTECTED_BINARY:
    protected_arrays[pa] = df[pa].values.astype(int)

# Stratified split (80/20) preserving target balance
split_data = train_test_split(
    X, y, *[protected_arrays[pa] for pa in PROTECTED_BINARY],
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Unpack: X_train, X_test, y_train, y_test, then pairs for each protected attr
X_train, X_test = split_data[0], split_data[1]
y_train, y_test = split_data[2], split_data[3]

pa_train = {}
pa_test = {}
for i, pa in enumerate(PROTECTED_BINARY):
    pa_train[pa] = split_data[4 + 2 * i]
    pa_test[pa]  = split_data[4 + 2 * i + 1]

print(f"   Train : {X_train.shape[0]}  (pass_bar=1: {y_train.sum()}, "
      f"pass_bar=0: {(1 - y_train).sum()})")
print(f"   Test  : {X_test.shape[0]}  (pass_bar=1: {y_test.sum()}, "
      f"pass_bar=0: {(1 - y_test).sum()})")

# 4b. Standardise numeric features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Individual proxy baseline audit (multi-metric)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. Individual proxy baseline audit (multi-metric)")
print("=" * 60)
print("   Computing baselines for each of 5 protected attributes...")

N_MI_BINS = 20


def compute_individual_baselines(X_data, y_binary, feature_names, label):
    """Compute AUC, MI, and correlation for each feature vs a binary target."""
    results = []
    for i, col in enumerate(feature_names):
        xi = X_data[:, i]

        if len(np.unique(y_binary)) < 2:
            results.append({
                "feature": col, "auc": 0.5, "mi": 0.0,
                "abs_corr": 0.0, "corr_pval": 1.0,
            })
            continue

        # --- AUC (logistic regression) ---
        try:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(xi.reshape(-1, 1), y_binary)
            proba = clf.predict_proba(xi.reshape(-1, 1))[:, 1]
            auc = roc_auc_score(y_binary, proba)
        except Exception:
            auc = 0.5

        # --- Mutual Information ---
        xi_finite = xi[np.isfinite(xi)]
        if len(np.unique(xi_finite)) > N_MI_BINS:
            xi_binned = pd.cut(xi, bins=N_MI_BINS, labels=False, duplicates="drop")
            xi_binned = xi_binned.fillna(0).astype(int) if hasattr(xi_binned, "fillna") else np.nan_to_num(xi_binned)
        else:
            xi_binned = xi.astype(int)
        mi = mutual_info_score(y_binary, xi_binned)

        # --- Point-biserial correlation ---
        try:
            corr, pval = pointbiserialr(y_binary, xi)
            corr = abs(corr)
        except Exception:
            corr, pval = 0.0, 1.0

        results.append({
            "feature": col,
            "auc": round(auc, 4),
            "mi": round(mi, 4),
            "abs_corr": round(corr, 4),
            "corr_pval": round(pval, 6),
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("auc", ascending=False).reset_index(drop=True)

    print(f"\n   --- {label} ---")
    print(f"   {'Feature':20s}  {'AUC':>7s}  {'MI':>7s}  {'|corr|':>7s}  {'p-val':>10s}  Flag")
    print(f"   {'─' * 20}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 10}  {'─' * 25}")
    for _, row in df_results.iterrows():
        flags = []
        if row["auc"] >= 0.65:
            flags.append("high-AUC")
        if row["mi"] >= df_results["mi"].quantile(0.75):
            flags.append("high-MI")
        if row["abs_corr"] >= 0.15:
            flags.append("high-corr")
        flag_str = ", ".join(flags) if flags else ""
        print(f"   {row['feature']:20s}  {row['auc']:7.4f}  {row['mi']:7.4f}  "
              f"{row['abs_corr']:7.4f}  {row['corr_pval']:10.6f}  {flag_str}")

    best = df_results.iloc[0]
    print(f"\n   Best AUC: {best['feature']} ({best['auc']:.4f})")
    return df_results


# Compute baselines for EACH protected attribute
all_baselines = {}
for pa, group in PROTECTED_BINARY.items():
    baselines_df = compute_individual_baselines(
        X_train, pa_train[pa], BASE_FEATURE_COLS,
        f"Predicting {pa.upper()} [{group}] (train set)"
    )
    all_baselines[pa] = baselines_df

print("\n   NOTE: All features are KEPT for GP regardless of individual scores.")
print("   These baselines define the bar that GP composite proxies must")
print("   exceed to qualify as genuinely multi-attribute discoveries.")

# ---------------------------------------------------------------------------
# 6. Save processed artefacts
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. Saving artefacts")
print("=" * 60)

# .npz -- scaled + raw feature matrices, all protected attribute arrays
npz_data = {
    "X_train": X_train_scaled,
    "X_test": X_test_scaled,
    "X_train_raw": X_train,
    "X_test_raw": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": np.array(BASE_FEATURE_COLS),
}
for pa in PROTECTED_BINARY:
    npz_data[f"{pa}_train"] = pa_train[pa]
    npz_data[f"{pa}_test"]  = pa_test[pa]

np.savez_compressed(OUT_DIR / "law_school_base_features.npz", **npz_data)
print(f"   -> {OUT_DIR / 'law_school_base_features.npz'}")

# CSV with base features + target + protected attributes
df_clean = df[BASE_FEATURE_COLS + [TARGET] + list(PROTECTED_BINARY.keys())].copy()
df_clean.to_csv(OUT_DIR / "law_school_clean.csv", index=False)
print(f"   -> {OUT_DIR / 'law_school_clean.csv'}")

# Build combined baselines CSV
combined_parts = []
for pa, bl_df in all_baselines.items():
    renamed = bl_df.rename(columns={
        "auc": f"auc_{pa}", "mi": f"mi_{pa}",
        "abs_corr": f"corr_{pa}", "corr_pval": f"pval_{pa}",
    })
    combined_parts.append(renamed.set_index("feature"))
combined_baselines = pd.concat(combined_parts, axis=1).reset_index()
combined_baselines.to_csv(OUT_DIR / "individual_proxy_baselines.csv", index=False)
print(f"   -> {OUT_DIR / 'individual_proxy_baselines.csv'}")

# Metadata JSON
individual_baselines_meta = {}
for pa, bl_df in all_baselines.items():
    individual_baselines_meta[pa] = {
        "group": PROTECTED_BINARY[pa],
        "per_feature": bl_df.set_index("feature").to_dict(orient="index"),
        "best_feature_auc": bl_df.iloc[0]["feature"],
        "best_auc": float(bl_df.iloc[0]["auc"]),
    }

metadata = {
    "task_target": TARGET,
    "preprocessing": {
        "column_taxonomy": "applied (leakage, redundancy, protected columns removed)",
        "imputation": "numeric=median, categorical=mode",
        "encoding": (
            "categorical features stored as integer codes for numpy compatibility; "
            "arithmetic grammar excludes them entirely; "
            "extended grammar uses equality operators (CatEquals/CatNotEquals) only"
        ),
        "scaling": "StandardScaler (z-score), fit on train only",
    },
    "protected_attributes": {
        pa: {"group": group, "type": "binary"}
        for pa, group in PROTECTED_BINARY.items()
    },
    "base_feature_columns": BASE_FEATURE_COLS,
    "numeric_features": numeric_base,
    "categorical_features": categorical_base,
    "label_encodings": {
        c: list(le.classes_) for c, le in label_encoders.items()
    },
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std": scaler.scale_.tolist(),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "random_state": 42,
    "individual_proxy_baselines": individual_baselines_meta,
}
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"   -> {OUT_DIR / 'metadata.json'}")

# ---------------------------------------------------------------------------
# 7. Generate HTML results overview
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. Generating HTML results overview")
print("=" * 60)


def strength_badge(auc):
    if auc >= 0.65:
        return "strong", "Strong"
    elif auc >= 0.55:
        return "moderate", "Moderate"
    else:
        return "weak", "Weak"


def fmt_pval(p):
    if p < 0.001:
        return "&lt;0.001"
    return f"{p:.3f}"


def build_table_rows(bl_df):
    rows = []
    for idx, row in bl_df.iterrows():
        cls, label = strength_badge(row["auc"])
        css = f' class="{cls}"' if cls != "weak" else ""
        rows.append(
            f'  <tr{css}><td>{idx + 1}</td><td>{row["feature"]}</td>'
            f'<td>{row["auc"]:.4f}</td><td>{row["mi"]:.4f}</td>'
            f'<td>{row["abs_corr"]:.4f}</td><td>{fmt_pval(row["corr_pval"])}</td>'
            f'<td><span class="badge badge-{cls}">{label}</span></td></tr>'
        )
    return "\n".join(rows)


# Build protected attribute summary cards
pa_cards = []
for pa, bl_df in all_baselines.items():
    best_auc = bl_df.iloc[0]["auc"]
    best_feat = bl_df.iloc[0]["feature"]
    n1 = int(pa_train[pa].sum())
    prevalence = 100 * n1 / len(pa_train[pa])
    pa_cards.append(
        f'  <div class="summary-card">'
        f'<h3>{pa} ({PROTECTED_BINARY[pa]})</h3>'
        f'<div class="val">Best AUC {best_auc:.4f}</div>'
        f'<div style="font-size:.85rem;color:#666">{best_feat} &middot; '
        f'prevalence {prevalence:.1f}%</div></div>'
    )

# Build per-protected-attribute HTML sections
pa_sections = []
for pa, bl_df in all_baselines.items():
    best_auc = bl_df.iloc[0]["auc"]
    table_rows = build_table_rows(bl_df)
    pa_sections.append(f"""
<h2>Proxy Baselines: {pa.upper()} ({PROTECTED_BINARY[pa]})</h2>
<p class="note">Sorted by AUC descending. GP must beat <strong>{best_auc:.4f}</strong> to claim multi-attribute discovery.</p>
<table>
  <tr><th>#</th><th>Feature</th><th>AUC</th><th>MI</th><th>|corr|</th><th>p-value</th><th>Strength</th></tr>
{table_rows}
</table>""")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MA-Proxies &mdash; Law School Baseline Results (Processed)</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem auto; max-width: 1100px; background: #f8f9fa; color: #222; }}
  h1 {{ border-bottom: 3px solid #333; padding-bottom: .4rem; }}
  h2 {{ margin-top: 2.5rem; color: #444; }}
  h3 {{ margin-top: 1.5rem; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  th, td {{ padding: .55rem .75rem; text-align: left; border: 1px solid #ddd; }}
  th {{ background: #343a40; color: #fff; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  tr:hover {{ background: #e8edf2; }}
  .strong {{ background: #d4edda !important; font-weight: 600; }}
  .moderate {{ background: #fff3cd !important; }}
  .weak {{ }}
  .note {{ font-size: .85rem; color: #666; margin-top: -.8rem; margin-bottom: 1.5rem; }}
  .badge {{ display: inline-block; padding: .15rem .5rem; border-radius: 3px; font-size: .78rem; font-weight: 600; }}
  .badge-strong {{ background: #28a745; color: #fff; }}
  .badge-moderate {{ background: #ffc107; color: #333; }}
  .badge-weak {{ background: #dee2e6; color: #555; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .summary-card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .summary-card h3 {{ margin: 0 0 .3rem 0; font-size: .9rem; color: #888; text-transform: uppercase; }}
  .summary-card .val {{ font-size: 1.4rem; font-weight: 700; color: #222; }}
</style>
</head>
<body>

<h1>MA-Proxies &mdash; Law School Baseline Results (Processed)</h1>

<!-- Summary cards -->
<div class="summary-grid">
  <div class="summary-card">
    <h3>Dataset</h3>
    <div class="val">{len(df):,} rows</div>
    <div style="font-size:.85rem;color:#666">Train {X_train.shape[0]:,} &middot; Test {X_test.shape[0]:,}</div>
  </div>
  <div class="summary-card">
    <h3>Base features</h3>
    <div class="val">{len(BASE_FEATURE_COLS)}</div>
    <div style="font-size:.85rem;color:#666">{len(numeric_base)} numeric &middot; {len(categorical_base)} categorical</div>
  </div>
{chr(10).join(pa_cards)}
</div>

<!-- Preprocessing approach -->
<h2>Preprocessing Approach &amp; Justification</h2>

<p>The processed version applies deliberate, researcher-driven curation of the raw 39-column dataset before
any proxy analysis. Every decision is motivated by a single goal: <strong>ensure the GP terminal set contains
only features that a real-world model builder could plausibly use</strong>, so that any proxy discovered by GP
reflects a genuine fairness risk rather than a trivial artefact of data leakage or redundancy.</p>

<h3>Step 1 &mdash; Column Taxonomy (39 &rarr; 19 base features)</h3>
<p>Each raw column is classified into one of eight categories. Only columns in the <em>Base features</em> category
enter the GP terminal set; all others are excluded with explicit justification.</p>

<table>
  <tr><th>Category</th><th>Columns</th><th>Rationale for inclusion / exclusion</th></tr>
  <tr><td>Task target</td><td><code>pass_bar</code></td><td>The downstream prediction task (Stage&nbsp;4). Excluded from features because the GP fitness function predicts <em>protected attributes</em>, not the target itself.</td></tr>
  <tr><td>Protected attributes</td><td><code>black</code>, <code>asian</code>, <code>hisp</code>, <code>other</code>, <code>male</code></td><td>The 5 binary protected attributes that GP optimises against. These are labels, not features.</td></tr>
  <tr><td>Other protected columns</td><td><code>race1</code>, <code>race</code>, <code>race2</code>, <code>sex</code>, <code>gender</code></td><td>Alternative encodings of the same protected information. Including them would let GP trivially achieve AUC&nbsp;&asymp;&nbsp;1.0 by selecting the protected column itself.</td></tr>
  <tr><td>Target leakage</td><td><code>bar1</code>, <code>bar2</code>, <code>bar1_yr</code>, <code>bar2_yr</code>, <code>bar</code>, <code>bar_passed</code>, <code>dnn_bar_pass_prediction</code></td><td>Encode the bar exam outcome. In a realistic deployment a model would not have access to the outcome it is trying to predict. Bar passage rates differ by race, so any bar-outcome column is a strong race proxy by construction.</td></tr>
  <tr><td>Redundant features</td><td><code>gpa</code></td><td>Numerically identical to <code>ugpa</code> in every row. Keeping both would double-count the same information.</td></tr>
  <tr><td>Metadata</td><td><code>ID</code></td><td>An arbitrary row identifier with no predictive or demographic content.</td></tr>
  <tr><td><strong>Base features</strong></td><td><strong>19 columns</strong></td><td>Everything remaining: academic performance, test scores, institutional characteristics, demographics, and enrolment status.</td></tr>
</table>

<h3>Step 2 &mdash; Protected Attribute Definition</h3>
<p>Five binary columns from the raw data are used as individual protected attributes, each analysed separately:</p>
<table>
  <tr><th>Attribute</th><th>Group</th><th>Type</th><th>Prevalence (=1)</th><th>n in dataset</th></tr>
  {''.join(f'<tr><td><code>{pa}</code></td><td>{group}</td><td>binary (0/1)</td><td>{100 * df[pa].sum() / len(df):.1f}%</td><td>{int(df[pa].sum()):,} / {len(df):,}</td></tr>' for pa, group in PROTECTED_BINARY.items())}
</table>
<p>This matches the non-processed version exactly, enabling direct comparison of proxy baselines between the two approaches.</p>

<h3>Step 3 &mdash; Missing Value Imputation</h3>
<p>Missing values are imputed to ensure GP trees can evaluate on every row:</p>
<ul style="margin-left:1.5rem">
  <li><strong>Numeric features</strong> &rarr; column <strong>median</strong> (robust to outliers and skewed distributions).</li>
  <li><strong>Categorical features</strong> &rarr; column <strong>mode</strong> (most frequent value).</li>
</ul>

<h3>Step 4 &mdash; Categorical Feature Handling</h3>
<p>Categorical features cannot be stored directly in a NumPy float64 array, so they are mapped
to integer codes purely for storage compatibility. <strong>No numeric relationship is implied</strong>:
the integer codes are never exposed to the Arithmetic grammar as arithmetic terminals.
The Extended (typed) grammar accesses categorical features exclusively via equality operators
(<code>CatEquals</code> / <code>CatNotEquals</code>), which translate back to the original
category names in all output expressions (e.g.&nbsp;<code>grad&nbsp;==&nbsp;Y</code>, not <code>grad&nbsp;==&nbsp;2</code>).</p>
<table>
  <tr><th>Feature</th><th>Original categories</th><th>Storage codes</th><th>Grammar exposure</th></tr>
  <tr><td><code>grad</code></td><td>O, X, Y</td><td>0, 1, 2</td><td>Extended only (equality)</td></tr>
  <tr><td><code>Dropout</code></td><td>NO, YES</td><td>0, 1</td><td>Extended only (equality)</td></tr>
  <tr><td><code>indxgrp</code></td><td>a&nbsp;under&nbsp;400 &hellip; g&nbsp;700+</td><td>0&ndash;6 (7 bands)</td><td>Extended only (equality)</td></tr>
  <tr><td><code>indxgrp2</code></td><td>a&nbsp;under&nbsp;400 &hellip; i&nbsp;820+</td><td>0&ndash;8 (9 bands)</td><td>Extended only (equality)</td></tr>
</table>

<h3>Step 5 &mdash; Feature Standardisation</h3>
<p>All 19 features are standardised to zero mean and unit variance (<code>StandardScaler</code>, fit on train only). Both scaled and unscaled versions are saved.</p>

<h3>Step 6 &mdash; Stratified Train/Test Split</h3>
<p>80/20 split stratified on <code>pass_bar</code> (random_state&nbsp;=&nbsp;42).</p>

<!-- Base features -->
<h2>Base Features ({len(BASE_FEATURE_COLS)})</h2>
<table>
  <tr><th>Type</th><th>Count</th><th>Features</th></tr>
  <tr><td>Numeric</td><td>{len(numeric_base)}</td><td><code>{'</code>, <code>'.join(numeric_base)}</code></td></tr>
  <tr><td>Categorical (equality-only in Extended grammar)</td><td>{len(categorical_base)}</td><td><code>{'</code>, <code>'.join(categorical_base)}</code></td></tr>
</table>

<!-- Feature descriptions -->
<h2>Feature Descriptions</h2>
<table>
  <tr><th>Feature</th><th>Type</th><th>Description</th></tr>
  <tr><td><code>decile1b</code></td><td>Numeric</td><td>Academic performance decile rank (1st year, alternate version). Peer-relative class standing on a 1&ndash;10 scale.</td></tr>
  <tr><td><code>decile3</code></td><td>Numeric</td><td>Academic performance decile rank (3rd year). Peer-relative class standing on a 1&ndash;10 scale.</td></tr>
  <tr><td><code>decile1</code></td><td>Numeric</td><td>Academic performance decile rank (1st year). Peer-relative class standing on a 1&ndash;10 scale.</td></tr>
  <tr><td><code>cluster</code></td><td>Numeric</td><td>Law school cluster grouping (1&ndash;6) based on institutional characteristics such as selectivity and resources.</td></tr>
  <tr><td><code>lsat</code></td><td>Numeric</td><td>Law School Admission Test score (range 11&ndash;48 in this dataset).</td></tr>
  <tr><td><code>ugpa</code></td><td>Numeric</td><td>Undergraduate grade point average (1.5&ndash;3.9 scale).</td></tr>
  <tr><td><code>zfygpa</code></td><td>Numeric</td><td>Standardised (z-score) first-year law school GPA.</td></tr>
  <tr><td><code>DOB_yr</code></td><td>Numeric</td><td>Year of birth (last two digits).</td></tr>
  <tr><td><code>zgpa</code></td><td>Numeric</td><td>Standardised (z-score) cumulative law school GPA.</td></tr>
  <tr><td><code>fulltime</code></td><td>Numeric</td><td>Full-time enrolment status (1&nbsp;=&nbsp;full-time, 2&nbsp;=&nbsp;not full-time).</td></tr>
  <tr><td><code>fam_inc</code></td><td>Numeric</td><td>Family income bracket on a 1&ndash;5 ordinal scale.</td></tr>
  <tr><td><code>age</code></td><td>Numeric</td><td>Age encoded as a negative offset from a reference year.</td></tr>
  <tr><td><code>parttime</code></td><td>Numeric</td><td>Part-time enrolment indicator (0&nbsp;=&nbsp;no, 1&nbsp;=&nbsp;yes).</td></tr>
  <tr><td><code>tier</code></td><td>Numeric</td><td>Law school tier ranking (1&ndash;6). Lower values = more prestigious.</td></tr>
  <tr><td><code>index6040</code></td><td>Numeric</td><td>Composite admission index: 60%&nbsp;LSAT + 40%&nbsp;UGPA, scaled.</td></tr>
  <tr><td><code>grad</code></td><td>Categorical</td><td>Graduation status: Y&nbsp;=&nbsp;graduated, X&nbsp;=&nbsp;transferred, O&nbsp;=&nbsp;did not complete.</td></tr>
  <tr><td><code>Dropout</code></td><td>Categorical</td><td>Whether the student dropped out: NO or YES.</td></tr>
  <tr><td><code>indxgrp</code></td><td>Categorical</td><td>Admission index group in 7 bands (based on <code>index6040</code>).</td></tr>
  <tr><td><code>indxgrp2</code></td><td>Categorical</td><td>Admission index group in 9 finer bands (based on <code>index6040</code>).</td></tr>
</table>

<!-- Metric definitions -->
<h2>Baseline Metric Definitions</h2>
<table>
  <tr><th>Column</th><th>Full Name</th><th>Description</th></tr>
  <tr><td><strong>AUC</strong></td><td>Area Under the ROC Curve</td><td>Measures how well a single feature can linearly separate the two groups (protected=1 vs protected=0) using logistic regression. 0.5&nbsp;=&nbsp;random; 1.0&nbsp;=&nbsp;perfect. Primary proxy strength metric.</td></tr>
  <tr><td><strong>MI</strong></td><td>Mutual Information</td><td>Measures any statistical dependency including non-linear relationships. Continuous features discretised into 20 bins. Higher&nbsp;=&nbsp;stronger dependency.</td></tr>
  <tr><td><strong>|corr|</strong></td><td>Absolute Point-Biserial Correlation</td><td>Absolute Pearson correlation between the feature and the binary protected attribute. 0&nbsp;=&nbsp;no linear relationship; 1&nbsp;=&nbsp;perfect.</td></tr>
  <tr><td><strong>p-value</strong></td><td>Correlation Significance</td><td>Statistical significance of the correlation. Values &lt;0.05 indicate the relationship is unlikely due to chance.</td></tr>
  <tr><td><strong>Strength</strong></td><td>Proxy Strength Category</td><td><span class="badge badge-strong">Strong</span> (AUC&nbsp;&ge;&nbsp;0.65), <span class="badge badge-moderate">Moderate</span> (AUC&nbsp;&ge;&nbsp;0.55), <span class="badge badge-weak">Weak</span> (AUC&nbsp;&lt;&nbsp;0.55).</td></tr>
</table>

<!-- Per-protected-attribute baseline tables -->
{''.join(pa_sections)}

<!-- Output artefacts -->
<h2>Output Artefacts</h2>
<table>
  <tr><th>File</th><th>Contents</th></tr>
  <tr><td><code>law_school_base_features.npz</code></td><td>X_train/X_test (scaled), X_train_raw/X_test_raw (unscaled), y_train/y_test, per-attribute train/test arrays ({', '.join(f'{pa}_train/test' for pa in PROTECTED_BINARY)}), feature_names</td></tr>
  <tr><td><code>law_school_clean.csv</code></td><td>Base features (imputed, encoded) + pass_bar + 5 protected attribute columns</td></tr>
  <tr><td><code>metadata.json</code></td><td>Column lists, label encodings, scaler params, split sizes, per-attribute baselines</td></tr>
  <tr><td><code>individual_proxy_baselines.csv</code></td><td>Per-feature AUC, MI, and correlation for all 5 protected attributes</td></tr>
</table>

<p style="margin-top:2rem;color:#999;font-size:.8rem;">Generated from MA-Proxies processed data.</p>
</body>
</html>"""

with open(OUT_DIR / "results_overview.html", "w") as f:
    f.write(html)
print(f"   -> {OUT_DIR / 'results_overview.html'}")

# ---------------------------------------------------------------------------
# 8. Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. Summary statistics")
print("=" * 60)

print(f"\n   --- Target by protected attributes ---")
for pa, group in PROTECTED_BINARY.items():
    cross = df.groupby(pa)[TARGET].agg(["count", "mean"])
    cross.columns = ["n", "pass_rate"]
    cross.index = [f"{pa}=0", f"{pa}=1"]
    print(f"\n   {pa} ({group}):")
    print(f"   {cross.to_string()}")

print(f"\n   --- Base feature stats (pre-scaling) ---")
feat_stats = pd.DataFrame(X, columns=BASE_FEATURE_COLS).describe().T
print(feat_stats[["mean", "std", "min", "max"]].round(3).to_string())

print("\n" + "=" * 60)
print("Done. Processed artefacts saved to:", OUT_DIR)
print("=" * 60)
