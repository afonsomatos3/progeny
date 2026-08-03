# PROGENY

**PRO**xy discovery via **GEN**etic programming.

Grammar-guided genetic programming for discovering **multi-attribute proxies**:
combinations of individually non-sensitive features that, taken together,
reconstruct a protected attribute (gender, race, age) even after it has been
removed from a dataset.

PROGENY evolves **interpretable symbolic expressions**, characterises each one
by its precision, recall, and coverage, and then measures, through post-hoc
attribution analysis, whether trained models actually rely on the discovered
proxies.

This repository contains the code for the MSc thesis *Finding Multi-Attribute
Proxies using Genetic Programming* (Instituto Superior Técnico, 2026).

## How it works

The pipeline processes a dataset in five stages:

1. **Single-feature baseline audit**: records how much of the protected
   attribute each feature reconstructs on its own.
2. **Stage 1, Arithmetic grammar**: evolves continuous numeric proxies.
3. **Stage 2, Extended grammar**: evolves boolean rules (categorical tests,
   conjunctions, interval predicates) unreachable by arithmetic alone.
4. **Acceptance criteria**: a candidate is reported as a proxy only if it meets
   a quality bar (precision >= 80% for logical rules, AUC >= 0.60 for continuous
   ones) and a recall floor (>= 5%).
5. **Attribution analysis**: injects each qualifying proxy into four classifiers
   (Logistic Regression, Random Forest, XGBoost, MLP) and measures its share of
   the model's attribution (SHAP for the first three, Integrated Gradients for
   the MLP).

The fitness function is configurable (precision / recall / coverage weights),
so the search can be steered toward narrow high-confidence proxies or broad
high-coverage ones.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11. The GP backend is
[GeneticEngine](https://github.com/alcides/GeneticEngine), which also provides
the `geml` helper package that the code imports. If `import geml` fails after
`pip install GeneticEngine`, install GeneticEngine from source and place the
checkout at `GeneticEngine/GeneticEngine/` (the code adds that path
automatically as a fallback).

## Datasets

The processed feature matrices (`.npz`) and baseline audits (`.csv`) are
included for four datasets, so they run with no setup:

| Dataset | Domain | Ships with data? |
|---------|--------|------------------|
| Law School (LSAC) | legal education | yes (`processed/`, `non_processed/`) |
| Adult Income | labour market (1994) | yes (`adult_test_dataset/`) |
| German Credit | credit scoring | yes (`German-Credit-Analysis-master/processed/`) |
| Folktables (ACS 2018) | labour market (2018) | yes (`folktables-main-Dataset/processed/`) |
| COMPAS | criminal justice | no, raw CSV required (see below) |
| OULAD | higher education | no, raw CSV required (see below) |

COMPAS and OULAD load from raw data at runtime. To run them, download the
sources and place them where the loaders expect:

- COMPAS: `compas-scores-raw.csv` into `compas_scores_raw_dataset/`
  (<https://www.kaggle.com/datasets/danofer/compass>).
- OULAD: `studentInfo.csv` into `OULAD-student_dataset/`
  (<https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad>).

Each dataset has a setup script that regenerates its processed artefacts from
the raw source (`law_school_setup.py`, `adult_test_dataset/gp_proxy_discovery_adult.py`,
`German-Credit-Analysis-master/setup_german_credit.py`,
`folktables-main-Dataset/setup_folktables.py`).

## Usage

```bash
# Discovery on a single dataset (writes to pipeline_results/{dataset}/)
python pipeline.py --dataset law_school
python pipeline.py --dataset adult   --classifier random_forest
python pipeline.py --dataset german_credit --penalty 0.05 --max-depth 4

# Full sweep over all datasets x fitness configurations, then attribution
python run_pipeline_sweep.py --attribution

# RQ1: leave-one-out grammar study, then its figures
python ablation_runner.py
python analysis_ablation.py pipeline_results/ablation --out figs/ablation

# RQ3: attribution of discovered proxies (auto-detects newest run)
python mlp_attribution_from_run.py

# Controlled validation on synthetic data (grammar-gap and fitness-steering)
python synthetic_experiments.py

# Three-stage comparison report (classifier vs arithmetic vs extended grammar)
python generate_3stage_report.py --classifier logistic_regression
```

## Repository layout

| Path | Purpose |
|------|---------|
| `pipeline.py` | End-to-end orchestration for one dataset: loading, GP stages, acceptance filtering, reporting. |
| `gp_proxy_discovery.py` | Grammar node definitions, fitness helpers, and the GP runner. |
| `run_pipeline_sweep.py` | Runs the pipeline across all datasets and fitness configurations, optionally launching attribution. |
| `ablation_runner.py` | Leave-one-out grammar study (six variants x three weight configs). |
| `analysis_ablation.py`, `analysis_evolution.py`, `analysis_stats.py` | Figures and tables from the study outputs. |
| `mlp_attribution_from_run.py`, `split_attribution_report.py` | Attribution analysis (SHAP + Integrated Gradients). |
| `simplify_pipeline_results.py` | Algebraic simplification of discovered expressions. |
| `generate_3stage_report.py` | Self-contained HTML comparison report. |
| `synthetic_experiments.py`, `synthetic_toy_datasets.py` | Controlled-validation experiments. |
| `*_setup.py` / `gp_proxy_discovery_{compas,oulad,adult}.py` | Per-dataset preprocessing, loaders, and grammars. |
| `test_fold.py` | Unit checks for the expression simplifier. |

Run outputs (`pipeline_results/`, `pipeline_runs/`, HTML reports) are
regenerated by the tool and are git-ignored.

## License

Released under the [MIT License](LICENSE).
