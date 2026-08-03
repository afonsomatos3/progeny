# PROGENY

**PRO**xy discovery via **GEN**etic programming.

Grammar-guided genetic programming for discovering **multi-attribute proxies**:
combinations of individually non-sensitive features that, taken together,
reconstruct a protected attribute (gender, race, age) even after it has been
removed from a dataset.

PROGENY evolves **interpretable symbolic expressions** and characterises each
one by its precision, recall, and coverage, so an auditor gets the proxy rule
itself, not just a verdict that one exists.

This repository contains the core discovery tool for the MSc thesis *Finding
Multi-Attribute Proxies using Genetic Programming* (Instituto Superior Técnico,
2026). The additional experiment drivers from the thesis (fitness sweep,
leave-one-out grammar study, cross-architecture attribution, and the figure
scripts) are not included here; this repository is the tool for a standard
discovery run.

## How it works

A run processes one dataset in four stages:

1. **Single-feature baseline audit**: records how much of the protected
   attribute each feature reconstructs on its own.
2. **Stage 1, Arithmetic grammar**: evolves continuous numeric proxies.
3. **Stage 2, Extended grammar**: evolves boolean rules (categorical tests,
   conjunctions, interval predicates) unreachable by arithmetic alone.
4. **Acceptance criteria**: a candidate is reported as a proxy only if it meets
   a quality bar (precision >= 80% for logical rules, AUC >= 0.60 for continuous
   ones) and a recall floor (>= 5%).

The fitness function is configurable (precision / recall / coverage weights),
so the search can be steered toward narrow high-confidence proxies or broad
high-coverage ones. Each run produces a self-contained HTML report.

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
python pipeline.py --dataset adult          --classifier random_forest
python pipeline.py --dataset german_credit  --penalty 0.05 --max-depth 4
python pipeline.py --dataset folktables

# Three-stage comparison report (classifier vs arithmetic vs extended grammar)
python generate_3stage_report.py --classifier logistic_regression
```

`pipeline.py --dataset` accepts `law_school`, `adult`, `german_credit`,
`folktables` (ready to run), and `compas`, `oulad` (after their raw CSV is in
place; see above). Use `--help` for the full option list (time budget,
population size, penalty, tree depth, fitness weights, splits).

## Repository layout

| Path | Purpose |
|------|---------|
| `pipeline.py` | End-to-end orchestration for one dataset: loading, GP stages, acceptance filtering, reporting. |
| `gp_proxy_discovery.py` | Grammar node definitions, fitness helpers, and the GP runner. |
| `simplify_pipeline_results.py` | Algebraic simplification of discovered expressions. |
| `generate_3stage_report.py` | Self-contained HTML comparison report. |
| `law_school_setup.py`, `German-Credit-Analysis-master/setup_german_credit.py`, `folktables-main-Dataset/setup_folktables.py`, `adult_test_dataset/gp_proxy_discovery_adult.py` | Per-dataset preprocessing (regenerate the processed inputs from raw). |
| `compas_scores_raw_dataset/gp_proxy_discovery_compas.py`, `OULAD-student_dataset/gp_proxy_discovery_oulad.py` | Loaders and grammars for the two datasets that read raw data at runtime. |

Run outputs (`pipeline_results/`, HTML reports) are regenerated by the tool and
are git-ignored.

## License

Released under the [MIT License](LICENSE).
