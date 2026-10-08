# ics-anomaly

Research pipeline for anomaly detection on industrial control system data
(SWaT), extended from an MSc dissertation that reproduced Fung et al.,
*Perspectives from a Comprehensive Evaluation of Reconstruction-based
Anomaly Detection in Industrial Control Systems* (ESORICS 2022).

This is Phase 1 of `ics_startup_brief_v2.md`: the dissertation pipeline,
ported so it can be extended without breaking. The dissertation repo stays
untouched as the reference implementation.

## Status

| | |
|---|---|
| Datasets | SWaT December 2015 only. Later releases wait on the Phase 0 inventory. |
| Models | Dense AE, CNN and LSTM predictors (small and reference configs), k-means, IsolationForest |
| Equivalence with dissertation code | **Verified on synthetic data.** AE, CNN and LSTM produce bit-identical error arrays and identical metrics to the dissertation's own functions; k-means agrees to 1e-12 (see below). `tests/test_legacy_equivalence.py` |
| Reproduction on real 2015 data | **Not yet run.** Needs the 2015 workbooks on the desktop. |

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
python -m pytest                  # ~40 s; -m "not slow" for the 1 s subset
```

For a CUDA build of torch on the desktop, install torch first following
pytorch.org, then `pip install -e ".[dev]"`.

## Running

Run commands from the project root; config paths are relative to it.

```
# 1. Build the 2015 tables once from the iTrust workbooks
#    (SWaT_Dataset_Normal_v1.xlsx, SWaT_Dataset_Attack_v0.xlsx)
python -m icsad prepare-swat2015 --raw-dir <folder with the .xlsx files>

# 2. Run an experiment. Finished (model, seed) runs are skipped, so an
#    interrupted run can simply be restarted.
python -m icsad run configs/dissertation_multiseed.yaml

# 3. Retune cached runs against another metric or score aggregation
#    (seconds, no retraining). Adds evaluations, never overwrites.
python -m icsad retune --experiment dissertation-multiseed --tuning-metric point_f1

# 4. Distributions across seeds
python -m icsad summary --experiment dissertation-multiseed

# Phase 1 acceptance check against the dissertation's output CSVs
python scripts/compare_dissertation.py --legacy-dir <folder with multiseed_full_metrics.csv>
```

## Layout

```
configs/                 one YAML per experiment: data, split, training, detection, models, seeds
src/icsad/
  data/swat2015.py       2015 workbook -> parquet, corrected attack ranges with target tags
  data/features.py       K-S feature selection (reference procedure, leak included)
  data/splits.py         four-way split; windowed partitions built lazily
  models/                registry, deep models, classical baselines
  training.py            training loop, per-feature error computation
  detection.py           tau/w rule and search (unchanged from dissertation)
  metrics.py             range/point/Numenta metrics (unchanged from dissertation)
  aggregation.py         mean / max / top-k scoring over per-feature errors
  evaluation.py          tune on attack-val, evaluate on test, from cached arrays
  store.py               SQLite results store
  pipeline.py            one run end to end; experiment loop
scripts/compare_dissertation.py
tests/
```

Each run records its config, seed, split seed, git commit and dirty flag,
host, device, library versions, training time, selected tau/w and every
test metric in `results/results.sqlite`. Its arrays go to `runs/<run_id>/`:
per-partition scores, the per-feature error matrix, labels, and the attack
row each sample scores.

## Differences from the dissertation code

All deliberate and covered by tests.

- **Windowed splits are lazy.** Windows are cut per batch from one scaled
  array instead of materialised up front (the old approach needed ~1.7 GB at
  history 30 and could not do history 200). Values, scaler and training
  order are identical.
- **Per-feature errors are cached.** The dissertation kept only the mean
  across features and had to rerun inference for its per-feature analysis.
- **Split seed can be separated from the initialisation seed.** By default
  one seed drives both, as before. Setting `split_seed` on a model fixes the
  train/val assignment so seed-to-seed variance reflects initialisation alone.
- **Baselines are seeded per run.** The dissertation ran k-means and
  IsolationForest at seed 42 only, so its "k-means is competitive" result is
  a single run.
- **k-means is not bit-reproducible** by either codebase: sklearn's
  multi-threaded fit moves centroids by ~1e-16 between identical runs. With
  `runtime.deterministic: true` it fits single-threaded and is exact.
- **Seeding was already per run** in the committed dissertation code
  (`run_ae` / `run_predictive` seed before splitting). The brief's
  "seed at import" defect predates that commit.
- Attack 14's targets are recorded as P203 and P205; the old comment's
  "P203/205" split into a non-existent tag "205".
- Not ported: the windowed-reconstruction branch of `four_way_split` and
  `eval_stride`, which no result used.

## Rules this code is built around

From the brief's standing methodological rules: never report a single run
or a best-of-N; state metric, dataset and seed count with every number;
pre-register comparisons before running them. `summary` reports n, mean,
std, min and max per model and nothing else. The alternative aggregations
in `aggregation.py` are implemented but are not to be used for reported
comparisons until `prereg/phase2.md` is committed.
