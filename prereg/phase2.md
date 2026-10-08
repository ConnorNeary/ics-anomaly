# Phase 2 pre-registration: score aggregation and normal-data calibration

**Status: final.** The git commit that adds this file is the
pre-registration timestamp; no Phase 2 run was started before it. The
**DECISION** items were settled by Connor on 2026-10-08 (proposals
accepted as drafted).

Author: Connor Neary. 2026-10-08.

## 1. What is already known (disclosure)

- The aggregation hypothesis (H1) came from the dissertation's per-feature
  analysis of the **2015 test partition**: three test attacks on LIT301 and
  LIT401 were missed by all five methods, although the targeted sensor
  ranked among the top contributors to the error. That partition therefore
  cannot be used to decide H1.
- The **2015 attack-val partition** (first 30% of the attack workbook,
  14 attacks) was used in the dissertation only to tune tau/w for
  mean-scored models. Its per-feature errors were never inspected. Its
  mean-scored tuning scores were seen.
- The Phase 1 reproduction run (configs/dissertation_multiseed.yaml) will
  again show mean-scored results tuned on attack-val. It shows nothing about
  max or top-k scoring, and no other Phase 2 quantity will be looked at
  before this document is committed.
- Jul 2019 data cannot be scored by models trained on 2015: several tags read
  on different scales (docs/data_inventory.md). It is used only as a
  secondary, within-2019 check.

## 2. Hypotheses

**H1 (aggregation).** Scoring each timestep by the mean of its 3 largest
standardised per-feature errors ("top3") detects more attacks than the
dissertation's mean over features, at the same false-alarm setting, without
a material increase in false alarms.

**H2 (calibration without attack labels).** A threshold set only from
normal data to a target exceedance rate keeps close to that rate on normal
operation recorded days later.

H2 matters independently of H1: it is the mechanism by which a new site
with no labelled attacks can set a false-alarm budget (brief v2, section 4.2).

## 3. Data and models

- Training and calibration: SWaT Dec 2015 normal workbook, split as in
  Phase 1 (src/icsad/data/splits.py), unchanged.
- H1 evaluation: 2015 **attack-val** partition, 14 attacks.
- H2 evaluation: normal-labelled rows of the 2015 attack workbook (both
  partitions), excluding the recovery window after each attack (below).
- Models: AE, CNN and LSTM (dissertation small configs, history 30) and
  k-means. IsolationForest is excluded: it has no per-feature error.
- Seeds: 10 per model, 42-51, in `configs/phase2_seeds.yaml`. Split seed
  follows the run seed (dissertation behaviour).
- **DECISION 6 (k-means k):** k is fixed in advance, because the pipeline's
  default chooses k by tuning on attack-val, which would use the evaluation
  labels. Rule: k = the value the dissertation's baselines.py selected at
  seed 42 (read from its saved `model_kmeans.pkl`, or by rerunning its
  `choose_k` on the 2015 data). That selection used attack-val labels once,
  for mean scoring, before Phase 2 existed; it is disclosed here and not
  revisited. k-means is added to `configs/phase2_seeds.yaml` in a separate
  commit that records the value, before any k-means Phase 2 run.

Seed count rationale: H1 is a paired comparison (same trained model, two
scoring rules), so its variance is the variance of the within-run
difference, which should be much smaller than the between-seed variance of
either score. No pilot is run, because any pilot would expose attack-val
results under top3. If the result is inconclusive at 10 seeds it is
reported as unresolved, and any follow-up gets its own pre-registration.

## 4. Threshold rule (no attack labels used)

For every run and every aggregation:

- Per-feature standardisation is fitted on that run's benign-val errors
  (src/icsad/aggregation.py).
- tau = quantile q of the run's benign-val aggregated scores.
- An alarm fires once the score exceeds tau for w consecutive timesteps
  (src/icsad/detection.py, unchanged).

**DECISION 1 (primary operating point):** q = 0.9995, w = 10.
Secondary: the full curve over q in {0.99, 0.995, 0.999, 0.9995, 0.9999}
at w = 10, reported for every model. This curve is the trade-off curve of
brief section 4.3 and is reported whatever H1's outcome.

Attack-val labels are never used to choose q, w, k, or anything else.

## 5. Outcomes

### H1

- **Primary outcome:** attacks detected out of 14 on attack-val (range
  recall, existence reward), per run, under top3 minus under mean. One
  paired difference per (model, seed): 40 differences.
- **Guard outcome:** false-alarm events on attack-val's normal stretches,
  per run, top3 versus mean.
- **Analysis:** mean paired difference, per model and pooled with model as
  a stratum (equal weight per model). 95% interval by bootstrap: resample
  seeds within model and attacks within the partition, 10,000 resamples,
  fixed bootstrap seed 0.

**DECISION 2 (meaningful improvement):** H1 is supported if the pooled
mean paired difference is at least **1 attack (1/14 recall)** and its 95%
interval excludes 0.

**DECISION 3 (false-alarm guard):** and the pooled mean false-alarm count
under top3 is no more than **25% higher** than under mean (or +2 events,
whichever is larger, to allow for small counts).

Outcomes are reported as:
- supported: both conditions met;
- not supported: interval excludes the 1-attack improvement, or the guard fails;
- unresolved: anything else.

Per-model results are reported in all cases. Per-model "wins" are not
claimed unless that model alone meets both conditions, and are described as
exploratory.

### H2

- **Outcome:** realised exceedance rate (fraction of normal rows with
  aggregated score > tau, before the w rule) on the 2015 attack workbook's
  normal rows, versus nominal 1 - q, at each q in section 4, mean aggregation.
- **DECISION 4 (tolerance):** calibration "holds" at a given q if the
  realised rate is within a **factor of 2** of nominal (e.g. 0.025% to 0.1%
  at q = 0.9995) for at least 8 of 10 seeds of each model.
- **DECISION 5 (recovery window):** rows within **10 minutes** after each
  labelled attack's end are excluded, since the plant takes time to return to
  normal and those rows are labelled normal. Results with no exclusion are
  also reported.

### Secondary, descriptive only (no claims)

- The 2015 test partition under mean, max, top2, top3 and top5, including
  whether the three motivating attacks are recovered. Labelled as the data
  that generated the hypothesis.
- max, top2 and top5 on attack-val (exploratory).
- Jul 2019: k-means and AE trained on the first 70% of the 2019 normal
  period (04:35-06:50 UTC), calibrated on the rest, scored on the six
  logged attacks, all aggregations. Six attacks cannot support a claim.

## 6. Fixed choices

These will not be revisited after seeing results: k = 3 for the primary
comparison; q, w and the margins above; the model list and seed list; the
recovery window; the bootstrap procedure. Any change is a deviation,
reported as such with its reason.

The analysis script (`scripts/phase2_analysis.py`) is written and committed
before `configs/phase2_seeds.yaml` is run, and is tested on synthetic data
first.

## 7. Reporting

Every number states its metric, dataset and partition, and seed count.
Distributions, not best runs. A null or unresolved result is reported with
the same prominence as a positive one.
