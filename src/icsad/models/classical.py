"""
Classical baselines: k-means (distance to the nearest centroid) and
IsolationForest (negated decision_function). Ported from the dissertation's
baselines.py. Point-level only, same split, scaler and tau/w rule as the
deep models, so all methods are compared on equal footing.

Changes from baselines.py:
  - The random seed is the run's seed rather than a fixed 42, so the
    baselines get the same multi-seed treatment as the deep models. The
    dissertation ran them at seed 42 only.
  - sklearn's KMeans reduces across OpenMP threads, so two fits of the same
    array can differ at ~1e-16 (the dissertation's k-means numbers are not
    bit-reproducible, by its own code either). deterministic=True fits on
    one thread, which makes them exactly repeatable at some cost in speed.
"""

import logging
from contextlib import nullcontext

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits

from ..detection import DEFAULT_PERCENTILES, DEFAULT_W_VALUES, search_tau_w

log = logging.getLogger(__name__)

K_RANGE = (2, 3, 4, 5, 6, 8, 10, 15)


def kmeans_score(model, X):
    """Anomaly score = distance to the nearest cluster centroid."""
    return model.transform(X).min(axis=1)


def kmeans_per_feature(model, X):
    """Per-feature squared distance to the nearest centroid. Row sums are
    the squared kmeans_score."""
    return (X - model.cluster_centers_[model.predict(X)]) ** 2


def isoforest_score(model, X):
    """Anomaly score = negated decision_function (sklearn: high = inlier)."""
    return -model.decision_function(X)


def choose_k(X_train, X_benign_val, X_attack_val, y_attack_val, metric_fn, seed,
             k_range=K_RANGE, percentiles=DEFAULT_PERCENTILES, w_values=DEFAULT_W_VALUES,
             deterministic=False):
    """Fit k-means per candidate k, tune tau/w on attack_val, keep the k with
    the best tuned score. Note the chosen model therefore depends on the
    tuning metric; retuning a cached run against another metric keeps this k,
    as the dissertation's tune_metrics.py did.

    Returns (model, k, table) where table has one row per candidate k."""
    rows = []
    best = None  # (model, k, score)
    for k in k_range:
        with threadpool_limits(limits=1, user_api='openmp') if deterministic else nullcontext():
            model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_train)
        tau, w, score, p = search_tau_w(kmeans_score(model, X_benign_val),
                                        kmeans_score(model, X_attack_val),
                                        y_attack_val, metric_fn, percentiles, w_values)
        rows.append({'k': k, 'tau_percentile': p, 'tau': tau, 'w': w, 'tuned_score': score})
        if best is None or score > best[2]:
            best = (model, k, score)

    table = pd.DataFrame(rows)
    log.info('k-means candidates:\n%s', table.to_string(index=False))
    model, k, score = best
    log.info('chosen k=%d (tuned score %.4f on attack_val)', k, score)
    return model, k, table


def fit_isoforest(X_train, seed, n_estimators=100):
    return IsolationForest(random_state=seed, n_estimators=n_estimators).fit(X_train)
