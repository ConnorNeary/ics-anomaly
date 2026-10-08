"""
Shared detection rule: alarm fires at t once the anomaly score has exceeded
threshold tau for w consecutive timesteps. tau = a percentile of BENIGN-
validation scores (never attack data). Per project brief (replaces the old
`mean + 3*std` rule): tau over percentiles [0.95, 0.99995], w over [1, 100],
tuned on the 30% attack-validation slice, reported on the 70% test slice.

Model-agnostic: takes raw score arrays only. Used identically by baselines.py
and train_models.py so every model is thresholded the same way.
"""

import numpy as np
import pandas as pd

DEFAULT_PERCENTILES = (0.95, 0.97, 0.99, 0.995, 0.999, 0.9999, 0.99995)
DEFAULT_W_VALUES = (1, 5, 10, 20, 30, 50, 100)


def percentile_threshold(benign_scores, percentile):
    """tau = the given percentile (in [0, 1]) of benign-validation scores."""
    return float(np.quantile(benign_scores, percentile))


def apply_threshold_rule(scores, tau, w):
    """alarm[t] = 1 iff scores[t-w+1:t+1] all exceed tau -- w consecutive
    timesteps required, not just one (unless w=1)."""
    scores = np.asarray(scores)
    exceeds = scores > tau
    if w <= 1:
        return exceeds.astype(int)

    alarm = np.zeros(len(scores), dtype=int)
    run = 0
    for i, e in enumerate(exceeds):
        run = run + 1 if e else 0
        if run >= w:
            alarm[i] = 1
    return alarm


def search_tau_w(benign_scores, val_scores, y_val, metric_fn,
                  percentiles=DEFAULT_PERCENTILES, w_values=DEFAULT_W_VALUES,
                  verbose=False):
    """
    Grid-search tau (percentile of benign_scores) x w, scored by
    metric_fn(y_true, y_pred) on (val_scores, y_val) -- attack-validation
    only, never test. Returns (tau, w, best_metric_value, percentile_used).

    7x7=49-point grid: a coarse but fast subset of the full [0.95, 0.99995]
    x [1, 100] search space. Pass finer percentiles/w_values for closer coverage.
    """
    rows = []
    best = None
    for p in percentiles:
        tau = percentile_threshold(benign_scores, p)
        for w in w_values:
            y_pred = apply_threshold_rule(val_scores, tau, w)
            score = metric_fn(y_val, y_pred)
            rows.append({'percentile': p, 'tau': tau, 'w': w, 'metric': score})
            if best is None or score > best[2]:
                best = (tau, w, score, p)

    if verbose:
        print(pd.DataFrame(rows).sort_values('metric', ascending=False).head(10)
              .to_string(index=False))

    return best  # (tau, w, score, percentile)
