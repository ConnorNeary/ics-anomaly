"""
Turn a per-feature error matrix (n_samples, n_features) into one anomaly
score per sample.

'mean' is the dissertation's scoring. It is read from the scores each run
wrote at error-computation time (computed in torch, identical to the
original .mean(dim=1)), never recomputed here, so it reproduces exactly.

The alternatives test the hypothesis that averaging across ~40 features
dilutes an attack on a single sensor (dissertation per-feature analysis:
the targeted sensor's error rises but the mean stays under threshold).
Each feature's error is first standardised against its own benign-val mean
and standard deviation; without that, max-over-features is dominated by
whichever sensor is noisiest.

  'max'      max over standardised features
  'top<k>'   mean of the k largest standardised features, e.g. 'top3'

Not to be used in any reported comparison until prereg/phase2.md is
committed (brief v2, section 4): the three attacks that motivated this were
found by inspecting the 2015 test partition, so the decision has to be made
on data that played no part in forming the hypothesis.
"""

import re

import numpy as np

_TOPK = re.compile(r'top(\d+)$')


def validate(method):
    if method in ('mean', 'max') or _TOPK.match(method):
        return method
    raise ValueError(f"aggregation must be 'mean', 'max' or 'top<k>', got {method!r}")


def fit_normaliser(benign_per_feature, eps=1e-12):
    """Per-feature mean and std of benign-val errors. eps guards constant
    features (e.g. an actuator that never changes state in normal data)."""
    mu = benign_per_feature.mean(axis=0, dtype=np.float64)
    sd = np.maximum(benign_per_feature.std(axis=0, dtype=np.float64), eps)
    return mu, sd


def aggregate(per_feature, method, normaliser):
    """Score each row of per_feature by `method` ('max' or 'top<k>')."""
    mu, sd = normaliser
    z = (per_feature - mu) / sd
    if method == 'max':
        return z.max(axis=1)
    m = _TOPK.match(method)
    if not m:
        raise ValueError(f'cannot aggregate with {method!r} here')
    k = int(m.group(1))
    if not 1 <= k <= z.shape[1]:
        raise ValueError(f'{method}: k must be between 1 and {z.shape[1]}')
    return np.partition(z, -k, axis=1)[:, -k:].mean(axis=1)
