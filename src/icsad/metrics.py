"""
Range-based anomaly-detection metrics.

Fung et al. (ESORICS 2022), Sec. 5.2: range-precision/recall via Tatbul et
al.'s (NeurIPS 2018) existence-reward framework, range-F1/Fbeta, and the
Numenta anomaly score (Lavin & Ahmad 2015) incl. the paper's NA-early variant.

All functions take same-length binary sequences as (y_true, y_pred), matching
sklearn's convention. "Existence reward" = a true attack segment counts as
detected if ANY predicted-positive point falls inside it, regardless of
coverage -- Fung et al. deliberately drop Tatbul's overlap-size/positional
terms (Sec. 5.2: "we only consider existence rewards").

Ported from the reference implementation (pwwl/ics-anomaly-detection,
metrics.py), not derived from the PDF's equations: the paper's prose
description of beta inverts the numeric beta actually used (see range_fbeta),
so the repo is ground truth. Reference code uses (y_pred, y_true) argument
order; flipped here to (y_true, y_pred) -- arithmetic unchanged.
"""

import numpy as np


def to_segments(sequence):
    """Convert a 0/1 sequence to (starts, ends) index lists; both inclusive."""
    seq = np.asarray(sequence).astype(bool)
    n = len(seq)
    starts, ends = [], []
    for i in range(n):
        if seq[i] and (i == 0 or not seq[i - 1]):
            starts.append(i)
        if seq[i] and (i == n - 1 or not seq[i + 1]):
            ends.append(i)
    return starts, ends


def true_positive_segments(y_true, y_pred):
    """Count true segments (from y_true) with >=1 overlapping predicted point."""
    y_pred = np.asarray(y_pred).astype(bool)
    starts, ends = to_segments(y_true)
    return sum(np.any(y_pred[s:e + 1]) for s, e in zip(starts, ends))


def false_positive_segments(y_true, y_pred):
    """Count predicted segments (from y_pred) with zero overlap with any true point."""
    y_true = np.asarray(y_true).astype(bool)
    starts, ends = to_segments(y_pred)
    return sum(not np.any(y_true[s:e + 1]) for s, e in zip(starts, ends))


def range_recall(y_true, y_pred):
    """Fraction of true attack segments with at least one overlapping detection."""
    starts, _ = to_segments(y_true)
    if not starts:
        return 0.0
    return true_positive_segments(y_true, y_pred) / len(starts)


def range_precision(y_true, y_pred):
    """(# true attacks detected) / (# true attacks detected + # false-alarm
    segments). Numerator reuses range_recall's count, not a count of correct
    predicted segments -- paper's own asymmetric definition (Sec. 5.2 /
    reference segment_ppv), not a bug."""
    tp = true_positive_segments(y_true, y_pred)
    fp = false_positive_segments(y_true, y_pred)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def _range_fbeta_raw(y_true, y_pred, beta):
    """Standard F-beta over range-precision/recall:
        (1 + beta^2) * P * R / (beta^2 * P + R)
    beta > 1 weights precision more (matches reference fb_score exactly).
    See range_fbeta for the paper's inverted prose convention."""
    prec = range_precision(y_true, y_pred)
    rec = range_recall(y_true, y_pred)
    if prec == 0 and rec == 0:
        return 0.0
    return (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)


def range_f1(y_true, y_pred):
    return _range_fbeta_raw(y_true, y_pred, beta=1.0)


def range_fbeta(y_true, y_pred, beta):
    """
    Range-F-beta using Fung et al.'s PROSE convention (Sec. 5.2):
        beta > 1  ->  precision weighted 'beta' times more than recall
        beta < 1  ->  recall weighted '1/beta' times more than precision

    This is the INVERSE of the beta fed to the raw F-beta formula above.
    Verified against the reference: fb13_score (paper's "beta=1/3", recall
    3x more important) = fb_score(beta=3); fb31_score (paper's "beta=3") =
    fb_score(beta=1/3). We invert internally so callers can pass the paper's
    own beta values directly.
    """
    return _range_fbeta_raw(y_true, y_pred, beta=1.0 / beta)


def range_fbeta_recall_weighted(y_true, y_pred):
    """Paper's range-Fbeta(1:3): recall weighted 3x more than precision."""
    return range_fbeta(y_true, y_pred, beta=1 / 3)


def range_fbeta_precision_weighted(y_true, y_pred):
    """Paper's range-Fbeta(3:1): precision weighted 3x more than recall."""
    return range_fbeta(y_true, y_pred, beta=3)


def numenta_score(y_true, y_pred, kappa=5, position_bias=0.5,
                   fp_weight=-1, fn_weight=-1, tp_weight=1):
    """
    Numenta Anomaly Benchmark score (Lavin & Ahmad 2015), adapted in Fung et
    al. Sec. 5.2 per Singh & Olinsky (2017). Ported from the reference
    metrics.py: numenta().

    Quirk (preserved for fidelity): for true segment [s, e], predicted
    region is y_pred[s:e], EXCLUSIVE of e -- unlike the inclusive
    y_pred[s:e+1] used above. A length-1 true segment therefore always
    scores as a miss. No practical effect here since SWaT attacks are
    multi-timestep.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_pred) != len(y_true):
        raise ValueError(f'length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}')

    starts, ends = to_segments(y_true)

    # Isolate false-positive-only points (pred=1, true=0) as their own segments.
    ydiff = y_pred - y_true
    ydiff[ydiff == -1] = 0
    dstarts, _ = to_segments(ydiff)

    score = len(dstarts) * fp_weight

    for s, e in zip(starts, ends):
        pred_region = y_pred[s:e]  # exclusive of e -- see docstring note above
        detect_idx = np.where(pred_region)[0]
        if len(detect_idx) == 0:
            sig_value = fn_weight
        else:
            first_detect = detect_idx.min()
            f_value = first_detect - int(position_bias * len(pred_region))
            sig_value = (tp_weight - fp_weight) / (1 + np.exp(kappa * f_value)) - fp_weight
        score += sig_value * tp_weight

    denom = len(starts) + len(dstarts)
    if denom == 0:
        return 0.0
    return score / denom


def na_early(y_true, y_pred, position_bias=0.2):
    """
    Fung et al.'s early-detection Numenta variant: sigmoid re-centred
    earlier, kappa=10 (stricter than the kappa=5 default), fp cost halved.

    position_bias defaults to 0.2, matching the reference numenta_early()
    exactly. Paper's prose (Sec. 5.2) says "25% point" instead -- repo and
    prose disagree; repo wins since it produced the published numbers.
    Pass position_bias=0.25 for the paper-prose value as a comparison.
    """
    return numenta_score(y_true, y_pred, kappa=10, position_bias=position_bias, fp_weight=-0.5)
