"""
Tune tau/w on attack-val and evaluate on test, from a run's cached score
arrays. Used both straight after training and later to retune a run against
a different metric or aggregation -- retuning takes seconds, retraining
takes minutes to hours, which is what makes many-seed work tractable.
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from . import aggregation as agg
from . import metrics
from .detection import DEFAULT_PERCENTILES, DEFAULT_W_VALUES, apply_threshold_rule, search_tau_w

log = logging.getLogger(__name__)

PARTITIONS = ('benign_val', 'attack_val', 'test')

TUNING_METRICS = {
    'point_f1': lambda yt, yp: f1_score(yt, yp, zero_division=0),
    'range_f1': metrics.range_f1,
    'range_fbeta_3': metrics.range_fbeta_precision_weighted,   # paper's range-Fbeta(3:1)
    'range_fbeta_1_3': metrics.range_fbeta_recall_weighted,    # paper's range-Fbeta(1:3)
    'numenta': metrics.numenta_score,
    'na_early': metrics.na_early,
}


def evaluate_predictions(y_true, y_pred):
    """Every metric reported for a test partition, plus the event counts
    behind the range metrics (false alarms are what operators see)."""
    return {
        'point_precision': precision_score(y_true, y_pred, zero_division=0),
        'point_recall': recall_score(y_true, y_pred, zero_division=0),
        'point_f1': f1_score(y_true, y_pred, zero_division=0),
        'range_precision': metrics.range_precision(y_true, y_pred),
        'range_recall': metrics.range_recall(y_true, y_pred),
        'range_f1': metrics.range_f1(y_true, y_pred),
        'range_fbeta_3': metrics.range_fbeta_precision_weighted(y_true, y_pred),
        'range_fbeta_1_3': metrics.range_fbeta_recall_weighted(y_true, y_pred),
        'numenta': metrics.numenta_score(y_true, y_pred),
        'na_early': metrics.na_early(y_true, y_pred),
        'false_alarms': metrics.false_positive_segments(y_true, y_pred),
        'attacks_detected': metrics.true_positive_segments(y_true, y_pred),
        'attacks_total': len(metrics.to_segments(y_true)[0]),
    }


def partition_scores(artifact_dir, aggregation):
    """Anomaly scores per partition for one run under `aggregation`."""
    d = Path(artifact_dir)
    if aggregation == 'mean':
        return {p: np.load(d / f'scores_{p}.npy') for p in PARTITIONS}
    if not (d / 'perfeat_benign_val.npy').exists():
        raise ValueError(f'{d.name} has no per-feature errors, so it can only be scored by mean')
    per = {p: np.load(d / f'perfeat_{p}.npy') for p in PARTITIONS}
    normaliser = agg.fit_normaliser(per['benign_val'])
    return {p: agg.aggregate(per[p], aggregation, normaliser) for p in PARTITIONS}


def evaluate_run(store, run_id, aggregation='mean', tuning_metric='range_f1',
                 percentiles=DEFAULT_PERCENTILES, w_values=DEFAULT_W_VALUES):
    """Tune on attack-val, score test, record the evaluation. Returns
    (tau, tau_percentile, w, tuning_score, test_metrics)."""
    run = store.get_run(run_id)
    d = Path(run['artifact_dir'])
    scores = partition_scores(d, agg.validate(aggregation))
    y_attack_val = np.load(d / 'labels_attack_val.npy')
    y_test = np.load(d / 'labels_test.npy')

    tau, w, tuning_score, p = search_tau_w(scores['benign_val'], scores['attack_val'], y_attack_val,
                                           TUNING_METRICS[tuning_metric], percentiles, w_values)
    results = evaluate_predictions(y_test, apply_threshold_rule(scores['test'], tau, w))
    store.add_evaluation(run_id, aggregation, tuning_metric, tau, p, w, tuning_score, results)
    log.info('%s [%s, tuned for %s]: tau_percentile=%s w=%d -> test range_f1=%.4f point_f1=%.4f '
             'false_alarms=%d detected=%d/%d', run_id, aggregation, tuning_metric, p, w,
             results['range_f1'], results['point_f1'], results['false_alarms'],
             results['attacks_detected'], results['attacks_total'])
    return tau, p, w, tuning_score, results
