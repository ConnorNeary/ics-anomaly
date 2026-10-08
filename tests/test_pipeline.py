"""End-to-end runs on synthetic data: determinism, resumability, retuning."""

import numpy as np
import pytest

from icsad.evaluation import evaluate_run
from icsad.pipeline import run_one
from icsad.store import ResultsStore

pytestmark = pytest.mark.slow

CNN = {'name': 'cnn', 'params': {'history': 30}}


def _scores(store, run_id, part='test'):
    return np.load(f"{store.get_run(run_id)['artifact_dir']}/scores_{part}.npy")


def test_same_seed_same_result_different_seed_different(make_cfg, synthetic, tmp_path):
    cfg = make_cfg([CNN])
    m = cfg.models[0]
    stores = [ResultsStore(tmp_path / f's{i}.sqlite') for i in range(3)]
    ids = []
    for i, (store, seed) in enumerate(zip(stores, [42, 42, 43])):
        cfg.runtime.artifact_root = str(tmp_path / f'runs{i}')
        ids.append(run_one(cfg, m, seed, synthetic, store))
    a, b, c = (_scores(s, r) for s, r in zip(stores, ids))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_finished_runs_are_skipped(make_cfg, synthetic, tmp_path):
    cfg = make_cfg([{'name': 'ae'}])
    store = ResultsStore(tmp_path / 'r.sqlite')
    run_id = run_one(cfg, cfg.models[0], 42, synthetic, store)
    finished_at = store.get_run(run_id)['finished_at']
    assert run_one(cfg, cfg.models[0], 42, synthetic, store) == run_id
    assert store.get_run(run_id)['finished_at'] == finished_at


def test_fixed_split_seed_isolates_initialisation(make_cfg, synthetic, tmp_path):
    cfg = make_cfg([{**CNN, 'split_seed': 7}])
    store = ResultsStore(tmp_path / 'r.sqlite')
    r1 = run_one(cfg, cfg.models[0], 42, synthetic, store)
    r2 = run_one(cfg, cfg.models[0], 43, synthetic, store)
    d1, d2 = (store.get_run(r)['artifact_dir'] for r in (r1, r2))
    np.testing.assert_array_equal(np.load(f'{d1}/rawidx_test.npy'), np.load(f'{d2}/rawidx_test.npy'))
    assert not np.array_equal(_scores(store, r1), _scores(store, r2))


def test_retune_adds_an_evaluation_without_touching_the_original(make_cfg, synthetic, tmp_path):
    cfg = make_cfg([{'name': 'kmeans', 'params': {'k_range': [2, 3]}}])
    store = ResultsStore(tmp_path / 'r.sqlite')
    run_id = run_one(cfg, cfg.models[0], 42, synthetic, store)
    before = store.results(aggregation='mean', tuning_metric='range_f1')
    evaluate_run(store, run_id, 'max', 'point_f1')
    after = store.results(aggregation='mean', tuning_metric='range_f1')
    assert before.equals(after)
    assert len(store.results(aggregation='max', tuning_metric='point_f1')) == 1


def test_deterministic_kmeans_is_exactly_repeatable(synthetic):
    from icsad.data.splits import point_split
    from icsad.evaluation import TUNING_METRICS
    from icsad.models.classical import choose_k

    s = point_split(synthetic.normal_df, synthetic.attack_df, synthetic.feature_cols, seed=42)
    fit = lambda: choose_k(s.train.X, s.benign_val.X, s.attack_val.X, s.attack_val_labels,
                           TUNING_METRICS['range_f1'], seed=42, k_range=(2, 3),
                           deterministic=True)[0].cluster_centers_
    np.testing.assert_array_equal(fit(), fit())


def test_summary_reports_distribution_not_best(make_cfg, synthetic, tmp_path):
    cfg = make_cfg([{'name': 'iforest', 'seeds': [1, 2, 3]}])
    store = ResultsStore(tmp_path / 'r.sqlite')
    for seed in cfg.models[0].seeds:
        run_one(cfg, cfg.models[0], seed, synthetic, store)
    table = store.summary(['range_f1'])
    assert list(table['range_f1'].columns) == ['count', 'mean', 'std', 'min', 'max']
    assert table['range_f1']['count'].iloc[0] == 3
