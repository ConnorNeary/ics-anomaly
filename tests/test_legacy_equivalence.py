"""
The port's acceptance test: run the dissertation's own code and this
pipeline on the same (synthetic) data with the same seed, and require
identical error arrays and identical test metrics.

Needs the dissertation repo; looks next to this project by default, or set
ICSAD_LEGACY_REPO. Skipped if not found. Everything runs on CPU: GPU
results are not bit-reproducible across code paths.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from icsad.pipeline import run_one
from icsad.store import ResultsStore

LEGACY = Path(os.environ.get('ICSAD_LEGACY_REPO',
                             Path(__file__).resolve().parents[2] / '40328065-msc-dissertation'))
EPOCHS, PATIENCE = 3, 2

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (LEGACY / 'train_models.py').exists(),
                       reason=f'dissertation repo not found at {LEGACY}'),
]

LEGACY_METRICS = ('point_precision', 'point_recall', 'point_f1', 'range_precision',
                  'range_recall', 'range_f1', 'numenta', 'na_early')


@pytest.fixture(scope='module')
def legacy():
    sys.path.insert(0, str(LEGACY))
    try:
        mods = {name: importlib.import_module(name)
                for name in ('data_split', 'models', 'train_models', 'baselines')}
    finally:
        sys.path.remove(str(LEGACY))
    tm = mods['train_models']
    tm.DEVICE = torch.device('cpu')
    # train_with_early_stopping binds epochs/batch_size/patience as defaults
    tm.train_with_early_stopping.__defaults__ = (EPOCHS, 512, PATIENCE)
    return mods


def _new_run(make_cfg, synthetic, tmp_path, model):
    cfg = make_cfg([model], epochs=EPOCHS, patience=PATIENCE)
    store = ResultsStore(tmp_path / 'new.sqlite')
    run_id = run_one(cfg, cfg.models[0], 42, synthetic, store)
    d = Path(store.get_run(run_id)['artifact_dir'])
    results = store.results().iloc[0]
    return d, results


def _assert_same_errors(new_dir, legacy_dir, legacy_name):
    for part in ('benign_val', 'attack_val', 'test'):
        np.testing.assert_array_equal(np.load(new_dir / f'scores_{part}.npy'),
                                      np.load(legacy_dir / f'errors_{legacy_name}_{part}.npy'),
                                      err_msg=f'{legacy_name} {part} errors differ')


def _assert_same_metrics(new_results, legacy_results):
    for k in LEGACY_METRICS:
        assert new_results[k] == pytest.approx(legacy_results[k], abs=1e-12), k


def test_ae_matches_dissertation_code(legacy, synthetic, make_cfg, tmp_path, monkeypatch):
    legacy_dir = tmp_path / 'legacy'
    legacy_dir.mkdir()
    monkeypatch.chdir(legacy_dir)
    legacy_res = legacy['train_models'].run_ae(synthetic.normal_df, synthetic.attack_df,
                                               synthetic.feature_cols, seed=42)
    new_dir, new_res = _new_run(make_cfg, synthetic, tmp_path, {'name': 'ae', 'params': {'enc_dim': 16}})
    _assert_same_errors(new_dir, legacy_dir, 'ae')
    _assert_same_metrics(new_res, legacy_res)


@pytest.mark.parametrize('name,new_params', [
    ('CNN', {'history': 30}),
    ('LSTM', {'history': 30, 'latent_dim': 32, 'num_layers': 1}),
])
def test_predictors_match_dissertation_code(legacy, synthetic, make_cfg, tmp_path, monkeypatch,
                                            name, new_params):
    lm = legacy['models']
    cls = {'CNN': lm.CNNPredictor, 'LSTM': lm.LSTMPredictor}[name]
    legacy_dir = tmp_path / 'legacy'
    legacy_dir.mkdir()
    monkeypatch.chdir(legacy_dir)
    legacy_res = legacy['train_models'].run_predictive(
        name, cls, lm.SMALL_CONFIGS[name], synthetic.normal_df, synthetic.attack_df,
        synthetic.feature_cols, seed=42)
    new_dir, new_res = _new_run(make_cfg, synthetic, tmp_path, {'name': name.lower(), 'params': new_params})
    _assert_same_errors(new_dir, legacy_dir, name.lower())
    _assert_same_metrics(new_res, legacy_res)


def test_baselines_match_dissertation_code(legacy, synthetic, make_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds, bl = legacy['data_split'], legacy['baselines']
    split = ds.four_way_split(synthetic.normal_df, synthetic.attack_df, synthetic.feature_cols, seed=42)

    # k-means is compared to a tolerance: sklearn's multi-threaded fit moves
    # centroids by ~1e-16 between runs of the same code on the same array
    # (see models/classical.py), so bit-equality is not achievable.
    km, k, tau, w = bl.choose_k(split.X_train, split.X_benign_val, split.X_attack_val, split.y_attack_val)
    new_dir, new_res = _new_run(make_cfg, synthetic, tmp_path / 'km', {'name': 'kmeans'})
    np.testing.assert_allclose(np.load(new_dir / 'scores_test.npy'), bl.kmeans_score(km, split.X_test),
                               rtol=1e-12, atol=0)
    run_extra = ResultsStore(tmp_path / 'km' / 'new.sqlite').runs()[0]['extra_json']
    assert f'"k": {k}' in run_extra
    assert int(new_res['w']) == w
    y_pred = legacy['train_models'].apply_threshold_rule(bl.kmeans_score(km, split.X_test), tau, w)
    _assert_same_metrics(new_res, bl.evaluate_on_test('k-means', split.y_test, y_pred))

    iso = bl.IsolationForest(random_state=42, n_estimators=100).fit(split.X_train)
    new_dir, _ = _new_run(make_cfg, synthetic, tmp_path / 'iso', {'name': 'iforest'})
    np.testing.assert_array_equal(np.load(new_dir / 'scores_test.npy'), bl.isoforest_score(iso, split.X_test))
