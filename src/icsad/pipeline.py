"""
Run an experiment: for every (model, seed) in the config, seed -> split ->
train or fit -> cache scores -> tune tau/w on attack-val -> evaluate on test
-> record in the results store.

Each run gets a deterministic run_id from its config, so re-running an
experiment skips runs already finished and redoes failed or interrupted
ones. Every run writes to its own artifact directory:

  scores_<partition>.npy    (n,) anomaly score, the dissertation's scoring
  perfeat_<partition>.npy   (n, n_features) per-feature error, if available
  labels_<partition>.npy    0/1 attack label per sample (attack_val, test)
  rawidx_<partition>.npy    attack-table row each sample scores (attack_val, test)
  features.json, scaler.pkl, model.pt | model.pkl
"""

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch

from .data.splits import point_split, predictive_split
from .evaluation import TUNING_METRICS, evaluate_run
from .models import get_spec
from .models.classical import (K_RANGE, choose_k, fit_isoforest, isoforest_score,
                               kmeans_per_feature, kmeans_score)
from .seeding import resolve_device, seed_everything
from .training import compute_errors, train_with_early_stopping

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    try:
        return subprocess.run(['git', *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(device, deterministic):
    import numpy, pandas, sklearn  # noqa: E401 -- versions only
    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain')
    return {
        'git_commit': commit,
        'git_dirty': None if status is None else int(bool(status)),
        'host': platform.node(),
        'device': str(device),
        'deterministic': int(deterministic),
        'versions': {'python': sys.version.split()[0], 'numpy': numpy.__version__,
                     'pandas': pandas.__version__, 'sklearn': sklearn.__version__,
                     'torch': torch.__version__},
    }


def run_identity(cfg, model_cfg, seed, split_seed, data):
    """Everything that determines a run's result, as a dict, and its hash."""
    spec = get_spec(model_cfg.name)
    ident = {
        'dataset': data.name, 'features': data.feature_hash, 'split': asdict(cfg.split),
        'model': model_cfg.name, 'params': model_cfg.params, 'seed': seed,
        'split_seed': split_seed, 'deterministic': cfg.runtime.deterministic,
    }
    if spec.kind == 'deep':
        ident['train'] = asdict(cfg.train)
    if model_cfg.name == 'kmeans':  # k is chosen by tuned score on attack_val
        ident['detection'] = {k: v for k, v in asdict(cfg.detection).items() if k != 'aggregation'}
    key = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    return ident, key


def _split_for(spec, params, cfg, data, split_seed):
    common = dict(train_frac=cfg.split.train_frac, attack_val_frac=cfg.split.attack_val_frac,
                  seed=split_seed)
    if spec.framing == 'point':
        return point_split(data.normal_df, data.attack_df, data.feature_cols, **common)
    return predictive_split(data.normal_df, data.attack_df, data.feature_cols,
                            history=params['history'], block_size=cfg.split.block_size, **common)


def _save_split(art, split, feature_cols):
    np.save(art / 'labels_attack_val.npy', split.attack_val_labels.astype(np.int8))
    np.save(art / 'labels_test.npy', split.test_labels.astype(np.int8))
    np.save(art / 'rawidx_attack_val.npy', split.attack_val_raw_idx)
    np.save(art / 'rawidx_test.npy', split.test_raw_idx)
    (art / 'features.json').write_text(json.dumps(feature_cols), encoding='utf-8')
    joblib.dump(split.scaler, art / 'scaler.pkl')


def _save_scores(art, partition, scores, per_feature):
    np.save(art / f'scores_{partition}.npy', scores)
    if per_feature is not None:
        np.save(art / f'perfeat_{partition}.npy', per_feature.astype(np.float32))


def _run_deep(cfg, model_cfg, spec, split, data, device, art):
    model = spec.build(len(data.feature_cols), dict(model_cfg.params))
    n_params = sum(p.numel() for p in model.parameters())
    log.info('%s: %d parameters, device %s', model_cfg.name, n_params, device)
    result = train_with_early_stopping(
        model, split.train, split.benign_val, epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size, patience=cfg.train.patience,
        val_batch_size=cfg.train.val_batch_size, device=device)
    for part in ('benign_val', 'attack_val', 'test'):
        scores, per = compute_errors(model, getattr(split, part), device=device,
                                     batch_size=cfg.train.eval_batch_size,
                                     per_feature=cfg.runtime.save_per_feature)
        _save_scores(art, part, scores, per)
    if cfg.runtime.save_model:
        torch.save(model.state_dict(), art / 'model.pt')
    return {'train_time_s': result.train_time_s, 'n_params': n_params,
            'epochs_run': result.epochs_run, 'best_val_loss': result.best_val_loss,
            'extra': {'loss_history': result.history}}


def _run_classical(cfg, model_cfg, seed, split, art):
    X = {p: getattr(split, p).X for p in ('train', 'benign_val', 'attack_val', 'test')}
    t0 = time.time()
    if model_cfg.name == 'kmeans':
        model, k, table = choose_k(
            X['train'], X['benign_val'], X['attack_val'], split.attack_val_labels,
            metric_fn=TUNING_METRICS[cfg.detection.tuning_metric], seed=seed,
            k_range=tuple(model_cfg.params.get('k_range', K_RANGE)),
            percentiles=cfg.detection.percentiles, w_values=cfg.detection.w_values,
            deterministic=cfg.runtime.deterministic)
        train_time = time.time() - t0
        for part in ('benign_val', 'attack_val', 'test'):
            per = kmeans_per_feature(model, X[part]) if cfg.runtime.save_per_feature else None
            _save_scores(art, part, kmeans_score(model, X[part]), per)
        extra = {'k': k, 'k_table': table.to_dict(orient='records')}
    else:
        model = fit_isoforest(X['train'], seed, model_cfg.params.get('n_estimators', 100))
        train_time = time.time() - t0
        for part in ('benign_val', 'attack_val', 'test'):
            # no per-feature breakdown: the score is a path length over random trees
            _save_scores(art, part, isoforest_score(model, X[part]), None)
        extra = {}
    if cfg.runtime.save_model:
        joblib.dump(model, art / 'model.pkl')
    return {'train_time_s': train_time, 'extra': extra}


def run_one(cfg, model_cfg, seed, data, store):
    """Train/fit one (model, seed), evaluate it, record it. Returns run_id."""
    spec = get_spec(model_cfg.name)
    split_seed = seed if model_cfg.split_seed is None else model_cfg.split_seed
    ident, key = run_identity(cfg, model_cfg, seed, split_seed, data)
    run_id = f'{cfg.experiment}__{model_cfg.name}__s{seed}__{key[:8]}'

    prior = store.get_run(run_id)
    if prior is not None and prior['status'] == 'done':
        log.info('skip %s (already done)', run_id)
        return run_id

    art = Path(cfg.runtime.artifact_root) / run_id
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    (art / 'run.json').write_text(json.dumps(ident, indent=2, sort_keys=True), encoding='utf-8')
    device = resolve_device(cfg.runtime.device)
    store.start_run(run_id=run_id, experiment=cfg.experiment, model=model_cfg.name, seed=seed,
                    split_seed=split_seed, run_key=key, config=ident, artifact_dir=art.as_posix(),
                    provenance=provenance(device, cfg.runtime.deterministic))
    log.info('=== %s ===', run_id)
    try:
        # seed immediately before this run's split and model construction
        seed_everything(seed, cfg.runtime.deterministic)
        split = _split_for(spec, model_cfg.params, cfg, data, split_seed)
        _save_split(art, split, data.feature_cols)
        if spec.kind == 'deep':
            info = _run_deep(cfg, model_cfg, spec, split, data, device, art)
        else:
            info = _run_classical(cfg, model_cfg, seed, split, art)
        store.finish_run(run_id, **info)
        evaluate_run(store, run_id, cfg.detection.aggregation, cfg.detection.tuning_metric,
                     cfg.detection.percentiles, cfg.detection.w_values)
    except BaseException:
        store.fail_run(run_id, traceback.format_exc())
        raise
    return run_id


def run_experiment(cfg, data, store, only_models=None):
    run_ids = []
    for model_cfg in cfg.models:
        if only_models and model_cfg.name not in only_models:
            continue
        for seed in model_cfg.seeds:
            run_ids.append(run_one(cfg, model_cfg, seed, data, store))
    return run_ids
