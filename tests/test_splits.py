import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

from icsad.data.splits import (LABEL_COL, _assign_blocks, _block_anchors, point_split,
                               predictive_split)

HISTORY = 30


@pytest.fixture(scope='module')
def pred(synthetic):
    return predictive_split(synthetic.normal_df, synthetic.attack_df, synthetic.feature_cols,
                            history=HISTORY, seed=42)


def test_windows_follow_reference_convention(pred):
    """x = values[i-history:i], y = values[i+1]; index i itself skipped."""
    part = pred.train
    idx = np.array([0, 1, 777, len(part) - 1])
    X, Y = part.batch(idx)
    for k, i in zip(range(len(idx)), part.anchors[idx]):
        np.testing.assert_array_equal(X[k], part.values[i - HISTORY:i])
        np.testing.assert_array_equal(Y[k], part.values[i + 1])


def test_block_anchors_keep_window_and_target_inside_block():
    a = _block_anchors(100, 200, HISTORY)
    assert a[0] - HISTORY == 100 and a[-1] + 1 == 199
    assert len(a) == 100 - HISTORY - 1
    assert len(_block_anchors(0, HISTORY + 1, HISTORY)) == 0


def test_scaler_matches_fit_on_materialised_windows(synthetic, pred):
    """data_split.py fit the scaler on the materialised training windows;
    the lazy split fits on the rows those windows read. Same min/max."""
    vals = synthetic.normal_df[synthetic.feature_cols].values.astype(np.float64)
    blocks, train_idx = _assign_blocks(len(vals), 14_400, 0.7, np.random.RandomState(42))
    windows = []
    for bi, (s, e) in enumerate(blocks):
        if bi in train_idx:
            n_valid = (e - s) - HISTORY - 1
            windows.append(sliding_window_view(vals[s:e], (HISTORY, vals.shape[1]))[:n_valid, 0])
    ref = MinMaxScaler().fit(np.concatenate(windows).reshape(-1, vals.shape[1]))
    np.testing.assert_array_equal(pred.scaler.data_min_, ref.data_min_)
    np.testing.assert_array_equal(pred.scaler.data_max_, ref.data_max_)


def test_boundary_does_not_cut_an_attack(synthetic, pred):
    is_attack = synthetic.attack_df[LABEL_COL].values == 'Attack'
    b = pred.attack_val_boundary_idx
    assert b != int(len(is_attack) * 0.3), 'synthetic data puts an attack on the naive boundary'
    assert not (is_attack[b - 1] and is_attack[b])


def test_labels_are_the_target_rows_labels(synthetic, pred):
    is_attack = (synthetic.attack_df[LABEL_COL].values == 'Attack').astype(int)
    np.testing.assert_array_equal(pred.test_labels, is_attack[pred.test_raw_idx])
    np.testing.assert_array_equal(pred.attack_val_labels, is_attack[pred.attack_val_raw_idx])


def test_point_split_sizes_and_raw_index(synthetic):
    s = point_split(synthetic.normal_df, synthetic.attack_df, synthetic.feature_cols, seed=42)
    assert len(s.train) == int(len(synthetic.normal_df) * 0.7)
    assert len(s.train) + len(s.benign_val) == len(synthetic.normal_df)
    assert len(s.attack_val) + len(s.test) == len(synthetic.attack_df)
    assert s.test_raw_idx[0] == s.attack_val_boundary_idx


def test_split_seed_changes_the_split(synthetic):
    a = predictive_split(synthetic.normal_df, synthetic.attack_df, synthetic.feature_cols, HISTORY, seed=1)
    b = predictive_split(synthetic.normal_df, synthetic.attack_df, synthetic.feature_cols, HISTORY, seed=2)
    assert not np.array_equal(a.train.anchors, b.train.anchors)
