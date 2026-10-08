import numpy as np
import pytest

from icsad import aggregation as agg


def test_single_sensor_spike_survives_max_but_not_mean():
    """The hypothesis in miniature: one sensor of 40 jumps; the mean barely
    moves, the standardised max flags it."""
    rng = np.random.default_rng(0)
    benign = rng.normal(1.0, 0.1, size=(5_000, 40)) ** 2
    sample = rng.normal(1.0, 0.1, size=(1, 40)) ** 2
    sample[0, 7] += 3.0
    norm = agg.fit_normaliser(benign)
    benign_max = agg.aggregate(benign, 'max', norm)
    assert agg.aggregate(sample, 'max', norm)[0] > np.quantile(benign_max, 0.9999)
    assert sample.mean() < np.quantile(benign.mean(axis=1), 1.0)


def test_topk_is_mean_of_k_largest():
    per = np.array([[1.0, 5.0, 3.0, 2.0]])
    norm = (np.zeros(4), np.ones(4))
    assert agg.aggregate(per, 'top2', norm)[0] == pytest.approx(4.0)
    assert agg.aggregate(per, 'top1', norm)[0] == agg.aggregate(per, 'max', norm)[0]


def test_constant_feature_does_not_divide_by_zero():
    benign = np.ones((10, 3))
    mu, sd = agg.fit_normaliser(benign)
    assert np.all(sd > 0)


@pytest.mark.parametrize('bad', ['median', 'top0x', 'topk'])
def test_validate_rejects_unknown(bad):
    with pytest.raises(ValueError):
        agg.validate(bad)


def test_topk_larger_than_features_is_an_error():
    with pytest.raises(ValueError):
        agg.aggregate(np.ones((2, 3)), 'top4', (np.zeros(3), np.ones(3)))
