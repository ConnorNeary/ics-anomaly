from pathlib import Path

import pytest

from icsad.config import config_from_dict, load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / 'configs').glob('*.yaml'))


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path):
    load_config(path)


def _minimal(**overrides):
    raw = {'experiment': 'x', 'models': [{'name': 'ae', 'params': {'enc_dim': 8}}]}
    raw.update(overrides)
    return raw


def test_config_saved_with_bom_loads(tmp_path):
    path = tmp_path / 'bom.yaml'
    path.write_bytes(b'\xef\xbb\xbfexperiment: x\nmodels:\n  - {name: ae}\n')
    assert load_config(path).experiment == 'x'


def test_seed_range_expands():
    cfg = config_from_dict(_minimal(models=[{'name': 'ae', 'seeds': {'start': 10, 'count': 3}}]))
    assert cfg.models[0].seeds == [10, 11, 12]


@pytest.mark.parametrize('raw', [
    _minimal(trian={'epochs': 1}),                                     # top-level typo
    _minimal(train={'epoch': 1}),                                      # section typo
    _minimal(models=[{'name': 'cnn', 'params': {}}]),                  # windowed model without history
    _minimal(models=[{'name': 'ae', 'params': {'units': 3}}]),         # bad model kwarg
    _minimal(models=[{'name': 'transformer'}]),                        # unregistered model
    _minimal(detection={'tuning_metric': 'accuracy'}),
    _minimal(detection={'aggregation': 'median'}),
])
def test_bad_configs_fail_at_load(raw):
    with pytest.raises((ValueError, KeyError, TypeError)):
        config_from_dict(raw)
