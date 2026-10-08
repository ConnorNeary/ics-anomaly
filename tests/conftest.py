"""
Synthetic SWaT-shaped data so the pipeline can be tested without the iTrust
files: same column layout (Timestamp, sensor/actuator columns,
'Normal/Attack'), 1 Hz timestamps, normal data long enough for three
14,400-row blocks, and labelled attacks that shift one tag each. One attack
straddles the naive 30% attack-val boundary to exercise the nudge.
"""

import numpy as np
import pandas as pd
import pytest

from icsad.config import config_from_dict
from icsad.data import LoadedData

# six tags: enough for the pre-registered top5 aggregation
FEATURES = ['FIT101', 'LIT101', 'P101', 'AIT201', 'AIT202', 'FIT201']
OFFSETS = {'FIT101': 1.0, 'LIT101': 250.0, 'P101': 1.0, 'AIT201': 20.0, 'AIT202': 0.8, 'FIT201': 0.6}
SYN_ATTACKS = [(800, 1100), (2400, 2800), (3500, 3700), (5200, 5600),
               (7000, 7500), (9100, 9400), (10800, 11300)]
N_NORMAL, N_ATTACK = 43_200, 12_000


def _signals(t, rng):
    return pd.DataFrame({
        'FIT101': 2.5 + 0.4 * np.sin(2 * np.pi * t / 700) + rng.normal(0, 0.03, len(t)),
        'LIT101': 600 + 120 * np.sin(2 * np.pi * t / 3_000) + rng.normal(0, 2.0, len(t)),
        'P101': np.where(np.sin(2 * np.pi * t / 1_100) > 0, 2.0, 1.0),
        'AIT201': 250 + 6 * np.sin(2 * np.pi * t / 5_000 + 0.5) + rng.normal(0, 0.2, len(t)),
        'AIT202': 8.5 + 0.3 * np.sin(2 * np.pi * t / 4_200 + 1.3) + rng.normal(0, 0.01, len(t)),
        'FIT201': 2.2 + 0.2 * np.sin(2 * np.pi * t / 900 + 0.7) + rng.normal(0, 0.02, len(t)),
    })


def make_synthetic(seed=0):
    rng = np.random.default_rng(seed)
    normal = _signals(np.arange(N_NORMAL), rng)
    normal.insert(0, 'Timestamp', pd.date_range('2015-12-22 16:00:00', periods=N_NORMAL, freq='s'))
    normal['Normal/Attack'] = 'Normal'

    attack = _signals(np.arange(N_NORMAL, N_NORMAL + N_ATTACK), rng)
    is_attack = np.zeros(N_ATTACK, dtype=bool)
    for i, (s, e) in enumerate(SYN_ATTACKS):
        is_attack[s:e] = True
        col = FEATURES[i % len(FEATURES)]
        attack.iloc[s:e, attack.columns.get_loc(col)] += OFFSETS[col]
    attack.insert(0, 'Timestamp', pd.date_range('2015-12-28 10:00:00', periods=N_ATTACK, freq='s'))
    attack['Normal/Attack'] = np.where(is_attack, 'Attack', 'Normal')
    return normal, attack


@pytest.fixture(scope='session')
def synthetic():
    normal, attack = make_synthetic()
    return LoadedData('synthetic', normal, attack, list(FEATURES))


@pytest.fixture
def make_cfg(tmp_path):
    """Build an ExperimentConfig writing into tmp_path; short training."""
    def _make(models, experiment='test', epochs=3, patience=2, **sections):
        raw = {
            'experiment': experiment,
            'train': {'epochs': epochs, 'patience': patience},
            'runtime': {'device': 'cpu', 'artifact_root': str(tmp_path / 'runs'),
                        'store': str(tmp_path / 'results.sqlite')},
            'models': models,
        }
        for name, values in sections.items():
            raw.setdefault(name, {}).update(values)
        return config_from_dict(raw)
    return _make
