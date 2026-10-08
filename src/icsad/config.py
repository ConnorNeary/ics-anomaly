"""
Experiment configuration. One YAML file declares the data, split, training,
detection and runtime settings plus the models and seeds to run; nothing
experiment-relevant is hardcoded elsewhere. Unknown keys are errors, so a
typo fails at load time instead of silently falling back to a default.

seeds may be a list ([42, 43, 44]) or a range ({start: 42, count: 36}).

split_seed (per model, optional) fixes the train/benign-val assignment
across seeds. Left unset, one seed drives both the split and the weight
initialisation, as in the dissertation; setting it isolates initialisation
variance from split variance.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from . import aggregation
from .detection import DEFAULT_PERCENTILES, DEFAULT_W_VALUES
from .evaluation import TUNING_METRICS
from .models import get_spec, validate_params


@dataclass
class DataConfig:
    dataset: str = 'swat2015'
    dir: str = 'data/swat2015'
    features: str = 'ks_filtered'     # 'ks_filtered' or 'all'


@dataclass
class SplitConfig:
    train_frac: float = 0.7
    attack_val_frac: float = 0.3
    block_size: int = 14_400


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 512
    patience: int = 5
    val_batch_size: int | None = None   # None = whole benign-val set per pass
    eval_batch_size: int = 4096


@dataclass
class DetectionConfig:
    tuning_metric: str = 'range_f1'
    aggregation: str = 'mean'
    percentiles: list = field(default_factory=lambda: list(DEFAULT_PERCENTILES))
    w_values: list = field(default_factory=lambda: list(DEFAULT_W_VALUES))


@dataclass
class RuntimeConfig:
    device: str = 'auto'
    deterministic: bool = False
    artifact_root: str = 'runs'
    store: str = 'results/results.sqlite'
    save_per_feature: bool = True
    save_model: bool = True


@dataclass
class ModelConfig:
    name: str
    params: dict = field(default_factory=dict)
    seeds: list = field(default_factory=lambda: [42])
    split_seed: int | None = None


@dataclass
class ExperimentConfig:
    experiment: str
    models: list
    description: str = ''
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self):
        return asdict(self)


def _build(cls, raw, where):
    raw = dict(raw or {})
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f'{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}')
    return cls(**raw)


def _seeds(value, where):
    if isinstance(value, int):
        return [value]
    if isinstance(value, list) and all(isinstance(s, int) for s in value):
        return list(value)
    if isinstance(value, dict) and set(value) == {'start', 'count'}:
        return list(range(value['start'], value['start'] + value['count']))
    raise ValueError(f'{where}: seeds must be an int, a list of ints, or {{start, count}}')


def config_from_dict(raw):
    raw = dict(raw)
    sections = {'data': DataConfig, 'split': SplitConfig, 'train': TrainConfig,
                'detection': DetectionConfig, 'runtime': RuntimeConfig}
    unknown = set(raw) - set(sections) - {'experiment', 'description', 'models'}
    if unknown:
        raise ValueError(f'unknown top-level key(s) {sorted(unknown)}')
    if not raw.get('experiment'):
        raise ValueError('config needs an `experiment` name')

    models = []
    for i, m in enumerate(raw.get('models') or []):
        where = f'models[{i}]'
        m = dict(m)
        m['seeds'] = _seeds(m.get('seeds', 42), where)
        mc = _build(ModelConfig, m, where)
        mc.params = dict(mc.params or {})
        validate_params(get_spec(mc.name), mc.params)
        models.append(mc)
    if not models:
        raise ValueError('config lists no models')

    cfg = ExperimentConfig(
        experiment=raw['experiment'], description=raw.get('description', ''), models=models,
        **{name: _build(cls, raw.get(name), name) for name, cls in sections.items()})

    if cfg.detection.tuning_metric not in TUNING_METRICS:
        raise ValueError(f'detection.tuning_metric must be one of {sorted(TUNING_METRICS)}')
    aggregation.validate(cfg.detection.aggregation)
    return cfg


def load_config(path):
    # utf-8-sig: Windows editors and PowerShell often save YAML with a BOM
    return config_from_dict(yaml.safe_load(Path(path).read_text(encoding='utf-8-sig')))
