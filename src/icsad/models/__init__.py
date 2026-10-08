"""
Model registry. Each entry says how a model is framed -- point-level
reconstruction, or next-step prediction from a window of `history` rows --
and how to build it from the `params` block of an experiment config.

New architectures are added here and nowhere else.
"""

from dataclasses import dataclass
from typing import Callable

from .deep import Autoencoder, CNNPredictor, CNNPredictorReference, LSTMPredictor

CLASSICAL_PARAMS = {
    'kmeans': {'k_range'},
    'iforest': {'n_estimators'},
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                        # 'deep' or 'classical'
    framing: str                     # 'point' or 'predictive'
    build: Callable | None = None    # (n_feat, params) -> nn.Module, deep models only


def _without_history(params):
    return {k: v for k, v in params.items() if k != 'history'}


REGISTRY = {spec.name: spec for spec in [
    ModelSpec('ae', 'deep', 'point',
              lambda n_feat, p: Autoencoder(n_feat, **p)),
    ModelSpec('cnn', 'deep', 'predictive',
              lambda n_feat, p: CNNPredictor(n_feat, window_size=p['history'], **_without_history(p))),
    ModelSpec('cnn_reference', 'deep', 'predictive',
              lambda n_feat, p: CNNPredictorReference(n_feat, window_size=p['history'],
                                                      **_without_history(p))),
    ModelSpec('lstm', 'deep', 'predictive',
              lambda n_feat, p: LSTMPredictor(n_feat, **_without_history(p))),
    ModelSpec('kmeans', 'classical', 'point'),
    ModelSpec('iforest', 'classical', 'point'),
]}


def get_spec(name):
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f'unknown model {name!r}; registered: {sorted(REGISTRY)}') from None


def validate_params(spec, params):
    """Fail at config-load time rather than hours into a run."""
    if spec.framing == 'predictive' and 'history' not in params:
        raise ValueError(f'model {spec.name!r} is windowed and needs params.history')
    if spec.kind == 'classical':
        unknown = set(params) - CLASSICAL_PARAMS[spec.name]
        if unknown:
            raise ValueError(f'model {spec.name!r}: unknown params {sorted(unknown)}')
    else:
        spec.build(1 if spec.framing == 'point' else 4, dict(params))  # raises on bad kwargs
