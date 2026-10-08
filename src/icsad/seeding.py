"""
Seeding and device selection.

seed_everything() is called once per run, immediately before that run's
split and model construction -- never once at import. The dissertation's
train_models.py seeded torch then numpy at the top of run_ae/run_predictive,
in that position; doing the same here is what lets the new pipeline
reproduce its numbers exactly.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    """Seed every RNG a run can draw from. torch.manual_seed also seeds CUDA.

    deterministic=True forces deterministic kernels (slower on GPU, and some
    ops then raise instead of running). CUBLAS_WORKSPACE_CONFIG only takes
    effect if set before CUDA initialises, so the CLI also sets it at
    start-up when a config asks for determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # set both ways, so one deterministic run can't leak into the next
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def resolve_device(name):
    """'auto' -> cuda if available, else cpu. Anything else is passed to torch."""
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)
