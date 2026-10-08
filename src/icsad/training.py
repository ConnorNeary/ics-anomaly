"""
Training loop and error computation for the deep models. Ported from the
dissertation's train_models.py: Adam with default settings, MSE loss,
early stopping on benign-validation loss, best weights restored.

Batches come from a split partition (data/splits.py) through a real
DataLoader with shuffle=True, so the shuffle order draws from the global
torch RNG exactly as the original TensorDataset loader did.

compute_errors keeps the per-feature squared-error matrix as well as the
per-sample mean. The mean is computed in torch exactly as the dissertation
did (.mean(dim=1)), so tau/w tuning on it reproduces the original numbers;
the matrix feeds alternative score aggregation and per-sensor attribution
without retraining.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


def _to_tensor(a, device):
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=device)


class _PartitionDataset(Dataset):
    """Adapter so DataLoader fetches whole batches from a partition."""

    def __init__(self, partition):
        self.partition = partition

    def __len__(self):
        return len(self.partition)

    def __getitems__(self, idx):
        X, Y = self.partition.batch(np.asarray(idx))
        return _to_tensor(X, 'cpu'), _to_tensor(Y, 'cpu')

    def __getitem__(self, i):
        X, Y = self.partition.batch(np.asarray([i]))
        return _to_tensor(X[0], 'cpu'), _to_tensor(Y[0], 'cpu')


def _keep_batch(batch):
    return batch


@dataclass
class TrainResult:
    epochs_run: int
    best_val_loss: float
    train_time_s: float
    history: dict = field(default_factory=dict)


def _validation_loss(model, val, loss_fn, device, val_batch_size):
    """val_batch_size=None scores the whole benign-val set in one forward
    pass, as the dissertation did; set a batch size for models or windows
    too large for that (the reference CNN script used 4096)."""
    if val_batch_size is None:
        X, Y = val.materialise()
        return loss_fn(model(_to_tensor(X, device)), _to_tensor(Y, device)).item()
    total = 0.0
    for start in range(0, len(val), val_batch_size):
        X, Y = val.batch(np.arange(start, min(start + val_batch_size, len(val))))
        total += loss_fn(model(_to_tensor(X, device)), _to_tensor(Y, device)).item() * len(X)
    return total / len(val)


def train_with_early_stopping(model, train, val, *, epochs, batch_size, patience,
                              val_batch_size, device):
    """Generic MSE training loop, early-stopping on benign-validation loss.
    Same loop for the AE (Y == X) and the predictors (Y = next-step target).
    The model is left holding its best weights."""
    model = model.to(device)
    loader = DataLoader(_PartitionDataset(train), batch_size=batch_size, shuffle=True,
                        collate_fn=_keep_batch)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    history = {'loss': [], 'val_loss': []}
    best_val = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    wait = 0
    epochs_run = 0
    t0 = time.time()
    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = _validation_loss(model, val, loss_fn, device, val_batch_size)
        history['loss'].append(total / len(train))
        history['val_loss'].append(val_loss)
        log.info('  epoch %3d/%d  loss %.6f  val %.6f  (%.0fs elapsed)',
                 epoch + 1, epochs, history['loss'][-1], val_loss, time.time() - t0)
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                log.info('  early stop at epoch %d (best val %.6f)', epoch + 1, best_val)
                break
    model.load_state_dict(best_state)
    return TrainResult(epochs_run=epochs_run, best_val_loss=best_val,
                       train_time_s=time.time() - t0, history=history)


def compute_errors(model, partition, *, device, batch_size=4096, per_feature=True):
    """Returns (scores, per_feature_errors):
      scores              (n,) float32, mean squared error over features
      per_feature_errors  (n, n_feat) float32 squared error, or None
    """
    n = len(partition)
    if n == 0:
        return np.array([], dtype=np.float32), None
    model.eval()
    scores, per = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            X, Y = partition.batch(np.arange(start, min(start + batch_size, n)))
            err = (_to_tensor(Y, device) - model(_to_tensor(X, device))) ** 2
            scores.append(err.mean(dim=1).cpu().numpy())
            if per_feature:
                per.append(err.cpu().numpy())
    return np.concatenate(scores), (np.concatenate(per) if per_feature else None)
