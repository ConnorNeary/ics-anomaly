"""
Deep model classes, ported unchanged from the dissertation's models.py
(which mirrors SWaT_A12_Models.ipynb). Hyperparameters now live in the
experiment configs (configs/*.yaml) rather than SMALL_CONFIGS /
REFERENCE_CONFIGS dicts.

Reference configurations, confirmed against the reference repo and paper:
  - AE: nH=5, cf=2.5 -- matches utils.get_argparser() defaults.
  - CNN: units=32, history=200, layers=8, kernel=3 -- Table 2's
    [18]/"KS'18" (Fig. 3 ties this to SWaT).
  - LSTM: units=512, history=100, layers=4 ([37]/"Zizzo 2019"), inter-layer
    dropout 0.5. history=100 has no basis in the paper's prose -- taken
    from the repo default paired with that config.
"""

import torch.nn as nn


class Autoencoder(nn.Module):
    """Point-level (h=0) reconstruction autoencoder. Matches the reference
    detector/autoencoder.py: flat Input(shape=(nI,)), input==target."""
    def __init__(self, n_feat, enc_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_feat, 32), nn.ReLU(),
                                      nn.Linear(32, enc_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(enc_dim, 32), nn.ReLU(),
                                      nn.Linear(32, n_feat), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CNNPredictor(nn.Module):
    """Next-step predictor: x = X[t-history:t-1], target = X_{t+1} (X_t
    itself never seen). Full derivation in the notebook's CNNPredictor cell."""
    def __init__(self, n_feat, window_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_feat, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 16, 1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * window_size, n_feat))

    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.encoder(x)
        return self.head(features)


class LSTMPredictor(nn.Module):
    """Next-step predictor, same convention as CNNPredictor above. dropout
    only matters for num_layers>1; matches the reference's inter-layer
    dropout=0.5 for the stacked reference config, left at 0 for the small
    model (PyTorch warns on nonzero dropout with num_layers=1 anyway)."""
    def __init__(self, n_feat, latent_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = nn.LSTM(n_feat, latent_dim, num_layers=num_layers,
                                dropout=dropout, batch_first=True)
        self.head = nn.Linear(latent_dim, n_feat)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return self.head(h[-1])


class CNNPredictorReference(nn.Module):
    """
    Next-step predictor matching the reference repo's actual CNN
    (detector/cnn.py): `layers` repeated Conv1d(`units`, kernel=`kernel`) +
    BatchNorm1d + ReLU blocks at uniform width, matching Keras'
    Conv1D+BatchNormalization stack.

    Keras' Conv1D default padding is 'valid' (none), so each layer shortens
    the temporal axis by (kernel-1). Flatten -> single linear output, no
    activation.
    """
    def __init__(self, n_feat, window_size, units=32, layers=8, kernel=3):
        super().__init__()
        blocks = []
        in_ch = n_feat
        for _ in range(layers):
            blocks += [nn.Conv1d(in_ch, units, kernel), nn.BatchNorm1d(units), nn.ReLU()]
            in_ch = units
        self.encoder = nn.Sequential(*blocks)

        final_length = window_size - layers * (kernel - 1)
        if final_length <= 0:
            raise ValueError(
                f'window_size={window_size} too small for {layers} layers of kernel={kernel} '
                f'with no padding (needs window_size > {layers * (kernel - 1)})')
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(units * final_length, n_feat))

    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.encoder(x)
        return self.head(features)
