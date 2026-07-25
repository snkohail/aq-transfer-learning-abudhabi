"""Backbones behind one interface: fit_sources / predict / finetune.

Two fixes from FINDINGS.md Section 6 are structural here rather than optional:

  * 6.1 target standardisation is INTERNAL to the LSTM. Training on raw PM2.5
    (mean ~39) against standardised inputs drives the output layer to a large
    bias and, with early stopping, badly underfits -- it was worth ~4 ug/m3 of
    RMSE. The (mu, sd) pair is stored on the model and carried through
    fine-tuning, so a caller cannot forget to inverse-transform.

  * 6.2 seasonal routing routes by the season of each predicted DAY. The
    original wrapper ran both seasonal models on the whole test set and
    returned whichever array was longer, which is not routing at all.

torch is imported lazily so the ridge stages run without it installed.
"""

from __future__ import annotations

import copy
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import config as cfg

__all__ = [
    "Backbone",
    "RidgeBackbone",
    "LSTMBackbone",
    "SeasonalRouter",
    "season_of",
    "COOL_MONTHS",
]

# UAE seasons as used throughout this project: cool October-April, hot May-September.
COOL_MONTHS = frozenset({10, 11, 12, 1, 2, 3, 4})


def season_of(dates) -> np.ndarray:
    """Season label per day: 'cool' or 'hot'."""
    months = pd.DatetimeIndex(dates).month
    return np.where(np.isin(months, list(COOL_MONTHS)), "cool", "hot")


class Backbone(Protocol):
    """The interface every model in this package satisfies."""

    def fit_sources(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> "Backbone": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class RidgeBackbone:
    """Standardise, then ridge. The scaler is fit on SOURCE data only.

    `predict` reuses that scaler, which is what keeps invariant 2 of the
    protocol (never refit on adaptation or test) true by construction.
    """

    def __init__(self, alpha: float = cfg.RIDGE_ALPHA):
        self.alpha = alpha
        self.scaler: StandardScaler | None = None
        self.model: Ridge | None = None

    def fit_sources(self, X, y, sample_weight=None) -> "RidgeBackbone":
        X = np.asarray(X, float)
        if X.ndim == 3:  # accept sequence input by flattening, as stage3 did
            X = X.reshape(len(X), -1)
        self.scaler = StandardScaler().fit(X)
        self.model = Ridge(alpha=self.alpha).fit(
            self.scaler.transform(X), np.asarray(y, float), sample_weight=sample_weight
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit_sources must be called before predict")
        X = np.asarray(X, float)
        if X.ndim == 3:
            X = X.reshape(len(X), -1)
        return self.model.predict(self.scaler.transform(X))


class LSTMBackbone:
    """LSTM(64) -> attention(32) -> dropout -> linear(1), on (n, SEQ_L, 5) input.

    Feature standardisation is applied by the caller (it is shared across
    targets); TARGET standardisation is applied here and is not optional.
    """

    def __init__(
        self,
        hidden: int = cfg.LSTM_HIDDEN,
        attention: int = cfg.LSTM_ATTENTION,
        dropout: float = cfg.LSTM_DROPOUT,
        seed: int = cfg.SEED,
    ):
        self.hidden = hidden
        self.attention = attention
        self.dropout = dropout
        self.seed = seed
        self.net = None
        self.y_mu = 0.0
        self.y_sd = 1.0

    # -- torch plumbing, imported lazily -----------------------------------
    @staticmethod
    def _torch():
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the LSTM backbone needs torch: pip install torch"
            ) from exc
        return torch, nn

    def _build(self):
        torch, nn = self._torch()

        class Attention(nn.Module):
            def __init__(self, hidden, size):
                super().__init__()
                self.W = nn.Linear(hidden, size, bias=False)
                self.U = nn.Linear(size, 1, bias=False)

            def forward(self, outputs):
                scores = self.U(torch.tanh(self.W(outputs))).squeeze(-1)
                weights = torch.softmax(scores, dim=1)
                return torch.sum(weights.unsqueeze(-1) * outputs, dim=1)

        class Net(nn.Module):
            def __init__(self, hidden, attention, dropout, n_channels=5):
                super().__init__()
                self.lstm = nn.LSTM(n_channels, hidden, num_layers=1, batch_first=True)
                self.att = Attention(hidden, attention)
                self.drop = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                outputs, _ = self.lstm(x)
                return self.fc(self.drop(self.att(outputs)))

        return Net(self.hidden, self.attention, self.dropout)

    # -- interface ---------------------------------------------------------
    def fit_sources(
        self,
        X,
        y,
        sample_weight=None,
        epochs: int = cfg.LSTM_EPOCHS,
        patience: int = cfg.LSTM_PATIENCE,
    ) -> "LSTMBackbone":
        torch, nn = self._torch()
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        torch.manual_seed(self.seed)
        y = np.asarray(y, float)

        # FINDINGS 6.1 -- standardise the target, remember how to undo it.
        self.y_mu = float(np.mean(y))
        sd = float(np.std(y))
        self.y_sd = sd if sd > 1e-6 else 1.0
        y_scaled = (y - self.y_mu) / self.y_sd

        cut = int((1.0 - cfg.LSTM_VAL_FRACTION) * len(X))
        X_train = torch.FloatTensor(np.asarray(X[:cut], dtype=np.float32))
        y_train = torch.FloatTensor(y_scaled[:cut].astype(np.float32))
        X_val = torch.FloatTensor(np.asarray(X[cut:], dtype=np.float32))
        y_val = torch.FloatTensor(y_scaled[cut:].astype(np.float32))

        dataset = TensorDataset(X_train, y_train)
        if sample_weight is not None:
            weights = torch.DoubleTensor(np.asarray(sample_weight, float)[:cut])
            loader = DataLoader(
                dataset,
                batch_size=cfg.LSTM_BATCH,
                sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
            )
        else:
            loader = DataLoader(dataset, batch_size=cfg.LSTM_BATCH, shuffle=True)

        net = self._build()
        optimiser = torch.optim.Adam(net.parameters(), lr=cfg.LSTM_LR)
        criterion = nn.MSELoss()

        best, best_state, bad = np.inf, None, 0
        for _ in range(epochs):
            net.train()
            for batch_x, batch_y in loader:
                optimiser.zero_grad()
                loss = criterion(net(batch_x).squeeze(-1), batch_y)
                loss.backward()
                optimiser.step()
            net.eval()
            with torch.no_grad():
                val = (
                    criterion(net(X_val).squeeze(-1), y_val).item()
                    if len(X_val)
                    else 0.0
                )
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            net.load_state_dict(best_state)
        self.net = net
        return self

    def predict(self, X) -> np.ndarray:
        torch, _ = self._torch()
        if self.net is None:
            raise RuntimeError("fit_sources must be called before predict")
        if len(X) == 0:
            return np.array([])
        self.net.eval()
        with torch.no_grad():
            z = (
                self.net(torch.FloatTensor(np.asarray(X, dtype=np.float32)))
                .squeeze(-1)
                .numpy()
            )
        # FINDINGS 6.1 -- always return raw scale.
        return z * self.y_sd + self.y_mu

    def finetune(
        self, X_adapt, y_adapt, *, head_only: bool, epochs: int, lr: float, k: int | None = None
    ) -> "LSTMBackbone":
        """Adapt on the target's adaptation window.

        `k` is the adaptation window length; when given it is validated, since
        a window shorter than MIN_K_FOR_ADAPTATION yields too few sequences for
        this to mean anything (FINDINGS.md Section 3).
        """
        torch, nn = self._torch()
        if k is not None:
            from .splits import require_adaptation_k

            require_adaptation_k(k)

        clone = LSTMBackbone(self.hidden, self.attention, self.dropout, self.seed)
        clone.net = copy.deepcopy(self.net)
        # carry the standardisation through -- forgetting this was the 6.1 bug
        clone.y_mu, clone.y_sd = self.y_mu, self.y_sd

        y_scaled = (np.asarray(y_adapt, float) - clone.y_mu) / clone.y_sd

        for param in clone.net.parameters():
            param.requires_grad = True
        if head_only:
            for name, param in clone.net.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

        optimiser = torch.optim.Adam(
            filter(lambda p: p.requires_grad, clone.net.parameters()), lr=lr
        )
        criterion = nn.MSELoss()
        X_tensor = torch.FloatTensor(np.asarray(X_adapt, dtype=np.float32))
        y_tensor = torch.FloatTensor(y_scaled.astype(np.float32))

        clone.net.train()
        for _ in range(epochs):
            optimiser.zero_grad()
            loss = criterion(clone.net(X_tensor).squeeze(-1), y_tensor)
            loss.backward()
            optimiser.step()
        return clone


class SeasonalRouter:
    """Two backbones, one per UAE season, routed by the season of each day.

    FINDINGS.md Section 6.2: the original implementation ran both models on the
    full test set and returned whichever produced more predictions, so it never
    routed. Here every prediction is taken from the model matching that day's
    season, with a documented fallback when one model has no output.
    """

    def __init__(self, factory):
        self.factory = factory
        self.models: dict[str, object] = {}

    def fit_sources(self, X, y, dates, sample_weight=None) -> "SeasonalRouter":
        labels = season_of(dates)
        for season in ("cool", "hot"):
            mask = labels == season
            if mask.sum() <= cfg.SEQ_L + 30:
                continue
            weights = None if sample_weight is None else np.asarray(sample_weight)[mask]
            self.models[season] = self.factory().fit_sources(
                X[mask], np.asarray(y)[mask], sample_weight=weights
            )
        if not self.models:
            raise RuntimeError("neither seasonal model had enough data to train")
        return self

    def predict(self, X, dates) -> np.ndarray:
        labels = season_of(dates)
        if len(labels) != len(X):
            raise ValueError("dates must align with X row-for-row")

        available = {s: m for s, m in self.models.items() if m is not None}
        if not available:
            raise RuntimeError("no seasonal model is fitted")
        if len(available) == 1:  # documented fallback: one season only
            return next(iter(available.values())).predict(X)

        out = np.empty(len(X), dtype=float)
        for season, model in available.items():
            mask = labels == season
            if mask.any():
                out[mask] = model.predict(X[mask])
        return out
