"""Vendored DUET-Prob inference package.

A minimal, self-contained copy of the parts of ``ts_proba_cuda`` that the
Eisbach forecast actually needs at inference time. See PROVENANCE.md for what
was taken, from where, and what was stripped.

Typical use::

    from eisbach.model import load_model, forecast, long_to_wide

    model, config = load_model()          # downloads + verifies the checkpoint
    df_wide = long_to_wide(df_long)
    df_forecast = forecast(model, config, df_wide)

Runtime dependencies: torch, numpy, pandas, einops, scipy (plus requests, only
when the checkpoint has to be downloaded).
"""

from .api import (
    QUANTILES,
    SERIES_ORDER,
    forecast,
    load_model,
    long_to_wide,
    pick_device,
)
from .checkpoint import (
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    ChecksumError,
    resolve_checkpoint,
)
from .config import TransformerConfig

__all__ = [
    "CHECKPOINT_SHA256",
    "CHECKPOINT_URL",
    "ChecksumError",
    "QUANTILES",
    "SERIES_ORDER",
    "TransformerConfig",
    "forecast",
    "load_model",
    "long_to_wide",
    "pick_device",
    "resolve_checkpoint",
]
