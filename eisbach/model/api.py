"""In-process inference API for the vendored DUET-Prob model.

This replaces the old ``subprocess`` call to
``ts_proba_cuda/run_single_forecast.py`` plus the CSV round-trip. The output
contract is identical to that script's CSV: one row per forecast step, columns
named ``f"{variable}_q{quantile}"`` in ``SERIES_ORDER`` x ``QUANTILES`` order.

Deliberate differences from the original script (see PROVENANCE.md):

* ``load_state_dict(..., strict=True)`` is explicit. The checkpoint matches the
  model exactly (137 tensors, zero missing, zero unexpected), so a mismatch
  should be an error rather than a quietly wrong forecast. Upstream HEAD
  downgraded this to ``strict=False``; pinning it here removes the hazard.
* No ``config_dict['distribution_family'] = 'ZIEGPD_M1'`` override. The
  checkpoint was trained with ``student_t`` and already carries it; upstream
  HEAD overrides it and then swallows the resulting ValueError.
* Failures raise instead of being printed and swallowed. The original returned
  ``None`` and exited 0 on a load failure, which made a broken run look
  successful.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from .checkpoint import resolve_checkpoint
from .config import TransformerConfig
from .duet.models.duet_prob_model import DUETProbModel

__all__ = [
    "SERIES_ORDER",
    "QUANTILES",
    "forecast",
    "load_model",
    "long_to_wide",
    "pick_device",
]

#: Channel order used during training. Must not be reordered.
SERIES_ORDER: Sequence[str] = ["wassertemp", "airtemp_96", "pressure_96"]

#: Quantiles the forecast is evaluated at, in output column order.
QUANTILES: Sequence[float] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def pick_device() -> torch.device:
    """Same device preference as the original ``run_single_forecast.py``."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[DUETProbModel, TransformerConfig]:
    """Load the trained model and its config from a checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint. If omitted, it is resolved (and if
        necessary downloaded and hash-verified) via
        :func:`eisbach.model.checkpoint.resolve_checkpoint`.
    device:
        Torch device to place the model on. Defaults to :func:`pick_device`.

    Returns
    -------
    (model, config)
        ``model`` is in ``eval()`` mode on ``device``. ``config.quantiles`` has
        been set to :data:`QUANTILES`.
    """
    resolved = resolve_checkpoint(checkpoint_path)
    torch_device = torch.device(device) if device is not None else pick_device()

    checkpoint = torch.load(resolved, map_location=torch_device, weights_only=False)
    if "config_dict" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint {resolved} is missing 'config_dict' and/or 'model_state_dict'; "
            f"found keys: {sorted(checkpoint)}"
        )

    config = TransformerConfig(**checkpoint["config_dict"])
    # The quantiles the model should evaluate at inference time.
    config.quantiles = list(QUANTILES)

    model = DUETProbModel(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch_device)
    model.eval()
    return model, config


def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the repo's long-format frame into the wide input the model expects.

    Mirrors ``run_single_forecast.py``: pivot on ``date``/``cols``/``data``,
    select :data:`SERIES_ORDER`, parse the index as datetimes, then forward-
    and back-fill gaps.
    """
    missing = set(SERIES_ORDER) - set(df_long["cols"].unique())
    if missing:
        raise ValueError(f"df_long is missing required channels: {sorted(missing)}")

    df_wide = df_long.pivot(index="date", columns="cols", values="data")[list(SERIES_ORDER)]
    df_wide.index = pd.to_datetime(df_wide.index)
    df_wide = df_wide.ffill().bfill()
    return df_wide


def forecast(
    model: DUETProbModel,
    config: TransformerConfig,
    df_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Run one probabilistic forecast.

    Parameters
    ----------
    model, config:
        As returned by :func:`load_model`.
    df_wide:
        DataFrame with a ``DatetimeIndex`` and at least the columns in
        :data:`SERIES_ORDER`. Must contain at least ``config.seq_len`` rows;
        the last ``seq_len`` rows are used as model input.

    Returns
    -------
    pandas.DataFrame
        Index: ``config.horizon`` future timestamps, starting one step after
        the last input timestamp, at the frequency inferred from ``df_wide``.
        Columns: ``f"{var}_q{q}"`` for every ``var`` in :data:`SERIES_ORDER`
        crossed with every ``q`` in :data:`QUANTILES`, variable-major.
    """
    if not isinstance(df_wide.index, pd.DatetimeIndex):
        raise TypeError(f"df_wide must have a DatetimeIndex, got {type(df_wide.index).__name__}")

    missing = [c for c in SERIES_ORDER if c not in df_wide.columns]
    if missing:
        raise ValueError(f"df_wide is missing required columns: {missing}")

    df_wide = df_wide[list(SERIES_ORDER)]

    if len(df_wide) < config.seq_len:
        raise ValueError(
            f"Not enough data: the model requires an input of length {config.seq_len}, "
            f"but only {len(df_wide)} rows were provided."
        )

    freq = pd.infer_freq(df_wide.index)
    if freq is None:
        raise ValueError(
            "Could not infer a frequency from the input index; the forecast index "
            "would be meaningless. Resample or reindex df_wide to a regular grid."
        )

    input_df = df_wide.iloc[-config.seq_len:]
    device = next(model.parameters()).device
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        distr = model(input_tensor)[0]
        q_tensor = torch.tensor(list(QUANTILES), device=device, dtype=torch.float32)
        # icdf returns [B, N_Vars, Horizon, N_Quantiles]
        quantile_predictions = distr.icdf(q_tensor)

    # [B, N, H, Q] -> [H, N, Q]
    prediction_array = quantile_predictions.squeeze(0).permute(1, 0, 2).cpu().numpy()

    last_timestamp = input_df.index[-1]
    step = pd.tseries.frequencies.to_offset(freq)
    forecast_index = pd.date_range(
        start=last_timestamp + step, periods=config.horizon, freq=freq
    )

    columns = [f"{var}_q{q}" for var in SERIES_ORDER for q in QUANTILES]
    reshaped = prediction_array.reshape(config.horizon, -1)
    return pd.DataFrame(reshaped, index=forecast_index, columns=columns)
