"""
glycosignal.schemas
===================

Canonical data contract and core data structures for GlycoSignal.

All public metric functions accept either:
  - A ``PreparedCGMData`` object (returned by ``prepare(df)``), or
  - A raw ``pd.DataFrame`` with columns **Timestamp** and **Glucose**.

When computing many features on the same window, call ``prepare(df)`` once
and pass the result to every function -- this avoids redundant validation
and timestamp parsing.

Time-based metrics (TIR, TOR, TAT, TBT, GRI, ...) use actual timestamp
intervals instead of assuming a fixed sampling rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Column name constants
# ─────────────────────────────────────────────────────────────────────────────

COL_TIMESTAMP: str = "Timestamp"
COL_GLUCOSE: str = "Glucose"

METADATA_COLUMNS: list[str] = [
    "subject",
    "filename",
    "sensor",
    "date",
    "window_id",
]

# Common alternative column name spellings (lowercase keys for matching)
_TIMESTAMP_ALIASES: tuple[str, ...] = (
    "timestamp",
    "time",
    "datetime",
    "date_time",
    "date",
)
_GLUCOSE_ALIASES: tuple[str, ...] = (
    "glucose",
    "glucose value (mg/dl)",
    "glucose_value",
    "gl",
    "sgv",
    "glucose_mg_dl",
    "gluc",
    "bg",
    "blood_glucose",
)


# ─────────────────────────────────────────────────────────────────────────────
# Core data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreparedCGMData:
    """Pre-validated CGM data ready for feature extraction.

    Attributes
    ----------
    glucose : np.ndarray
        Glucose values (float64, no NaNs).
    timestamps : np.ndarray
        Datetime64[ns] array, monotonically sorted.
    weights : np.ndarray
        Per-reading time weight in minutes (derived from forward timestamp diffs).
    total_minutes : float
        Sum of weights -- approximately equal to the monitoring duration in minutes.
    n_readings : int
        Number of valid readings.
    """

    glucose: np.ndarray
    timestamps: np.ndarray
    weights: np.ndarray
    total_minutes: float
    n_readings: int


# Type alias: anything that a metric function can accept
CGMInput = Union[pd.DataFrame, PreparedCGMData]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, NaN-free DataFrame with Timestamp and Glucose columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must already have ``Timestamp`` and ``Glucose`` columns.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with only ``Timestamp`` and ``Glucose``, sorted by time.

    Raises
    ------
    ValueError
        If ``Timestamp`` or ``Glucose`` columns are missing.
    TypeError
        If *df* is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}.")

    required = {COL_TIMESTAMP, COL_GLUCOSE}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Call preprocessing.standardize_columns(df) first if your data uses "
            f"different column names."
        )

    out = df[[COL_TIMESTAMP, COL_GLUCOSE]].copy()
    out[COL_TIMESTAMP] = pd.to_datetime(out[COL_TIMESTAMP], errors="coerce")
    out[COL_GLUCOSE] = pd.to_numeric(out[COL_GLUCOSE], errors="coerce")
    out = out.dropna().sort_values(COL_TIMESTAMP).reset_index(drop=True)
    return out


def _time_weights(timestamps: np.ndarray) -> np.ndarray:
    """Compute per-reading time weights (minutes) from forward timestamp differences.

    weight[i] = timestamp[i+1] - timestamp[i]   for i < n-1
    weight[n-1] = weight[n-2]                    (replicate the last interval)

    This approach attributes time to each reading proportional to the gap that
    follows it, which is appropriate for time-weighted metric calculations (TIR,
    TAR, TBR, GRI, etc.).

    Parameters
    ----------
    timestamps : np.ndarray
        Sorted datetime64 array.

    Returns
    -------
    np.ndarray
        Float64 array of per-reading weights in minutes, same length as input.
    """
    ts = pd.to_datetime(pd.Series(timestamps, copy=False)).reset_index(drop=True)
    n = len(ts)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([5.0])  # assume typical 5-minute CGM interval

    diffs_sec = np.diff(ts.values).astype("timedelta64[s]").astype(np.float64)
    weights = np.empty(n, dtype=np.float64)
    weights[:-1] = diffs_sec / 60.0
    weights[-1] = weights[-2]
    return np.maximum(weights, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def prepare(df: pd.DataFrame) -> PreparedCGMData:
    """Validate *df* and return a :class:`PreparedCGMData` for fast feature extraction.

    Call this once when computing many features on the same dataset.  Pass the
    returned object instead of the raw DataFrame to every metric function --
    this avoids repeated validation, sorting, and NaN filtering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Timestamp`` and ``Glucose`` columns.

    Returns
    -------
    PreparedCGMData
        Immutable container ready for metric computation.

    Examples
    --------
    >>> from glycosignal import metrics
    >>> from glycosignal.schemas import prepare
    >>> prepared = prepare(df)
    >>> metrics.mean_glucose(prepared)
    """
    vdf = _validate_dataframe(df)
    if vdf.empty:
        return PreparedCGMData(
            glucose=np.array([], dtype=np.float64),
            timestamps=np.array([], dtype="datetime64[ns]"),
            weights=np.array([], dtype=np.float64),
            total_minutes=0.0,
            n_readings=0,
        )
    weights = _time_weights(vdf[COL_TIMESTAMP].values)
    return PreparedCGMData(
        glucose=vdf[COL_GLUCOSE].values.astype(np.float64),
        timestamps=vdf[COL_TIMESTAMP].values,
        weights=weights,
        total_minutes=float(weights.sum()),
        n_readings=len(vdf),
    )


def _ensure_prepared(data: CGMInput) -> PreparedCGMData:
    """Accept a DataFrame or PreparedCGMData; always return PreparedCGMData.

    Internal helper used by every metric function so callers can pass either
    a raw DataFrame or a pre-prepared object.
    """
    if isinstance(data, PreparedCGMData):
        return data
    return prepare(data)
