"""
glycosignal.preprocessing
==========================

Functions for cleaning, standardizing, validating, and transforming CGM data.

All functions operate on DataFrames with canonical column names:
  - ``Timestamp`` (datetime-coercible)
  - ``Glucose``   (numeric, mg/dL by default)

Call :func:`standardize_columns` first when working with raw data that uses
non-standard column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .schemas import (
    COL_GLUCOSE,
    COL_TIMESTAMP,
    _GLUCOSE_ALIASES,
    _TIMESTAMP_ALIASES,
)
from .utils import require_columns, require_dataframe


# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion
# ─────────────────────────────────────────────────────────────────────────────

_MMOL_TO_MGDL: float = 18.0182
_VALID_UNITS = {"mg/dL", "mmol/L"}


def convert_units(
    values: pd.DataFrame | pd.Series | np.ndarray,
    from_unit: str,
    to_unit: str,
) -> pd.DataFrame | pd.Series | np.ndarray:
    """Convert glucose values between mg/dL and mmol/L.

    Parameters
    ----------
    values : DataFrame, Series, or ndarray
        Glucose values or a DataFrame containing a ``Glucose`` column.
        When a DataFrame is passed, only the ``Glucose`` column is converted
        (in a copy) and the full DataFrame is returned.
    from_unit : str
        Source unit: ``"mg/dL"`` or ``"mmol/L"``.
    to_unit : str
        Target unit: ``"mg/dL"`` or ``"mmol/L"``.

    Returns
    -------
    Same type as input.
        Converted values.  Original data is never modified.

    Raises
    ------
    ValueError
        If *from_unit* or *to_unit* is not one of the supported units, or if
        they are the same (no conversion needed -- use the values as-is).
    """
    if from_unit not in _VALID_UNITS:
        raise ValueError(f"from_unit must be one of {_VALID_UNITS}, got {from_unit!r}.")
    if to_unit not in _VALID_UNITS:
        raise ValueError(f"to_unit must be one of {_VALID_UNITS}, got {to_unit!r}.")
    if from_unit == to_unit:
        raise ValueError(
            f"from_unit and to_unit are both {from_unit!r} -- no conversion needed."
        )

    factor = _MMOL_TO_MGDL if from_unit == "mmol/L" else (1.0 / _MMOL_TO_MGDL)

    if isinstance(values, pd.DataFrame):
        require_columns(values, [COL_GLUCOSE])
        out = values.copy()
        out[COL_GLUCOSE] = out[COL_GLUCOSE] * factor
        return out

    if isinstance(values, pd.Series):
        return values * factor

    return np.asarray(values, dtype=np.float64) * factor


# ─────────────────────────────────────────────────────────────────────────────
# Column standardization
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_COLUMN_MAP: dict[str, str] = {
    alias: COL_TIMESTAMP for alias in _TIMESTAMP_ALIASES
}
_DEFAULT_COLUMN_MAP.update({alias: COL_GLUCOSE for alias in _GLUCOSE_ALIASES})


def standardize_columns(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rename non-standard columns to the canonical ``Timestamp`` / ``Glucose`` names.

    Matching is case-insensitive.  The function returns a copy; the original
    DataFrame is not modified.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CGM DataFrame with potentially non-standard column names.
    column_map : dict[str, str] | None
        Optional additional or overriding mappings of ``{original_name: new_name}``.
        When provided, this map is merged with the built-in defaults (user
        entries take precedence).

    Returns
    -------
    pd.DataFrame
        Copy with renamed columns.

    Examples
    --------
    >>> standardize_columns(df)  # renames "gl" -> "Glucose", "time" -> "Timestamp"
    >>> standardize_columns(df, column_map={"BG": "Glucose"})
    """
    require_dataframe(df, "df")
    combined_map = dict(_DEFAULT_COLUMN_MAP)
    if column_map:
        combined_map.update({k.lower(): v for k, v in column_map.items()})

    rename = {}
    for col in df.columns:
        lower = col.lower()
        if lower in combined_map and col not in (COL_TIMESTAMP, COL_GLUCOSE):
            rename[col] = combined_map[lower]

    return df.rename(columns=rename)


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_cgm(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    sort: bool = True,
    enforce_positive: bool = True,
) -> pd.DataFrame:
    """Apply standard CGM cleaning steps and return a cleaned copy.

    Steps performed (in order):
    1. Coerce ``Timestamp`` to datetime, drop un-parseable rows.
    2. Coerce ``Glucose`` to numeric, drop NaN rows.
    3. Optionally enforce ``Glucose > 0`` (removes physiologically impossible values).
    4. Optionally drop duplicate timestamps (keep the first occurrence).
    5. Optionally sort by ``Timestamp``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Timestamp`` and ``Glucose`` columns.
    drop_duplicates : bool
        Drop rows with duplicate ``Timestamp`` values, keeping the first.
    sort : bool
        Sort by ``Timestamp`` ascending.
    enforce_positive : bool
        Drop rows where ``Glucose <= 0``.

    Returns
    -------
    pd.DataFrame
        Cleaned copy with reset integer index.
    """
    require_dataframe(df, "df")
    require_columns(df, [COL_TIMESTAMP, COL_GLUCOSE])

    out = df.copy()
    out[COL_TIMESTAMP] = pd.to_datetime(out[COL_TIMESTAMP], errors="coerce")
    out[COL_GLUCOSE] = pd.to_numeric(out[COL_GLUCOSE], errors="coerce")
    out = out.dropna(subset=[COL_TIMESTAMP, COL_GLUCOSE])

    if enforce_positive:
        out = out[out[COL_GLUCOSE] > 0]

    if drop_duplicates:
        out = out.drop_duplicates(subset=[COL_TIMESTAMP], keep="first")

    if sort:
        out = out.sort_values(COL_TIMESTAMP)

    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Structured result from :func:`validate_cgm`.

    Attributes
    ----------
    n_readings : int
        Total number of rows in the input.
    n_valid : int
        Rows with valid Timestamp and Glucose (not NaN, positive).
    n_missing_timestamp : int
        Rows where Timestamp could not be parsed.
    n_missing_glucose : int
        Rows where Glucose is NaN or non-numeric.
    n_non_positive : int
        Rows where Glucose <= 0.
    n_duplicate_timestamps : int
        Number of duplicate Timestamp values.
    time_start : pd.Timestamp | None
        Earliest valid timestamp.
    time_end : pd.Timestamp | None
        Latest valid timestamp.
    duration_days : float | None
        Total span of monitoring in days.
    is_monotonic : bool
        True if timestamps are monotonically increasing after sorting.
    gaps : pd.DataFrame
        DataFrame of detected gaps (see :func:`detect_gaps`), empty if none.
    warnings : list[str]
        Human-readable warning strings.
    """

    n_readings: int = 0
    n_valid: int = 0
    n_missing_timestamp: int = 0
    n_missing_glucose: int = 0
    n_non_positive: int = 0
    n_duplicate_timestamps: int = 0
    time_start: pd.Timestamp | None = None
    time_end: pd.Timestamp | None = None
    duration_days: float | None = None
    is_monotonic: bool = True
    gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Return True if the dataset has no critical issues."""
        return self.n_valid > 0 and len(self.warnings) == 0

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        lines = [
            f"Readings: {self.n_readings} total, {self.n_valid} valid",
            f"Duration: {self.duration_days:.1f} days" if self.duration_days else "Duration: unknown",
            f"Gaps detected: {len(self.gaps)}",
            f"Duplicate timestamps: {self.n_duplicate_timestamps}",
        ]
        if self.warnings:
            lines.append("Warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)


def validate_cgm(
    df: pd.DataFrame,
    expected_interval_minutes: float = 5.0,
) -> ValidationReport:
    """Inspect a CGM DataFrame and return a structured :class:`ValidationReport`.

    This function does not modify *df* and never raises exceptions for data
    quality issues -- problems are reported in the returned object instead.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Timestamp`` and ``Glucose`` columns.
    expected_interval_minutes : float
        Expected sampling interval used for gap detection.

    Returns
    -------
    ValidationReport
        Detailed quality report.
    """
    require_dataframe(df, "df")
    report = ValidationReport(n_readings=len(df))

    if COL_TIMESTAMP not in df.columns:
        report.warnings.append(f"Missing column '{COL_TIMESTAMP}'.")
        return report
    if COL_GLUCOSE not in df.columns:
        report.warnings.append(f"Missing column '{COL_GLUCOSE}'.")
        return report

    ts = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce")
    gl = pd.to_numeric(df[COL_GLUCOSE], errors="coerce")

    report.n_missing_timestamp = int(ts.isna().sum())
    report.n_missing_glucose = int(gl.isna().sum())

    valid_mask = ts.notna() & gl.notna()
    gl_valid = gl[valid_mask]
    ts_valid = ts[valid_mask]

    report.n_non_positive = int((gl_valid <= 0).sum())
    positive_mask = valid_mask & (gl > 0)
    ts_pos = ts[positive_mask].sort_values()
    report.n_valid = int(positive_mask.sum())

    if report.n_valid > 0:
        report.time_start = ts_pos.iloc[0]
        report.time_end = ts_pos.iloc[-1]
        report.duration_days = (report.time_end - report.time_start).total_seconds() / 86400

    report.n_duplicate_timestamps = int(ts_valid.duplicated().sum())
    report.is_monotonic = bool(ts_valid.is_monotonic_increasing)

    # Gap detection using a clean subset
    clean = df.copy()
    clean[COL_TIMESTAMP] = ts
    clean[COL_GLUCOSE] = gl
    clean = clean.dropna(subset=[COL_TIMESTAMP, COL_GLUCOSE])
    clean = clean[clean[COL_GLUCOSE] > 0].sort_values(COL_TIMESTAMP).reset_index(drop=True)
    if len(clean) >= 2:
        report.gaps = detect_gaps(clean, expected_interval_minutes=expected_interval_minutes)

    # Build warnings
    if report.n_missing_timestamp:
        report.warnings.append(f"{report.n_missing_timestamp} un-parseable Timestamp value(s).")
    if report.n_missing_glucose:
        report.warnings.append(f"{report.n_missing_glucose} non-numeric Glucose value(s).")
    if report.n_non_positive:
        report.warnings.append(f"{report.n_non_positive} non-positive Glucose reading(s).")
    if report.n_duplicate_timestamps:
        report.warnings.append(f"{report.n_duplicate_timestamps} duplicate Timestamp(s).")
    if not report.is_monotonic:
        report.warnings.append("Timestamps are not monotonically increasing.")
    if len(report.gaps) > 0:
        report.warnings.append(f"{len(report.gaps)} gap(s) detected in the time series.")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Gap detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_gaps(
    df: pd.DataFrame,
    expected_interval_minutes: float = 5.0,
) -> pd.DataFrame:
    """Identify gaps in CGM data that are larger than the expected interval.

    A gap is defined as any consecutive timestamp difference greater than
    ``expected_interval_minutes * 1.5`` (50% tolerance).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with ``Timestamp`` and ``Glucose`` columns, sorted by time.
    expected_interval_minutes : float
        Normal sampling interval in minutes (default 5 for most CGM devices).

    Returns
    -------
    pd.DataFrame
        One row per gap with columns:
        - ``gap_start`` -- timestamp of the reading before the gap
        - ``gap_end`` -- timestamp of the reading after the gap
        - ``duration_minutes`` -- gap length in minutes
        - ``expected_readings`` -- how many readings are missing
        Empty DataFrame if no gaps are found or fewer than 2 valid readings.
    """
    require_dataframe(df, "df")
    require_columns(df, [COL_TIMESTAMP])

    ts = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce").dropna().sort_values().reset_index(drop=True)

    if len(ts) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "duration_minutes", "expected_readings"])

    diffs = ts.diff().iloc[1:]
    threshold = pd.Timedelta(minutes=expected_interval_minutes * 1.5)
    gap_mask = diffs > threshold

    gap_starts = ts.iloc[:-1][gap_mask.values].values
    gap_ends = ts.iloc[1:][gap_mask.values].values
    durations = (gap_ends - gap_starts).astype("timedelta64[s]").astype(float) / 60.0
    expected = np.floor(durations / expected_interval_minutes).astype(int) - 1

    return pd.DataFrame(
        {
            "gap_start": gap_starts,
            "gap_end": gap_ends,
            "duration_minutes": durations,
            "expected_readings": np.maximum(expected, 0),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_cgm(
    df: pd.DataFrame,
    freq: str = "5min",
    method: Literal["nearest", "mean", "interpolate"] = "nearest",
) -> pd.DataFrame:
    """Resample CGM data to a regular time grid.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with ``Timestamp`` and ``Glucose`` columns.
    freq : str
        Target frequency string (e.g. ``"5min"``, ``"15min"``).
    method : {"nearest", "mean", "interpolate"}
        How to aggregate / fill values on the target grid:
        - ``"nearest"`` -- assign the closest observed reading to each grid point.
        - ``"mean"`` -- average all readings within each grid interval.
        - ``"interpolate"`` -- linearly interpolate after resampling with mean.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with ``Timestamp`` and ``Glucose`` columns.
        Metadata columns present in *df* are preserved at their first value
        per group.

    Raises
    ------
    ValueError
        If *method* is not one of the supported options.
    """
    require_dataframe(df, "df")
    require_columns(df, [COL_TIMESTAMP, COL_GLUCOSE])

    valid_methods = {"nearest", "mean", "interpolate"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}.")

    out = df.copy()
    out[COL_TIMESTAMP] = pd.to_datetime(out[COL_TIMESTAMP], errors="coerce")
    out[COL_GLUCOSE] = pd.to_numeric(out[COL_GLUCOSE], errors="coerce")
    out = out.dropna(subset=[COL_TIMESTAMP, COL_GLUCOSE]).sort_values(COL_TIMESTAMP)
    out = out.set_index(COL_TIMESTAMP)

    if method == "nearest":
        grid = pd.date_range(out.index[0].floor(freq), out.index[-1].ceil(freq), freq=freq)
        result = out[[COL_GLUCOSE]].reindex(grid, method="nearest", tolerance=pd.Timedelta(freq))
    elif method in ("mean", "interpolate"):
        result = out[[COL_GLUCOSE]].resample(freq).mean()
        if method == "interpolate":
            result[COL_GLUCOSE] = result[COL_GLUCOSE].interpolate(method="time")

    result = result.reset_index()
    result.columns = [COL_TIMESTAMP, COL_GLUCOSE]
    return result.dropna(subset=[COL_GLUCOSE]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_cgm(
    df: pd.DataFrame,
    method: Literal["pchip", "linear", "time"] = "pchip",
    max_gap_points: int = 12,
) -> pd.DataFrame:
    """Fill short gaps in a regularly-sampled CGM DataFrame using interpolation.

    Gaps longer than ``max_gap_points`` consecutive missing values are left as
    NaN.  Observed values are never modified.

    This function assumes *df* is already on a regular time grid (e.g. after
    :func:`resample_cgm`).  For long-format sparse data, call
    :func:`resample_cgm` first to establish the grid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Timestamp`` (index or column) and ``Glucose`` columns.
        May contain NaN in the ``Glucose`` column.
    method : {"pchip", "linear", "time"}
        Interpolation method:
        - ``"pchip"`` -- shape-preserving cubic spline (recommended for CGM).
        - ``"linear"`` -- simple linear interpolation.
        - ``"time"`` -- pandas time-aware linear interpolation.
    max_gap_points : int
        Maximum consecutive NaN points to fill.  Gaps longer than this are
        kept as NaN.  Default 12 (= 1 hour at 5-min sampling).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with short gaps filled.  Columns and index unchanged.

    Raises
    ------
    ValueError
        If *method* is not supported.
    """
    require_dataframe(df, "df")
    require_columns(df, [COL_GLUCOSE])

    valid_methods = {"pchip", "linear", "time"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}.")

    out = df.copy()
    glucose = out[COL_GLUCOSE].to_numpy(dtype=float)
    filled = _fill_gaps_pchip(glucose, max_gap_points) if method == "pchip" else \
             _fill_gaps_pandas(glucose, out, method, max_gap_points)
    out[COL_GLUCOSE] = filled
    return out


def _fill_gaps_pchip(values: np.ndarray, max_gap_points: int) -> np.ndarray:
    """PCHIP interpolation for short consecutive NaN runs."""
    y = np.asarray(values, dtype=float)
    n = y.size
    x = np.arange(n, dtype=float)
    missing_mask = np.isnan(y)

    if not missing_mask.any():
        return y.copy()

    mask_known = ~missing_mask
    if int(mask_known.sum()) < 2:
        return y.copy()

    filled = y.copy()
    runs = _find_nan_runs(missing_mask)

    try:
        interp = PchipInterpolator(x[mask_known], y[mask_known], extrapolate=False)
        for start, end in runs:
            run_len = end - start + 1
            if run_len <= max_gap_points:
                xi = x[start:end + 1]
                filled[start:end + 1] = np.round(interp(xi), 1)
    except Exception:
        # Fall back to linear interpolation on failure
        for start, end in runs:
            run_len = end - start + 1
            if run_len <= max_gap_points:
                xi = x[start:end + 1]
                filled[start:end + 1] = np.round(
                    np.interp(xi, x[mask_known], y[mask_known]), 1
                )

    return filled


def _fill_gaps_pandas(
    values: np.ndarray,
    df: pd.DataFrame,
    method: str,
    max_gap_points: int,
) -> np.ndarray:
    """Pandas-backed interpolation (linear or time) for short gaps."""
    s = pd.Series(values)
    runs = _find_nan_runs(np.isnan(values))
    fill_indices: set[int] = set()
    for start, end in runs:
        if (end - start + 1) <= max_gap_points:
            fill_indices.update(range(start, end + 1))

    if not fill_indices:
        return values.copy()

    # Only interpolate at positions that qualify; mask the rest
    temp = s.copy()
    for start, end in runs:
        if (end - start + 1) > max_gap_points:
            pass  # leave as NaN -- will not be interpolated

    kw = {"method": "time"} if method == "time" else {"method": method}
    interpolated = temp.interpolate(**kw)

    result = values.copy()
    for idx in fill_indices:
        if not np.isnan(interpolated.iloc[idx]):
            result[idx] = round(float(interpolated.iloc[idx]), 1)
    return result


def _find_nan_runs(missing_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) index pairs for consecutive NaN runs."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for idx, is_missing in enumerate(missing_mask):
        if is_missing and run_start is None:
            run_start = idx
        elif (not is_missing) and run_start is not None:
            runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(missing_mask) - 1))
    return runs
