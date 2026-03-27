"""
glycosignal.detect
==================

Glycemic event detection: identify hypoglycemic and hyperglycemic episodes,
nocturnal events, and postprandial excursions from CGM data.

All functions accept a ``pd.DataFrame`` with ``Timestamp`` and ``Glucose``
columns and return a DataFrame of detected episodes.

Assumptions and limitations
----------------------------
- Detection is based on simple threshold crossing + duration rules.
- No meal data is used; postprandial excursions are heuristic.
- Events at the edges of the recording may be truncated.
- Readings are assumed to be sorted by time; call
  :func:`~glycosignal.preprocessing.clean_cgm` first if unsure.

Usage
-----
    >>> from glycosignal import detect
    >>> hypo = detect.detect_hypoglycemia(df, threshold=70, min_duration_minutes=15)
    >>> hyper = detect.detect_hyperglycemia(df, threshold=180)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import COL_GLUCOSE, COL_TIMESTAMP
from .utils import require_columns, require_dataframe


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_threshold_events(
    df: pd.DataFrame,
    threshold: float,
    direction: str,  # "below" or "above"
    min_duration_minutes: float,
    event_type: str,
    value_col_name: str,  # "nadir_glucose" or "peak_glucose"
) -> pd.DataFrame:
    """Core episode detection logic based on threshold crossing.

    An episode is a maximal contiguous run of readings satisfying the threshold
    condition that spans at least ``min_duration_minutes``.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned CGM data, sorted by Timestamp.
    threshold : float
        Glucose threshold in mg/dL.
    direction : {"below", "above"}
        Whether to detect readings below or above the threshold.
    min_duration_minutes : float
        Minimum episode duration to report.
    event_type : str
        String label for the ``event_type`` column.
    value_col_name : str
        Column name for the nadir or peak value.

    Returns
    -------
    pd.DataFrame
        Episodes with columns: start_time, end_time, duration_minutes,
        <value_col_name>, event_type.
    """
    require_dataframe(df, "df")
    require_columns(df, [COL_TIMESTAMP, COL_GLUCOSE])

    ts = pd.to_datetime(df[COL_TIMESTAMP]).reset_index(drop=True)
    gl = pd.to_numeric(df[COL_GLUCOSE], errors="coerce").reset_index(drop=True)

    valid = gl.notna()
    ts = ts[valid].reset_index(drop=True)
    gl = gl[valid].reset_index(drop=True)

    if len(gl) == 0:
        return _empty_episodes(value_col_name)

    if direction == "below":
        condition = gl <= threshold
    else:
        condition = gl >= threshold

    episodes: list[dict] = []

    # Find contiguous runs
    in_episode = False
    ep_start_idx: int = 0

    for i, cond in enumerate(condition):
        if cond and not in_episode:
            in_episode = True
            ep_start_idx = i
        elif not cond and in_episode:
            in_episode = False
            _append_episode(
                episodes, ts, gl, ep_start_idx, i - 1,
                min_duration_minutes, event_type, value_col_name, direction,
            )

    # Close any open episode at end of data
    if in_episode:
        _append_episode(
            episodes, ts, gl, ep_start_idx, len(ts) - 1,
            min_duration_minutes, event_type, value_col_name, direction,
        )

    if not episodes:
        return _empty_episodes(value_col_name)

    return pd.DataFrame(episodes)


def _append_episode(
    episodes: list,
    ts: pd.Series,
    gl: pd.Series,
    start_idx: int,
    end_idx: int,
    min_duration_minutes: float,
    event_type: str,
    value_col_name: str,
    direction: str,
) -> None:
    start_time = ts.iloc[start_idx]
    end_time = ts.iloc[end_idx]
    duration = (end_time - start_time).total_seconds() / 60.0

    if duration < min_duration_minutes:
        return

    segment = gl.iloc[start_idx:end_idx + 1]
    peak_nadir = float(segment.min() if direction == "below" else segment.max())

    episodes.append({
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": round(duration, 1),
        value_col_name: round(peak_nadir, 1),
        "event_type": event_type,
    })


def _empty_episodes(value_col_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["start_time", "end_time", "duration_minutes", value_col_name, "event_type"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public detection functions
# ─────────────────────────────────────────────────────────────────────────────

def detect_hypoglycemia(
    df: pd.DataFrame,
    threshold: float = 70.0,
    min_duration_minutes: float = 15.0,
) -> pd.DataFrame:
    """Detect hypoglycemic episodes in CGM data.

    An episode is a contiguous run of readings at or below *threshold* lasting
    at least *min_duration_minutes*.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns.
    threshold : float
        Glucose threshold in mg/dL.  Default 70 (ADA Level 1 hypoglycemia).
    min_duration_minutes : float
        Minimum episode duration to report.  Default 15 minutes.

    Returns
    -------
    pd.DataFrame
        One row per episode with columns:
        - ``start_time`` -- episode start timestamp
        - ``end_time`` -- episode end timestamp
        - ``duration_minutes`` -- episode duration
        - ``nadir_glucose`` -- minimum glucose during episode
        - ``event_type`` -- ``"hypoglycemia"``

    Notes
    -----
    The threshold crossing is inclusive (glucose <= threshold).
    Duration is measured from first to last reading in the episode; actual
    time below threshold may differ if readings are sparse.
    """
    return _detect_threshold_events(
        df=df,
        threshold=threshold,
        direction="below",
        min_duration_minutes=min_duration_minutes,
        event_type="hypoglycemia",
        value_col_name="nadir_glucose",
    )


def detect_hyperglycemia(
    df: pd.DataFrame,
    threshold: float = 180.0,
    min_duration_minutes: float = 15.0,
) -> pd.DataFrame:
    """Detect hyperglycemic episodes in CGM data.

    An episode is a contiguous run of readings at or above *threshold* lasting
    at least *min_duration_minutes*.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns.
    threshold : float
        Glucose threshold in mg/dL.  Default 180 (ADA Level 1 hyperglycemia).
    min_duration_minutes : float
        Minimum episode duration to report.  Default 15 minutes.

    Returns
    -------
    pd.DataFrame
        One row per episode with columns:
        - ``start_time`` -- episode start timestamp
        - ``end_time`` -- episode end timestamp
        - ``duration_minutes`` -- episode duration
        - ``peak_glucose`` -- maximum glucose during episode
        - ``event_type`` -- ``"hyperglycemia"``
    """
    return _detect_threshold_events(
        df=df,
        threshold=threshold,
        direction="above",
        min_duration_minutes=min_duration_minutes,
        event_type="hyperglycemia",
        value_col_name="peak_glucose",
    )


def detect_nocturnal_events(
    df: pd.DataFrame,
    start_hour: int = 0,
    end_hour: int = 6,
    hypo_threshold: float = 70.0,
    hyper_threshold: float = 180.0,
    min_duration_minutes: float = 15.0,
) -> pd.DataFrame:
    """Detect nocturnal hypoglycemic and hyperglycemic events.

    Filters CGM data to the nocturnal window [start_hour, end_hour) and then
    applies standard threshold detection.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns.
    start_hour : int
        Start of the nocturnal window (0-based, inclusive).  Default 0 (midnight).
    end_hour : int
        End of the nocturnal window (0-based, exclusive).  Default 6 (6am).
    hypo_threshold : float
        Hypoglycemia threshold in mg/dL.  Default 70.
    hyper_threshold : float
        Hyperglycemia threshold in mg/dL.  Default 180.
    min_duration_minutes : float
        Minimum episode duration.  Default 15.

    Returns
    -------
    pd.DataFrame
        Combined nocturnal hypo and hyper events.  Includes all columns from
        :func:`detect_hypoglycemia` and :func:`detect_hyperglycemia`, with
        a unified ``value_glucose`` column (nadir or peak) and an
        ``event_type`` column.

    Notes
    -----
    Nocturnal window crossing midnight (e.g. 22:00–06:00) can be specified by
    setting ``start_hour=22, end_hour=6`` -- in this case hour < 6 OR hour >= 22
    is selected.
    """
    require_dataframe(df, "df")
    require_columns(df, [COL_TIMESTAMP, COL_GLUCOSE])

    ts = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce")
    hour = ts.dt.hour

    if start_hour < end_hour:
        night_mask = (hour >= start_hour) & (hour < end_hour)
    else:
        # Window crosses midnight (e.g. 22:00–06:00)
        night_mask = (hour >= start_hour) | (hour < end_hour)

    night_df = df.loc[night_mask.values].copy()

    if night_df.empty:
        return pd.DataFrame(
            columns=["start_time", "end_time", "duration_minutes",
                     "value_glucose", "event_type"]
        )

    hypo = detect_hypoglycemia(night_df, threshold=hypo_threshold,
                               min_duration_minutes=min_duration_minutes)
    hyper = detect_hyperglycemia(night_df, threshold=hyper_threshold,
                                 min_duration_minutes=min_duration_minutes)

    # Harmonise column names and combine
    if not hypo.empty:
        hypo = hypo.rename(columns={"nadir_glucose": "value_glucose"})
    if not hyper.empty:
        hyper = hyper.rename(columns={"peak_glucose": "value_glucose"})

    combined = pd.concat([hypo, hyper], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(
            columns=["start_time", "end_time", "duration_minutes",
                     "value_glucose", "event_type"]
        )
    return combined.sort_values("start_time").reset_index(drop=True)


def detect_postprandial_excursions(
    df: pd.DataFrame,
    rise_threshold: float = 50.0,
    window_minutes: int = 120,
    min_baseline_readings: int = 3,
) -> pd.DataFrame:
    """Detect postprandial glucose excursions using a heuristic rise-detection approach.

    For each local nadir in the glucose trace, this function looks forward up to
    ``window_minutes`` and records an excursion if the glucose rises by at least
    ``rise_threshold`` mg/dL.

    This is a **heuristic** approach -- it does not use meal timestamps.  The
    results are approximate and may include non-postprandial excursions.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns.
    rise_threshold : float
        Minimum glucose rise (mg/dL) within the window to qualify as an
        excursion.  Default 50.
    window_minutes : int
        Look-ahead window in minutes.  Default 120.
    min_baseline_readings : int
        Minimum number of readings below the nadir + 5 mg/dL required to
        establish a baseline.  Prevents false positives at data edges.

    Returns
    -------
    pd.DataFrame
        One row per detected excursion with columns:
        - ``start_time`` -- nadir timestamp
        - ``end_time`` -- peak timestamp within window
        - ``duration_minutes`` -- time from nadir to peak
        - ``nadir_glucose`` -- baseline glucose level
        - ``peak_glucose`` -- maximum glucose within window
        - ``rise_mg_dl`` -- glucose rise (peak - nadir)
        - ``event_type`` -- ``"postprandial_excursion"``

    Notes
    -----
    This function identifies *local minima* as potential pre-meal baselines.
    False positives will occur during hypoglycemic recovery.  Clinical
    interpretation should consider the overall glucose context.
    """
    require_dataframe(df, "df")
    require_columns(df, [COL_TIMESTAMP, COL_GLUCOSE])

    ts = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce").reset_index(drop=True)
    gl = pd.to_numeric(df[COL_GLUCOSE], errors="coerce").reset_index(drop=True)
    valid = gl.notna() & ts.notna()
    ts = ts[valid].reset_index(drop=True)
    gl = gl[valid].reset_index(drop=True)
    n = len(gl)

    if n < 3:
        return pd.DataFrame(
            columns=["start_time", "end_time", "duration_minutes",
                     "nadir_glucose", "peak_glucose", "rise_mg_dl", "event_type"]
        )

    window_td = pd.Timedelta(minutes=window_minutes)
    excursions: list[dict] = []
    prev, cur, nxt = gl.iloc[:-2].values, gl.iloc[1:-1].values, gl.iloc[2:].values
    nadir_mask_inner = (cur < prev) & (cur < nxt)
    nadir_indices = np.where(nadir_mask_inner)[0] + 1  # indices into gl

    for idx in nadir_indices:
        nadir_ts = ts.iloc[idx]
        nadir_gl = float(gl.iloc[idx])

        # Look-ahead window
        end_ts = nadir_ts + window_td
        window_mask = (ts > nadir_ts) & (ts <= end_ts)
        window_gl = gl[window_mask]

        if window_gl.empty:
            continue

        peak_val = float(window_gl.max())
        rise = peak_val - nadir_gl

        if rise < rise_threshold:
            continue

        peak_ts = ts[window_mask].iloc[window_gl.values.argmax()]
        duration = (peak_ts - nadir_ts).total_seconds() / 60.0

        excursions.append({
            "start_time": nadir_ts,
            "end_time": peak_ts,
            "duration_minutes": round(duration, 1),
            "nadir_glucose": round(nadir_gl, 1),
            "peak_glucose": round(peak_val, 1),
            "rise_mg_dl": round(rise, 1),
            "event_type": "postprandial_excursion",
        })

    if not excursions:
        return pd.DataFrame(
            columns=["start_time", "end_time", "duration_minutes",
                     "nadir_glucose", "peak_glucose", "rise_mg_dl", "event_type"]
        )

    return pd.DataFrame(excursions).sort_values("start_time").reset_index(drop=True)
