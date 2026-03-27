"""
glycosignal.windows
===================

Sliding-window creation for CGM data.

The primary output format is **long-format**: one row per (window, timepoint)
observation, with a ``window_id`` column identifying each window.  This
integrates naturally with :func:`~glycosignal.features.build_feature_map`.

For backward compatibility with wide-format workflows, use
:func:`pivot_windows_wide` to convert to the ``HH:MM``-column format.

Usage
-----
    >>> from glycosignal import windows, features
    >>> windowed = windows.create_sliding_windows(df, window_hours=24)
    >>> X = features.build_feature_map(windowed)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocessing import interpolate_cgm
from .schemas import COL_GLUCOSE, COL_TIMESTAMP, METADATA_COLUMNS
from .utils import require_columns, require_dataframe


# ─────────────────────────────────────────────────────────────────────────────
# Return type for create_sliding_windows
# ─────────────────────────────────────────────────────────────────────────────

class WindowResult(NamedTuple):
    """Return value of :func:`create_sliding_windows`.

    Attributes
    ----------
    windows : pd.DataFrame
        Long-format DataFrame.  Columns: ``window_id``, ``Timestamp``,
        ``Glucose``, plus any preserved metadata columns.
    metadata : dict
        Processing statistics (n_subjects, n_valid_windows, n_discarded, etc.).
    """

    windows: pd.DataFrame
    metadata: dict


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_window_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format a window as a human-readable label.

    Parameters
    ----------
    start : pd.Timestamp
        Window start time.
    end : pd.Timestamp
        Window end time (exclusive).

    Returns
    -------
    str
        Label like ``"2023-01-01/2023-01-02"``.
    """
    return f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"


def _format_window_params(window_hours: float, overlap_hours: float) -> str:
    """Format window parameters for use in filenames."""
    w = f"{int(window_hours)}h" if window_hours == int(window_hours) else f"{window_hours}h"
    o = f"{int(overlap_hours)}h" if overlap_hours == int(overlap_hours) else f"{overlap_hours}h"
    return f"{w}_{o}"


# ─────────────────────────────────────────────────────────────────────────────
# Core sliding window function
# ─────────────────────────────────────────────────────────────────────────────

def create_sliding_windows(
    df: pd.DataFrame,
    window_hours: float = 24.0,
    overlap_hours: float = 0.0,
    min_fraction: float = 0.7,
    group_col: str = "subject",
    id_cols: list[str] | tuple[str, ...] | None = None,
    tolerance_minutes: float = 2.5,
    interpolate: bool = True,
    max_gap_points: int = 12,
    show_progress: bool = True,
) -> WindowResult:
    """Transform a CGM DataFrame into long-format sliding windows.

    Each window is a contiguous time block of ``window_hours`` hours, stepped
    by ``window_hours - overlap_hours`` hours.  Windows are aligned to midnight
    of the first complete calendar day for each group.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns plus any grouping
        columns.  Run through :func:`~glycosignal.preprocessing.clean_cgm`
        before calling this function.
    window_hours : float
        Duration of each window in hours.  Default 24.
    overlap_hours : float
        Overlap between consecutive windows in hours.  Must be less than
        ``window_hours``.  Default 0.
    min_fraction : float
        Minimum fraction of the 5-minute grid points that must have observed
        data for a window to be kept (before interpolation).  Default 0.7
        (70%).  Range: 0.0–1.0.
    group_col : str
        Column used to group data per subject / recording.  Default
        ``"subject"``.  Use ``"filename"`` for per-file recordings.
    id_cols : list[str] | None
        Additional identifier columns to carry through to the output.  If None,
        defaults to ``(group_col,)``.
    tolerance_minutes : float
        Maximum distance (minutes) between an actual CGM reading and a 5-minute
        grid point.  Readings outside this tolerance are treated as missing.
        Default 2.5 (half a 5-minute interval).
    interpolate : bool
        Whether to fill short gaps with PCHIP interpolation after gridding.
        Default True.
    max_gap_points : int
        Maximum consecutive missing grid points to interpolate.  Default 12
        (= 1 hour at 5-min sampling).
    show_progress : bool
        Show a tqdm progress bar.  Default True.

    Returns
    -------
    WindowResult
        Named tuple ``(windows, metadata)``:
        - ``windows``: long-format DataFrame with columns ``window_id``,
          ``Timestamp``, ``Glucose``, and any id_cols.
        - ``metadata``: dict with processing statistics.

    Raises
    ------
    ValueError
        If ``overlap_hours >= window_hours``, ``min_fraction`` is out of range,
        or required columns are missing from *df*.
    """
    require_dataframe(df, "df")

    if overlap_hours >= window_hours:
        raise ValueError(
            f"overlap_hours ({overlap_hours}) must be less than window_hours ({window_hours})."
        )
    if not (0.0 <= min_fraction <= 1.0):
        raise ValueError(
            f"min_fraction must be between 0.0 and 1.0, got {min_fraction}."
        )

    require_columns(df, [COL_TIMESTAMP, COL_GLUCOSE])

    # Auto-add grouping column if missing (single-subject use case)
    if group_col not in df.columns:
        df = df.copy()
        df[group_col] = "default"

    if id_cols is None:
        id_cols = (group_col,)
    id_cols = list(id_cols)

    working = df.copy()
    working[COL_TIMESTAMP] = pd.to_datetime(working[COL_TIMESTAMP], errors="coerce")
    working[COL_GLUCOSE] = pd.to_numeric(working[COL_GLUCOSE], errors="coerce")
    working = working.dropna(subset=[COL_TIMESTAMP, COL_GLUCOSE, group_col])

    # Grid parameters
    grid_freq = "5min"
    points_per_window = int(window_hours * 60 / 5)
    min_observed = int(np.ceil(min_fraction * points_per_window))
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    step = pd.Timedelta(hours=window_hours - overlap_hours)
    window_delta = pd.Timedelta(hours=window_hours)

    all_rows: list[dict] = []
    n_discarded = 0
    n_dropped_min_frac = 0
    n_dropped_after_interp = 0
    groups_seen: set = set()

    iterator = working.groupby(group_col)
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"Creating windows ({group_col})")

    for group_value, grp in iterator:
        groups_seen.add(group_value)
        grp = grp.sort_values(COL_TIMESTAMP).reset_index(drop=True)

        # Collect extra id_col values (take first occurrence)
        extra_meta = {}
        for col in id_cols:
            if col == group_col:
                extra_meta[col] = group_value
            elif col in grp.columns:
                extra_meta[col] = grp[col].iloc[0]
            else:
                extra_meta[col] = None

        first_ts = grp[COL_TIMESTAMP].iloc[0]
        first_midnight = first_ts.normalize()

        # Discard first partial calendar day
        if first_ts != first_midnight:
            first_midnight += pd.Timedelta(days=1)
            n_discarded += 1

        last_ts = grp[COL_TIMESTAMP].iloc[-1]
        if first_midnight > last_ts:
            continue

        # Pre-round readings to nearest 5 min for fast grid matching
        grp = grp.copy()
        grp["_rounded"] = grp[COL_TIMESTAMP].dt.round(grid_freq)
        grp["_dist"] = (grp[COL_TIMESTAMP] - grp["_rounded"]).abs()
        grp = grp.loc[grp["_dist"] <= tolerance]

        if grp.empty:
            continue

        window_start = first_midnight
        while window_start <= last_ts:
            window_end = window_start + window_delta
            targets = pd.date_range(
                start=window_start, periods=points_per_window, freq=grid_freq
            )

            # Select and average readings within this window
            mask = (grp["_rounded"] >= window_start) & (grp["_rounded"] < window_end)
            w = grp.loc[mask]
            resampled = w.groupby("_rounded")[COL_GLUCOSE].mean().reindex(targets)

            non_na = int(resampled.notna().sum())
            if non_na < min_observed:
                n_dropped_min_frac += 1
                window_start += step
                continue

            # Optional PCHIP interpolation for short gaps
            if interpolate:
                glucose_arr = resampled.values.copy()
                from .preprocessing import _fill_gaps_pchip
                glucose_arr = _fill_gaps_pchip(glucose_arr, max_gap_points)
                resampled = pd.Series(glucose_arr, index=targets)

            # Drop window if still contains NaN after interpolation
            if resampled.isna().any():
                n_dropped_after_interp += 1
                window_start += step
                continue

            # Emit one long-format row per time point
            window_id = (
                f"{group_value}_{window_start.strftime('%Y-%m-%d')}"
            )
            for ts, gl in zip(targets, resampled.values):
                row = {"window_id": window_id, COL_TIMESTAMP: ts, COL_GLUCOSE: gl}
                row.update(extra_meta)
                row["date"] = window_start.strftime("%Y-%m-%d")
                all_rows.append(row)

            window_start += step

    # Assemble output
    if not all_rows:
        cols = ["window_id", COL_TIMESTAMP, COL_GLUCOSE] + id_cols + ["date"]
        windows_df = pd.DataFrame(columns=cols)
    else:
        windows_df = pd.DataFrame(all_rows)
        # Reorder columns: window_id, id_cols, date, Timestamp, Glucose
        front = ["window_id"] + id_cols + ["date", COL_TIMESTAMP, COL_GLUCOSE]
        remaining = [c for c in windows_df.columns if c not in front]
        windows_df = windows_df[front + remaining]

    meta = {
        "n_groups": len(groups_seen),
        "n_valid_windows": int(windows_df["window_id"].nunique()) if not windows_df.empty else 0,
        "n_discarded_partial_days": n_discarded,
        "n_dropped_min_fraction": n_dropped_min_frac,
        "n_dropped_after_interpolation": n_dropped_after_interp,
    }

    return WindowResult(windows=windows_df, metadata=meta)


# ─────────────────────────────────────────────────────────────────────────────
# Format conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def pivot_windows_wide(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format windowed output to the legacy wide format.

    Each window becomes a single row, with time-of-day columns like
    ``00:00``, ``00:05``, ..., ``23:55``.  This format is accepted by
    :func:`~glycosignal.features.build_feature_map_wide`.

    Parameters
    ----------
    windows_df : pd.DataFrame
        Long-format output from :func:`create_sliding_windows`.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with one row per window.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    require_dataframe(windows_df, "windows_df")
    require_columns(windows_df, ["window_id", COL_TIMESTAMP, COL_GLUCOSE])

    # Compute time-of-day offset from start of window
    ts = pd.to_datetime(windows_df[COL_TIMESTAMP])
    day = ts.dt.floor("D")
    offset_min = ((ts - day).dt.total_seconds() // 60).astype(int)
    hours, mins = divmod(offset_min, 60)
    time_col = hours.astype(str).str.zfill(2) + ":" + mins.astype(str).str.zfill(2)

    temp = windows_df[["window_id", COL_GLUCOSE]].copy()
    temp["_time_col"] = time_col

    wide = temp.pivot_table(
        index="window_id", columns="_time_col", values=COL_GLUCOSE, aggfunc="first"
    )
    wide.columns.name = None
    wide = wide.reset_index()

    # Re-attach metadata columns
    meta_cols = [c for c in windows_df.columns
                 if c not in ("window_id", COL_TIMESTAMP, COL_GLUCOSE)]
    meta = windows_df.drop_duplicates("window_id")[["window_id"] + meta_cols]
    return meta.merge(wide, on="window_id").reset_index(drop=True)


def windows_to_records(
    windows_df: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    """Split long-format windowed data into a list of (window_id, sub-df) tuples.

    Parameters
    ----------
    windows_df : pd.DataFrame
        Long-format output from :func:`create_sliding_windows`.

    Returns
    -------
    list of (str, pd.DataFrame)
        Each element is ``(window_id, df)`` where *df* contains only the
        ``Timestamp`` and ``Glucose`` columns for that window.
    """
    require_dataframe(windows_df, "windows_df")
    require_columns(windows_df, ["window_id", COL_TIMESTAMP, COL_GLUCOSE])

    result = []
    for wid, grp in windows_df.groupby("window_id"):
        sub = grp[[COL_TIMESTAMP, COL_GLUCOSE]].reset_index(drop=True)
        result.append((str(wid), sub))
    return result
