"""
glycosignal.plotting
====================

Matplotlib-based visualization utilities for CGM data.

All functions:
  - Return a ``(fig, ax)`` tuple (or ``(fig, axes)`` for multi-panel plots).
  - Never call ``plt.show()`` automatically -- the caller controls display.
  - Accept an optional ``ax`` argument to plot into an existing Axes.

Usage
-----
    >>> from glycosignal import plotting
    >>> fig, ax = plotting.plot_glucose_timeseries(df)
    >>> fig.savefig("timeseries.png", dpi=150)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .schemas import COL_GLUCOSE, COL_TIMESTAMP
from .utils import require_columns, require_dataframe


# Clinical glucose range reference bands (mg/dL)
_TIR_LOW = 70.0
_TIR_HIGH = 180.0
_HYPO2 = 54.0
_HYPER2 = 250.0

_BAND_COLOR_GREEN = "#c8e6c9"
_BAND_COLOR_YELLOW_LOW = "#fff9c4"
_BAND_COLOR_YELLOW_HIGH = "#fff9c4"
_BAND_COLOR_RED = "#ffcdd2"


def _get_matplotlib():
    """Import matplotlib lazily to keep import overhead low."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with parsed Timestamp and numeric Glucose, sorted."""
    require_dataframe(df, "df")
    require_columns(df, [COL_TIMESTAMP, COL_GLUCOSE])
    out = df[[COL_TIMESTAMP, COL_GLUCOSE]].copy()
    out[COL_TIMESTAMP] = pd.to_datetime(out[COL_TIMESTAMP], errors="coerce")
    out[COL_GLUCOSE] = pd.to_numeric(out[COL_GLUCOSE], errors="coerce")
    return out.dropna().sort_values(COL_TIMESTAMP).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Glucose time series
# ─────────────────────────────────────────────────────────────────────────────

def plot_glucose_timeseries(
    df: pd.DataFrame,
    subject: Optional[str] = None,
    show_tir_bands: bool = True,
    ax=None,
) -> tuple:
    """Plot glucose values over time as a line chart.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns.
    subject : str | None
        Optional subject label for the title.
    show_tir_bands : bool
        Shade clinical glucose range bands (hypo red, target green, hyper yellow).
    ax : matplotlib.axes.Axes | None
        Axes to plot into.  Creates a new figure if None.

    Returns
    -------
    tuple
        ``(fig, ax)`` -- matplotlib Figure and Axes objects.
    """
    plt = _get_matplotlib()
    data = _prep_df(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.get_figure()

    ts = data[COL_TIMESTAMP]
    gl = data[COL_GLUCOSE]

    if show_tir_bands:
        ax.axhspan(_HYPO2, _TIR_LOW, alpha=0.15, color=_BAND_COLOR_RED, label="Level 2 hypo (<54)")
        ax.axhspan(0, _HYPO2, alpha=0.25, color=_BAND_COLOR_RED)
        ax.axhspan(_TIR_LOW, _TIR_HIGH, alpha=0.12, color=_BAND_COLOR_GREEN, label="Target (70–180)")
        ax.axhspan(_TIR_HIGH, _HYPER2, alpha=0.12, color=_BAND_COLOR_YELLOW_HIGH, label="Level 1 hyper (180–250)")
        ax.axhspan(_HYPER2, max(float(gl.max()) + 20, _HYPER2 + 20),
                   alpha=0.20, color=_BAND_COLOR_RED, label="Level 2 hyper (>250)")

    ax.plot(ts, gl, color="#1565c0", linewidth=0.9, alpha=0.85)
    ax.axhline(_TIR_LOW, color="#e53935", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.axhline(_TIR_HIGH, color="#fb8c00", linewidth=0.7, linestyle="--", alpha=0.7)

    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_xlabel("Time")
    title = "Glucose Time Series"
    if subject:
        title = f"{title} — {subject}"
    ax.set_title(title)
    ax.set_ylim(bottom=max(0, float(gl.min()) - 10))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Daily overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_daily_overlay(
    df: pd.DataFrame,
    alpha: float = 0.3,
    show_tir_bands: bool = True,
    ax=None,
) -> tuple:
    """Overlay individual days on a single 24-hour axis.

    Each calendar day is plotted as a separate translucent line.  This reveals
    intraday patterns across multiple days.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns.
    alpha : float
        Transparency of individual day lines.  Default 0.3.
    show_tir_bands : bool
        Shade clinical glucose range bands.
    ax : matplotlib.axes.Axes | None
        Axes to plot into.  Creates a new figure if None.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    plt = _get_matplotlib()
    data = _prep_df(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    data["_minutes"] = (
        data[COL_TIMESTAMP].dt.hour * 60 + data[COL_TIMESTAMP].dt.minute
    )
    data["_date"] = data[COL_TIMESTAMP].dt.date

    if show_tir_bands:
        ax.axhspan(_TIR_LOW, _TIR_HIGH, alpha=0.12, color=_BAND_COLOR_GREEN)
        ax.axhline(_TIR_LOW, color="#e53935", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(_TIR_HIGH, color="#fb8c00", linewidth=0.8, linestyle="--", alpha=0.7)

    for date, day_data in data.groupby("_date"):
        ax.plot(
            day_data["_minutes"],
            day_data[COL_GLUCOSE],
            color="#1565c0",
            alpha=alpha,
            linewidth=0.8,
        )

    # Tick marks at 4-hour intervals
    xticks = list(range(0, 1441, 240))
    xlabels = [f"{h:02d}:00" for h in range(0, 25, 4)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Daily Glucose Overlay")
    ax.set_xlim(0, 1440)

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Ambulatory Glucose Profile (AGP)
# ─────────────────────────────────────────────────────────────────────────────

def plot_agp(
    df: pd.DataFrame,
    percentiles: tuple[float, ...] = (10, 25, 50, 75, 90),
    bin_minutes: int = 15,
    ax=None,
) -> tuple:
    """Plot the Ambulatory Glucose Profile (AGP).

    Bins glucose readings by time-of-day and plots the median with IQR and
    10th–90th percentile bands.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns.  Multiple days
        are needed for a meaningful AGP.
    percentiles : tuple of float
        Percentile levels to compute.  Default ``(10, 25, 50, 75, 90)``.
    bin_minutes : int
        Width of each time-of-day bin in minutes.  Default 15.
    ax : matplotlib.axes.Axes | None
        Axes to plot into.  Creates a new figure if None.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    plt = _get_matplotlib()
    data = _prep_df(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    # Bin by time-of-day
    minutes_of_day = data[COL_TIMESTAMP].dt.hour * 60 + data[COL_TIMESTAMP].dt.minute
    data["_bin"] = (minutes_of_day // bin_minutes) * bin_minutes

    grouped = data.groupby("_bin")[COL_GLUCOSE]
    bins = sorted(data["_bin"].unique())

    pct_data = {p: [] for p in percentiles}
    bin_centers = []

    for b in bins:
        values = grouped.get_group(b).dropna()
        if len(values) >= 3:
            for p in percentiles:
                pct_data[p].append(np.nanpercentile(values, p))
            bin_centers.append(b)

    if not bin_centers:
        ax.set_title("AGP (insufficient data)")
        return fig, ax

    x = np.array(bin_centers)
    pct_arrays = {p: np.array(pct_data[p]) for p in percentiles}

    # TIR reference bands
    ax.axhspan(_TIR_LOW, _TIR_HIGH, alpha=0.10, color=_BAND_COLOR_GREEN)
    ax.axhline(_TIR_LOW, color="#e53935", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(_TIR_HIGH, color="#fb8c00", linewidth=0.8, linestyle="--", alpha=0.6)

    sorted_pcts = sorted(percentiles)
    n_pcts = len(sorted_pcts)

    # Outer band: lowest and highest percentiles
    if n_pcts >= 2:
        ax.fill_between(x, pct_arrays[sorted_pcts[0]], pct_arrays[sorted_pcts[-1]],
                        alpha=0.15, color="#1565c0",
                        label=f"{sorted_pcts[0]}th–{sorted_pcts[-1]}th %ile")

    # Inner band: 2nd and 2nd-to-last percentiles (IQR-equivalent)
    if n_pcts >= 4:
        ax.fill_between(x, pct_arrays[sorted_pcts[1]], pct_arrays[sorted_pcts[-2]],
                        alpha=0.30, color="#1565c0",
                        label=f"{sorted_pcts[1]}th–{sorted_pcts[-2]}th %ile (IQR)")

    # Median: middle percentile
    mid_pct = sorted_pcts[n_pcts // 2]
    ax.plot(x, pct_arrays[mid_pct], color="#1565c0", linewidth=2.0,
            label=f"Median ({mid_pct}th %ile)")

    # X-axis formatting
    xticks = list(range(0, 1441, 240))
    xlabels = [f"{h:02d}:00" for h in range(0, 25, 4)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(0, max(bin_centers) + bin_minutes)
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Ambulatory Glucose Profile (AGP)")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_histogram(
    df: pd.DataFrame,
    bins: int = 50,
    show_tir_lines: bool = True,
    ax=None,
) -> tuple:
    """Plot a glucose value distribution histogram.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data with ``Timestamp`` and ``Glucose`` columns.
    bins : int
        Number of histogram bins.  Default 50.
    show_tir_lines : bool
        Draw vertical lines at 70 and 180 mg/dL.
    ax : matplotlib.axes.Axes | None
        Axes to plot into.  Creates a new figure if None.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    plt = _get_matplotlib()
    data = _prep_df(df)
    gl = data[COL_GLUCOSE].dropna()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.hist(gl, bins=bins, color="#1565c0", alpha=0.75, edgecolor="white", linewidth=0.5)

    if show_tir_lines:
        ax.axvline(_TIR_LOW, color="#e53935", linewidth=1.5, linestyle="--", label="70 mg/dL")
        ax.axvline(_TIR_HIGH, color="#fb8c00", linewidth=1.5, linestyle="--", label="180 mg/dL")
        ax.legend(fontsize=9)

    ax.set_xlabel("Glucose (mg/dL)")
    ax.set_ylabel("Count")
    ax.set_title("Glucose Distribution")

    fig.tight_layout()
    return fig, ax
