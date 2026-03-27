"""
cgmquantify_stuart — CGM glycemic-feature computation
======================================================

All public feature functions accept either:
  - A ``PreparedCGMData`` object (returned by ``prepare(df)``), or
  - A raw ``pd.DataFrame`` with columns **Timestamp** and **Glucose**.

When computing many features on the same window, call ``prepare(df)`` once
and pass the result to every function — this avoids redundant validation
and timestamp parsing.

Time-based metrics (TIR, TOR, TAT, TBT, GRI, …) use actual timestamp
intervals instead of assuming a fixed sampling rate.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PreparedCGMData:
    """Pre-validated CGM data ready for feature extraction."""
    glucose: np.ndarray       # glucose values (float64, no NaNs)
    timestamps: np.ndarray    # datetime64[ns] array, sorted
    weights: np.ndarray       # per-reading time weight in minutes
    total_minutes: float      # sum of weights (≈ monitoring duration)
    n_readings: int


def _validate(df):
    """Return a sorted, NaN-free DataFrame with Timestamp and Glucose."""
    required = {"Timestamp", "Glucose"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[["Timestamp", "Glucose"]].copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out["Glucose"] = pd.to_numeric(out["Glucose"], errors="coerce")
    out = out.dropna().sort_values("Timestamp").reset_index(drop=True)
    return out


def _time_weights(timestamps):
    """Per-reading time weights (minutes) from forward timestamp differences.

    weight[i] = timestamp[i+1] − timestamp[i]   for i < n-1
    weight[n-1] = weight[n-2]                    (reuse last interval)
    """
    ts = pd.to_datetime(pd.Series(timestamps, copy=False)).reset_index(drop=True)
    n = len(ts)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([5.0])

    diffs_sec = np.diff(ts.values).astype("timedelta64[s]").astype(np.float64)
    weights = np.empty(n, dtype=np.float64)
    weights[:-1] = diffs_sec / 60.0
    weights[-1] = weights[-2]
    return np.maximum(weights, 0.0)


def prepare(df):
    """Validate *df* and return a ``PreparedCGMData`` for fast feature extraction."""
    vdf = _validate(df)
    if vdf.empty:
        return PreparedCGMData(
            glucose=np.array([], dtype=np.float64),
            timestamps=np.array([], dtype="datetime64[ns]"),
            weights=np.array([], dtype=np.float64),
            total_minutes=0.0,
            n_readings=0,
        )
    weights = _time_weights(vdf["Timestamp"])
    return PreparedCGMData(
        glucose=vdf["Glucose"].values.astype(np.float64),
        timestamps=vdf["Timestamp"].values,
        weights=weights,
        total_minutes=float(weights.sum()),
        n_readings=len(vdf),
    )


def _ensure_prepared(data):
    """Accept a DataFrame or PreparedCGMData; always return PreparedCGMData."""
    if isinstance(data, PreparedCGMData):
        return data
    return prepare(data)


# ─────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────

def mean_glucose(data):
    """Interday mean glucose."""
    d = _ensure_prepared(data)
    return float(np.nanmean(d.glucose)) if d.n_readings else np.nan


def median_glucose(data):
    """Interday median glucose."""
    d = _ensure_prepared(data)
    return float(np.nanmedian(d.glucose)) if d.n_readings else np.nan


def min_glucose(data):
    """Minimum glucose."""
    d = _ensure_prepared(data)
    return float(np.nanmin(d.glucose)) if d.n_readings else np.nan


def max_glucose(data):
    """Maximum glucose."""
    d = _ensure_prepared(data)
    return float(np.nanmax(d.glucose)) if d.n_readings else np.nan


def q1_glucose(data):
    """First quartile (25th percentile) of glucose."""
    d = _ensure_prepared(data)
    return float(np.nanpercentile(d.glucose, 25)) if d.n_readings else np.nan


def q3_glucose(data):
    """Third quartile (75th percentile) of glucose."""
    d = _ensure_prepared(data)
    return float(np.nanpercentile(d.glucose, 75)) if d.n_readings else np.nan


def summary(data):
    """Return (mean, median, min, max, Q1, Q3) — backward-compatible tuple."""
    d = _ensure_prepared(data)
    return (
        mean_glucose(d),
        median_glucose(d),
        min_glucose(d),
        max_glucose(d),
        q1_glucose(d),
        q3_glucose(d),
    )


# ─────────────────────────────────────────────────────────────────────
# Variability metrics
# ─────────────────────────────────────────────────────────────────────

def sd(data):
    """Standard deviation of glucose."""
    d = _ensure_prepared(data)
    return float(np.nanstd(d.glucose)) if d.n_readings else np.nan


def cv(data):
    """Coefficient of variation of glucose (%)."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    m = float(np.nanmean(d.glucose))
    return float((np.nanstd(d.glucose) / m) * 100) if m != 0 else np.nan


def J_index(data):
    """J-index: 0.001 × (mean + SD)²."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    return float(0.001 * (np.nanmean(d.glucose) + np.nanstd(d.glucose)) ** 2)


def CONGA24(data, window_hours=24, tolerance_minutes=30):
    """Continuous Overall Net Glycemic Action over *window_hours*.

    For each glucose reading at time *t*, finds the reading closest to
    *t* − ``window_hours`` (within ±``tolerance_minutes``).  Returns the
    standard deviation of all such paired differences G(t) − G(t−Δ).

    A ``tolerance_minutes`` of 30 means a ±30-minute window around the
    target look-back time is accepted; tighten or loosen as needed for
    your CGM sampling rate.

    Returns ``np.nan`` when fewer than two valid pairs are found.

    .. note:: For sub-daily windows the default 24-hour look-back will
       never find a match.  Pass a shorter ``window_hours`` (e.g. 1 or 4)
       for intraday CONGA variants.
    """
    d = _ensure_prepared(data)
    if d.n_readings < 2:
        return np.nan

    # Work in integer nanoseconds for fully-vectorised search
    ts_int = d.timestamps.astype("datetime64[ns]").astype(np.int64)
    window_ns = int(window_hours * 3600e9)
    tol_ns = int(tolerance_minutes * 60e9)

    targets = ts_int - window_ns
    idx = np.searchsorted(ts_int, targets).clip(0, d.n_readings - 1)
    idx_prev = np.maximum(idx - 1, 0)

    delta_idx = np.abs(ts_int[idx] - targets)
    delta_prev = np.abs(ts_int[idx_prev] - targets)

    use_prev = delta_prev < delta_idx
    best_j = np.where(use_prev, idx_prev, idx)
    best_delta = np.where(use_prev, delta_prev, delta_idx)

    i_arr = np.arange(d.n_readings)
    valid = (best_delta <= tol_ns) & (best_j != i_arr)

    if valid.sum() < 2:
        return np.nan

    diffs = d.glucose[valid] - d.glucose[best_j[valid]]
    return float(np.std(diffs))


# ─────────────────────────────────────────────────────────────────────
# Time-in-range metrics  (minutes, using actual timestamps)
# ─────────────────────────────────────────────────────────────────────

def TIR(data, sd=1):
    """Time in range (mean ± sd × std), in minutes."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    m, s = float(np.nanmean(d.glucose)), float(np.nanstd(d.glucose))
    mask = (d.glucose >= m - sd * s) & (d.glucose <= m + sd * s)
    return float(d.weights[mask].sum())


def TOR(data, sd=1):
    """Time outside range (mean ± sd × std), in minutes."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    m, s = float(np.nanmean(d.glucose)), float(np.nanstd(d.glucose))
    mask = (d.glucose < m - sd * s) | (d.glucose > m + sd * s)
    return float(d.weights[mask].sum())


def TIR_lo_hi(data, upper=180, lower=70):
    """Time in a fixed range [lower, upper], in minutes."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    mask = (d.glucose >= lower) & (d.glucose <= upper)
    return float(d.weights[mask].sum())


def TAT(data, threshold):
    """Time above threshold, in minutes."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    mask = d.glucose >= float(threshold)
    return float(d.weights[mask].sum())


def TBT(data, threshold):
    """Time below threshold, in minutes."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    mask = d.glucose <= float(threshold)
    return float(d.weights[mask].sum())


def POR(data, sd=1):
    """Percent time outside range (mean ± sd × std)."""
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    return float(TOR(d, sd=sd) / d.total_minutes * 100)


def PIR(data, sd=1):
    """Percent time in range (mean ± sd × std)."""
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    return float(TIR(d, sd=sd) / d.total_minutes * 100)


# ─────────────────────────────────────────────────────────────────────
# Risk indices  (vectorised — no Python loops)
# ─────────────────────────────────────────────────────────────────────

def _bgi_components(glucose):
    """Compute per-reading BG risk values (rl, rh) from a glucose array.

    Readings ≤ 0 are silently excluded (log requires positive values).
    """
    gl = np.asarray(glucose, dtype=np.float64)
    gl = gl[gl > 0]
    if len(gl) == 0:
        return np.array([]), np.array([])
    f = (np.log(gl) ** 1.084) - 5.381
    rl = np.where(f <= 0, 22.77 * f ** 2, 0.0)
    rh = np.where(f > 0, 22.77 * f ** 2, 0.0)
    return rl, rh


def LBGI(data):
    """Low Blood Glucose Index."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    rl, _ = _bgi_components(d.glucose)
    return float(np.mean(rl)) if len(rl) else np.nan


def HBGI(data):
    """High Blood Glucose Index."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    _, rh = _bgi_components(d.glucose)
    return float(np.mean(rh)) if len(rh) else np.nan


def ADRR(data):
    """Average Daily Risk Range (max low-risk + max high-risk)."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    rl, rh = _bgi_components(d.glucose)
    if len(rl) == 0:
        return np.nan
    return float(np.max(rl) + np.max(rh))


# ─────────────────────────────────────────────────────────────────────
# Excursion metrics
# ─────────────────────────────────────────────────────────────────────

def MGE(data, sd=1):
    """Mean Glucose Excursion — mean of readings outside (mean ± sd × std)."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    m, s = float(np.nanmean(d.glucose)), float(np.nanstd(d.glucose))
    mask = (d.glucose < m - sd * s) | (d.glucose > m + sd * s)
    excursion = d.glucose[mask]
    return float(np.nanmean(excursion)) if len(excursion) else np.nan


def MGN(data, sd=1):
    """Mean Glucose Normal — mean of readings inside (mean ± sd × std)."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    m, s = float(np.nanmean(d.glucose)), float(np.nanstd(d.glucose))
    mask = (d.glucose >= m - sd * s) & (d.glucose <= m + sd * s)
    normal = d.glucose[mask]
    return float(np.nanmean(normal)) if len(normal) else np.nan


def MAGE(data, sd=1):
    """Mean Amplitude of Glucose Excursions (MAGE).  Vectorised."""
    d = _ensure_prepared(data)
    gl = d.glucose
    n = len(gl)
    if n < 3:
        return 0.0

    threshold = sd * float(np.nanstd(gl))

    # Vectorised local turning-point detection
    prev, cur, nxt = gl[:-2], gl[1:-1], gl[2:]
    is_peak = (cur > prev) & (cur > nxt)
    is_nadir = (cur < prev) & (cur < nxt)
    tp_mask = is_peak | is_nadir

    tp_idx = np.where(tp_mask)[0] + 1          # indices into gl
    if len(tp_idx) < 2:
        return 0.0

    tp_vals = gl[tp_idx]
    tp_is_peak = is_peak[tp_idx - 1]           # True = peak, False = nadir

    # Keep only consecutive pairs that alternate peak/nadir
    alternates = tp_is_peak[:-1] != tp_is_peak[1:]
    amplitudes = np.abs(np.diff(tp_vals))
    valid = alternates & (amplitudes >= threshold)

    return float(np.mean(amplitudes[valid])) if valid.any() else 0.0


# ─────────────────────────────────────────────────────────────────────
# Composite scores
# ─────────────────────────────────────────────────────────────────────

def GRI(data):
    """Glucose Risk Index (Klonoff et al. 2023).

    GRI = 3.0 × %TBR<54  +  2.4 × %TBR<70  +  1.6 × %TAR>250  +  0.8 × %TAR>180

    All components are **percentages** of total monitoring time.  The final
    score is capped at 100.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    total = d.total_minutes
    pct_tar180 = TAT(d, threshold=180) / total * 100
    pct_tar250 = TAT(d, threshold=250) / total * 100
    pct_tbr70  = TBT(d, threshold=70)  / total * 100
    pct_tbr54  = TBT(d, threshold=54)  / total * 100
    raw = 3.0 * pct_tbr54 + 2.4 * pct_tbr70 + 1.6 * pct_tar250 + 0.8 * pct_tar180
    return float(min(raw, 100.0))


# ─────────────────────────────────────────────────────────────────────
# Peak analysis
# ─────────────────────────────────────────────────────────────────────

def count_peaks(data, threshold):
    """Count episodes where glucose rises above *threshold* and returns below.

    Vectorised: counts rising-edge transitions on the boolean above-mask.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0
    above = d.glucose > float(threshold)
    # Prepend False so that starting above counts as an entry
    entries = np.diff(np.concatenate(([False], above)).astype(np.int8))
    return int(np.sum(entries == 1))


def count_peaks_in_range(data, lower, upper):
    """Count episodes where glucose enters [lower, upper] and exits.

    Vectorised: counts rising-edge transitions on the boolean inside-mask.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0
    lower, upper = float(lower), float(upper)
    inside = (d.glucose >= lower) & (d.glucose <= upper)
    entries = np.diff(np.concatenate(([False], inside)).astype(np.int8))
    return int(np.sum(entries == 1))
