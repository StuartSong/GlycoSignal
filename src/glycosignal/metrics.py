"""
glycosignal.metrics
===================

Individual glycemic metric functions and grouped summary helpers.

Every function accepts either a ``pd.DataFrame`` (with ``Timestamp`` and
``Glucose`` columns) or a :class:`~glycosignal.schemas.PreparedCGMData` object.
When computing many features on the same dataset, call
:func:`~glycosignal.schemas.prepare` once and pass the result:

    >>> from glycosignal.schemas import prepare
    >>> from glycosignal import metrics
    >>> prepared = prepare(df)
    >>> metrics.mean_glucose(prepared)
    >>> metrics.cv(prepared)

Time-based metrics (TIR, TAR, TBR, GRI, ...) use actual timestamp intervals
derived from the data rather than assuming a fixed sampling rate.

Naming conventions
------------------
- ``*_minutes`` suffixed functions return time in minutes.
- ``*_percent`` suffixed functions return percentage of total monitoring time.
- Risk indices (lbgi, hbgi, adrr, gri) follow their published definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import CGMInput, PreparedCGMData, _ensure_prepared


# ─────────────────────────────────────────────────────────────────────────────
# Basic statistics
# ─────────────────────────────────────────────────────────────────────────────

def mean_glucose(data: CGMInput) -> float:
    """Mean glucose value (mg/dL).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
        CGM data.

    Returns
    -------
    float
        Mean glucose in mg/dL, or ``np.nan`` if no valid readings.
    """
    d = _ensure_prepared(data)
    return float(np.nanmean(d.glucose)) if d.n_readings else np.nan


def median_glucose(data: CGMInput) -> float:
    """Median glucose value (mg/dL).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        Median glucose in mg/dL, or ``np.nan`` if no valid readings.
    """
    d = _ensure_prepared(data)
    return float(np.nanmedian(d.glucose)) if d.n_readings else np.nan


def min_glucose(data: CGMInput) -> float:
    """Minimum glucose value (mg/dL).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        Minimum glucose in mg/dL, or ``np.nan`` if no valid readings.
    """
    d = _ensure_prepared(data)
    return float(np.nanmin(d.glucose)) if d.n_readings else np.nan


def max_glucose(data: CGMInput) -> float:
    """Maximum glucose value (mg/dL).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        Maximum glucose in mg/dL, or ``np.nan`` if no valid readings.
    """
    d = _ensure_prepared(data)
    return float(np.nanmax(d.glucose)) if d.n_readings else np.nan


def q1_glucose(data: CGMInput) -> float:
    """First quartile (25th percentile) of glucose (mg/dL).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        25th percentile in mg/dL, or ``np.nan`` if no valid readings.
    """
    d = _ensure_prepared(data)
    return float(np.nanpercentile(d.glucose, 25)) if d.n_readings else np.nan


def q3_glucose(data: CGMInput) -> float:
    """Third quartile (75th percentile) of glucose (mg/dL).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        75th percentile in mg/dL, or ``np.nan`` if no valid readings.
    """
    d = _ensure_prepared(data)
    return float(np.nanpercentile(d.glucose, 75)) if d.n_readings else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Variability metrics
# ─────────────────────────────────────────────────────────────────────────────

def sd(data: CGMInput) -> float:
    """Standard deviation of glucose (mg/dL).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        SD in mg/dL, or ``np.nan`` if no valid readings.
    """
    d = _ensure_prepared(data)
    return float(np.nanstd(d.glucose)) if d.n_readings else np.nan


def cv(data: CGMInput) -> float:
    """Coefficient of variation of glucose (%).

    Defined as ``(SD / mean) * 100``.  The CV is a dimensionless measure of
    glucose variability; values ≥ 36% are generally considered high risk.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        CV in percent, or ``np.nan`` if mean is zero or no valid readings.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    m = float(np.nanmean(d.glucose))
    return float((np.nanstd(d.glucose) / m) * 100) if m != 0 else np.nan


def j_index(data: CGMInput) -> float:
    """J-index: ``0.001 × (mean + SD)²``.

    A composite measure of glycemic control that penalizes both high mean
    glucose and high variability.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        J-index value, or ``np.nan`` if no valid readings.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    return float(0.001 * (np.nanmean(d.glucose) + np.nanstd(d.glucose)) ** 2)


def conga24(
    data: CGMInput,
    window_hours: float = 24,
    tolerance_minutes: float = 30,
) -> float:
    """Continuous Overall Net Glycemic Action (CONGA).

    For each glucose reading at time *t*, finds the reading closest to
    *t* − ``window_hours`` (within ±``tolerance_minutes``).  Returns the
    standard deviation of all paired differences G(t) − G(t−Δ).

    A ``tolerance_minutes`` of 30 means a ±30-minute window around the
    target look-back time is accepted.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
        CGM data spanning at least ``window_hours`` in time for non-NaN results.
    window_hours : float
        Look-back window in hours (default 24 for CONGA24).
    tolerance_minutes : float
        Acceptable deviation from the exact look-back target in minutes.

    Returns
    -------
    float
        CONGA value (SD of paired differences), or ``np.nan`` when fewer than
        two valid pairs are found.

    Notes
    -----
    For sub-daily windows the default 24-hour look-back will never find a match.
    Pass a shorter ``window_hours`` (e.g. 1 or 4) for intraday CONGA variants.
    """
    d = _ensure_prepared(data)
    if d.n_readings < 2:
        return np.nan

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


# ─────────────────────────────────────────────────────────────────────────────
# Time-in-range: minutes (using actual timestamps)
# ─────────────────────────────────────────────────────────────────────────────

def time_in_range_minutes(
    data: CGMInput,
    low: float = 70.0,
    high: float = 180.0,
) -> float:
    """Minutes spent within a fixed glucose range [low, high].

    Uses actual timestamp intervals, not assumed sampling rate.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    low : float
        Lower bound (inclusive), mg/dL.  Default 70.
    high : float
        Upper bound (inclusive), mg/dL.  Default 180.

    Returns
    -------
    float
        Time in range in minutes.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    mask = (d.glucose >= float(low)) & (d.glucose <= float(high))
    return float(d.weights[mask].sum())


def time_in_range_percent(
    data: CGMInput,
    low: float = 70.0,
    high: float = 180.0,
) -> float:
    """Percentage of monitoring time within a fixed glucose range [low, high].

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    low : float
        Lower bound (inclusive), mg/dL.  Default 70.
    high : float
        Upper bound (inclusive), mg/dL.  Default 180.

    Returns
    -------
    float
        Time in range as percentage (0–100), or ``np.nan`` if no monitoring time.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    return float(time_in_range_minutes(d, low=low, high=high) / d.total_minutes * 100)


def time_below_range_minutes(
    data: CGMInput,
    threshold: float = 70.0,
) -> float:
    """Minutes spent with glucose below *threshold*.

    Uses actual timestamp intervals, not assumed sampling rate.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    threshold : float
        Upper bound (inclusive) for "below range", mg/dL.  Default 70.

    Returns
    -------
    float
        Time below threshold in minutes.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    mask = d.glucose <= float(threshold)
    return float(d.weights[mask].sum())


def time_below_range_percent(
    data: CGMInput,
    threshold: float = 70.0,
) -> float:
    """Percentage of monitoring time with glucose below *threshold*.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    threshold : float
        Upper bound for "below range", mg/dL.  Default 70.

    Returns
    -------
    float
        Percent time below threshold, or ``np.nan`` if no monitoring time.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    return float(time_below_range_minutes(d, threshold=threshold) / d.total_minutes * 100)


def time_above_range_minutes(
    data: CGMInput,
    threshold: float = 180.0,
) -> float:
    """Minutes spent with glucose above *threshold*.

    Uses actual timestamp intervals, not assumed sampling rate.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    threshold : float
        Lower bound (inclusive) for "above range", mg/dL.  Default 180.

    Returns
    -------
    float
        Time above threshold in minutes.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    mask = d.glucose >= float(threshold)
    return float(d.weights[mask].sum())


def time_above_range_percent(
    data: CGMInput,
    threshold: float = 180.0,
) -> float:
    """Percentage of monitoring time with glucose above *threshold*.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    threshold : float
        Lower bound for "above range", mg/dL.  Default 180.

    Returns
    -------
    float
        Percent time above threshold, or ``np.nan`` if no monitoring time.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    return float(time_above_range_minutes(d, threshold=threshold) / d.total_minutes * 100)


def time_outside_range_minutes(
    data: CGMInput,
    low: float = 70.0,
    high: float = 180.0,
) -> float:
    """Minutes spent outside a fixed glucose range [low, high].

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    low : float
        Lower bound (inclusive) of in-range, mg/dL.  Default 70.
    high : float
        Upper bound (inclusive) of in-range, mg/dL.  Default 180.

    Returns
    -------
    float
        Time outside range in minutes.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    mask = (d.glucose < float(low)) | (d.glucose > float(high))
    return float(d.weights[mask].sum())


def time_outside_range_percent(
    data: CGMInput,
    low: float = 70.0,
    high: float = 180.0,
) -> float:
    """Percentage of monitoring time outside a fixed glucose range [low, high].

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    low : float
        Lower bound of in-range, mg/dL.  Default 70.
    high : float
        Upper bound of in-range, mg/dL.  Default 180.

    Returns
    -------
    float
        Percent time outside range, or ``np.nan`` if no monitoring time.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    return float(time_outside_range_minutes(d, low=low, high=high) / d.total_minutes * 100)


# Legacy SD-band based TIR (kept for backward compatibility)
def _tir_sd_minutes(data: CGMInput, sd_multiplier: float = 1.0) -> float:
    """Time in range defined as mean ± sd_multiplier × SD, in minutes."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    m, s = float(np.nanmean(d.glucose)), float(np.nanstd(d.glucose))
    mask = (d.glucose >= m - sd_multiplier * s) & (d.glucose <= m + sd_multiplier * s)
    return float(d.weights[mask].sum())


def _tor_sd_minutes(data: CGMInput, sd_multiplier: float = 1.0) -> float:
    """Time outside range defined as mean ± sd_multiplier × SD, in minutes."""
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0.0
    m, s = float(np.nanmean(d.glucose)), float(np.nanstd(d.glucose))
    mask = (d.glucose < m - sd_multiplier * s) | (d.glucose > m + sd_multiplier * s)
    return float(d.weights[mask].sum())


# ─────────────────────────────────────────────────────────────────────────────
# Risk indices (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _bgi_components(glucose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-reading Blood Glucose risk values (rl, rh).

    Readings ≤ 0 are excluded (log requires positive values).

    Based on Kovatchev et al. (2001).  The symmetric function f transforms
    glucose on a logarithmic scale, and rl / rh penalize low and high values.
    """
    gl = np.asarray(glucose, dtype=np.float64)
    gl = gl[gl > 0]
    if len(gl) == 0:
        return np.array([]), np.array([])
    f = (np.log(gl) ** 1.084) - 5.381
    rl = np.where(f <= 0, 22.77 * f ** 2, 0.0)
    rh = np.where(f > 0, 22.77 * f ** 2, 0.0)
    return rl, rh


def lbgi(data: CGMInput) -> float:
    """Low Blood Glucose Index (LBGI).

    Higher values indicate greater risk of hypoglycemia.
    Values > 2.5 are considered elevated risk.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        LBGI value, or ``np.nan`` if insufficient data.

    References
    ----------
    Kovatchev, B.P. et al. (2001).  Symmetrization of the blood glucose
    measurement scale and its applications.  Diabetes Care, 24(11), 1936-1941.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    rl, _ = _bgi_components(d.glucose)
    return float(np.mean(rl)) if len(rl) else np.nan


def hbgi(data: CGMInput) -> float:
    """High Blood Glucose Index (HBGI).

    Higher values indicate greater risk of hyperglycemia.
    Values > 4.5 are considered elevated risk.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        HBGI value, or ``np.nan`` if insufficient data.

    References
    ----------
    Kovatchev, B.P. et al. (2001).  Symmetrization of the blood glucose
    measurement scale and its applications.  Diabetes Care, 24(11), 1936-1941.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    _, rh = _bgi_components(d.glucose)
    return float(np.mean(rh)) if len(rh) else np.nan


def adrr(data: CGMInput) -> float:
    """Average Daily Risk Range (ADRR).

    Computed as the sum of the maximum low-risk and maximum high-risk values
    from the BGI components.  Higher values indicate greater overall glycemic risk.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        ADRR value, or ``np.nan`` if insufficient data.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    rl, rh = _bgi_components(d.glucose)
    if len(rl) == 0:
        return np.nan
    return float(np.max(rl) + np.max(rh))


def gri(data: CGMInput) -> float:
    """Glucose Risk Index (GRI).

    GRI = 3.0 × %TBR<54  +  2.4 × %TBR<70  +  1.6 × %TAR>250  +  0.8 × %TAR>180

    All components are percentages of total monitoring time.  The final score
    is capped at 100.  Higher GRI indicates greater glycemic risk.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    float
        GRI score (0–100), or ``np.nan`` if no monitoring time.

    References
    ----------
    Klonoff, D.C. et al. (2023).  A glycemia risk index (GRI) of hypoglycemia
    and hyperglycemia for continuous glucose monitoring validated by clinician
    ratings.  Journal of Diabetes Science and Technology, 17(5), 1131-1140.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    total = d.total_minutes
    pct_tar180 = time_above_range_minutes(d, threshold=180) / total * 100
    pct_tar250 = time_above_range_minutes(d, threshold=250) / total * 100
    pct_tbr70 = time_below_range_minutes(d, threshold=70) / total * 100
    pct_tbr54 = time_below_range_minutes(d, threshold=54) / total * 100
    raw = 3.0 * pct_tbr54 + 2.4 * pct_tbr70 + 1.6 * pct_tar250 + 0.8 * pct_tar180
    return float(min(raw, 100.0))


# ─────────────────────────────────────────────────────────────────────────────
# Excursion metrics
# ─────────────────────────────────────────────────────────────────────────────

def mean_glucose_excursion(data: CGMInput, sd_multiplier: float = 1.0) -> float:
    """Mean glucose value of readings *outside* (mean ± sd_multiplier × SD).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    sd_multiplier : float
        Number of standard deviations to define the "normal" range.

    Returns
    -------
    float
        Mean of excursion readings, or ``np.nan`` if none exist.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    m, s = float(np.nanmean(d.glucose)), float(np.nanstd(d.glucose))
    mask = (d.glucose < m - sd_multiplier * s) | (d.glucose > m + sd_multiplier * s)
    excursion = d.glucose[mask]
    return float(np.nanmean(excursion)) if len(excursion) else np.nan


def mean_glucose_normal(data: CGMInput, sd_multiplier: float = 1.0) -> float:
    """Mean glucose value of readings *inside* (mean ± sd_multiplier × SD).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    sd_multiplier : float
        Number of standard deviations to define the "normal" range.

    Returns
    -------
    float
        Mean of normal readings, or ``np.nan`` if none exist.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return np.nan
    m, s = float(np.nanmean(d.glucose)), float(np.nanstd(d.glucose))
    mask = (d.glucose >= m - sd_multiplier * s) & (d.glucose <= m + sd_multiplier * s)
    normal = d.glucose[mask]
    return float(np.nanmean(normal)) if len(normal) else np.nan


def mage(data: CGMInput, sd_multiplier: float = 1.0) -> float:
    """Mean Amplitude of Glucose Excursions (MAGE).

    Computes amplitudes of alternating peak-nadir pairs that exceed
    ``sd_multiplier × SD``.  Returns the mean of qualifying amplitudes.

    This vectorised implementation detects local turning points and filters
    to consecutive alternating peak/nadir pairs.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    sd_multiplier : float
        Threshold multiplier applied to the standard deviation.

    Returns
    -------
    float
        MAGE value in mg/dL, or ``0.0`` if fewer than two valid excursions.

    References
    ----------
    Service, F.J. et al. (1970).  Mean amplitude of glycemic excursions,
    a measure of diabetic instability.  Diabetes, 19(9), 644-655.
    """
    d = _ensure_prepared(data)
    gl = d.glucose
    n = len(gl)
    if n < 3:
        return 0.0

    threshold = sd_multiplier * float(np.nanstd(gl))

    prev, cur, nxt = gl[:-2], gl[1:-1], gl[2:]
    is_peak = (cur > prev) & (cur > nxt)
    is_nadir = (cur < prev) & (cur < nxt)
    tp_mask = is_peak | is_nadir

    tp_idx = np.where(tp_mask)[0] + 1
    if len(tp_idx) < 2:
        return 0.0

    tp_vals = gl[tp_idx]
    tp_is_peak = is_peak[tp_idx - 1]

    alternates = tp_is_peak[:-1] != tp_is_peak[1:]
    amplitudes = np.abs(np.diff(tp_vals))
    valid = alternates & (amplitudes >= threshold)

    return float(np.mean(amplitudes[valid])) if valid.any() else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Peak counting
# ─────────────────────────────────────────────────────────────────────────────

def count_peaks(data: CGMInput, threshold: float = 180.0) -> int:
    """Count episodes where glucose rises above *threshold* and returns below.

    A new episode is counted each time glucose crosses from below to above the
    threshold (rising edge on the boolean mask).

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    threshold : float
        Glucose threshold in mg/dL.

    Returns
    -------
    int
        Number of episodes above threshold.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0
    above = d.glucose > float(threshold)
    entries = np.diff(np.concatenate(([False], above)).astype(np.int8))
    return int(np.sum(entries == 1))


def count_peaks_in_range(
    data: CGMInput,
    lower: float,
    upper: float,
) -> int:
    """Count episodes where glucose enters [lower, upper] and exits.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    lower : float
        Lower bound of the range (inclusive), mg/dL.
    upper : float
        Upper bound of the range (inclusive), mg/dL.

    Returns
    -------
    int
        Number of entry-into-range episodes.
    """
    d = _ensure_prepared(data)
    if d.n_readings == 0:
        return 0
    inside = (d.glucose >= float(lower)) & (d.glucose <= float(upper))
    entries = np.diff(np.concatenate(([False], inside)).astype(np.int8))
    return int(np.sum(entries == 1))


# ─────────────────────────────────────────────────────────────────────────────
# Grouped summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def basic_stats(data: CGMInput) -> dict[str, float]:
    """Return a dict of basic glucose summary statistics.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``min``, ``max``, ``q1``, ``q3``.
    """
    d = _ensure_prepared(data)
    return {
        "mean": mean_glucose(d),
        "median": median_glucose(d),
        "min": min_glucose(d),
        "max": max_glucose(d),
        "q1": q1_glucose(d),
        "q3": q3_glucose(d),
    }


def variability_metrics(data: CGMInput) -> dict[str, float]:
    """Return a dict of glucose variability metrics.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    dict
        Keys: ``sd``, ``cv``, ``j_index``, ``mage``.
    """
    d = _ensure_prepared(data)
    return {
        "sd": sd(d),
        "cv": cv(d),
        "j_index": j_index(d),
        "mage": mage(d),
    }


def time_in_ranges(
    data: CGMInput,
    ranges: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Return time-in-range metrics for multiple glucose bands.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData
    ranges : list of (low, high) tuples | None
        Glucose ranges to compute TIR for.  Defaults to the standard clinical
        ranges: ``[(70, 180), (70, 140), (54, 70), (180, 250), (250, 999)]``.

    Returns
    -------
    dict
        Keys like ``"tir_70_180_pct"`` and ``"tir_70_180_min"`` for each range.
        Also includes ``"tbr_70_pct"``, ``"tbr_54_pct"``, ``"tar_180_pct"``,
        ``"tar_250_pct"`` for the standard thresholds.
    """
    d = _ensure_prepared(data)
    if ranges is None:
        ranges = [(70.0, 180.0), (70.0, 140.0)]

    result: dict[str, float] = {}
    for low, high in ranges:
        key = f"tir_{int(low)}_{int(high)}"
        result[f"{key}_min"] = time_in_range_minutes(d, low=low, high=high)
        result[f"{key}_pct"] = time_in_range_percent(d, low=low, high=high)

    # Standard clinical thresholds
    for thr in (70.0, 54.0):
        result[f"tbr_{int(thr)}_min"] = time_below_range_minutes(d, threshold=thr)
        result[f"tbr_{int(thr)}_pct"] = time_below_range_percent(d, threshold=thr)
    for thr in (180.0, 250.0):
        result[f"tar_{int(thr)}_min"] = time_above_range_minutes(d, threshold=thr)
        result[f"tar_{int(thr)}_pct"] = time_above_range_percent(d, threshold=thr)

    return result


def risk_indices(data: CGMInput) -> dict[str, float]:
    """Return a dict of glycemic risk index values.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    dict
        Keys: ``lbgi``, ``hbgi``, ``adrr``, ``gri``.
    """
    d = _ensure_prepared(data)
    return {
        "lbgi": lbgi(d),
        "hbgi": hbgi(d),
        "adrr": adrr(d),
        "gri": gri(d),
    }


def summary_dict(data: CGMInput) -> dict[str, float]:
    """Return a comprehensive dict of all glycemic metrics.

    Combines :func:`basic_stats`, :func:`variability_metrics`,
    :func:`time_in_ranges`, and :func:`risk_indices` into a single flat dict.

    Parameters
    ----------
    data : DataFrame or PreparedCGMData

    Returns
    -------
    dict
        All metrics keyed by their descriptive names.
    """
    d = _ensure_prepared(data)
    result: dict[str, float] = {}
    result.update(basic_stats(d))
    result.update(variability_metrics(d))
    result.update(time_in_ranges(d))
    result.update(risk_indices(d))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible aliases (old uppercase names from glycosignal.py)
# ─────────────────────────────────────────────────────────────────────────────

#: Alias for :func:`time_in_range_minutes` (SD-band based).
TIR = _tir_sd_minutes

#: Alias for :func:`time_outside_range_minutes` (SD-band based).
TOR = _tor_sd_minutes

#: Alias for :func:`time_in_range_minutes`.
TIR_lo_hi = time_in_range_minutes

#: Alias for :func:`time_above_range_minutes`.
TAT = time_above_range_minutes

#: Alias for :func:`time_below_range_minutes`.
TBT = time_below_range_minutes

#: Alias for :func:`time_outside_range_percent` (SD-band based).
def POR(data: CGMInput, sd: float = 1) -> float:  # noqa: N802
    """Deprecated alias. Use :func:`time_outside_range_percent`."""
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    return float(_tor_sd_minutes(d, sd_multiplier=sd) / d.total_minutes * 100)


#: Alias for :func:`time_in_range_percent` (SD-band based).
def PIR(data: CGMInput, sd: float = 1) -> float:  # noqa: N802
    """Deprecated alias. Use :func:`time_in_range_percent`."""
    d = _ensure_prepared(data)
    if d.n_readings == 0 or d.total_minutes == 0:
        return np.nan
    return float(_tir_sd_minutes(d, sd_multiplier=sd) / d.total_minutes * 100)


LBGI = lbgi
HBGI = hbgi
ADRR = adrr
GRI = gri
MAGE = mage
MGE = mean_glucose_excursion
MGN = mean_glucose_normal
J_index = j_index
CONGA24 = conga24

#: Backward-compatible summary tuple (mean, median, min, max, q1, q3).
def summary(data: CGMInput) -> tuple[float, float, float, float, float, float]:
    """Return (mean, median, min, max, Q1, Q3) -- backward-compatible tuple."""
    d = _ensure_prepared(data)
    return (
        mean_glucose(d),
        median_glucose(d),
        min_glucose(d),
        max_glucose(d),
        q1_glucose(d),
        q3_glucose(d),
    )
