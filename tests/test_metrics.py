"""Tests for glycosignal.metrics -- individual metrics and grouped summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glycosignal import metrics
from glycosignal.schemas import PreparedCGMData, prepare


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_simple_df(glucose_values: list[float], freq: str = "5min") -> pd.DataFrame:
    """Create a CGM DataFrame from a list of glucose values."""
    ts = pd.date_range("2023-01-01", periods=len(glucose_values), freq=freq)
    return pd.DataFrame({"Timestamp": ts, "Glucose": glucose_values})


def _uniform_df(value: float = 120.0, n: int = 100) -> pd.DataFrame:
    """All readings at the same glucose value (zero variability)."""
    return _make_simple_df([value] * n)


# ─────────────────────────────────────────────────────────────────────────────
# Basic statistics
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicStats:
    def test_mean_glucose(self):
        df = _make_simple_df([100.0, 200.0])
        assert metrics.mean_glucose(df) == pytest.approx(150.0)

    def test_median_glucose(self):
        df = _make_simple_df([100.0, 120.0, 200.0])
        assert metrics.median_glucose(df) == pytest.approx(120.0)

    def test_min_glucose(self):
        df = _make_simple_df([100.0, 50.0, 200.0])
        assert metrics.min_glucose(df) == pytest.approx(50.0)

    def test_max_glucose(self):
        df = _make_simple_df([100.0, 50.0, 200.0])
        assert metrics.max_glucose(df) == pytest.approx(200.0)

    def test_q1_glucose(self):
        df = _make_simple_df(list(range(1, 101)))
        q1 = metrics.q1_glucose(df)
        assert 24 <= q1 <= 26

    def test_q3_glucose(self):
        df = _make_simple_df(list(range(1, 101)))
        q3 = metrics.q3_glucose(df)
        assert 74 <= q3 <= 76

    def test_nan_on_empty(self):
        df = _make_simple_df([])
        assert np.isnan(metrics.mean_glucose(df))
        assert np.isnan(metrics.median_glucose(df))
        assert np.isnan(metrics.min_glucose(df))
        assert np.isnan(metrics.max_glucose(df))

    def test_accepts_prepared(self, cgm_prepared):
        val = metrics.mean_glucose(cgm_prepared)
        assert isinstance(val, float)
        assert not np.isnan(val)


# ─────────────────────────────────────────────────────────────────────────────
# Variability
# ─────────────────────────────────────────────────────────────────────────────

class TestVariability:
    def test_sd_zero_for_uniform(self):
        df = _uniform_df(100.0)
        assert metrics.sd(df) == pytest.approx(0.0)

    def test_cv_zero_for_uniform(self):
        df = _uniform_df(100.0)
        assert metrics.cv(df) == pytest.approx(0.0)

    def test_cv_formula(self):
        df = _make_simple_df([80.0, 120.0])
        expected_cv = (np.std([80.0, 120.0]) / np.mean([80.0, 120.0])) * 100
        assert metrics.cv(df) == pytest.approx(expected_cv, rel=1e-5)

    def test_j_index_positive(self, cgm_df):
        val = metrics.j_index(cgm_df)
        assert val > 0

    def test_mage_zero_for_monotone(self):
        # Strictly increasing sequence has no alternating peaks/nadirs
        df = _make_simple_df(list(range(50, 150)))
        assert metrics.mage(df) == pytest.approx(0.0)

    def test_mage_nonzero_for_oscillating(self):
        # Alternating high-low pattern
        gl = [100, 200, 100, 200, 100, 200, 100, 200, 100, 200]
        df = _make_simple_df(gl)
        assert metrics.mage(df) > 0

    def test_conga24_returns_nan_for_short_window(self):
        df = _make_simple_df([120.0] * 10)  # only ~50 min of data
        val = metrics.conga24(df)
        assert np.isnan(val)


# ─────────────────────────────────────────────────────────────────────────────
# Time-in-range (minutes)
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeInRangeMinutes:
    def test_all_in_range(self):
        df = _uniform_df(120.0, n=288)  # 24h at 120 mg/dL
        tir = metrics.time_in_range_minutes(df, low=70, high=180)
        assert tir == pytest.approx(24 * 60, rel=0.01)

    def test_none_in_range(self):
        df = _uniform_df(200.0, n=288)
        tir = metrics.time_in_range_minutes(df, low=70, high=180)
        assert tir == pytest.approx(0.0)

    def test_all_below_threshold(self):
        df = _uniform_df(50.0, n=288)
        tbr = metrics.time_below_range_minutes(df, threshold=70)
        total = metrics.time_above_range_minutes(df, threshold=1000)
        assert total == pytest.approx(0.0)
        assert tbr > 0

    def test_all_above_threshold(self):
        df = _uniform_df(300.0, n=288)
        tar = metrics.time_above_range_minutes(df, threshold=180)
        assert tar > 0

    def test_tor_complements_tir(self):
        df = _make_simple_df([100.0, 200.0])  # 1 in, 1 out
        tir = metrics.time_in_range_minutes(df, low=70, high=180)
        tor = metrics.time_outside_range_minutes(df, low=70, high=180)
        assert tir > 0
        assert tor > 0


# ─────────────────────────────────────────────────────────────────────────────
# Time-in-range (percent)
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeInRangePercent:
    def test_all_in_range_is_100(self):
        df = _uniform_df(120.0)
        pct = metrics.time_in_range_percent(df, low=70, high=180)
        assert pct == pytest.approx(100.0, abs=0.5)

    def test_none_in_range_is_zero(self):
        df = _uniform_df(50.0)
        pct = metrics.time_in_range_percent(df, low=70, high=180)
        assert pct == pytest.approx(0.0, abs=0.5)

    def test_percent_bounded_0_to_100(self, cgm_df):
        pct = metrics.time_in_range_percent(cgm_df, low=70, high=180)
        assert 0 <= pct <= 100

    def test_empty_returns_nan(self):
        df = _make_simple_df([])
        assert np.isnan(metrics.time_in_range_percent(df))


# ─────────────────────────────────────────────────────────────────────────────
# Risk indices
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskIndices:
    def test_lbgi_zero_for_high_glucose(self):
        df = _uniform_df(300.0)
        assert metrics.lbgi(df) == pytest.approx(0.0)

    def test_hbgi_zero_for_low_glucose(self):
        df = _uniform_df(50.0)
        assert metrics.hbgi(df) == pytest.approx(0.0)

    def test_lbgi_positive_for_low_glucose(self):
        df = _uniform_df(50.0)
        assert metrics.lbgi(df) > 0

    def test_hbgi_positive_for_high_glucose(self):
        df = _uniform_df(300.0)
        assert metrics.hbgi(df) > 0

    def test_gri_bounded_0_100(self, cgm_df):
        val = metrics.gri(cgm_df)
        assert 0 <= val <= 100

    def test_adrr_nonnegative(self, cgm_df):
        val = metrics.adrr(cgm_df)
        assert val >= 0

    def test_risk_indices_dict_keys(self, cgm_df):
        d = metrics.risk_indices(cgm_df)
        assert set(d.keys()) == {"lbgi", "hbgi", "adrr", "gri"}


# ─────────────────────────────────────────────────────────────────────────────
# Excursion metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestExcursionMetrics:
    def test_mage_symmetric(self):
        gl = [120, 60, 120, 60, 120, 60, 120, 60, 120, 60]
        df = _make_simple_df(gl)
        assert metrics.mage(df) > 0

    def test_mean_glucose_excursion_nan_when_no_excursions(self):
        # Uniform data: all readings are at mean, no excursions
        df = _uniform_df(120.0)
        val = metrics.mean_glucose_excursion(df)
        assert np.isnan(val)

    def test_mean_glucose_normal_equals_mean_for_uniform(self):
        df = _uniform_df(120.0)
        val = metrics.mean_glucose_normal(df)
        assert val == pytest.approx(120.0, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Peak counting
# ─────────────────────────────────────────────────────────────────────────────

class TestPeakCounting:
    def test_count_peaks_threshold_above(self):
        gl = [100, 100, 200, 200, 100, 100, 200, 200, 100]
        df = _make_simple_df(gl)
        assert metrics.count_peaks(df, threshold=150) == 2

    def test_count_peaks_none(self):
        df = _uniform_df(100.0)
        assert metrics.count_peaks(df, threshold=150) == 0

    def test_count_peaks_in_range(self):
        gl = [100, 160, 100, 160, 100]
        df = _make_simple_df(gl)
        assert metrics.count_peaks_in_range(df, lower=140, upper=180) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Grouped summaries
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupedSummaries:
    def test_basic_stats_keys(self, cgm_df):
        d = metrics.basic_stats(cgm_df)
        assert set(d.keys()) == {"mean", "median", "min", "max", "q1", "q3"}

    def test_variability_metrics_keys(self, cgm_df):
        d = metrics.variability_metrics(cgm_df)
        assert set(d.keys()) == {"sd", "cv", "j_index", "mage"}

    def test_time_in_ranges_contains_tir(self, cgm_df):
        d = metrics.time_in_ranges(cgm_df)
        assert "tir_70_180_pct" in d
        assert "tbr_70_pct" in d
        assert "tar_180_pct" in d

    def test_summary_dict_merges_all(self, cgm_df):
        d = metrics.summary_dict(cgm_df)
        assert "mean" in d
        assert "cv" in d
        assert "tir_70_180_pct" in d
        assert "lbgi" in d

    def test_summary_dict_no_nan_on_clean_data(self, cgm_df):
        d = metrics.summary_dict(cgm_df)
        # Most entries should be non-nan for clean 24h data
        non_nan = {k: v for k, v in d.items() if not (isinstance(v, float) and np.isnan(v))}
        assert len(non_nan) > 10


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat aliases
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatAliases:
    def test_lbgi_alias(self, cgm_df):
        assert metrics.LBGI(cgm_df) == metrics.lbgi(cgm_df)

    def test_hbgi_alias(self, cgm_df):
        assert metrics.HBGI(cgm_df) == metrics.hbgi(cgm_df)

    def test_mage_alias(self, cgm_df):
        assert metrics.MAGE(cgm_df) == metrics.mage(cgm_df)

    def test_summary_tuple(self, cgm_df):
        t = metrics.summary(cgm_df)
        assert len(t) == 6
        assert t[0] == metrics.mean_glucose(cgm_df)
