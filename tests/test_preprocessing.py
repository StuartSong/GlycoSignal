"""Tests for glycosignal.preprocessing -- cleaning, validation, interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glycosignal import preprocessing
from glycosignal.schemas import COL_GLUCOSE, COL_TIMESTAMP


# ─────────────────────────────────────────────────────────────────────────────
# standardize_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardizeColumns:
    def test_renames_time_to_timestamp(self):
        df = pd.DataFrame({"time": ["2023-01-01"], "Glucose": [100]})
        result = preprocessing.standardize_columns(df)
        assert COL_TIMESTAMP in result.columns

    def test_renames_gl_to_glucose(self):
        df = pd.DataFrame({"Timestamp": ["2023-01-01"], "gl": [100]})
        result = preprocessing.standardize_columns(df)
        assert COL_GLUCOSE in result.columns

    def test_handles_glucose_value_mg_dl(self):
        df = pd.DataFrame({"Timestamp": ["2023-01-01"], "Glucose Value (mg/dL)": [100]})
        result = preprocessing.standardize_columns(df)
        assert COL_GLUCOSE in result.columns

    def test_already_canonical_unchanged(self):
        df = pd.DataFrame({"Timestamp": ["2023-01-01"], "Glucose": [100], "subject": ["S1"]})
        result = preprocessing.standardize_columns(df)
        assert list(result.columns) == list(df.columns)

    def test_custom_column_map(self):
        df = pd.DataFrame({"BG": [120], "ts": ["2023-01-01"]})
        result = preprocessing.standardize_columns(df, column_map={"bg": "Glucose", "ts": "Timestamp"})
        assert COL_GLUCOSE in result.columns
        assert COL_TIMESTAMP in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"time": ["2023-01-01"], "gl": [100]})
        original_cols = list(df.columns)
        preprocessing.standardize_columns(df)
        assert list(df.columns) == original_cols


# ─────────────────────────────────────────────────────────────────────────────
# clean_cgm
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanCgm:
    def test_drops_nan_glucose(self):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:00", "2023-01-01 00:05"],
            "Glucose": [120.0, np.nan],
        })
        result = preprocessing.clean_cgm(df)
        assert len(result) == 1
        assert result["Glucose"].iloc[0] == 120.0

    def test_drops_non_positive(self):
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2023-01-01", periods=3, freq="5min"),
            "Glucose": [100.0, 0.0, -5.0],
        })
        result = preprocessing.clean_cgm(df)
        assert len(result) == 1

    def test_drops_duplicates(self):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:00", "2023-01-01 00:00"],
            "Glucose": [120.0, 130.0],
        })
        result = preprocessing.clean_cgm(df)
        assert len(result) == 1

    def test_sorts_by_timestamp(self):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:10", "2023-01-01 00:00", "2023-01-01 00:05"],
            "Glucose": [130.0, 120.0, 125.0],
        })
        result = preprocessing.clean_cgm(df)
        assert result[COL_TIMESTAMP].is_monotonic_increasing

    def test_returns_copy(self):
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "Glucose": [100.0, 110.0, 120.0, 130.0, 140.0],
        })
        result = preprocessing.clean_cgm(df)
        result.loc[0, "Glucose"] = 999.0
        assert df.loc[0, "Glucose"] != 999.0


# ─────────────────────────────────────────────────────────────────────────────
# validate_cgm
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCgm:
    def test_valid_data_no_warnings(self, cgm_df):
        report = preprocessing.validate_cgm(cgm_df)
        assert report.n_readings == len(cgm_df)
        assert report.n_valid == len(cgm_df)
        assert report.n_missing_glucose == 0
        assert report.n_duplicate_timestamps == 0

    def test_detects_missing_glucose(self):
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "Glucose": [100.0, np.nan, 120.0, np.nan, 140.0],
        })
        report = preprocessing.validate_cgm(df)
        assert report.n_missing_glucose == 2

    def test_detects_duplicates(self):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:00"] * 2 + ["2023-01-01 00:05"],
            "Glucose": [100.0, 110.0, 120.0],
        })
        report = preprocessing.validate_cgm(df)
        assert report.n_duplicate_timestamps >= 1

    def test_summary_string(self, cgm_df):
        report = preprocessing.validate_cgm(cgm_df)
        s = report.summary()
        assert "Readings" in s

    def test_missing_column_warns(self):
        df = pd.DataFrame({"Timestamp": ["2023-01-01"]})
        report = preprocessing.validate_cgm(df)
        assert any("Glucose" in w for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# detect_gaps
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectGaps:
    def test_no_gaps_when_regular(self, cgm_df):
        gaps = preprocessing.detect_gaps(cgm_df, expected_interval_minutes=5)
        assert len(gaps) == 0

    def test_detects_gap(self):
        ts = pd.date_range("2023-01-01", periods=10, freq="5min").tolist()
        ts.insert(5, pd.Timestamp("2023-01-01 01:00"))  # big gap
        df = pd.DataFrame({"Timestamp": ts, "Glucose": [100.0] * 11})
        gaps = preprocessing.detect_gaps(df)
        assert len(gaps) >= 1

    def test_gap_columns_present(self):
        ts = pd.date_range("2023-01-01", periods=5, freq="5min").tolist()
        ts.append(pd.Timestamp("2023-01-01 02:00"))
        df = pd.DataFrame({"Timestamp": ts, "Glucose": [100.0] * 6})
        gaps = preprocessing.detect_gaps(df)
        assert set(gaps.columns) >= {"gap_start", "gap_end", "duration_minutes"}


# ─────────────────────────────────────────────────────────────────────────────
# interpolate_cgm
# ─────────────────────────────────────────────────────────────────────────────

class TestInterpolateCgm:
    def test_fills_short_gap(self, cgm_df_with_gaps):
        df = cgm_df_with_gaps.copy()
        df_clean = df.dropna()
        # Create a small gap to fill
        small_gap = pd.DataFrame({
            "Timestamp": pd.date_range("2023-01-01", periods=20, freq="5min"),
            "Glucose": [100.0] * 8 + [np.nan] * 4 + [100.0] * 8,
        })
        result = preprocessing.interpolate_cgm(small_gap, max_gap_points=6)
        assert result["Glucose"].isna().sum() == 0

    def test_leaves_long_gap_as_nan(self):
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2023-01-01", periods=30, freq="5min"),
            "Glucose": [100.0] * 5 + [np.nan] * 20 + [100.0] * 5,
        })
        result = preprocessing.interpolate_cgm(df, max_gap_points=10)
        assert result["Glucose"].isna().any()

    def test_observed_values_unchanged(self):
        original = [100.0, np.nan, np.nan, 110.0, 120.0]
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "Glucose": original,
        })
        result = preprocessing.interpolate_cgm(df, max_gap_points=5)
        assert result["Glucose"].iloc[0] == pytest.approx(100.0)
        assert result["Glucose"].iloc[3] == pytest.approx(110.0)
        assert result["Glucose"].iloc[4] == pytest.approx(120.0)


# ─────────────────────────────────────────────────────────────────────────────
# resample_cgm
# ─────────────────────────────────────────────────────────────────────────────

class TestResampleCgm:
    def test_outputs_regular_grid(self, cgm_df):
        result = preprocessing.resample_cgm(cgm_df, freq="5min", method="nearest")
        ts = pd.to_datetime(result["Timestamp"])
        diffs = ts.diff().iloc[1:]
        assert (diffs <= pd.Timedelta("6min")).all()

    def test_mean_method(self, cgm_df):
        result = preprocessing.resample_cgm(cgm_df, freq="15min", method="mean")
        assert len(result) < len(cgm_df)
        assert result["Glucose"].notna().any()

    def test_invalid_method_raises(self, cgm_df):
        with pytest.raises(ValueError, match="method"):
            preprocessing.resample_cgm(cgm_df, method="spline")


# ─────────────────────────────────────────────────────────────────────────────
# convert_units
# ─────────────────────────────────────────────────────────────────────────────

class TestConvertUnits:
    def test_mmol_to_mgdl(self):
        val = preprocessing.convert_units(np.array([5.0]), from_unit="mmol/L", to_unit="mg/dL")
        assert val == pytest.approx(5.0 * 18.0182, rel=1e-4)

    def test_mgdl_to_mmol(self):
        val = preprocessing.convert_units(np.array([90.0]), from_unit="mg/dL", to_unit="mmol/L")
        assert val == pytest.approx(90.0 / 18.0182, rel=1e-4)

    def test_same_unit_raises(self):
        with pytest.raises(ValueError):
            preprocessing.convert_units(np.array([100.0]), from_unit="mg/dL", to_unit="mg/dL")

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            preprocessing.convert_units(np.array([100.0]), from_unit="mmol/L", to_unit="g/L")

    def test_dataframe_converts_glucose_column(self):
        df = pd.DataFrame({"Timestamp": ["2023-01-01"], "Glucose": [5.0]})
        result = preprocessing.convert_units(df, from_unit="mmol/L", to_unit="mg/dL")
        assert result["Glucose"].iloc[0] == pytest.approx(5.0 * 18.0182, rel=1e-4)
        # Original not modified
        assert df["Glucose"].iloc[0] == 5.0

    def test_series_converts(self):
        s = pd.Series([5.0, 6.0])
        result = preprocessing.convert_units(s, from_unit="mmol/L", to_unit="mg/dL")
        assert result.iloc[0] == pytest.approx(5.0 * 18.0182, rel=1e-4)
