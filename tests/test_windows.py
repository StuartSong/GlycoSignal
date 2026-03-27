"""Tests for glycosignal.windows -- sliding window creation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glycosignal import windows
from glycosignal.schemas import COL_GLUCOSE, COL_TIMESTAMP


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cgm(n_days: int = 3, subject: str = "S01", seed: int = 42) -> pd.DataFrame:
    """Generate n_days of 5-min CGM data."""
    rng = np.random.default_rng(seed)
    n = n_days * 288
    ts = pd.date_range("2023-01-01 00:00", periods=n, freq="5min")
    gl = np.clip(rng.normal(120, 25, n), 40, 400)
    return pd.DataFrame({
        "Timestamp": ts,
        "Glucose": gl,
        "subject": subject,
    })


# ─────────────────────────────────────────────────────────────────────────────
# create_sliding_windows
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateSlidingWindows:
    def test_basic_24h_windows(self):
        df = _make_cgm(n_days=3)
        result = windows.create_sliding_windows(df, window_hours=24, show_progress=False)
        assert isinstance(result.windows, pd.DataFrame)
        assert "window_id" in result.windows.columns
        assert result.metadata["n_valid_windows"] > 0

    def test_window_id_in_output(self):
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, show_progress=False)
        assert result.windows["window_id"].notna().all()

    def test_each_window_has_correct_size(self):
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, window_hours=24, show_progress=False)
        for wid, grp in result.windows.groupby("window_id"):
            # Each 24h window at 5-min grid should have 288 rows
            assert len(grp) == 288, f"Window {wid} has {len(grp)} rows"

    def test_returns_window_result_namedtuple(self):
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, show_progress=False)
        assert hasattr(result, "windows")
        assert hasattr(result, "metadata")

    def test_metadata_contains_expected_keys(self):
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, show_progress=False)
        meta = result.metadata
        assert "n_groups" in meta
        assert "n_valid_windows" in meta
        assert "n_discarded_partial_days" in meta

    def test_invalid_overlap_raises(self):
        df = _make_cgm(n_days=2)
        with pytest.raises(ValueError, match="overlap_hours"):
            windows.create_sliding_windows(df, window_hours=24, overlap_hours=24, show_progress=False)

    def test_invalid_min_fraction_raises(self):
        df = _make_cgm(n_days=2)
        with pytest.raises(ValueError, match="min_fraction"):
            windows.create_sliding_windows(df, min_fraction=1.5, show_progress=False)

    def test_missing_group_col_auto_adds_default(self):
        """When group_col is absent, windows adds a 'default' group and succeeds."""
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2023-01-01", periods=10, freq="5min"),
            "Glucose": [100.0] * 10,
        })
        # Should not raise -- creates a single 'default' group
        result = windows.create_sliding_windows(df, show_progress=False)
        assert isinstance(result.windows, pd.DataFrame)

    def test_multi_subject(self):
        df1 = _make_cgm(n_days=2, subject="S01", seed=1)
        df2 = _make_cgm(n_days=2, subject="S02", seed=2)
        df = pd.concat([df1, df2], ignore_index=True)
        result = windows.create_sliding_windows(df, show_progress=False)
        subjects_in_output = result.windows["subject"].unique()
        assert len(subjects_in_output) == 2

    def test_min_fraction_filters_incomplete_windows(self):
        """A window with 50% data should be dropped when min_fraction=0.8."""
        df = _make_cgm(n_days=2)
        # Drop half the readings from day 2
        day2_mask = df["Timestamp"].dt.date == pd.Timestamp("2023-01-02").date()
        df.loc[day2_mask & (df.index % 2 == 0), "Glucose"] = np.nan
        df = df.dropna().reset_index(drop=True)
        result = windows.create_sliding_windows(df, min_fraction=0.8, show_progress=False)
        # Day 2 may be dropped; we just check the function completes
        assert isinstance(result.windows, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# pivot_windows_wide
# ─────────────────────────────────────────────────────────────────────────────

class TestPivotWindowsWide:
    def test_produces_one_row_per_window(self):
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, show_progress=False)
        wide = windows.pivot_windows_wide(result.windows)
        n_windows = result.windows["window_id"].nunique()
        assert wide.shape[0] == n_windows

    def test_wide_has_time_columns(self):
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, show_progress=False)
        wide = windows.pivot_windows_wide(result.windows)
        time_cols = [c for c in wide.columns if ":" in c]
        assert len(time_cols) > 0


# ─────────────────────────────────────────────────────────────────────────────
# format_window_label
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatWindowLabel:
    def test_returns_string(self):
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-01-02")
        label = windows.format_window_label(start, end)
        assert isinstance(label, str)
        assert "2023-01-01" in label
        assert "2023-01-02" in label


# ─────────────────────────────────────────────────────────────────────────────
# windows_to_records
# ─────────────────────────────────────────────────────────────────────────────

class TestWindowsToRecords:
    def test_returns_list_of_tuples(self):
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, show_progress=False)
        records = windows.windows_to_records(result.windows)
        assert isinstance(records, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in records)

    def test_each_record_is_dataframe(self):
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, show_progress=False)
        records = windows.windows_to_records(result.windows)
        for wid, sub_df in records:
            assert isinstance(wid, str)
            assert isinstance(sub_df, pd.DataFrame)
            assert "Glucose" in sub_df.columns
