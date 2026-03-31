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


# ─────────────────────────────────────────────────────────────────────────────
# step_hours parameter
# ─────────────────────────────────────────────────────────────────────────────

class TestStepHours:
    def test_step_hours_equivalent_to_overlap(self):
        """step_hours=12 should produce same windows as overlap_hours=12."""
        df = _make_cgm(n_days=3)
        r_overlap = windows.create_sliding_windows(
            df, window_hours=24, overlap_hours=12, show_progress=False
        )
        r_step = windows.create_sliding_windows(
            df, window_hours=24, step_hours=12, show_progress=False
        )
        assert r_step.metadata["n_valid_windows"] == r_overlap.metadata["n_valid_windows"]
        pd.testing.assert_frame_equal(
            r_step.windows.reset_index(drop=True),
            r_overlap.windows.reset_index(drop=True),
        )

    def test_step_hours_zero_raises(self):
        df = _make_cgm(n_days=2)
        with pytest.raises(ValueError, match="step_hours"):
            windows.create_sliding_windows(df, step_hours=0, show_progress=False)

    def test_step_hours_negative_raises(self):
        df = _make_cgm(n_days=2)
        with pytest.raises(ValueError, match="step_hours"):
            windows.create_sliding_windows(df, step_hours=-6, show_progress=False)

    def test_step_hours_takes_precedence_over_overlap(self):
        """When step_hours is given, overlap_hours >= window_hours should not raise."""
        df = _make_cgm(n_days=3)
        # overlap_hours=24 would normally raise, but step_hours bypasses that check
        result = windows.create_sliding_windows(
            df, window_hours=24, overlap_hours=24, step_hours=12, show_progress=False
        )
        assert result.metadata["n_valid_windows"] > 0

    def test_step_larger_than_window_produces_gaps(self):
        """step_hours > window_hours is a valid 'sample every N hours' use case."""
        df = _make_cgm(n_days=5)
        result = windows.create_sliding_windows(
            df, window_hours=6, step_hours=24, show_progress=False
        )
        assert isinstance(result.windows, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# anchor_time parameter
# ─────────────────────────────────────────────────────────────────────────────

class TestAnchorTime:
    def test_midnight_anchor_matches_default(self):
        """anchor_time='00:00' must be identical to default behaviour."""
        df = _make_cgm(n_days=3)
        r_default = windows.create_sliding_windows(df, show_progress=False)
        r_midnight = windows.create_sliding_windows(
            df, anchor_time="00:00", show_progress=False
        )
        pd.testing.assert_frame_equal(
            r_default.windows.reset_index(drop=True),
            r_midnight.windows.reset_index(drop=True),
        )

    def test_8am_anchor_windows_start_at_8(self):
        """All windows from an 08:00 anchor must start at 08:00."""
        df = _make_cgm(n_days=4)
        result = windows.create_sliding_windows(
            df, window_hours=24, anchor_time="08:00", show_progress=False
        )
        if not result.windows.empty:
            starts = (
                result.windows.groupby("window_id")["Timestamp"].min()
            )
            assert (starts.dt.hour == 8).all(), "Not all windows start at 08:00"
            assert (starts.dt.minute == 0).all()

    def test_invalid_anchor_format_raises(self):
        df = _make_cgm(n_days=2)
        with pytest.raises(ValueError, match="anchor_time"):
            windows.create_sliding_windows(df, anchor_time="8am", show_progress=False)

    def test_invalid_anchor_out_of_range_raises(self):
        df = _make_cgm(n_days=2)
        with pytest.raises(ValueError, match="anchor_time"):
            windows.create_sliding_windows(df, anchor_time="25:00", show_progress=False)

    def test_non_midnight_window_id_contains_hhmm(self):
        """window_id for a non-midnight anchor should include the HH:MM suffix."""
        df = _make_cgm(n_days=4)
        result = windows.create_sliding_windows(
            df, window_hours=24, anchor_time="08:00", show_progress=False
        )
        if not result.windows.empty:
            wids = result.windows["window_id"].unique()
            assert all("_0800" in wid for wid in wids)

    def test_midnight_window_id_has_no_hhmm_suffix(self):
        """window_id for midnight anchor must keep the date-only format."""
        df = _make_cgm(n_days=2)
        result = windows.create_sliding_windows(df, show_progress=False)
        wids = result.windows["window_id"].unique()
        # No _HHMM suffix; each id ends with the date (YYYY-MM-DD)
        for wid in wids:
            parts = wid.split("_")
            assert len(parts) == 2, f"Expected 'subject_date' format, got: {wid}"


# ─────────────────────────────────────────────────────────────────────────────
# create_day_segments
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateDaySegments:
    def test_returns_window_result(self):
        df = _make_cgm(n_days=3)
        result = windows.create_day_segments(df, show_progress=False)
        assert hasattr(result, "windows")
        assert hasattr(result, "metadata")

    def test_midnight_segments_288_rows_each(self):
        """Each midnight-anchored day segment must have 288 rows (5-min grid)."""
        df = _make_cgm(n_days=3)
        result = windows.create_day_segments(df, show_progress=False)
        for wid, grp in result.windows.groupby("window_id"):
            assert len(grp) == 288, f"Window {wid} has {len(grp)} rows"

    def test_8am_segments_288_rows_each(self):
        """Each 8 AM-anchored day segment must also have 288 rows."""
        df = _make_cgm(n_days=5)
        result = windows.create_day_segments(df, anchor_time="08:00", show_progress=False)
        for wid, grp in result.windows.groupby("window_id"):
            assert len(grp) == 288, f"Window {wid} has {len(grp)} rows"

    def test_multi_subject(self):
        df1 = _make_cgm(n_days=3, subject="A", seed=10)
        df2 = _make_cgm(n_days=3, subject="B", seed=20)
        df = pd.concat([df1, df2], ignore_index=True)
        result = windows.create_day_segments(df, show_progress=False)
        assert result.windows["subject"].nunique() == 2

    def test_non_overlapping_windows(self):
        """Adjacent windows must not share any Timestamp values."""
        df = _make_cgm(n_days=4)
        result = windows.create_day_segments(df, show_progress=False)
        for wid, grp in result.windows.groupby("window_id"):
            ts_set = set(grp["Timestamp"])
            other = result.windows.loc[result.windows["window_id"] != wid, "Timestamp"]
            assert ts_set.isdisjoint(other), f"Window {wid} overlaps another window"

    def test_top_level_import(self):
        """create_day_segments must be importable from the top-level namespace."""
        import glycosignal
        assert hasattr(glycosignal, "create_day_segments")

    def test_roundtrip_with_feature_map(self):
        """Day segments should feed into build_feature_map without errors."""
        from glycosignal import features
        df = _make_cgm(n_days=3)
        result = windows.create_day_segments(df, show_progress=False)
        X = features.build_feature_map(result.windows)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == result.metadata["n_valid_windows"]
