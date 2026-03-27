"""Tests for glycosignal.detect -- event detection."""

from __future__ import annotations

import pandas as pd
import pytest

from glycosignal import detect


# ─────────────────────────────────────────────────────────────────────────────
# detect_hypoglycemia
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectHypoglycemia:
    def test_detects_clear_episode(self, cgm_df_with_hypo):
        episodes = detect.detect_hypoglycemia(cgm_df_with_hypo, threshold=70, min_duration_minutes=10)
        assert len(episodes) >= 1

    def test_no_episodes_in_clean_data(self):
        ts = pd.date_range("2023-01-01", periods=100, freq="5min")
        df = pd.DataFrame({"Timestamp": ts, "Glucose": [120.0] * 100})
        episodes = detect.detect_hypoglycemia(df, threshold=70)
        assert len(episodes) == 0

    def test_episode_columns_present(self, cgm_df_with_hypo):
        episodes = detect.detect_hypoglycemia(cgm_df_with_hypo, min_duration_minutes=5)
        if not episodes.empty:
            assert "start_time" in episodes.columns
            assert "end_time" in episodes.columns
            assert "duration_minutes" in episodes.columns
            assert "nadir_glucose" in episodes.columns
            assert "event_type" in episodes.columns

    def test_nadir_below_threshold(self, cgm_df_with_hypo):
        episodes = detect.detect_hypoglycemia(cgm_df_with_hypo, threshold=70, min_duration_minutes=5)
        if not episodes.empty:
            assert (episodes["nadir_glucose"] <= 70).all()

    def test_event_type_label(self, cgm_df_with_hypo):
        episodes = detect.detect_hypoglycemia(cgm_df_with_hypo, min_duration_minutes=5)
        if not episodes.empty:
            assert (episodes["event_type"] == "hypoglycemia").all()

    def test_duration_filter_respected(self, cgm_df_with_hypo):
        long_min = detect.detect_hypoglycemia(cgm_df_with_hypo, min_duration_minutes=120)
        short_min = detect.detect_hypoglycemia(cgm_df_with_hypo, min_duration_minutes=5)
        assert len(long_min) <= len(short_min)

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"Timestamp": [], "Glucose": []})
        result = detect.detect_hypoglycemia(df)
        assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# detect_hyperglycemia
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectHyperglycemia:
    def test_detects_clear_episode(self, cgm_df_with_hyper):
        episodes = detect.detect_hyperglycemia(cgm_df_with_hyper, threshold=180, min_duration_minutes=10)
        assert len(episodes) >= 1

    def test_no_episodes_in_normal_data(self):
        ts = pd.date_range("2023-01-01", periods=100, freq="5min")
        df = pd.DataFrame({"Timestamp": ts, "Glucose": [120.0] * 100})
        episodes = detect.detect_hyperglycemia(df, threshold=180)
        assert len(episodes) == 0

    def test_peak_above_threshold(self, cgm_df_with_hyper):
        episodes = detect.detect_hyperglycemia(cgm_df_with_hyper, threshold=180, min_duration_minutes=5)
        if not episodes.empty:
            assert (episodes["peak_glucose"] >= 180).all()

    def test_event_type_label(self, cgm_df_with_hyper):
        episodes = detect.detect_hyperglycemia(cgm_df_with_hyper, min_duration_minutes=5)
        if not episodes.empty:
            assert (episodes["event_type"] == "hyperglycemia").all()


# ─────────────────────────────────────────────────────────────────────────────
# detect_nocturnal_events
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectNocturnalEvents:
    def test_returns_dataframe(self, cgm_df):
        result = detect.detect_nocturnal_events(cgm_df)
        assert isinstance(result, pd.DataFrame)

    def test_events_within_nocturnal_window(self, cgm_df_with_hypo):
        result = detect.detect_nocturnal_events(
            cgm_df_with_hypo, start_hour=0, end_hour=6
        )
        if not result.empty:
            start_hours = pd.to_datetime(result["start_time"]).dt.hour
            assert ((start_hours >= 0) & (start_hours < 6)).all()

    def test_empty_result_outside_window(self, cgm_df):
        # All data at 12:00+; nocturnal window 0-6 should be empty
        ts = pd.date_range("2023-01-01 10:00", periods=100, freq="5min")
        df = pd.DataFrame({"Timestamp": ts, "Glucose": [50.0] * 100})
        result = detect.detect_nocturnal_events(df, start_hour=0, end_hour=6)
        assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# detect_postprandial_excursions
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectPostprandialExcursions:
    def test_returns_dataframe(self, cgm_df):
        result = detect.detect_postprandial_excursions(cgm_df)
        assert isinstance(result, pd.DataFrame)

    def test_detects_large_rise(self):
        # Create a clear rise: nadir at 70, peak at 200
        gl = [120] * 10 + [70] + [200] * 10 + [120] * 10
        ts = pd.date_range("2023-01-01 12:00", periods=len(gl), freq="5min")
        df = pd.DataFrame({"Timestamp": ts, "Glucose": gl})
        result = detect.detect_postprandial_excursions(df, rise_threshold=100, window_minutes=120)
        assert len(result) >= 1

    def test_rise_column_present(self, cgm_df):
        result = detect.detect_postprandial_excursions(cgm_df)
        if not result.empty:
            assert "rise_mg_dl" in result.columns
            assert "nadir_glucose" in result.columns
            assert "peak_glucose" in result.columns

    def test_event_type_label(self, cgm_df):
        result = detect.detect_postprandial_excursions(cgm_df)
        if not result.empty:
            assert (result["event_type"] == "postprandial_excursion").all()

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"Timestamp": [], "Glucose": []})
        result = detect.detect_postprandial_excursions(df)
        assert len(result) == 0
