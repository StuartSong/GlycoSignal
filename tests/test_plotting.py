"""Smoke tests for glycosignal.plotting -- verify plots return fig/ax without errors."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import pytest

from glycosignal import plotting


class TestPlotGlucoseTimeseries:
    def test_returns_fig_ax(self, cgm_df):
        fig, ax = plotting.plot_glucose_timeseries(cgm_df)
        assert fig is not None
        assert ax is not None

    def test_with_subject_label(self, cgm_df):
        fig, ax = plotting.plot_glucose_timeseries(cgm_df, subject="TestSubject")
        assert "TestSubject" in ax.get_title()

    def test_without_tir_bands(self, cgm_df):
        fig, ax = plotting.plot_glucose_timeseries(cgm_df, show_tir_bands=False)
        assert fig is not None

    def test_uses_provided_ax(self, cgm_df):
        import matplotlib.pyplot as plt
        fig_pre, ax_pre = plt.subplots()
        fig, ax = plotting.plot_glucose_timeseries(cgm_df, ax=ax_pre)
        assert ax is ax_pre
        plt.close("all")

    def test_missing_column_raises(self):
        import pandas as pd
        df = pd.DataFrame({"Timestamp": ["2023-01-01"], "bad_col": [100]})
        with pytest.raises((ValueError, KeyError)):
            plotting.plot_glucose_timeseries(df)


class TestPlotDailyOverlay:
    def test_returns_fig_ax(self, cgm_df_multi):
        fig, ax = plotting.plot_daily_overlay(cgm_df_multi)
        assert fig is not None
        assert ax is not None

    def test_single_day_does_not_raise(self, cgm_df):
        fig, ax = plotting.plot_daily_overlay(cgm_df)
        assert fig is not None


class TestPlotAgp:
    def test_returns_fig_ax(self, cgm_df_multi):
        fig, ax = plotting.plot_agp(cgm_df_multi)
        assert fig is not None
        assert ax is not None

    def test_single_day_does_not_raise(self, cgm_df):
        fig, ax = plotting.plot_agp(cgm_df)
        assert fig is not None

    def test_custom_percentiles(self, cgm_df_multi):
        fig, ax = plotting.plot_agp(cgm_df_multi, percentiles=(5, 50, 95))
        assert fig is not None


class TestPlotHistogram:
    def test_returns_fig_ax(self, cgm_df):
        fig, ax = plotting.plot_histogram(cgm_df)
        assert fig is not None
        assert ax is not None

    def test_custom_bins(self, cgm_df):
        fig, ax = plotting.plot_histogram(cgm_df, bins=20)
        assert fig is not None

    def test_without_tir_lines(self, cgm_df):
        fig, ax = plotting.plot_histogram(cgm_df, show_tir_lines=False)
        assert fig is not None
