"""Tests for glycosignal.features -- feature map builders and registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glycosignal import features, registry
from glycosignal.schemas import COL_GLUCOSE, COL_TIMESTAMP


# ─────────────────────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_list_features_nonempty(self):
        names = registry.list_features()
        assert len(names) > 10
        assert "mean_glucose" in names

    def test_list_features_by_category(self):
        risk = registry.list_features(category="risk")
        assert "lbgi" in risk
        assert "gri" in risk

    def test_get_feature_returns_descriptor(self):
        desc = registry.get_feature("cv")
        assert desc.name == "cv"
        assert callable(desc.func)
        assert desc.category == "variability"

    def test_get_feature_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            registry.get_feature("nonexistent_feature_xyz")

    def test_get_feature_metadata_dataframe(self):
        meta = registry.get_feature_metadata()
        assert isinstance(meta, pd.DataFrame)
        assert "name" in meta.columns
        assert "category" in meta.columns
        assert "description" in meta.columns

    def test_get_feature_names_alias(self):
        assert registry.get_feature_names() == registry.list_features()

    def test_registry_contains(self):
        from glycosignal.registry import DEFAULT_REGISTRY
        assert "mean_glucose" in DEFAULT_REGISTRY
        assert "nonexistent_xyz" not in DEFAULT_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# build_feature_vector
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureVector:
    def test_default_features_returned(self, cgm_df):
        vec = features.build_feature_vector(cgm_df)
        assert isinstance(vec, dict)
        assert "mean_glucose" in vec
        assert "cv" in vec
        assert "lbgi" in vec

    def test_custom_feature_names(self, cgm_df):
        vec = features.build_feature_vector(cgm_df, feature_names=["mean_glucose", "sd"])
        assert set(vec.keys()) == {"mean_glucose", "sd"}

    def test_raises_on_unknown_feature(self, cgm_df):
        with pytest.raises(KeyError):
            features.build_feature_vector(cgm_df, feature_names=["bad_feature_xyz"])

    def test_accepts_prepared(self, cgm_prepared):
        vec = features.build_feature_vector(cgm_prepared, feature_names=["mean_glucose"])
        assert "mean_glucose" in vec

    def test_values_are_numeric(self, cgm_df):
        vec = features.build_feature_vector(cgm_df, feature_names=["mean_glucose", "cv", "gri"])
        for name, val in vec.items():
            assert isinstance(val, (int, float)), f"{name} is not numeric"


# ─────────────────────────────────────────────────────────────────────────────
# build_feature_map (long-format)
# ─────────────────────────────────────────────────────────────────────────────

def _make_windowed_df(n_windows: int = 3) -> pd.DataFrame:
    """Create a synthetic long-format windowed DataFrame."""
    rows = []
    for i in range(n_windows):
        ts = pd.date_range(f"2023-01-0{i+1}", periods=288, freq="5min")
        gl = np.random.default_rng(i).normal(120, 25, 288).clip(40, 400)
        for t, g in zip(ts, gl):
            rows.append({
                "window_id": f"S01_2023-01-0{i+1}",
                "subject": "S01",
                "date": f"2023-01-0{i+1}",
                "Timestamp": t,
                "Glucose": g,
            })
    return pd.DataFrame(rows)


class TestBuildFeatureMap:
    def test_output_shape(self):
        wdf = _make_windowed_df(3)
        X = features.build_feature_map(wdf, show_progress=False)
        assert X.shape[0] == 3  # 3 windows

    def test_metadata_included_by_default(self):
        wdf = _make_windowed_df(2)
        X = features.build_feature_map(wdf, show_progress=False)
        assert "window_id" in X.columns
        assert "subject" in X.columns

    def test_metadata_excluded(self):
        wdf = _make_windowed_df(2)
        X = features.build_feature_map(wdf, include_metadata=False, show_progress=False)
        assert "window_id" not in X.columns

    def test_custom_feature_names(self):
        wdf = _make_windowed_df(2)
        X = features.build_feature_map(
            wdf, feature_names=["mean_glucose", "cv"], show_progress=False
        )
        feature_cols = [c for c in X.columns if c in ("mean_glucose", "cv")]
        assert len(feature_cols) == 2

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"window_id": ["w1"], "Timestamp": ["2023-01-01"]})
        with pytest.raises(ValueError, match="Glucose"):
            features.build_feature_map(df, show_progress=False)


# ─────────────────────────────────────────────────────────────────────────────
# build_feature_table
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureTable:
    def test_multiple_records(self, cgm_df):
        records = [cgm_df.copy(), cgm_df.copy()]
        result = features.build_feature_table(records, show_progress=False)
        assert result.shape[0] == 2

    def test_with_record_ids(self, cgm_df):
        records = [cgm_df.copy(), cgm_df.copy()]
        result = features.build_feature_table(
            records, record_ids=["R1", "R2"], show_progress=False
        )
        assert "record_id" in result.columns
        assert list(result["record_id"]) == ["R1", "R2"]

    def test_mismatched_ids_raises(self, cgm_df):
        with pytest.raises(ValueError, match="length"):
            features.build_feature_table([cgm_df], record_ids=["A", "B"], show_progress=False)


# ─────────────────────────────────────────────────────────────────────────────
# build_feature_map_wide (backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def _make_wide_windowed_df(n_rows: int = 2) -> pd.DataFrame:
    """Synthetic wide-format windowed DataFrame (HH:MM columns)."""
    time_cols = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
    rng = np.random.default_rng(99)
    rows = []
    for i in range(n_rows):
        row = {"subject": f"S0{i}", "date": f"2023-01-0{i+1}"}
        row.update({tc: float(rng.normal(120, 25)) for tc in time_cols})
        rows.append(row)
    return pd.DataFrame(rows)


class TestBuildFeatureMapWide:
    def test_output_shape(self):
        df = _make_wide_windowed_df(3)
        X = features.build_feature_map_wide(df, show_progress=False)
        assert X.shape[0] == 3

    def test_feature_columns_present(self):
        df = _make_wide_windowed_df(2)
        X = features.build_feature_map_wide(
            df, feature_names=["mean_glucose", "cv"], show_progress=False
        )
        assert "mean_glucose" in X.columns
        assert "cv" in X.columns

    def test_no_time_cols_raises(self):
        df = pd.DataFrame({"subject": ["S1"], "Glucose": [120.0]})
        with pytest.raises(ValueError, match="No time-grid columns"):
            features.build_feature_map_wide(df, show_progress=False)
