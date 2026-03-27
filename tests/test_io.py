"""Tests for glycosignal.io -- data loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from glycosignal import io
from glycosignal.schemas import COL_GLUCOSE, COL_TIMESTAMP


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# load_csv
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCsv:
    def test_canonical_columns(self, tmp_path):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:00", "2023-01-01 00:05"],
            "Glucose": [120.0, 125.0],
        })
        path = tmp_path / "test.csv"
        _write_csv(df, path)
        result = io.load_csv(path)
        assert COL_TIMESTAMP in result.columns
        assert COL_GLUCOSE in result.columns
        assert len(result) == 2

    def test_auto_detect_alternative_names(self, tmp_path):
        df = pd.DataFrame({
            "time": ["2023-01-01 00:00", "2023-01-01 00:05"],
            "gl": [120.0, 125.0],
        })
        path = tmp_path / "test.csv"
        _write_csv(df, path)
        result = io.load_csv(path)
        assert COL_TIMESTAMP in result.columns
        assert COL_GLUCOSE in result.columns

    def test_explicit_column_names(self, tmp_path):
        df = pd.DataFrame({
            "time_utc": ["2023-01-01 00:00", "2023-01-01 00:05"],
            "bg_mg": [120.0, 125.0],
        })
        path = tmp_path / "test.csv"
        _write_csv(df, path)
        result = io.load_csv(path, timestamp_col="time_utc", glucose_col="bg_mg")
        assert COL_TIMESTAMP in result.columns
        assert COL_GLUCOSE in result.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            io.load_csv(tmp_path / "nonexistent.csv")

    def test_missing_glucose_col_raises(self, tmp_path):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:00"],
            "other_col": [1.0],
        })
        path = tmp_path / "bad.csv"
        _write_csv(df, path)
        with pytest.raises(ValueError, match="Glucose"):
            io.load_csv(path)

    def test_subject_col_detected(self, tmp_path):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:00"],
            "Glucose": [120.0],
            "id": ["P001"],
        })
        path = tmp_path / "test.csv"
        _write_csv(df, path)
        result = io.load_csv(path, subject_col="id")
        assert "subject" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# load_cgm_folder
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCgmFolder:
    def test_loads_multiple_files(self, tmp_path):
        for i, name in enumerate(["subj_001.csv", "subj_002.csv"]):
            df = pd.DataFrame({
                "Timestamp": [f"2023-01-0{i+1} 00:00", f"2023-01-0{i+1} 00:05"],
                "Glucose": [120.0 + i, 125.0 + i],
            })
            _write_csv(df, tmp_path / name)

        result = io.load_cgm_folder(tmp_path, show_progress=False)
        assert len(result) == 4
        assert "filename" in result.columns
        assert "subject" in result.columns

    def test_empty_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            io.load_cgm_folder(tmp_path, show_progress=False)

    def test_nonexistent_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            io.load_cgm_folder("/nonexistent/path/", show_progress=False)


# ─────────────────────────────────────────────────────────────────────────────
# load_cgm_file
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCgmFile:
    def test_multi_subject_file(self, tmp_path):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:00", "2023-01-01 00:00"],
            "Glucose": [100.0, 110.0],
            "id": ["P001", "P002"],
        })
        path = tmp_path / "multi.csv"
        _write_csv(df, path)
        result = io.load_cgm_file(path, subject_col="id")
        assert "subject" in result.columns
        assert result["subject"].dtype == object

    def test_no_subject_col_raises(self, tmp_path):
        df = pd.DataFrame({
            "Timestamp": ["2023-01-01 00:00"],
            "Glucose": [100.0],
        })
        path = tmp_path / "no_subject.csv"
        _write_csv(df, path)
        with pytest.raises(ValueError, match="subject"):
            io.load_cgm_file(path)
