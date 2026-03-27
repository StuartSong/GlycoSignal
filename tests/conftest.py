"""
Shared pytest fixtures for GlycoSignal test suite.

All fixtures use small synthetic DataFrames -- no external data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_cgm_df(
    n: int = 288,
    start: str = "2023-01-01 00:00",
    freq: str = "5min",
    mean: float = 120.0,
    std: float = 30.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic CGM DataFrame with realistic glucose values."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n, freq=freq)
    gl = np.clip(rng.normal(mean, std, n), 40, 400)
    return pd.DataFrame({"Timestamp": ts, "Glucose": gl})


@pytest.fixture
def cgm_df():
    """288-reading (24h) synthetic CGM DataFrame."""
    return _make_cgm_df()


@pytest.fixture
def cgm_df_multi():
    """Two-day CGM DataFrame for multi-day metric tests."""
    return _make_cgm_df(n=576, start="2023-01-01 00:00")


@pytest.fixture
def cgm_df_with_hypo():
    """CGM DataFrame with a clear hypoglycemic episode."""
    df = _make_cgm_df(n=288, mean=140, std=20)
    # Insert a 60-min hypo episode at reading 100–111 (5 * 12 = 60 min)
    df.loc[100:111, "Glucose"] = 55.0
    return df


@pytest.fixture
def cgm_df_with_hyper():
    """CGM DataFrame with a clear hyperglycemic episode."""
    df = _make_cgm_df(n=288, mean=100, std=15)
    df.loc[50:70, "Glucose"] = 280.0
    return df


@pytest.fixture
def cgm_df_with_gaps():
    """CGM DataFrame with a large gap (60 missing readings)."""
    df = _make_cgm_df(n=288)
    df.loc[120:179, "Glucose"] = np.nan
    return df


@pytest.fixture
def cgm_prepared(cgm_df):
    """PreparedCGMData built from the standard 24h fixture."""
    from glycosignal.schemas import prepare
    return prepare(cgm_df)


@pytest.fixture
def multi_subject_df():
    """Two-subject long-format CGM DataFrame."""
    df1 = _make_cgm_df(n=288, start="2023-01-01 00:00", seed=1)
    df1["subject"] = "S01"
    df2 = _make_cgm_df(n=288, start="2023-01-01 00:00", seed=2)
    df2["subject"] = "S02"
    return pd.concat([df1, df2], ignore_index=True)
