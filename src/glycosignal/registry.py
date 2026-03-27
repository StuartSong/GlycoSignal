"""
glycosignal.registry
====================

Feature registry: a catalogue of every registered glycemic feature with
metadata, and the machinery to compute features by name.

The module-level :data:`DEFAULT_REGISTRY` is populated at import time with
all built-in features from :mod:`glycosignal.metrics`.

Usage
-----
    >>> from glycosignal import registry
    >>> registry.list_features()
    ['mean_glucose', 'median_glucose', ...]
    >>> registry.get_feature_metadata()  # returns a DataFrame
    >>> registry.compute('cv', df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .schemas import CGMInput


# ─────────────────────────────────────────────────────────────────────────────
# Feature descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureDescriptor:
    """Metadata for a single registered glycemic feature.

    Attributes
    ----------
    name : str
        Public feature name (used as dict key / column name in feature maps).
    func : Callable
        Callable that accepts a ``DataFrame`` or ``PreparedCGMData`` and returns
        a scalar (float or int).
    description : str
        Short human-readable description of the feature.
    category : str
        Feature category: ``"basic_stats"``, ``"variability"``,
        ``"time_in_range"``, ``"risk"``, ``"excursion"``, or ``"peak"``.
    output_type : str
        Python type name of the return value: ``"float"`` or ``"int"``.
    """

    name: str
    func: Callable[[CGMInput], float | int]
    description: str
    category: str
    output_type: str = "float"


# ─────────────────────────────────────────────────────────────────────────────
# Registry class
# ─────────────────────────────────────────────────────────────────────────────

class FeatureRegistry:
    """A catalogue of glycemic feature functions with metadata.

    Features are registered by name and can be retrieved, listed, or computed
    individually or in bulk.

    Examples
    --------
    >>> reg = FeatureRegistry()
    >>> reg.register("mean_glucose", metrics.mean_glucose, "Mean glucose", "basic_stats")
    >>> reg.compute("mean_glucose", df)
    """

    def __init__(self) -> None:
        self._features: dict[str, FeatureDescriptor] = {}

    def register(
        self,
        name: str,
        func: Callable[[CGMInput], float | int],
        description: str,
        category: str,
        output_type: str = "float",
    ) -> None:
        """Register a feature.

        Parameters
        ----------
        name : str
            Unique feature name.  Registering an existing name overwrites the
            previous entry.
        func : Callable
            Function accepting ``(data: DataFrame | PreparedCGMData)`` and
            returning a scalar.
        description : str
            Short description shown in :meth:`get_feature_metadata`.
        category : str
            Feature category for grouping.
        output_type : str
            Return type (``"float"`` or ``"int"``).
        """
        self._features[name] = FeatureDescriptor(
            name=name,
            func=func,
            description=description,
            category=category,
            output_type=output_type,
        )

    def get(self, name: str) -> FeatureDescriptor:
        """Return the :class:`FeatureDescriptor` for *name*.

        Parameters
        ----------
        name : str
            Feature name.

        Returns
        -------
        FeatureDescriptor

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._features:
            available = sorted(self._features)
            raise KeyError(
                f"Feature {name!r} is not registered. "
                f"Call list_features() to see available features. "
                f"Available: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
        return self._features[name]

    def list_features(self, category: str | None = None) -> list[str]:
        """Return sorted list of registered feature names.

        Parameters
        ----------
        category : str | None
            If provided, filter to features in that category.

        Returns
        -------
        list[str]
            Sorted feature names.
        """
        if category is None:
            return sorted(self._features)
        return sorted(
            name for name, desc in self._features.items()
            if desc.category == category
        )

    def get_feature_names(self, category: str | None = None) -> list[str]:
        """Alias for :meth:`list_features`.

        Parameters
        ----------
        category : str | None
            Optional category filter.

        Returns
        -------
        list[str]
        """
        return self.list_features(category=category)

    def get_feature_metadata(self) -> pd.DataFrame:
        """Return a DataFrame describing all registered features.

        Returns
        -------
        pd.DataFrame
            Columns: ``name``, ``description``, ``category``, ``output_type``.
            Sorted by category then name.
        """
        rows = [
            {
                "name": desc.name,
                "description": desc.description,
                "category": desc.category,
                "output_type": desc.output_type,
            }
            for desc in self._features.values()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["category", "name"]).reset_index(drop=True)
        return df

    def compute(self, name: str, data: CGMInput) -> float | int:
        """Compute a single feature by name.

        Parameters
        ----------
        name : str
            Registered feature name.
        data : DataFrame or PreparedCGMData
            Input data.

        Returns
        -------
        float or int
            Feature value.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        return self.get(name).func(data)

    def compute_all(
        self,
        data: CGMInput,
        names: list[str] | None = None,
    ) -> dict[str, float | int]:
        """Compute multiple features and return a dict.

        Parameters
        ----------
        data : DataFrame or PreparedCGMData
            Input data.  When computing many features, pass a
            :class:`~glycosignal.schemas.PreparedCGMData` for efficiency.
        names : list[str] | None
            Feature names to compute.  ``None`` computes all registered features.

        Returns
        -------
        dict[str, float | int]
            Mapping from feature name to value.

        Raises
        ------
        KeyError
            If any name in *names* is not registered.
        """
        from .schemas import _ensure_prepared

        d = _ensure_prepared(data)
        target_names = names if names is not None else sorted(self._features)
        return {name: self.get(name).func(d) for name in target_names}

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self._features)} features)"


# ─────────────────────────────────────────────────────────────────────────────
# Default registry population
# ─────────────────────────────────────────────────────────────────────────────

def _build_default_registry() -> FeatureRegistry:
    """Construct and return the default registry with all built-in features."""
    from . import metrics as m

    reg = FeatureRegistry()

    # Basic stats
    for name, func, desc in [
        ("mean_glucose", m.mean_glucose, "Mean glucose (mg/dL)"),
        ("median_glucose", m.median_glucose, "Median glucose (mg/dL)"),
        ("min_glucose", m.min_glucose, "Minimum glucose (mg/dL)"),
        ("max_glucose", m.max_glucose, "Maximum glucose (mg/dL)"),
        ("q1_glucose", m.q1_glucose, "25th percentile glucose (mg/dL)"),
        ("q3_glucose", m.q3_glucose, "75th percentile glucose (mg/dL)"),
    ]:
        reg.register(name, func, desc, "basic_stats")

    # Variability
    for name, func, desc in [
        ("sd", m.sd, "Standard deviation of glucose (mg/dL)"),
        ("cv", m.cv, "Coefficient of variation (%)"),
        ("j_index", m.j_index, "J-index: 0.001 × (mean + SD)²"),
        ("mage", m.mage, "Mean Amplitude of Glucose Excursions (mg/dL)"),
        ("conga24", m.conga24, "CONGA24: SD of glucose differences 24h apart (mg/dL)"),
    ]:
        reg.register(name, func, desc, "variability")

    # Time-in-range (standard clinical thresholds -- fixed-parameter versions)
    reg.register(
        "tir_70_180_min",
        lambda d: m.time_in_range_minutes(d, low=70, high=180),
        "Minutes in target range 70–180 mg/dL",
        "time_in_range",
    )
    reg.register(
        "tir_70_180_pct",
        lambda d: m.time_in_range_percent(d, low=70, high=180),
        "Percent time in target range 70–180 mg/dL",
        "time_in_range",
    )
    reg.register(
        "tir_70_140_min",
        lambda d: m.time_in_range_minutes(d, low=70, high=140),
        "Minutes in tight range 70–140 mg/dL",
        "time_in_range",
    )
    reg.register(
        "tir_70_140_pct",
        lambda d: m.time_in_range_percent(d, low=70, high=140),
        "Percent time in tight range 70–140 mg/dL",
        "time_in_range",
    )
    reg.register(
        "tbr_70_min",
        lambda d: m.time_below_range_minutes(d, threshold=70),
        "Minutes below 70 mg/dL (level 1 hypoglycemia)",
        "time_in_range",
    )
    reg.register(
        "tbr_70_pct",
        lambda d: m.time_below_range_percent(d, threshold=70),
        "Percent time below 70 mg/dL",
        "time_in_range",
    )
    reg.register(
        "tbr_54_min",
        lambda d: m.time_below_range_minutes(d, threshold=54),
        "Minutes below 54 mg/dL (level 2 hypoglycemia)",
        "time_in_range",
    )
    reg.register(
        "tbr_54_pct",
        lambda d: m.time_below_range_percent(d, threshold=54),
        "Percent time below 54 mg/dL",
        "time_in_range",
    )
    reg.register(
        "tar_180_min",
        lambda d: m.time_above_range_minutes(d, threshold=180),
        "Minutes above 180 mg/dL (level 1 hyperglycemia)",
        "time_in_range",
    )
    reg.register(
        "tar_180_pct",
        lambda d: m.time_above_range_percent(d, threshold=180),
        "Percent time above 180 mg/dL",
        "time_in_range",
    )
    reg.register(
        "tar_250_min",
        lambda d: m.time_above_range_minutes(d, threshold=250),
        "Minutes above 250 mg/dL (level 2 hyperglycemia)",
        "time_in_range",
    )
    reg.register(
        "tar_250_pct",
        lambda d: m.time_above_range_percent(d, threshold=250),
        "Percent time above 250 mg/dL",
        "time_in_range",
    )

    # Risk indices
    for name, func, desc in [
        ("lbgi", m.lbgi, "Low Blood Glucose Index"),
        ("hbgi", m.hbgi, "High Blood Glucose Index"),
        ("adrr", m.adrr, "Average Daily Risk Range"),
        ("gri", m.gri, "Glucose Risk Index (Klonoff et al. 2023)"),
    ]:
        reg.register(name, func, desc, "risk")

    # Excursion metrics
    for name, func, desc in [
        ("mean_glucose_excursion", m.mean_glucose_excursion, "Mean of readings outside mean ± 1SD (mg/dL)"),
        ("mean_glucose_normal", m.mean_glucose_normal, "Mean of readings inside mean ± 1SD (mg/dL)"),
    ]:
        reg.register(name, func, desc, "excursion")

    # Peak counts
    reg.register(
        "peaks_above_180",
        lambda d: m.count_peaks(d, threshold=180),
        "Number of hyperglycemic excursion episodes (>180 mg/dL)",
        "peak",
        output_type="int",
    )
    reg.register(
        "peaks_above_250",
        lambda d: m.count_peaks(d, threshold=250),
        "Number of severe hyperglycemic episodes (>250 mg/dL)",
        "peak",
        output_type="int",
    )
    reg.register(
        "peaks_above_140",
        lambda d: m.count_peaks(d, threshold=140),
        "Number of postprandial-range excursions (>140 mg/dL)",
        "peak",
        output_type="int",
    )

    return reg


#: The global default feature registry used by :mod:`glycosignal.features`.
DEFAULT_REGISTRY: FeatureRegistry = _build_default_registry()

#: Default core feature names used when ``feature_names=None``.
DEFAULT_FEATURE_NAMES: list[str] = [
    "mean_glucose",
    "median_glucose",
    "min_glucose",
    "max_glucose",
    "q1_glucose",
    "q3_glucose",
    "sd",
    "cv",
    "j_index",
    "mage",
    "tir_70_180_pct",
    "tir_70_140_pct",
    "tbr_70_pct",
    "tbr_54_pct",
    "tar_180_pct",
    "tar_250_pct",
    "lbgi",
    "hbgi",
    "adrr",
    "gri",
]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions (delegate to DEFAULT_REGISTRY)
# ─────────────────────────────────────────────────────────────────────────────

def list_features(category: str | None = None) -> list[str]:
    """List all registered feature names.

    Parameters
    ----------
    category : str | None
        Optional filter by category.

    Returns
    -------
    list[str]
        Sorted feature names.
    """
    return DEFAULT_REGISTRY.list_features(category=category)


def get_feature(name: str) -> FeatureDescriptor:
    """Get the :class:`FeatureDescriptor` for a feature by name.

    Parameters
    ----------
    name : str
        Feature name.

    Returns
    -------
    FeatureDescriptor

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    return DEFAULT_REGISTRY.get(name)


def get_feature_names(category: str | None = None) -> list[str]:
    """Return sorted list of registered feature names.

    Parameters
    ----------
    category : str | None
        Optional category filter.

    Returns
    -------
    list[str]
    """
    return DEFAULT_REGISTRY.get_feature_names(category=category)


def get_feature_metadata() -> pd.DataFrame:
    """Return a DataFrame with metadata for all registered features.

    Returns
    -------
    pd.DataFrame
        Columns: ``name``, ``description``, ``category``, ``output_type``.
    """
    return DEFAULT_REGISTRY.get_feature_metadata()
