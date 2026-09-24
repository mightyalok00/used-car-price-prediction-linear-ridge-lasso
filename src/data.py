"""Loading, inspection, cleaning, and feature classification."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MISSING_TEXT = {"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA, "n/a": pd.NA}


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV and fail with a clear path-specific error."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def inspect_dataset(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Return the core Q1 inspection facts, including df.shape logic."""
    rows = int(df.shape[0])
    columns = int(df.shape[1])
    return {
        "dataset": name,
        "rows": rows,
        "columns": columns,
        "total_cells": rows * columns,
        "shape_formula": "rows = df.shape[0]; columns = df.shape[1]; total cells = rows * columns",
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def schema_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "missing_pct": [float(df[c].isna().mean() * 100) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def _parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_dataset(df: pd.DataFrame, target: str, reference_year: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean without learning imputation values; model imputers are fit on train only."""
    clean = df.copy()
    raw_rows = len(clean)
    raw_duplicates = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().copy()

    # Normalize free-text categories before encoding.
    for col in clean.select_dtypes(include=["object", "string"]).columns:
        clean[col] = clean[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        clean[col] = clean[col].str.lower().replace(MISSING_TEXT)
        # scikit-learn imputers expect np.nan rather than pandas' nullable pd.NA.
        clean[col] = clean[col].astype(object).where(clean[col].notna(), np.nan)

    if target not in clean.columns:
        raise KeyError(f"Required target column '{target}' was not found. Available: {list(clean.columns)}")
    clean[target] = _parse_numeric(clean[target])

    invalid_counts: dict[str, int] = {}
    rules = {
        "year": lambda s: (s < 1886) | (s > datetime.now().year + 1),
        "mileage": lambda s: s < 0,
        "engine_size": lambda s: s <= 0,
        "min_mpg": lambda s: s <= 0,
        "max_mpg": lambda s: s <= 0,
    }
    for col, invalid_rule in rules.items():
        if col in clean.columns:
            numeric = pd.to_numeric(clean[col], errors="coerce")
            bad = invalid_rule(numeric).fillna(False)
            invalid_counts[col] = int(bad.sum())
            clean[col] = numeric.mask(bad)

    valid_target = clean[target].notna() & (clean[target] > 0)
    removed_invalid_target = int((~valid_target).sum())
    clean = clean.loc[valid_target].copy()

    if reference_year is None:
        observed_max = pd.to_numeric(clean.get("year"), errors="coerce").max() if "year" in clean else np.nan
        reference_year = int(observed_max) if pd.notna(observed_max) else datetime.now().year
    if "year" in clean.columns:
        clean["vehicle_age"] = reference_year - clean["year"]
        clean.loc[clean["vehicle_age"] < 0, "vehicle_age"] = np.nan
    if {"min_mpg", "max_mpg"}.issubset(clean.columns):
        clean["avg_mpg"] = clean[["min_mpg", "max_mpg"]].mean(axis=1)

    summary = {
        "raw_rows": int(raw_rows),
        "duplicate_rows_removed": raw_duplicates,
        "invalid_or_missing_target_rows_removed": removed_invalid_target,
        "clean_rows": int(len(clean)),
        "reference_year_for_vehicle_age": int(reference_year),
        "invalid_values_set_to_missing": invalid_counts,
        "remaining_missing_values": int(clean.isna().sum().sum()),
        "note": "Remaining predictor gaps are imputed inside leakage-safe pipelines fit only on training data.",
    }
    return clean, summary


def classify_features(df: pd.DataFrame, target: str) -> dict[str, list[str]]:
    features = [c for c in df.columns if c != target]
    categorical = [c for c in features if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object"]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    binary = [c for c in numeric if set(df[c].dropna().unique()).issubset({0, 1})]
    continuous = [c for c in numeric if c not in binary]
    identifiers = [c for c in features if c.lower() in {"id", "vin", "listing_id", "url"}]
    ordinal = [c for c in ("year", "vehicle_age") if c in features]
    return {
        "numerical_continuous": continuous,
        "binary_indicator": binary,
        "categorical_nominal": categorical,
        "ordinal": ordinal,
        "identifier": identifiers,
        "target": [target],
    }


def align_feature_columns(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use only predictors present in both files, in train order."""
    common = [c for c in train.columns if c != target and c in test.columns]
    if not common:
        raise ValueError("Train and test have no common predictor columns.")
    return train[common].copy(), test[common].copy()
