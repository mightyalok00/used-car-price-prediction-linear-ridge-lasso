"""Deterministic checks for cleaning, EDA summaries, diagnostics, and pricing logic."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data import clean_dataset, inspect_dataset
from src.modeling import (
    build_preprocessor,
    make_pipeline,
    numeric_multicollinearity_diagnostics,
    ridge_coefficient_path,
    select_model_features,
)
from src.reporting import categorical_price_summary, pricing_opportunities, segment_retention_table


def test_shape_formula() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = inspect_dataset(frame, "sample")
    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["total_cells"] == 4


def test_cleaning_removes_invalid_target_and_adds_age() -> None:
    frame = pd.DataFrame({"year": [2020, 2019], "price": ["$10,000", None], "mileage": [100, -1]})
    clean, summary = clean_dataset(frame, "price", reference_year=2023)
    assert len(clean) == 1
    assert clean.iloc[0]["price"] == 10000
    assert clean.iloc[0]["vehicle_age"] == 3
    assert summary["invalid_or_missing_target_rows_removed"] == 1


def test_categorical_summary_includes_binary_features() -> None:
    frame = pd.DataFrame(
        {
            "brand": ["a", "a", "b", "b"],
            "damaged": [0, 1, 0, 1],
            "price": [10_000, 8_000, 20_000, 15_000],
        }
    )
    result = categorical_price_summary(frame, "price")
    assert {"brand", "damaged"}.issubset(set(result["feature"]))


def test_multicollinearity_detects_perfect_numeric_relation() -> None:
    x = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [0, 1, 0, 1]})
    vif, summary = numeric_multicollinearity_diagnostics(x)
    assert np.isinf(vif.loc[vif["feature"] == "a", "vif"]).iloc[0]
    assert summary["features_vif_gt_10"] >= 2


def test_ridge_coefficient_path_reports_each_alpha() -> None:
    x = pd.DataFrame({"mileage": [1, 2, 3, 4, 5], "brand": ["a", "a", "b", "b", "b"]})
    y = pd.Series([10000, 9000, 12000, 11000, 10500], dtype=float)
    model_x, groups = select_model_features(x)
    pipeline = make_pipeline(build_preprocessor(groups, min_frequency=1), Ridge(solver="lsqr"))
    result = ridge_coefficient_path(pipeline, model_x, y, (0.1, 1.0, 10.0))
    assert result["alpha"].tolist() == [0.1, 1.0, 10.0]
    assert (result["l2_coefficient_norm"] >= 0).all()


def test_segment_retention_covers_multiple_segment_families() -> None:
    frame = pd.DataFrame(
        {
            "vehicle_age": [1, 2, 3, 4] * 4,
            "brand": ["a"] * 8 + ["b"] * 8,
            "model": ["m1"] * 8 + ["m2"] * 8,
            "fuel_type": ["gas"] * 8 + ["diesel"] * 8,
            "drivetrain": ["fwd"] * 8 + ["awd"] * 8,
            "price": [20000, 18000, 16000, 14000] * 4,
        }
    )
    result = segment_retention_table(frame, "price", min_listings=4)
    assert {"brand", "model", "fuel_type", "drivetrain"}.issubset(set(result["segment_feature"]))


def test_pricing_opportunity_flags_are_directionally_correct() -> None:
    frame = pd.DataFrame({"price": [80.0, 100.0, 130.0]})
    result = pricing_opportunities(frame, "price", np.array([100.0, 100.0, 100.0]))
    flags = dict(zip(result["price"], result["pricing_flag"]))
    assert flags[80.0] == "potentially_underpriced"
    assert flags[100.0] == "within_20_percent"
    assert flags[130.0] == "potentially_overpriced"
