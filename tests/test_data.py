"""Deterministic checks for cleaning, preprocessing, diagnostics, metrics, and pricing logic."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data import align_feature_columns, clean_dataset, inspect_dataset
from src.modeling import (
    build_preprocessor,
    make_pipeline,
    numeric_multicollinearity_diagnostics,
    predict_price,
    regression_metrics,
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


def test_align_feature_columns_uses_only_shared_predictors() -> None:
    train = pd.DataFrame({"a": [1], "b": [2], "price": [100]})
    test = pd.DataFrame({"a": [3], "c": [4], "price": [110]})
    train_x, test_x = align_feature_columns(train, test, "price")
    assert list(train_x.columns) == ["a"]
    assert list(test_x.columns) == ["a"]


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


def test_regression_metrics_match_known_values() -> None:
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 2.0
    assert metrics["MSE"] == 4.0
    assert metrics["RMSE"] == 2.0


def test_preprocessing_handles_unseen_category() -> None:
    x = pd.DataFrame(
        {
            "mileage": [10_000, 20_000, 30_000, 40_000],
            "brand": ["a", "a", "b", "b"],
        }
    )
    y = pd.Series([20_000, 18_000, 16_000, 14_000], dtype=float)
    model_x, groups = select_model_features(x)
    pipeline = make_pipeline(build_preprocessor(groups, min_frequency=1), Ridge(alpha=1.0, solver="lsqr"))
    fitted = pipeline.fit(model_x, np.log1p(y))
    unseen = pd.DataFrame({"mileage": [25_000], "brand": ["never_seen_before"]})
    prediction = predict_price(fitted, unseen)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_predict_price_clips_negative_values_to_zero() -> None:
    class FakeModel:
        def predict(self, x):
            return np.array([-10.0, 0.0, 1.0])

    prediction = predict_price(FakeModel(), pd.DataFrame({"x": [1, 2, 3]}))
    assert (prediction >= 0).all()
    assert prediction[0] == 0.0


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
