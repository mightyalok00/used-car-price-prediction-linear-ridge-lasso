"""End-to-end orchestration for all 16 project questions."""

from __future__ import annotations

import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from .config import ProjectConfig
from .data import align_feature_columns, classify_features, clean_dataset, inspect_dataset, load_csv, schema_table
from .modeling import (
    build_preprocessor,
    coefficient_table,
    fit_and_evaluate,
    make_pipeline,
    make_price_stratified_cv,
    numeric_multicollinearity_diagnostics,
    residual_diagnostics,
    ridge_coefficient_path,
    select_model_features,
    tune_model,
)
from .reporting import (
    categorical_price_summary,
    depreciation_table,
    make_eda_plots,
    make_residual_plot,
    numeric_correlations,
    outlier_summary,
    pricing_opportunities,
    save_json,
    segment_retention_table,
    write_summary_report,
)


def _write_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)


def run_analysis(config: ProjectConfig) -> dict[str, Any]:
    """Run, validate, and save the full analysis without fabricating results."""
    # Unknown holdout categories are expected and routed to the infrequent bucket.
    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories.*",
        category=UserWarning,
        module="sklearn.preprocessing._encoders",
    )
    train_raw = load_csv(config.train_path)
    test_raw = load_csv(config.test_path)

    dimensions = pd.DataFrame(
        [inspect_dataset(train_raw, "train.csv"), inspect_dataset(test_raw, "test.csv")]
    )
    _write_csv(dimensions, config.tables_dir / "dataset_dimensions.csv")
    _write_csv(schema_table(train_raw), config.tables_dir / "train_schema.csv")
    _write_csv(schema_table(test_raw), config.tables_dir / "test_schema.csv")

    if list(train_raw.columns) != list(test_raw.columns):
        schema_note = {
            "same_column_order": False,
            "train_only": [c for c in train_raw.columns if c not in test_raw.columns],
            "test_only": [c for c in test_raw.columns if c not in train_raw.columns],
            "behavior": "Only common predictors are used; target must exist in both for external evaluation.",
        }
    else:
        schema_note = {"same_column_order": True, "train_only": [], "test_only": [], "behavior": "All predictors align."}
    save_json(schema_note, config.reports_dir / "schema_alignment.json")

    # Use the observed maximum year across both files so vehicle age is consistent.
    year_values = []
    for frame in (train_raw, test_raw):
        if "year" in frame:
            year_values.append(pd.to_numeric(frame["year"], errors="coerce").max())
    reference_year = int(np.nanmax(year_values)) if year_values else datetime.now().year
    train, train_cleaning = clean_dataset(train_raw, config.target, reference_year)
    test, test_cleaning = clean_dataset(test_raw, config.target, reference_year)
    clean_summaries = {"train": train_cleaning, "test": test_cleaning}
    save_json(clean_summaries, config.reports_dir / "cleaning_summary.json")

    missing = pd.concat(
        [
            train.isna().sum().rename("train_missing"),
            test.isna().sum().rename("test_missing"),
        ],
        axis=1,
    ).reset_index(names="column")
    _write_csv(missing, config.tables_dir / "missing_values_after_row_cleaning.csv")

    classified = classify_features(train, config.target)
    save_json(classified, config.reports_dir / "feature_classification.json")

    eda_columns = [config.target, "mileage", "year", "vehicle_age", "engine_size", "min_mpg", "max_mpg", "avg_mpg"]
    outliers = outlier_summary(train, eda_columns)
    correlations = numeric_correlations(train, config.target)
    categorical = categorical_price_summary(train, config.target)
    _write_csv(outliers, config.tables_dir / "outlier_summary_iqr.csv")
    _write_csv(correlations, config.tables_dir / "numerical_price_correlations.csv")
    _write_csv(categorical, config.tables_dir / "categorical_price_summary_top_levels.csv")
    make_eda_plots(train, config.target, config.figures_dir)

    X_train_raw, X_test_raw = align_feature_columns(train, test, config.target)
    X_train, model_groups = select_model_features(X_train_raw)
    X_test = X_test_raw[X_train.columns].copy()
    y_train = train[config.target].astype(float)
    y_test = test[config.target].astype(float)
    save_json(model_groups, config.reports_dir / "model_feature_plan.json")

    vif_table, multicollinearity_summary = numeric_multicollinearity_diagnostics(X_train)
    _write_csv(vif_table, config.tables_dir / "numeric_multicollinearity_vif.csv")
    save_json(multicollinearity_summary, config.reports_dir / "multicollinearity_summary.json")

    preprocessor = build_preprocessor(model_groups, config.one_hot_min_frequency)
    cv, cv_fold_balance = make_price_stratified_cv(
        y_train, config.cv_folds, config.random_state
    )
    _write_csv(cv_fold_balance, config.tables_dir / "cv_fold_price_balance.csv")
    linear_template = make_pipeline(preprocessor, LinearRegression(n_jobs=config.n_jobs))
    ridge_template = make_pipeline(preprocessor, Ridge(solver="lsqr"))
    lasso_template = make_pipeline(
        preprocessor,
        Lasso(max_iter=5000, tol=1e-3, selection="random", random_state=config.random_state),
    )

    ridge_best, ridge_search = tune_model(
        ridge_template, X_train, np.log1p(y_train), config.ridge_alphas, cv, config.n_jobs
    )
    lasso_best, lasso_search = tune_model(
        lasso_template, X_train, np.log1p(y_train), config.lasso_alphas, cv, config.n_jobs
    )
    _write_csv(ridge_search, config.tables_dir / "ridge_alpha_search.csv")
    _write_csv(lasso_search, config.tables_dir / "lasso_alpha_search.csv")
    ridge_path = ridge_coefficient_path(ridge_template, X_train, y_train, config.ridge_alphas)
    _write_csv(ridge_path, config.tables_dir / "ridge_coefficient_path.csv")

    best_ridge_alpha = float(ridge_best.named_steps["model"].alpha)
    best_lasso_alpha = float(lasso_best.named_steps["model"].alpha)
    templates = [
        ("Linear Regression", linear_template, None),
        ("Ridge Regression", ridge_best, best_ridge_alpha),
        ("Lasso Regression", lasso_best, best_lasso_alpha),
    ]

    fitted_models = {}
    comparison_rows = []
    predictions = {}
    for name, template, alpha in templates:
        fitted, row, pred_train, pred_test = fit_and_evaluate(
            name, template, X_train, y_train, X_test, y_test, cv, alpha
        )
        fitted_models[name] = fitted
        predictions[name] = {"train": pred_train, "test": pred_test}
        comparison_rows.append(row)
        safe_name = name.lower().replace(" ", "_")
        coefs = coefficient_table(fitted)
        _write_csv(coefs, config.tables_dir / f"{safe_name}_coefficients.csv")
        joblib.dump(fitted, config.models_dir / f"{safe_name}.joblib", compress=3)

    comparison = pd.DataFrame(comparison_rows)
    comparison["Generalization_Gap_R2"] = comparison["Train_R2"] - comparison["Test_R2"]
    comparison["Overfit_Assessment"] = np.select(
        [comparison["Generalization_Gap_R2"] > 0.10, comparison["Generalization_Gap_R2"] < -0.05],
        ["possible overfit", "test set easier or sampling shift"],
        default="similar train/test fit",
    )
    _write_csv(comparison, config.tables_dir / "model_comparison.csv")

    linear_residuals = residual_diagnostics(y_test, predictions["Linear Regression"]["test"])
    save_json(linear_residuals, config.reports_dir / "linear_residual_diagnostics.json")
    make_residual_plot(
        y_test,
        predictions["Linear Regression"]["test"],
        config.figures_dir / "linear_regression_residuals.png",
    )

    depreciation = depreciation_table(train, config.target)
    retention = segment_retention_table(train, config.target)
    _write_csv(depreciation, config.tables_dir / "depreciation_by_age_band.csv")
    _write_csv(retention, config.tables_dir / "segment_value_retention.csv")

    best_model_name = comparison.sort_values("RMSE").iloc[0]["Model"]
    pricing = pricing_opportunities(test, config.target, predictions[best_model_name]["test"])
    _write_csv(pricing, config.tables_dir / "pricing_opportunities_all_test_rows.csv")
    _write_csv(pricing.head(100), config.tables_dir / "top_100_potentially_underpriced.csv")
    _write_csv(pricing.tail(100).sort_values("actual_minus_predicted", ascending=False), config.tables_dir / "top_100_potentially_overpriced.csv")

    write_summary_report(
        config.reports_dir / "analysis_summary.md",
        dimensions,
        clean_summaries,
        model_groups,
        correlations,
        comparison,
        linear_residuals,
        pricing,
    )

    metadata = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "train_path": str(config.train_path),
        "test_path": str(config.test_path),
        "target": config.target,
        "cv_folds": config.cv_folds,
        "cv_strategy": "StratifiedKFold on quantile-binned training price",
        "random_state": config.random_state,
        "best_external_test_model": best_model_name,
        "raw_dimensions": dimensions.to_dict("records"),
        "status": "completed",
    }
    save_json(metadata, config.reports_dir / "run_metadata.json")

    required = [
        config.tables_dir / "dataset_dimensions.csv",
        config.tables_dir / "model_comparison.csv",
        config.tables_dir / "pricing_opportunities_all_test_rows.csv",
        config.tables_dir / "numeric_multicollinearity_vif.csv",
        config.tables_dir / "ridge_coefficient_path.csv",
        config.tables_dir / "cv_fold_price_balance.csv",
        config.reports_dir / "analysis_summary.md",
        config.reports_dir / "run_metadata.json",
    ]
    missing_outputs = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f"Validation failed; required outputs missing or empty: {missing_outputs}")

    validation_lines = [
        "VALIDATION STATUS: PASS",
        f"Raw train shape: {tuple(train_raw.shape)}",
        f"Raw test shape: {tuple(test_raw.shape)}",
        f"Clean train rows: {len(train)}",
        f"Clean test rows: {len(test)}",
        f"Models evaluated: {', '.join(comparison['Model'])}",
        f"Best external-test RMSE model: {best_model_name}",
        f"Required output files checked: {len(required)}",
        "No source CSV was copied or modified.",
    ]
    (config.reports_dir / "validation_report.txt").write_text("\n".join(validation_lines), encoding="utf-8")

    return {
        "dimensions": dimensions,
        "cleaning": clean_summaries,
        "feature_classification": classified,
        "model_groups": model_groups,
        "outliers": outliers,
        "correlations": correlations,
        "categorical_summary": categorical,
        "multicollinearity_vif": vif_table,
        "multicollinearity_summary": multicollinearity_summary,
        "ridge_search": ridge_search,
        "ridge_coefficient_path": ridge_path,
        "cv_fold_price_balance": cv_fold_balance,
        "lasso_search": lasso_search,
        "comparison": comparison,
        "residual_diagnostics": linear_residuals,
        "depreciation": depreciation,
        "retention": retention,
        "pricing": pricing,
        "best_model": best_model_name,
        "project_root": str(config.project_root),
    }
