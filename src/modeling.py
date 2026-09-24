"""Leakage-safe Linear, Ridge, and Lasso modeling utilities."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_model_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Remove deterministic/redundant EDA fields while adapting to available columns."""
    drop_if_present = [c for c in ("year", "min_mpg", "max_mpg") if c in X.columns]
    model_X = X.drop(columns=drop_if_present)
    categorical = [c for c in model_X.columns if pd.api.types.is_string_dtype(model_X[c]) or model_X[c].dtype == "object"]
    numeric_all = [c for c in model_X.columns if c not in categorical]
    binary = [c for c in numeric_all if set(model_X[c].dropna().unique()).issubset({0, 1})]
    continuous = [c for c in numeric_all if c not in binary]
    return model_X, {
        "continuous": continuous,
        "binary": binary,
        "categorical": categorical,
        "dropped_redundant": drop_if_present,
    }


def build_preprocessor(groups: dict[str, list[str]], min_frequency: int) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]
    )
    binary_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    drop="first",
                    sparse_output=True,
                ),
            ),
        ]
    )
    transformers = []
    if groups["continuous"]:
        transformers.append(("num", numeric_pipe, groups["continuous"]))
    if groups["binary"]:
        transformers.append(("binary", binary_pipe, groups["binary"]))
    if groups["categorical"]:
        transformers.append(("cat", categorical_pipe, groups["categorical"]))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def make_pipeline(preprocessor: ColumnTransformer, model: Any) -> Pipeline:
    return Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])


def _original_price_rmse_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSE in original price units when model target is log1p(price)."""
    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.maximum(0.0, np.expm1(np.asarray(y_pred_log)))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


ORIGINAL_PRICE_RMSE_SCORER = make_scorer(
    _original_price_rmse_from_log,
    greater_is_better=False,
)



def make_price_stratified_cv(
    y: pd.Series | np.ndarray,
    n_splits: int,
    random_state: int,
    n_bins: int = 10,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Create regression CV folds stratified by quantile-binned target price.

    Stratifying on target quantiles keeps the price distribution more comparable
    across validation folds while preserving shuffled, seeded reproducibility.
    """
    y_series = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
    if len(y_series) < n_splits * 2:
        raise ValueError("Not enough rows to build stable stratified CV folds.")

    max_bins = max(2, min(n_bins, len(y_series) // n_splits))
    # Rank first so duplicate price values cannot collapse qcut bins unpredictably.
    bins = pd.qcut(y_series.rank(method="first"), q=max_bins, labels=False, duplicates="drop")
    counts = pd.Series(bins).value_counts()
    if counts.empty or int(counts.min()) < n_splits:
        raise ValueError("Target bins do not contain enough samples for the requested folds.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(np.zeros(len(y_series)), bins))

    rows = []
    for fold, (_, valid_idx) in enumerate(splits, start=1):
        fold_y = y_series.iloc[valid_idx]
        rows.append(
            {
                "fold": fold,
                "rows": int(len(valid_idx)),
                "price_mean": float(fold_y.mean()),
                "price_median": float(fold_y.median()),
                "price_std": float(fold_y.std(ddof=1)),
                "price_min": float(fold_y.min()),
                "price_max": float(fold_y.max()),
            }
        )
    return splits, pd.DataFrame(rows)


def tune_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_log: np.ndarray,
    alphas: tuple[float, ...],
    cv: list[tuple[np.ndarray, np.ndarray]],
    n_jobs: int,
) -> tuple[Pipeline, pd.DataFrame]:
    search = GridSearchCV(
        pipeline,
        {"model__alpha": list(alphas)},
        scoring=ORIGINAL_PRICE_RMSE_SCORER,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y_log)
    table = pd.DataFrame(search.cv_results_)[
        ["param_model__alpha", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    ].copy()
    table.columns = ["alpha", "mean_train_neg_rmse_price", "mean_cv_neg_rmse_price", "cv_std_price", "rank"]
    return search.best_estimator_, table.sort_values("rank")


def predict_price(fitted: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, np.expm1(fitted.predict(X)))


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_price(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, list[float]]:
    """Evaluate every fold on original-dollar RMSE, despite log-target training."""
    scores: list[float] = []
    y_array = np.asarray(y)
    for train_idx, valid_idx in cv:
        fitted = clone(pipeline).fit(X.iloc[train_idx], np.log1p(y_array[train_idx]))
        pred = predict_price(fitted, X.iloc[valid_idx])
        scores.append(float(np.sqrt(mean_squared_error(y_array[valid_idx], pred))))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), scores


def fit_and_evaluate(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: list[tuple[np.ndarray, np.ndarray]],
    best_alpha: float | None,
) -> tuple[Pipeline, dict[str, Any], np.ndarray, np.ndarray]:
    start = time.perf_counter()
    fitted = clone(pipeline).fit(X_train, np.log1p(y_train))
    fit_seconds = time.perf_counter() - start
    pred_train = predict_price(fitted, X_train)
    pred_test = predict_price(fitted, X_test)
    test_metrics = regression_metrics(y_test, pred_test)
    cv_mean, cv_std, cv_scores = cross_validate_price(pipeline, X_train, y_train, cv)

    coef = np.asarray(fitted.named_steps["model"].coef_).ravel()
    row: dict[str, Any] = {
        "Model": name,
        **test_metrics,
        "CV_RMSE_Mean": cv_mean,
        "CV_RMSE_Std": cv_std,
        "Train_R2": float(r2_score(y_train, pred_train)),
        "Test_R2": test_metrics["R2"],
        "Best_Alpha": best_alpha if best_alpha is not None else "N/A",
        "Active_Features": int(np.sum(np.abs(coef) > 1e-10)),
        "Zero_Coefficients": int(np.sum(np.abs(coef) <= 1e-10)),
        "Fit_Seconds": float(fit_seconds),
        "CV_Fold_RMSE": ", ".join(f"{x:.2f}" for x in cv_scores),
    }
    return fitted, row, pred_train, pred_test


def coefficient_table(fitted: Pipeline) -> pd.DataFrame:
    names = fitted.named_steps["preprocessor"].get_feature_names_out()
    values = np.asarray(fitted.named_steps["model"].coef_).ravel()
    table = pd.DataFrame({"feature": names, "coefficient_log_price": values})
    table["absolute_coefficient"] = table["coefficient_log_price"].abs()
    table["is_zero"] = table["absolute_coefficient"] <= 1e-10
    return table.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)



def numeric_multicollinearity_diagnostics(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute VIF-style diagnostics for numeric predictors without extra dependencies.

    Missing values are median-filled for this diagnostic only. Binary indicators are
    included because they can also be linearly redundant with other numeric fields.
    """
    numeric = X.select_dtypes(include=np.number).copy()
    if numeric.empty:
        return pd.DataFrame(columns=["feature", "vif"]), {"condition_number": float("nan"), "n_numeric_features": 0}

    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.empty:
        return pd.DataFrame(columns=["feature", "vif"]), {"condition_number": float("nan"), "n_numeric_features": 0}

    filled = numeric.fillna(numeric.median(numeric_only=True))
    scaled = (filled - filled.mean()) / filled.std(ddof=0).replace(0, 1)
    matrix = scaled.to_numpy(dtype=float)
    condition_number = float(np.linalg.cond(matrix))

    rows = []
    for col in scaled.columns:
        y = scaled[col].to_numpy(dtype=float)
        other_cols = [c for c in scaled.columns if c != col]
        if not other_cols:
            vif = 1.0
        else:
            x_other = scaled[other_cols].to_numpy(dtype=float)
            r2 = LinearRegression().fit(x_other, y).score(x_other, y)
            vif = float("inf") if r2 >= 0.999999 else float(1.0 / (1.0 - r2))
        rows.append({"feature": col, "vif": vif})

    table = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
    summary = {
        "condition_number": condition_number,
        "n_numeric_features": int(len(scaled.columns)),
        "features_vif_gt_5": int((table["vif"] > 5).sum()),
        "features_vif_gt_10": int((table["vif"] > 10).sum()),
    }
    return table, summary


def ridge_coefficient_path(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    alphas: tuple[float, ...],
) -> pd.DataFrame:
    """Fit Ridge at each alpha and summarize coefficient shrinkage."""
    rows = []
    y_log = np.log1p(np.asarray(y))
    for alpha in alphas:
        candidate = clone(pipeline)
        candidate.set_params(model__alpha=float(alpha))
        fitted = candidate.fit(X, y_log)
        coef = np.asarray(fitted.named_steps["model"].coef_).ravel()
        rows.append(
            {
                "alpha": float(alpha),
                "l1_coefficient_norm": float(np.abs(coef).sum()),
                "l2_coefficient_norm": float(np.sqrt(np.square(coef).sum())),
                "max_absolute_coefficient": float(np.abs(coef).max()) if coef.size else 0.0,
                "active_features": int(np.sum(np.abs(coef) > 1e-10)),
            }
        )
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)

def residual_diagnostics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(y_true) - predictions
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "residual_skew": float(pd.Series(residuals).skew()),
        "residual_kurtosis": float(pd.Series(residuals).kurtosis()),
        "fitted_abs_residual_correlation": float(np.corrcoef(predictions, np.abs(residuals))[0, 1]),
        "interpretation": "A fitted-vs-absolute-residual correlation far from zero suggests non-constant error variance.",
    }
