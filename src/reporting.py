"""EDA, plots, business tables, and plain-language report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook")


def save_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "outlier_count": int(((s < lower) | (s > upper)).sum()),
                "outlier_pct": float(((s < lower) | (s > upper)).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def numeric_correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if target not in numeric.columns:
        return pd.DataFrame(columns=["feature", "pearson_with_price", "spearman_with_price"])
    pearson = numeric.corr(numeric_only=True)[target]
    spearman = numeric.corr(method="spearman", numeric_only=True)[target]
    return (
        pd.DataFrame({"feature": pearson.index, "pearson_with_price": pearson.values, "spearman_with_price": spearman.values})
        .query("feature != @target")
        .assign(abs_pearson=lambda x: x["pearson_with_price"].abs())
        .sort_values("abs_pearson", ascending=False)
        .reset_index(drop=True)
    )


def categorical_price_summary(df: pd.DataFrame, target: str, max_levels: int = 15) -> pd.DataFrame:
    """Summarize nominal and binary categories against price.

    Binary indicators are included because assignment Q6 explicitly asks how
    characteristics such as accident/damage history affect average/median price.
    """
    tables = []
    candidates = []
    for col in df.columns:
        if col == target:
            continue
        is_text = pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object"
        non_null = df[col].dropna()
        is_binary = (
            pd.api.types.is_numeric_dtype(df[col])
            and not non_null.empty
            and set(non_null.unique()).issubset({0, 1})
        )
        if is_text or is_binary:
            candidates.append(col)

    for col in candidates:
        series = df[col].astype(object).where(df[col].notna(), "unknown")
        top = series.value_counts().head(max_levels).index
        part = (
            df.assign(_category=series)
            .loc[lambda x: x["_category"].isin(top)]
            .groupby("_category", dropna=False)[target]
            .agg(count="size", mean_price="mean", median_price="median")
            .reset_index()
            .rename(columns={"_category": "category"})
        )
        part.insert(0, "feature", col)
        tables.append(part)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def make_eda_plots(df: pd.DataFrame, target: str, figures_dir: Path) -> None:
    price = df[target].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(price, bins=60, ax=axes[0], color="#2a6f97")
    axes[0].set_title("Listing price distribution")
    axes[0].set_xlabel("Price")
    sns.histplot(np.log1p(price), bins=60, ax=axes[1], color="#d97706")
    axes[1].set_title("log(1 + price) distribution")
    axes[1].set_xlabel("log(1 + price)")
    fig.tight_layout()
    fig.savefig(figures_dir / "price_distribution_and_log.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    available = [c for c in ["mileage", "vehicle_age", "engine_size", "avg_mpg"] if c in df.columns]
    if available:
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, available):
            sns.scatterplot(data=sample, x=col, y=target, alpha=0.3, s=14, ax=ax)
            ax.set_title(f"{col} vs price")
        fig.tight_layout()
        fig.savefig(figures_dir / "numeric_features_vs_price.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def make_residual_plot(y_true: pd.Series, predictions: np.ndarray, path: Path) -> None:
    residuals = np.asarray(y_true) - predictions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.3, s=15, ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Residuals vs fitted values", xlabel="Predicted price", ylabel="Residual")
    sns.histplot(residuals, bins=60, ax=axes[1], color="#6a4c93")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def depreciation_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if "vehicle_age" not in df.columns:
        return pd.DataFrame({"note": ["Skipped: vehicle_age could not be derived because year is unavailable."]})
    temp = df.dropna(subset=["vehicle_age", target]).copy()
    temp["age_band"] = pd.cut(temp["vehicle_age"], bins=[-1, 2, 5, 8, 12, 20, np.inf], labels=["0-2", "3-5", "6-8", "9-12", "13-20", "21+"])
    return temp.groupby("age_band", observed=True)[target].agg(listings="size", mean_price="mean", median_price="median").reset_index()


def segment_retention_table(
    df: pd.DataFrame,
    target: str,
    segment_features: tuple[str, ...] = ("brand", "model", "fuel_type", "drivetrain"),
    min_listings: int = 100,
) -> pd.DataFrame:
    """Estimate age-related value retention across every supported segment type."""
    if "vehicle_age" not in df.columns:
        return pd.DataFrame({"note": ["Skipped: vehicle_age is unavailable."]})

    rows = []
    for segment in [c for c in segment_features if c in df.columns]:
        temp = df.dropna(subset=[segment, "vehicle_age", target]).copy()
        counts = temp[segment].value_counts()
        keep = counts[counts >= min_listings].index
        temp = temp[temp[segment].isin(keep)]
        for value, group in temp.groupby(segment):
            if group["vehicle_age"].nunique() < 2:
                continue
            slope = np.polyfit(group["vehicle_age"], np.log1p(group[target]), 1)[0]
            rows.append(
                {
                    "segment_feature": segment,
                    "segment": value,
                    "listings": len(group),
                    "annual_log_price_slope": slope,
                    "approx_annual_pct_change": np.expm1(slope) * 100,
                }
            )
    if not rows:
        return pd.DataFrame({"note": ["No supported segments met the minimum-listing threshold."]})
    return pd.DataFrame(rows).sort_values(
        ["segment_feature", "approx_annual_pct_change"],
        ascending=[True, False],
    ).reset_index(drop=True)


def pricing_opportunities(test_df: pd.DataFrame, target: str, predictions: np.ndarray) -> pd.DataFrame:
    result = test_df.copy()
    result["predicted_price"] = predictions
    result["actual_minus_predicted"] = result[target] - result["predicted_price"]
    result["predicted_minus_actual"] = -result["actual_minus_predicted"]
    result["difference_pct_of_predicted"] = np.where(
        result["predicted_price"] > 0,
        result["actual_minus_predicted"] / result["predicted_price"] * 100,
        np.nan,
    )
    result["pricing_flag"] = np.select(
        [result["difference_pct_of_predicted"] <= -20, result["difference_pct_of_predicted"] >= 20],
        ["potentially_underpriced", "potentially_overpriced"],
        default="within_20_percent",
    )
    return result.sort_values("predicted_minus_actual", ascending=False)


def write_summary_report(
    path: Path,
    dimensions: pd.DataFrame,
    clean_summaries: dict[str, Any],
    feature_groups: dict[str, list[str]],
    correlations: pd.DataFrame,
    model_comparison: pd.DataFrame,
    residuals: dict[str, float],
    pricing: pd.DataFrame,
) -> None:
    best = model_comparison.sort_values("RMSE").iloc[0]
    top_pos = correlations.sort_values("pearson_with_price", ascending=False).head(3)
    top_neg = correlations.sort_values("pearson_with_price").head(3)
    under = int((pricing["pricing_flag"] == "potentially_underpriced").sum())
    over = int((pricing["pricing_flag"] == "potentially_overpriced").sum())
    lines = [
        "# Analysis Summary",
        "",
        "## Dataset dimensions and formula",
        "",
        "Pandas returns `(rows, columns)` through `df.shape`. Therefore `rows = df.shape[0]`, `columns = df.shape[1]`, and `total cells = rows × columns`.",
        "",
    ]
    for row in dimensions.to_dict("records"):
        lines.append(f"- {row['dataset']}: {row['rows']:,} rows × {row['columns']:,} columns = {row['total_cells']:,} cells.")
    lines += [
        "",
        "## Cleaning and feature handling",
        "",
        f"- Train cleaning: {clean_summaries['train']}",
        f"- Test cleaning: {clean_summaries['test']}",
        f"- Continuous model features: {feature_groups['continuous']}",
        f"- Binary model features: {feature_groups['binary']}",
        f"- Categorical model features: {feature_groups['categorical']}",
        f"- Redundant fields omitted from modeling: {feature_groups['dropped_redundant']}",
        "- Numerical medians, category modes, scaling, and one-hot categories are learned only from each training fold. Unknown and rare categories are handled by the encoder.",
        "",
        "## Main numerical relationships",
        "",
        f"Strongest positive Pearson relationships: {top_pos[['feature','pearson_with_price']].to_dict('records')}",
        f"Strongest negative Pearson relationships: {top_neg[['feature','pearson_with_price']].to_dict('records')}",
        "",
        "## Model comparison",
        "",
        model_comparison.to_markdown(index=False),
        "",
        f"The lowest external-test RMSE belongs to **{best['Model']}**. All models train on `log1p(price)` to reduce target skew; predictions are converted back to price units before MAE, MSE, RMSE, and R² are calculated.",
        f"Linear residual diagnostics: {residuals}",
        "",
        "## Business use",
        "",
        f"The ±20% screening rule flags {under:,} potentially underpriced and {over:,} potentially overpriced test listings. These are review candidates, not automatic buy/sell decisions.",
        "Dealerships can use predicted value as a consistent first-pass benchmark for acquisitions, trade-in negotiation, and listing-price review. Age, mileage, configuration, condition indicators, brand/model, and equipment patterns should be combined with inspection and local market evidence.",
        "",
        "## Limitations",
        "",
        "- Listings are observational; coefficients are associations, not causal effects.",
        "- The data may omit geography, trim, service history, title status detail, seller type, local demand, taxes, and negotiation outcomes.",
        "- High-cardinality text values are grouped when rare. This improves stability but can hide niche trims or colors.",
        "- Extreme prices remain in the analysis. Log-target training reduces their influence, but unusual vehicles can still have large dollar errors.",
        "- A flagged price difference can reflect an unobserved feature or data error. Inspect the vehicle and comparable listings before acting.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
