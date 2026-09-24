"""Build the main notebook from maintainable source cells."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_complete_used_car_analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# Used Car Listings: Features and Price Prediction

This notebook answers the 16 finalized questions using the actual source schemas. It uses the provided `train.csv` for training/cross-validation and the provided labeled `test.csv` as the external holdout. All monetary metrics are derived from the files during execution.
"""
    ),
    code(
        r"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import ProjectConfig, run_analysis

TRAIN_PATH = Path(r"D:\Used Car Listings Features and Price Prediction\train.csv")
TEST_PATH = Path(r"D:\Used Car Listings Features and Price Prediction\test.csv")
config = ProjectConfig(project_root=PROJECT_ROOT, train_path=TRAIN_PATH, test_path=TEST_PATH)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
"""
    ),
    md(
        """
## Q1. Dataset Understanding and Data Structures

Load the data and inspect shape, columns, types, sample records, descriptive statistics, unique values, and memory usage. A `DataFrame` is the full table; selecting one column produces a `Series`; `list(df.columns)` is a Python list; summary mappings are dictionaries; and model matrices/predictions are NumPy arrays.

The shape formula is explicit: `rows = df.shape[0]`, `columns = df.shape[1]`, and `total_cells = rows * columns`.
"""
    ),
    code(
        """
train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
test_raw = pd.read_csv(TEST_PATH, low_memory=False)

for name, df in {"train": train_raw, "test": test_raw}.items():
    rows = df.shape[0]
    columns = df.shape[1]
    total_cells = rows * columns
    print(f"{name}: rows={rows:,}, columns={columns:,}, total cells={total_cells:,}")
    print(f"memory={df.memory_usage(deep=True).sum():,} bytes")

display(train_raw.head(), train_raw.tail())
display(train_raw.dtypes.rename("dtype").to_frame())
display(train_raw.describe(include="all").T)
display(train_raw.nunique(dropna=False).rename("unique_including_missing").to_frame())
"""
    ),
    md(
        """
### Execute the reusable workflow

The next cell performs all cleaning, EDA, tuning, model fitting, evaluation, plots, business tables, model serialization, and validation. Re-running it refreshes every generated result from the source CSVs.
"""
    ),
    code("results = run_analysis(config)\nprint('Completed. Best external-test RMSE model:', results['best_model'])"),
    md(
        """
## Q2. Data Quality and Cleaning

Review missing values, duplicates, malformed targets, impossible numerical values, and text normalization. Predictor imputation is intentionally deferred to training-only pipelines.
"""
    ),
    code(
        """
display(pd.read_csv(PROJECT_ROOT / "outputs/tables/missing_values_after_row_cleaning.csv"))
display(results["cleaning"])
"""
    ),
    md(
        """
## Q3. Feature Classification and Selection

Classify continuous, binary, categorical, ordinal, identifier, and target fields. `vehicle_age` and `avg_mpg` are engineered when their source columns exist. Redundant originals are retained for EDA but omitted from the model matrix.
"""
    ),
    code("display(results['feature_classification'])\ndisplay(results['model_groups'])"),
    md(
        """
## Q4. Price Distribution and Outlier Analysis

IQR fences identify unusual values without automatically deleting them. The log-price view tests whether `log1p(price)` reduces positive skew.
"""
    ),
    code(
        """
price_numeric = pd.to_numeric(train_raw["price"], errors="coerce").dropna()
print({"raw_price_skew": price_numeric.skew(), "log1p_price_skew": np.log1p(price_numeric).skew()})
display(results["outliers"])
display(Image(filename=str(PROJECT_ROOT / "outputs/figures/price_distribution_and_log.png")))
"""
    ),
    md(
        """
## Q5. Numerical Features vs Price

Pearson correlation summarizes linear association; Spearman correlation summarizes monotonic association. Correlation does not prove causation.
"""
    ),
    code(
        """
display(results["correlations"])
display(Image(filename=str(PROJECT_ROOT / "outputs/figures/numeric_features_vs_price.png")))
"""
    ),
    md(
        """
## Q6. Categorical Features vs Price

For each available text category, compare count, mean price, and median price. The saved table includes the most frequent levels so sparse categories do not dominate the presentation.
"""
    ),
    code(
        """
cat_summary = results["categorical_summary"]
for feature in [c for c in ["brand", "model", "fuel_type", "transmission", "drivetrain"] if c in cat_summary["feature"].unique()]:
    display(Markdown(f"### {feature}"))
    display(cat_summary[cat_summary["feature"] == feature].sort_values("median_price", ascending=False).head(10))
"""
    ),
    md(
        """
## Q7. Encoding, Scaling, and Data Leakage Prevention

Continuous values use training-fold median imputation and standardization. Binary flags use training-fold mode imputation. Text categories use training-fold mode imputation and one-hot encoding with rare-category grouping. Scaling puts continuous predictors on comparable units, which makes Ridge and Lasso penalties meaningful. All transformations sit inside the pipeline; the holdout test file never teaches the preprocessing or model.
"""
    ),
    code("display(results['model_groups'])"),
    md(
        """
## Q8. Baseline Multiple Linear Regression

MAE is average absolute dollar error. MSE squares errors and heavily weights large misses. RMSE is the square root of MSE and returns to price units. R² is the share of holdout variance explained relative to predicting the mean.
"""
    ),
    code("display(results['comparison'].query(\"Model == 'Linear Regression'\"))"),
    md(
        """
## Q9. Linear Regression Interpretation and Diagnostics

Coefficients are on the standardized/encoded feature space and log-price target. Positive values raise predicted log-price; negative values lower it, holding other encoded inputs fixed. Large residual skew or a strong fitted-vs-absolute-residual correlation warns that assumptions are imperfect. Correlated vehicle descriptors can make individual baseline coefficients unstable; Ridge is designed to reduce that instability.
"""
    ),
    code(
        """
linear_coef = pd.read_csv(PROJECT_ROOT / "outputs/tables/linear_regression_coefficients.csv")
display(linear_coef.head(15))
display(linear_coef.sort_values("coefficient_log_price").head(15))
display(results["residual_diagnostics"])
display(Image(filename=str(PROJECT_ROOT / "outputs/figures/linear_regression_residuals.png")))
"""
    ),
    md(
        """
## Q10. Ridge Regression and Alpha Tuning

GridSearchCV uses the same seeded folds and chooses alpha by validation RMSE on the log target. Ridge shrinks correlated coefficients but normally retains them all.
"""
    ),
    code("display(results['ridge_search'])\ndisplay(results['comparison'].query(\"Model == 'Ridge Regression'\"))"),
    md(
        """
## Q11. Lasso Regression and Feature Selection

Lasso uses L1 regularization. Its exact zero coefficients remove encoded features from the fitted equation, making active/zero counts a direct feature-selection summary.
"""
    ),
    code(
        """
display(results["lasso_search"])
display(results["comparison"].query("Model == 'Lasso Regression'"))
lasso_coef = pd.read_csv(PROJECT_ROOT / "outputs/tables/lasso_regression_coefficients.csv")
display(lasso_coef[lasso_coef["is_zero"]].head(20))
"""
    ),
    md(
        """
## Q12. Cross-Validation of All Three Models

All models use the same external holdout and the same shuffled, seeded folds. `CV_RMSE_Mean` and `CV_RMSE_Std` are calculated in original price units after reversing the log transform.
"""
    ),
    code("display(results['comparison'][['Model','CV_RMSE_Mean','CV_RMSE_Std','CV_Fold_RMSE']])"),
    md(
        """
## Q13. Comprehensive Linear vs Ridge vs Lasso Comparison

The table includes MAE, MSE, RMSE, R², CV mean/std, train/test R², best alpha, active features, zero coefficients, fit time, and a train-test generalization assessment.
"""
    ),
    code("display(results['comparison'].sort_values('RMSE'))"),
    md(
        """
## Q14. Business Analysis — Vehicle Valuation

The strongest numerical relationships and model coefficients provide a consistent valuation starting point. A dealership can combine these signals with acquisition cost, reconditioning cost, local comparables, and desired margin when evaluating purchases, trade-ins, and listing prices.
"""
    ),
    code(
        """
display(results["correlations"].head(12))
best_key = results["best_model"].lower().replace(" ", "_")
best_coef = pd.read_csv(PROJECT_ROOT / f"outputs/tables/{best_key}_coefficients.csv")
display(best_coef.head(20))
"""
    ),
    md(
        """
## Q15. Business Analysis — Depreciation and Inventory Strategy

Age bands summarize price decay. Segment slopes estimate the association between one additional year of age and log price for sufficiently represented segments. Better apparent retention can still reflect trim mix, condition, or selection effects.
"""
    ),
    code("display(results['depreciation'])\ndisplay(results['retention'].head(20))"),
    md(
        """
## Q16. Business Analysis — Pricing Strategy and Opportunities

`actual_minus_predicted` is positive when the listing is above the model benchmark. `predicted_minus_actual` is positive when it appears below the benchmark. A ±20% rule creates review queues. These are not automatic transactions: omitted trim, history, condition, geography, seller urgency, taxes, and market changes can explain a large difference.
"""
    ),
    code(
        """
pricing = results["pricing"]
display(pricing[pricing["pricing_flag"] == "potentially_underpriced"].head(20))
display(pricing[pricing["pricing_flag"] == "potentially_overpriced"].sort_values("actual_minus_predicted", ascending=False).head(20))
display(pricing["pricing_flag"].value_counts().rename_axis("flag").to_frame("listings"))
"""
    ),
    md(
        """
## Final files

The workflow saves detailed tables, figures, fitted pipelines, an analysis summary, run metadata, and a validation report under `outputs/`. See `README.md` for the run order and `docs/methodology.md` for availability and fallback rules.
"""
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(nb, OUT)
print(OUT)
