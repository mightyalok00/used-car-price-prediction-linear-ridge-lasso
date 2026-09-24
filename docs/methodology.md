# Methodology and Data-Availability Rules

## Source files

The project reads the source datasets in place and does not modify or duplicate them:

- `D:\Used Car Listings Features and Price Prediction\train.csv`
- `D:\Used Car Listings Features and Price Prediction\test.csv`

Use `pathlib.Path` or raw strings on Windows. Command-line overrides are available if the files move.

## Shape formula

For each DataFrame:

```python
rows = df.shape[0]
columns = df.shape[1]
total_cells = rows * columns
```

`df.shape` is a two-item tuple in `(rows, columns)` order. The row and column counts in generated reports come from this code, not from manual counting.

## Cleaning policy

- Exact duplicate rows are removed.
- Text is trimmed, internal whitespace is standardized, and case is normalized.
- `price` is converted to numeric after removing currency punctuation. Missing, nonnumeric, or nonpositive targets are removed.
- Impossible year, mileage, engine-size, and MPG values are set to missing.
- Predictor gaps are not filled globally. Imputation happens inside the model pipeline and is fitted only on training folds.
- IQR outliers are reported but not automatically deleted. All models use `log1p(price)` and convert predictions back to price units.

## Schema adaptation

Every optional analysis checks column availability. If a requested field is absent, the code skips that component or uses the nearest supported alternative and writes a note instead of raising an avoidable error. Model predictors are restricted to columns shared by train and test.

The actual schema contains `brand`, `model`, `year`, `mileage`, `engine`, `engine_size`, `transmission`, fuel/drive fields, many binary equipment flags, colors, and `price`. There is no explicit listing identifier. The test file includes `price`, so it is used as a labeled external holdout.

## Leakage prevention

The source train file is used for training and cross-validation. The source test file is held out for final metrics and pricing-opportunity analysis. Median imputation, mode imputation, scaling, rare-category grouping, and one-hot encoding are contained in scikit-learn pipelines, so each fit learns them from training data only.

## Modeling choices

- `year` is represented by engineered `vehicle_age` in the model to avoid exact redundancy.
- `min_mpg` and `max_mpg` are replaced by `avg_mpg` in the model. The original fields remain available for EDA.
- Rare categories are grouped by `OneHotEncoder(min_frequency=20)`.
- A shuffled, seeded three-fold strategy is used consistently for tuning and model comparison.
- Model fitting defaults to one process for reliable execution on Windows; this changes runtime, not results.
- Alpha tuning uses `GridSearchCV` on log-target RMSE. Final CV mean and standard deviation are computed as RMSE in original price units.
- Coefficients are associations on the log-price scale. They are not causal effects.

## Pricing flags

A listing is a review candidate when actual price differs from predicted price by at least 20% of predicted price. This threshold is a screening convention, not a guarantee of value. Inspection, local comparables, trim details, title/service history, taxes, reconditioning cost, and market liquidity remain necessary.
