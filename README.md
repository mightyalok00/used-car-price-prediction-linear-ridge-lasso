# Used Car Listings: Features and Price Prediction

An end-to-end, reproducible machine-learning portfolio project for used-car valuation. It compares Multiple Linear Regression, Ridge, and Lasso on real listing data, then converts model output into practical dealership insights for valuation, inventory selection, depreciation, and pricing opportunities.

## Project highlights

- Answers all 16 analysis questions in one fully executed notebook.
- Inspects the actual train/test schemas instead of assuming columns.
- Uses leakage-safe imputation, scaling, rare-category handling, and one-hot encoding.
- Tunes Ridge and Lasso with `GridSearchCV` and evaluates all models with consistent cross-validation.
- Reports MAE, MSE, RMSE, R², CV stability, train/test fit, coefficient activity, and generalization gaps.
- Produces reusable Python modules, fitted-model artifacts, figures, business tables, and validation reports.
- Identifies potentially underpriced and overpriced listings while documenting decision risks and limitations.

## Model results

| Model | Test MAE | Test RMSE | Test R² | Best alpha | Active features | Zero coefficients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear Regression | 5,898 | 13,943 | 0.7322 | N/A | 630 | 0 |
| Ridge Regression | 5,904 | 13,942 | 0.7322 | 0.1 | 630 | 0 |
| Lasso Regression | 5,987 | 14,103 | 0.7260 | 0.0001 | 357 | 273 |

Ridge produced the lowest external-test RMSE, narrowly outperforming the unregularized baseline. Lasso delivered a smaller model by setting 273 encoded coefficients to zero, with a modest reduction in predictive accuracy.

## Verified source dimensions

Pandas returns `(rows, columns)` from `df.shape`:

```python
rows = df.shape[0]
columns = df.shape[1]
total_cells = rows * columns
```

- Raw train: 19,109 rows × 36 columns = 687,924 cells.
- Raw test: 4,778 rows × 36 columns = 172,008 cells.

These counts include rows with unusable targets. Cleaning removes 34 train rows (32 nonnumeric `ot Priced` values and 2 missing prices) and 10 test rows (9 nonnumeric `ot Priced` values and 1 missing price). Generated reports preserve both raw and clean counts.

## Project structure

```text
Used_Car_Price_Prediction_Project/
├── README.md
├── requirements.txt
├── data/raw/README.md
├── docs/
│   ├── questions.md
│   └── methodology.md
├── notebooks/
│   └── 01_complete_used_car_analysis.ipynb
├── scripts/
│   ├── build_notebook.py
│   └── run_analysis.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── reporting.py
│   └── workflow.py
├── tests/test_data.py
└── outputs/
    ├── figures/
    ├── tables/
    ├── models/
    └── reports/
```

## Run order

1. Open PowerShell in the project root.
2. Create and activate an environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Confirm the source files exist at the default paths shown in `data/raw/README.md`.
4. Run either the notebook or the command-line workflow:

   ```powershell
   jupyter notebook notebooks\01_complete_used_car_analysis.ipynb
   ```

   or

   ```powershell
   python scripts\run_analysis.py
   ```

5. Review `outputs/reports/analysis_summary.md`, `outputs/tables/model_comparison.csv`, and the generated figures and business tables.
6. Optionally run the lightweight tests:

   ```powershell
   python -m pytest -q
   ```

## Important interpretation notes

- The supplied test file includes `price`, so it is used as a labeled external holdout rather than being mixed into training.
- The models learn a log-price target, then convert predictions back to price units before calculating MAE, MSE, RMSE, and R².
- Rare categorical levels are grouped, and unseen test categories are handled safely.
- Underpriced/overpriced labels are screening signals. They require vehicle inspection and local-market validation.
