# Changelog

All notable project changes are documented here.

## v1.0.0-ready

### Added
- Price-stratified 5-fold cross-validation using quantile-binned training prices.
- Fold-balance diagnostics in `outputs/tables/cv_fold_price_balance.csv`.
- Median Absolute Error (MedianAE) as a robust companion to RMSE.
- Explicit limitations and future-work guidance, including prediction-interval/conformal uncertainty as a future extension.
- Deterministic tests for target-balanced CV and MedianAE.
- Expanded README verification, topics, methodology, CI, licensing, and reproducibility documentation.

### Improved
- Reduced Ridge CV RMSE standard deviation from approximately 12,325 to 8,660 after target-balanced fold construction.
- Kept all preprocessing leakage-safe inside sklearn pipelines.
- Clarified that Ridge and Linear Regression are effectively tied on the external holdout.
- Clarified that pricing opportunity labels are screening signals rather than transactional recommendations.

### Scope
- This repository is intentionally kept as a reproducible local/portfolio ML project.
- No deployment layer is included.
- The raw dataset is not redistributed.

### Final local regeneration
After metric or workflow changes, regenerate committed analytical outputs with:

```bash
python scripts/run_analysis.py
```

Then run:

```bash
python -m pytest -q
```
