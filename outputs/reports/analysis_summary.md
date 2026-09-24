# Analysis Summary

## Dataset dimensions and formula

Pandas returns `(rows, columns)` through `df.shape`. Therefore `rows = df.shape[0]`, `columns = df.shape[1]`, and `total cells = rows × columns`.

- train.csv: 19,109 rows × 36 columns = 687,924 cells.
- test.csv: 4,778 rows × 36 columns = 172,008 cells.

## Cleaning and feature handling

- Train cleaning: {'raw_rows': 19109, 'duplicate_rows_removed': 0, 'invalid_or_missing_target_rows_removed': 34, 'clean_rows': 19075, 'reference_year_for_vehicle_age': 2024, 'invalid_values_set_to_missing': {'year': 0, 'mileage': 0, 'engine_size': 14, 'min_mpg': 179, 'max_mpg': 89}, 'remaining_missing_values': 12775, 'note': 'Remaining predictor gaps are imputed inside leakage-safe pipelines fit only on training data.'}
- Test cleaning: {'raw_rows': 4778, 'duplicate_rows_removed': 0, 'invalid_or_missing_target_rows_removed': 10, 'clean_rows': 4768, 'reference_year_for_vehicle_age': 2024, 'invalid_values_set_to_missing': {'year': 0, 'mileage': 0, 'engine_size': 1, 'min_mpg': 48, 'max_mpg': 22}, 'remaining_missing_values': 3210, 'note': 'Remaining predictor gaps are imputed inside leakage-safe pipelines fit only on training data.'}
- Continuous model features: ['mileage', 'engine_size', 'vehicle_age', 'avg_mpg']
- Binary model features: ['automatic_transmission', 'damaged', 'first_owner', 'personal_using', 'turbo', 'alloy_wheels', 'adaptive_cruise_control', 'navigation_system', 'power_liftgate', 'backup_camera', 'keyless_start', 'remote_start', 'sunroof/moonroof', 'automatic_emergency_braking', 'stability_control', 'leather_seats', 'memory_seat', 'third_row_seating', 'apple_car_play/android_auto', 'bluetooth', 'usb_port', 'heated_seats']
- Categorical model features: ['brand', 'model', 'engine', 'transmission', 'fuel_type', 'drivetrain', 'interior_color', 'exterior_color']
- Redundant fields omitted from modeling: ['year', 'min_mpg', 'max_mpg']
- Numerical medians, category modes, scaling, and one-hot categories are learned only from each training fold. Unknown and rare categories are handled by the encoder.

## Main numerical relationships

Strongest positive Pearson relationships: [{'feature': 'navigation_system', 'pearson_with_price': 0.2843221610702425}, {'feature': 'year', 'pearson_with_price': 0.24059091875449598}, {'feature': 'memory_seat', 'pearson_with_price': 0.21127637782816452}]
Strongest negative Pearson relationships: [{'feature': 'mileage', 'pearson_with_price': -0.38297102322565363}, {'feature': 'vehicle_age', 'pearson_with_price': -0.24059091875448735}, {'feature': 'avg_mpg', 'pearson_with_price': -0.20271625592527032}]

## Model comparison

| Model             |     MAE |         MSE |    RMSE |       R2 |   CV_RMSE_Mean |   CV_RMSE_Std |   Train_R2 |   Test_R2 | Best_Alpha   |   Active_Features |   Zero_Coefficients |   Fit_Seconds | CV_Fold_RMSE                                     |   Generalization_Gap_R2 | Overfit_Assessment                |
|:------------------|--------:|------------:|--------:|---------:|---------------:|--------------:|-----------:|----------:|:-------------|------------------:|--------------------:|--------------:|:-------------------------------------------------|------------------------:|:----------------------------------|
| Linear Regression | 5898.19 | 1.94394e+08 | 13942.5 | 0.732227 |        24062.5 |       8655.68 |   0.492668 |  0.732227 | N/A          |               630 |                   0 |      1.19235  | 31395.59, 15048.24, 31272.12, 28276.49, 14320.03 |               -0.239559 | test set easier or sampling shift |
| Ridge Regression  | 5903.92 | 1.94383e+08 | 13942.1 | 0.732241 |        24061   |       8660.2  |   0.492695 |  0.732241 | 0.1          |               630 |                   0 |      0.57049  | 31398.06, 15034.35, 31274.66, 28277.00, 14320.79 |               -0.239546 | test set easier or sampling shift |
| Lasso Regression  | 5987.16 | 1.98893e+08 | 14102.9 | 0.726029 |        24190.8 |       8625.61 |   0.485277 |  0.726029 | 0.0001       |               357 |                 273 |      0.523554 | 31527.56, 15219.12, 31329.96, 28408.97, 14468.39 |               -0.240752 | test set easier or sampling shift |

The lowest external-test RMSE belongs to **Ridge Regression**. All models train on `log1p(price)` to reduce target skew; predictions are converted back to price units before MAE, MSE, RMSE, and R² are calculated.
Linear residual diagnostics: {'residual_mean': 1019.982083196775, 'residual_std': 13906.61476049489, 'residual_skew': 8.132469763145549, 'residual_kurtosis': 113.30684197126139, 'fitted_abs_residual_correlation': 0.40872037107022, 'interpretation': 'A fitted-vs-absolute-residual correlation far from zero suggests non-constant error variance.'}

## Business use

The ±20% screening rule flags 565 potentially underpriced and 693 potentially overpriced test listings. These are review candidates, not automatic buy/sell decisions.
Dealerships can use predicted value as a consistent first-pass benchmark for acquisitions, trade-in negotiation, and listing-price review. Age, mileage, configuration, condition indicators, brand/model, and equipment patterns should be combined with inspection and local market evidence.

## Limitations

- Listings are observational; coefficients are associations, not causal effects.
- The data may omit geography, trim, service history, title status detail, seller type, local demand, taxes, and negotiation outcomes.
- High-cardinality text values are grouped when rare. This improves stability but can hide niche trims or colors.
- Extreme prices remain in the analysis. Log-target training reduces their influence, but unusual vehicles can still have large dollar errors.
- A flagged price difference can reflect an unobserved feature or data error. Inspect the vehicle and comparable listings before acting.