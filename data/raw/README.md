# Raw Data Location

The large source CSVs are intentionally not committed to this repository.

## Files required

- train.csv
- test.csv

The recorded analysis run used:

    D:\Used Car Listings Features and Price Prediction\train.csv
    D:\Used Car Listings Features and Price Prediction\test.csv

Expected raw dimensions:

| File | Rows | Columns |
| --- | ---: | ---: |
| train.csv | 19,109 | 36 |
| test.csv | 4,778 | 36 |

## Run with the default paths

    python scripts\run_analysis.py

## Run with custom paths

    python scripts\run_analysis.py --train "D:\path\to\train.csv" --test "D:\path\to\test.csv"

The workflow reads the source files and writes generated artifacts under outputs/. It does **not** overwrite the original CSV files.

For a reproduction check, compare your generated files with:

- outputs/reports/run_metadata.json
- outputs/reports/validation_report.txt
- outputs/tables/dataset_dimensions.csv
- outputs/tables/model_comparison.csv
