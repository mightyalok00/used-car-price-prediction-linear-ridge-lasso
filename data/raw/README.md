# Raw Data Location

The large source CSVs are intentionally not copied into this project.

Default paths:

- `D:\Used Car Listings Features and Price Prediction\train.csv`
- `D:\Used Car Listings Features and Price Prediction\test.csv`

If the files move, pass new paths:

```powershell
python scripts/run_analysis.py --train "D:\new\train.csv" --test "D:\new\test.csv"
```

The analysis never overwrites the source files.
