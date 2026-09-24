"""Small deterministic checks for shape logic and cleaning behavior."""

import pandas as pd

from src.data import clean_dataset, inspect_dataset


def test_shape_formula() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = inspect_dataset(frame, "sample")
    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["total_cells"] == 4


def test_cleaning_removes_invalid_target_and_adds_age() -> None:
    frame = pd.DataFrame({"year": [2020, 2019], "price": ["$10,000", None], "mileage": [100, -1]})
    clean, summary = clean_dataset(frame, "price", reference_year=2023)
    assert len(clean) == 1
    assert clean.iloc[0]["price"] == 10000
    assert clean.iloc[0]["vehicle_age"] == 3
    assert summary["invalid_or_missing_target_rows_removed"] == 1
