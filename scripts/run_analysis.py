"""Command-line entry point for the complete project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import ProjectConfig, run_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all 16 used-car price analysis questions.")
    parser.add_argument("--train", type=Path, default=Path(r"D:\Used Car Listings Features and Price Prediction\train.csv"))
    parser.add_argument("--test", type=Path, default=Path(r"D:\Used Car Listings Features and Price Prediction\test.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ProjectConfig(project_root=PROJECT_ROOT, train_path=args.train, test_path=args.test)
    result = run_analysis(cfg)
    print("Analysis completed.")
    print(result["comparison"].to_string(index=False))
    print(f"Outputs: {PROJECT_ROOT / 'outputs'}")
