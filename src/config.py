"""Central project configuration with Windows-safe pathlib paths."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    project_root: Path
    train_path: Path = Path(r"D:\Used Car Listings Features and Price Prediction\train.csv")
    test_path: Path = Path(r"D:\Used Car Listings Features and Price Prediction\test.csv")
    target: str = "price"
    random_state: int = 42
    cv_folds: int = 5
    one_hot_min_frequency: int = 20
    # Single-process fitting avoids Windows joblib worker exhaustion on modest machines.
    n_jobs: int = 1
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    lasso_alphas: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1)
    figures_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.train_path = Path(self.train_path)
        self.test_path = Path(self.test_path)
        output_root = self.project_root / "outputs"
        self.figures_dir = output_root / "figures"
        self.tables_dir = output_root / "tables"
        self.models_dir = output_root / "models"
        self.reports_dir = output_root / "reports"
        for folder in (self.figures_dir, self.tables_dir, self.models_dir, self.reports_dir):
            folder.mkdir(parents=True, exist_ok=True)
