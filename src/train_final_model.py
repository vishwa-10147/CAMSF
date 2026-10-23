"""
===============================================================
Train the Final Localization-Error Prediction Model

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
===============================================================

Loads ``results/merged/merged_localization_ml_dataset.csv`` (built by
``build_ml_dataset.py``) and trains several regressors to predict the
adaptive fusion's per-frame position error from the confidence/quality
features:

    * Linear Regression
    * Random Forest
    * Gradient Boosting
    * XGBoost (optional -- used only if the ``xgboost`` package is installed)

The best model (lowest test RMSE) is saved to
``models/best_localization_model.joblib``. A model-comparison table,
feature-importance chart, residual plot and prediction plot are written
to ``results/`` and ``figures/``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_XGBOOST = False

LOGGER = logging.getLogger(__name__)


class TrainFinalModelError(RuntimeError):
    """Raised when the final model cannot be trained safely."""


DEFAULT_TARGET_COLUMN = "Adaptive Position Error (m)"

#: Non-predictive / leaky columns dropped from the feature matrix before
#: training. Mirrors the exclusions applied in build_ml_dataset.py.
DEFAULT_DROP_COLUMNS: tuple[str, ...] = (
    "Drive",
    "Frame",
    "Confidence Level",
    "Baseline Position Error (m)",
)

RANDOM_STATE = 42


@dataclass
class ModelResult:
    """Test-set performance for one trained model."""

    name: str
    model: object
    rmse: float
    mae: float
    r2: float
    predictions: np.ndarray


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_features_and_target(
    dataset_path: Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_columns: Sequence[str] = DEFAULT_DROP_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the ML dataset and split it into a numeric feature matrix and target."""

    if not dataset_path.exists():
        raise TrainFinalModelError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise TrainFinalModelError(f"Target column '{target_column}' not in {dataset_path}")

    feature_df = df.drop(columns=[c for c in drop_columns if c in df.columns] + [target_column], errors="ignore")

    # Keep numeric columns only; a text "Confidence Level" style column
    # could sneak in if the drop list is customized upstream.
    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    dropped_non_numeric = sorted(set(feature_df.columns) - set(numeric_df.columns))
    if dropped_non_numeric:
        LOGGER.info("Dropping non-numeric columns from features: %s", dropped_non_numeric)

    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    target = df[target_column]

    LOGGER.info("Loaded %s rows, %s feature columns, target='%s'.", len(df), numeric_df.shape[1], target_column)
    return numeric_df, target, list(numeric_df.columns)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def _build_candidate_models() -> dict[str, object]:
    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    if _HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        LOGGER.warning("xgboost is not installed -- skipping the XGBoost model.")
    return models


def train_and_compare(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> list[ModelResult]:
    """Fit every candidate model and score it on the held-out test split."""

    results: list[ModelResult] = []
    for name, model in _build_candidate_models().items():
        LOGGER.info("Training %s ...", name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        LOGGER.info("%-20s RMSE=%.6f  MAE=%.6f  R2=%.6f", name, rmse, mae, r2)
        results.append(ModelResult(name=name, model=model, rmse=rmse, mae=mae, r2=r2, predictions=predictions))

    results.sort(key=lambda r: r.rmse)
    return results


# ---------------------------------------------------------------------
# Reporting artifacts
# ---------------------------------------------------------------------

def write_model_comparison(results: Sequence[ModelResult], output_path: Path) -> Path:
    table = pd.DataFrame(
        [{"Model": r.name, "RMSE": r.rmse, "MAE": r.mae, "R2": r.r2} for r in results]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    LOGGER.info("Model comparison written -> %s", output_path)
    return output_path


def plot_feature_importance(model, feature_names: Sequence[str], output_path: Path, top_n: int = 20) -> Path | None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        if isinstance(model, LinearRegression):
            importances = np.abs(model.coef_)
        else:
            LOGGER.info("Model has no feature_importances_ / coef_; skipping importance plot.")
            return None

    order = np.argsort(importances)[::-1][:top_n]
    names = np.array(feature_names)[order]
    values = np.array(importances)[order]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
    ax.barh(names[::-1], values[::-1], color="#2E7D32")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance -- Best Localization Model")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Feature importance plot written -> %s", output_path)
    return output_path


def plot_residuals(y_test: pd.Series, predictions: np.ndarray, output_path: Path) -> Path:
    residuals = y_test.to_numpy() - predictions

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(predictions, residuals, s=10, alpha=0.5, color="#1565C0")
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Predicted Position Error (m)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=40, color="#1565C0", alpha=0.8)
    axes[1].set_xlabel("Residual (m)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.suptitle("Best Model -- Residual Analysis")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Residual plot written -> %s", output_path)
    return output_path


def plot_predictions(y_test: pd.Series, predictions: np.ndarray, output_path: Path) -> Path:
    y_true = y_test.to_numpy()
    lims = [0.0, max(float(y_true.max()), float(predictions.max())) * 1.05]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_true, predictions, s=10, alpha=0.5, color="#C62828")
    axes[0].plot(lims, lims, color="black", linewidth=1, linestyle="--", label="Ideal")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("Actual Position Error (m)")
    axes[0].set_ylabel("Predicted Position Error (m)")
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend()

    order = np.argsort(y_true)
    axes[1].plot(y_true[order], label="Actual", color="black", linewidth=1)
    axes[1].plot(predictions[order], label="Predicted", color="#C62828", linewidth=1, alpha=0.8)
    axes[1].set_xlabel("Test Sample (sorted by actual error)")
    axes[1].set_ylabel("Position Error (m)")
    axes[1].set_title("Predicted vs Actual (sorted)")
    axes[1].legend()

    fig.suptitle("Best Model -- Prediction Quality")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Prediction plot written -> %s", output_path)
    return output_path


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_training(
    project_root: Path,
    dataset_path: Path | None = None,
    model_output: Path | None = None,
    test_size: float = 0.2,
) -> tuple[list[ModelResult], list[str]]:
    dataset_path = dataset_path or project_root / "results" / "merged" / "merged_localization_ml_dataset.csv"
    model_output = model_output or project_root / "models" / "best_localization_model.joblib"
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    X, y, feature_names = load_features_and_target(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    LOGGER.info("Train/test split: %s / %s rows.", len(X_train), len(X_test))

    results = train_and_compare(X_train, X_test, y_train, y_test)
    best = results[0]
    LOGGER.info("Best model: %s (RMSE=%.6f, R2=%.6f)", best.name, best.rmse, best.r2)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best.model, "feature_names": feature_names, "model_name": best.name}, model_output)
    LOGGER.info("Saved best model -> %s", model_output)

    write_model_comparison(results, results_dir / "model_comparison.csv")
    plot_feature_importance(best.model, feature_names, figures_dir / "feature_importance.png")
    plot_residuals(y_test, best.predictions, figures_dir / "residual_plot.png")
    plot_predictions(y_test, best.predictions, figures_dir / "prediction_plot.png")

    return results, feature_names


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare regression models on the localization ML dataset."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=None, help="Path to merged_localization_ml_dataset.csv.")
    parser.add_argument("--model-output", type=Path, default=None, help="Where to save the best model (.joblib).")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        results, _ = run_training(
            project_root=args.project_root,
            dataset_path=args.dataset,
            model_output=args.model_output,
            test_size=args.test_size,
        )
    except TrainFinalModelError:
        LOGGER.exception("Training failed.")
        return 1
    except Exception:
        LOGGER.exception("Unexpected training failure.")
        return 1

    print("=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"{'Model':20s}{'RMSE':>12s}{'MAE':>12s}{'R2':>12s}")
    for r in results:
        print(f"{r.name:20s}{r.rmse:12.6f}{r.mae:12.6f}{r.r2:12.6f}")
    print("-" * 70)
    print(f"Best model: {results[0].name}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
