"""
============================================================================
ML Model Train + Drive-Holdout Evaluation -- FIXED

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
============================================================================

Fixes vs. the original ``train_final_model.py`` / ``evaluate_final_model.py``
-----------------------------------------------------------------------------
The original code split the *frame-level* dataset randomly with
``train_test_split``.  Because consecutive frames of the same drive are
strongly autocorrelated (sensor noise, driving mode, and the fused error all
persist frame-to-frame), a random frame split puts nearly-identical samples
from the same drive into both train and test, inflating R2 (the paper reported
0.94-0.98 from three different scripts).  That is not honest generalization.

This module replaces it with a **leave-one-drive-out** evaluation:

* train on every combination of 8 drives, evaluate on the 1 held-out drive,
* pool the 9 held-out predictions into a single canonical metric set,
* report one RMSE / MAE / R2 / MAPE for the whole dataset,
* and a per-drive breakdown for traceability.

Only a *single* metrics file is produced (``results/ml/metrics_summary.csv``),
so no three-way numeric disagreement is possible.

The final model is also re-fit on all drives and saved to
``models/best_localization_model.joblib`` for reproducibility.
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

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_XGBOOST = False

LOGGER = logging.getLogger(__name__)


class MLPipelineError(RuntimeError):
    """Raised when the ML stage cannot complete safely."""


DEFAULT_TARGET_COLUMN = "Adaptive Position Error (m)"

#: Non-predictive / leaky columns dropped from the feature matrix.
#: ``X GT (m)`` / ``Y GT (m)`` are pristine ground-truth coordinates; the
#: regression target (Adaptive Position Error) is computed from them, so
#: they would let a model reconstruct the target and must never be features.
#: The same applies to the pristine OXTS geographic/kinematic streams
#: (``lat``/``lon``/``alt``/``roll``/``pitch``/``yaw`` and the raw
#: velocity/acceleration channels): they are ground-truth measurements of
#: the true position and motion and therefore leak the target. Fused
#: outputs and raw GPS fix coordinates are not pre-fusion features and are
#: also excluded. Only pre-fusion quality metrics, confidence scores,
#: stability/reliability indicators, the Sensor Confidence Index, and the
#: adaptive sensor weights are retained as features.
DEFAULT_DROP_COLUMNS: tuple[str, ...] = (
    "Drive",
    "Frame",
    "Confidence Level",
    "Baseline Position Error (m)",
    "X GT (m)",
    "Y GT (m)",
    # Pristine OXTS ground-truth geographic coordinates.
    "lat",
    "lon",
    "alt",
    # Pristine OXTS kinematic / attitude streams (ground truth).
    "roll",
    "pitch",
    "yaw",
    "vn",
    "ve",
    "vf",
    "vl",
    "vu",
    "ax",
    "ay",
    "az",
    "af",
    "al",
    "au",
    "wx",
    "wy",
    "wz",
    "wf",
    "wl",
    "wu",
    # Fused outputs and raw fix coordinates are not pre-fusion features.
    "X Adaptive",
    "Y Adaptive",
    "Yaw Adaptive",
    "V Adaptive",
    "X Baseline",
    "Y Baseline",
    "Yaw Baseline",
    "V Baseline",
    "X GPS",
    "Y GPS",
    "V GPS",
    # Injected-protocol labels (not pre-fusion sensor information).
    "GPS Available",
    "GPS State",
)

RANDOM_STATE = 42


@dataclass
class ModelResult:
    """Performance of one model under leave-one-drive-out evaluation."""

    name: str
    rmse: float
    mae: float
    r2: float
    mape: float


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_features_and_target(
    dataset_path: Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_columns: Sequence[str] = DEFAULT_DROP_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load the ML dataset, returning (X, y, drive_ids, feature_names)."""

    if not dataset_path.exists():
        raise MLPipelineError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise MLPipelineError(f"Target column '{target_column}' not in {dataset_path}")
    if "Drive" not in df.columns:
        raise MLPipelineError(f"Dataset has no 'Drive' column: {dataset_path}")

    feature_df = df.drop(
        columns=[c for c in drop_columns if c in df.columns] + [target_column],
        errors="ignore",
    )

    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    dropped_non_numeric = sorted(set(feature_df.columns) - set(numeric_df.columns))
    if dropped_non_numeric:
        LOGGER.info("Dropping non-numeric columns from features: %s", dropped_non_numeric)

    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    target = df[target_column]
    drive_ids = df["Drive"].astype(str)

    LOGGER.info(
        "Loaded %s rows, %s feature columns, %s drives, target='%s'.",
        len(df), numeric_df.shape[1], drive_ids.nunique(), target_column,
    )
    return numeric_df, target, drive_ids, list(numeric_df.columns)


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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    safe = np.abs(y_true) > 1e-8
    if safe.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs((y_true[safe] - y_pred[safe]) / y_true[safe])) * 100.0
        )
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def evaluate_leave_one_drive_out(
    X: pd.DataFrame,
    y: pd.Series,
    drive_ids: pd.Series,
    model: object,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fit ``model`` on every 8-drive subset and predict the held-out drive.

    Returns
    -------
    (pooled_predictions, per_drive_metrics)
        ``pooled_predictions`` is a full-length array aligned with ``X`` (one
        row per frame, each predicted by a model that never saw its drive).
        ``per_drive_metrics`` has one row per drive with RMSE/MAE/R2/MAPE.
    """

    drives = sorted(drive_ids.unique())
    predictions = np.empty(len(X), dtype=float)
    per_drive: list[dict[str, object]] = []

    for holdout in drives:
        train_mask = (drive_ids != holdout).to_numpy()
        test_mask = (drive_ids == holdout).to_numpy()

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y.to_numpy()[train_mask], y.to_numpy()[test_mask]

        fitted = model
        if hasattr(model, "set_params"):
            fitted = model.__class__(**model.get_params(deep=True))
        fitted.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        predictions[test_mask] = y_pred

        metrics = _compute_metrics(y_test, y_pred)
        per_drive.append(
            {
                "Holdout Drive": holdout,
                "N Frames": int(test_mask.sum()),
                "RMSE (m)": metrics["rmse"],
                "MAE (m)": metrics["mae"],
                "R2": metrics["r2"],
                "MAPE (%)": metrics["mape"],
            }
        )

    return predictions, pd.DataFrame(per_drive)


def train_and_compare_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    drive_ids: pd.Series,
) -> list[ModelResult]:
    """Evaluate every candidate model with leave-one-drive-out CV."""

    results: list[ModelResult] = []
    for name, model in _build_candidate_models().items():
        LOGGER.info("Evaluating %s with leave-one-drive-out CV ...", name)
        predictions, per_drive = evaluate_leave_one_drive_out(X, y, drive_ids, model)

        metrics = _compute_metrics(y.to_numpy(), predictions)
        LOGGER.info(
            "%-20s pooled RMSE=%.6f  MAE=%.6f  R2=%.6f  MAPE=%.2f%%",
            name, metrics["rmse"], metrics["mae"], metrics["r2"], metrics["mape"],
        )

        results.append(
            ModelResult(
                name=name,
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                r2=metrics["r2"],
                mape=metrics["mape"],
            )
        )

    results.sort(key=lambda r: r.rmse)
    return results


# ---------------------------------------------------------------------
# Reporting artifacts
# ---------------------------------------------------------------------

def write_metrics_summary(
    results: Sequence[ModelResult],
    per_drive: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write a single canonical metrics file (models + per-drive breakdown)."""

    table = pd.DataFrame(
        [{
            "Model": r.name,
            "Pooled RMSE (m)": r.rmse,
            "Pooled MAE (m)": r.mae,
            "Pooled R2": r.r2,
            "Pooled MAPE (%)": r.mape,
        } for r in results]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    per_drive_path = output_path.parent / "per_drive_holdout_metrics.csv"
    per_drive.to_csv(per_drive_path, index=False)
    LOGGER.info("ML metrics written -> %s (+ %s)", output_path, per_drive_path.name)
    return output_path


def plot_feature_importance(
    model, feature_names: Sequence[str], output_path: Path, top_n: int = 20
) -> Path | None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        if isinstance(model, LinearRegression):
            importances = np.abs(model.coef_)
        else:
            LOGGER.info("Model has no feature_importances_ / coef_; skipping plot.")
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


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_ml_stage(
    project_root: Path,
    dataset_path: Path | None = None,
    model_output: Path | None = None,
) -> tuple[list[ModelResult], pd.DataFrame]:
    """
    Run the corrected ML stage end to end.

    Returns
    -------
    (results, per_drive_metrics)
        Sorted model comparison and the hold-out-drive breakdown of the
        best model (used to persist a merged summary).
    """

    project_root = Path(project_root)
    dataset_path = dataset_path or (
        project_root / "results" / "merged" / "merged_localization_ml_dataset.csv"
    )
    model_output = model_output or project_root / "models" / "best_localization_model.joblib"

    X, y, drive_ids, feature_names = load_features_and_target(dataset_path)

    results = train_and_compare_holdout(X, y, drive_ids)
    best = results[0]
    LOGGER.info("Best model: %s (pooled RMSE=%.6f, R2=%.6f)", best.name, best.rmse, best.r2)

    # Per-drive table for the best model.
    best_model = _build_candidate_models()[best.name]
    _, per_drive = evaluate_leave_one_drive_out(X, y, drive_ids, best_model)

    write_metrics_summary(results, per_drive, project_root / "results" / "ml" / "metrics_summary.csv")

    # Final model fit on ALL drives, saved for reproducibility.
    final_model = _build_candidate_models()[best.name]
    final_model.fit(X, y)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final_model, "feature_names": feature_names, "model_name": best.name},
        model_output,
    )
    LOGGER.info("Saved final model -> %s", model_output)

    plot_feature_importance(
        final_model, feature_names, project_root / "figures" / "feature_importance.png"
    )

    return results, per_drive


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models with leave-one-drive-out CV."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=None, help="Path to merged_localization_ml_dataset.csv.")
    parser.add_argument("--model-output", type=Path, default=None, help="Where to save the best model (.joblib).")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        results, per_drive = run_ml_stage(
            project_root=args.project_root,
            dataset_path=args.dataset,
            model_output=args.model_output,
        )
    except MLPipelineError:
        LOGGER.exception("ML stage failed.")
        return 1
    except Exception:
        LOGGER.exception("Unexpected ML-stage failure.")
        return 1

    print("=" * 78)
    print("ML STAGE COMPLETE -- LEAVE-ONE-DRIVE-OUT CROSS-VALIDATION")
    print("=" * 78)
    print(f"{'Model':20s}{'RMSE (m)':>12s}{'MAE (m)':>12s}{'R2':>12s}{'MAPE (%)':>12s}")
    for r in results:
        print(f"{r.name:20s}{r.rmse:12.6f}{r.mae:12.6f}{r.r2:12.6f}{r.mape:12.2f}")
    print("-" * 78)
    print(f"Best model: {results[0].name}")
    print()
    print("Per-drive holdout metrics (best model):")
    print(per_drive.to_string(index=False))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
