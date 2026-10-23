"""
===============================================================
Evaluate the Saved Localization-Error Model on Unseen Data

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
===============================================================

Loads ``models/best_localization_model.joblib`` (saved by
``train_final_model.py``) and scores it on data it did not see during
training, reported separately from the training run for reproducible
research.

"Unseen data" is either:

* one or more whole drives held out with ``--holdout-drive`` (evaluates
  generalization to a new drive), or
* a fresh random split of ``merged_localization_ml_dataset.csv`` using a
  different random seed than training (``--eval-seed``), if no
  ``--holdout-drive`` is given.

Generates:

    RMSE, MAE, R2, MAPE
    Prediction Error Distribution (figure)
    Worst Frames / Best Frames (CSV, ranked by absolute error)
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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


class EvaluateFinalModelError(RuntimeError):
    """Raised when evaluation cannot complete safely."""


DEFAULT_TARGET_COLUMN = "Adaptive Position Error (m)"
DEFAULT_DROP_COLUMNS: tuple[str, ...] = (
    "Drive",
    "Frame",
    "Confidence Level",
    "Baseline Position Error (m)",
)
EVAL_RANDOM_STATE = 4242  # deliberately different from training's RANDOM_STATE


@dataclass
class EvaluationReport:
    rmse: float
    mae: float
    r2: float
    mape: float
    n_samples: int
    holdout_drives: tuple[str, ...]


# ---------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------

def load_model(model_path: Path) -> tuple[object, list[str], str]:
    if not model_path.exists():
        raise EvaluateFinalModelError(f"Model file not found: {model_path}")
    payload = joblib.load(model_path)
    return payload["model"], payload["feature_names"], payload.get("model_name", "Unknown")


def load_evaluation_split(
    dataset_path: Path,
    feature_names: Sequence[str],
    target_column: str,
    holdout_drives: Sequence[str] | None,
    eval_seed: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Return (X_eval, y_eval, id_eval) where id_eval carries Drive/Frame for
    traceability (worst/best-frame reporting).
    """

    if not dataset_path.exists():
        raise EvaluateFinalModelError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise EvaluateFinalModelError(f"Target column '{target_column}' not in {dataset_path}")

    if holdout_drives:
        missing = set(holdout_drives) - set(df["Drive"].unique())
        if missing:
            raise EvaluateFinalModelError(f"Requested holdout drives not in dataset: {sorted(missing)}")
        eval_df = df[df["Drive"].isin(holdout_drives)].reset_index(drop=True)
        LOGGER.info("Evaluating on %s held-out drive(s): %s rows.", len(holdout_drives), len(eval_df))
    else:
        _, eval_df = train_test_split(df, test_size=test_size, random_state=eval_seed)
        eval_df = eval_df.reset_index(drop=True)
        LOGGER.info(
            "No --holdout-drive given; evaluating on a fresh %.0f%% random split (seed=%s): %s rows.",
            test_size * 100, eval_seed, len(eval_df),
        )

    missing_features = [c for c in feature_names if c not in eval_df.columns]
    if missing_features:
        raise EvaluateFinalModelError(f"Evaluation data is missing model features: {missing_features}")

    X_eval = eval_df[list(feature_names)].apply(pd.to_numeric, errors="coerce")
    X_eval = X_eval.fillna(X_eval.median(numeric_only=True))
    y_eval = eval_df[target_column]
    id_eval = eval_df[["Drive", "Frame"]].copy() if {"Drive", "Frame"}.issubset(eval_df.columns) else pd.DataFrame(index=eval_df.index)

    return X_eval, y_eval, id_eval


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(model, X_eval: pd.DataFrame, y_eval: pd.Series) -> tuple[np.ndarray, EvaluationReport]:
    predictions = model.predict(X_eval)

    rmse = float(np.sqrt(mean_squared_error(y_eval, predictions)))
    mae = float(mean_absolute_error(y_eval, predictions))
    r2 = float(r2_score(y_eval, predictions))

    # MAPE is undefined for near-zero targets; guard against a division blow-up
    # the way sklearn's mean_absolute_percentage_error would not.
    safe_mask = np.abs(y_eval.to_numpy()) > 1e-8
    if safe_mask.sum() == 0:
        LOGGER.warning("All target values are ~0; MAPE is not meaningful and will be reported as NaN.")
        mape = float("nan")
    else:
        mape = float(
            mean_absolute_percentage_error(y_eval.to_numpy()[safe_mask], predictions[safe_mask]) * 100.0
        )

    report = EvaluationReport(rmse=rmse, mae=mae, r2=r2, mape=mape, n_samples=len(y_eval), holdout_drives=())
    return predictions, report


def build_ranked_frames(
    id_eval: pd.DataFrame, y_eval: pd.Series, predictions: np.ndarray, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = id_eval.copy()
    table["Actual Position Error (m)"] = y_eval.to_numpy()
    table["Predicted Position Error (m)"] = predictions
    table["Absolute Error (m)"] = np.abs(table["Actual Position Error (m)"] - table["Predicted Position Error (m)"])

    worst = table.sort_values("Absolute Error (m)", ascending=False).head(top_n).reset_index(drop=True)
    best = table.sort_values("Absolute Error (m)", ascending=True).head(top_n).reset_index(drop=True)
    return worst, best


def plot_error_distribution(y_eval: pd.Series, predictions: np.ndarray, output_path: Path) -> Path:
    errors = y_eval.to_numpy() - predictions
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(errors, bins=40, color="#6A1B9A", alpha=0.85)
    axes[0].axvline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Prediction Error (Actual - Predicted, m)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Error Distribution")

    sorted_abs = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    axes[1].plot(sorted_abs, cumulative, color="#6A1B9A")
    axes[1].set_xlabel("Absolute Error (m)")
    axes[1].set_ylabel("Cumulative Fraction of Frames")
    axes[1].set_title("Cumulative Absolute Error")

    fig.suptitle("Held-Out Evaluation -- Prediction Error")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Error-distribution plot written -> %s", output_path)
    return output_path


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_evaluation(
    project_root: Path,
    model_path: Path | None = None,
    dataset_path: Path | None = None,
    holdout_drives: Sequence[str] | None = None,
    eval_seed: int = EVAL_RANDOM_STATE,
    test_size: float = 0.2,
    top_n: int = 15,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> EvaluationReport:
    model_path = model_path or project_root / "models" / "best_localization_model.joblib"
    dataset_path = dataset_path or project_root / "results" / "merged" / "merged_localization_ml_dataset.csv"
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    model, feature_names, model_name = load_model(model_path)
    LOGGER.info("Loaded model '%s' with %s features from %s", model_name, len(feature_names), model_path)

    X_eval, y_eval, id_eval = load_evaluation_split(
        dataset_path, feature_names, target_column, holdout_drives, eval_seed, test_size
    )

    predictions, report = evaluate(model, X_eval, y_eval)
    report.holdout_drives = tuple(holdout_drives) if holdout_drives else ()

    metrics_path = results_dir / "evaluation_metrics.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "Model": model_name,
        "RMSE": report.rmse,
        "MAE": report.mae,
        "R2": report.r2,
        "MAPE (%)": report.mape,
        "N Samples": report.n_samples,
        "Holdout Drives": ", ".join(report.holdout_drives) if report.holdout_drives else "random split",
    }]).to_csv(metrics_path, index=False)
    LOGGER.info("Evaluation metrics written -> %s", metrics_path)

    plot_error_distribution(y_eval, predictions, figures_dir / "prediction_error_distribution.png")

    if not id_eval.empty:
        worst, best = build_ranked_frames(id_eval, y_eval, predictions, top_n)
        worst.to_csv(results_dir / "worst_frames.csv", index=False)
        best.to_csv(results_dir / "best_frames.csv", index=False)
        LOGGER.info("Worst/best frame tables written -> %s / %s", results_dir / "worst_frames.csv", results_dir / "best_frames.csv")
    else:
        LOGGER.warning("Evaluation data has no Drive/Frame columns; skipping worst/best frame tables.")

    return report


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved localization model on unseen data.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, default=None, help="Path to best_localization_model.joblib.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to merged_localization_ml_dataset.csv.")
    parser.add_argument(
        "--holdout-drive",
        action="append",
        dest="holdout_drives",
        help="Drive to evaluate on as unseen data. Repeat for multiple drives. "
             "If omitted, a fresh random split (different seed than training) is used.",
    )
    parser.add_argument("--eval-seed", type=int, default=EVAL_RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=15, help="Number of worst/best frames to report.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        report = run_evaluation(
            project_root=args.project_root,
            model_path=args.model,
            dataset_path=args.dataset,
            holdout_drives=args.holdout_drives,
            eval_seed=args.eval_seed,
            test_size=args.test_size,
            top_n=args.top_n,
        )
    except EvaluateFinalModelError:
        LOGGER.exception("Evaluation failed.")
        return 1
    except Exception:
        LOGGER.exception("Unexpected evaluation failure.")
        return 1

    print("=" * 70)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Samples evaluated : {report.n_samples}")
    print(f"Evaluated on      : {', '.join(report.holdout_drives) if report.holdout_drives else 'random split'}")
    print(f"RMSE              : {report.rmse:.6f}")
    print(f"MAE               : {report.mae:.6f}")
    print(f"R2                : {report.r2:.6f}")
    print(f"MAPE              : {report.mape:.2f}%")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
