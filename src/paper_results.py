"""
===============================================================
Generate Publication-Ready Tables and Figures

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
===============================================================

Regenerates every paper-facing table and figure directly from the final
merged data (``results/merged/``, ``results/model_comparison.csv``,
``results/evaluation_metrics.csv``) rather than from copy-pasted
notebook output, so the paper can always be reproduced from
``run_pipeline.py`` -> ``merge_results.py`` -> ``training/*.py`` alone.

Generates, under ``results/paper/`` and ``figures/paper/``:

    Tables (CSV + LaTeX)
    ---------------------
    model_comparison_table.csv / .tex
        Test-set RMSE / MAE / R2 for every trained regressor.
    fusion_error_summary_table.csv / .tex
        Adaptive vs. baseline position-error statistics, per drive and overall.
    sensor_confidence_summary_table.csv / .tex
        Mean camera / GPS / IMU confidence and weight, per drive.

    Figures
    -------
    fig_adaptive_vs_baseline_error.png
        Bar chart of mean position error, adaptive fusion vs. baseline, per drive.
    fig_confidence_distributions.png
        Boxplots of per-sensor confidence scores across all drives.
    fig_model_comparison.png
        Bar chart of RMSE for every trained regressor.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)


class PaperResultsError(RuntimeError):
    """Raised when paper artifacts cannot be regenerated safely."""


# ---------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------

def _write_table(df: pd.DataFrame, output_dir: Path, name: str, float_format: str = "%.4f") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"
    df.to_csv(csv_path, index=False, float_format=float_format)
    df.to_latex(tex_path, index=False, float_format=float_format, na_rep="--")
    LOGGER.info("Table '%s' written -> %s, %s", name, csv_path, tex_path)


def build_model_comparison_table(results_dir: Path, tables_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "model_comparison.csv"
    if not path.exists():
        LOGGER.warning("model_comparison.csv not found; skipping model-comparison table. Run train_final_model.py first.")
        return None
    table = pd.read_csv(path).sort_values("RMSE").reset_index(drop=True)
    _write_table(table, tables_dir, "model_comparison_table")
    return table


def build_fusion_error_summary_table(merged_dir: Path, tables_dir: Path) -> pd.DataFrame | None:
    path = merged_dir / "merged_fusion_trajectory.csv"
    if not path.exists():
        LOGGER.warning("merged_fusion_trajectory.csv not found; skipping fusion-error summary table.")
        return None

    df = pd.read_csv(path)
    per_drive = df.groupby("Drive").agg(
        **{
            "Adaptive Mean Error (m)": ("Adaptive Position Error (m)", "mean"),
            "Adaptive Std Error (m)": ("Adaptive Position Error (m)", "std"),
            "Baseline Mean Error (m)": ("Baseline Position Error (m)", "mean"),
            "Baseline Std Error (m)": ("Baseline Position Error (m)", "std"),
            "Frames": ("Frame", "count"),
        }
    ).reset_index()
    per_drive["Improvement (%)"] = (
        (per_drive["Baseline Mean Error (m)"] - per_drive["Adaptive Mean Error (m)"])
        / per_drive["Baseline Mean Error (m)"].replace(0, pd.NA)
        * 100.0
    )

    overall = pd.DataFrame([{
        "Drive": "ALL DRIVES",
        "Adaptive Mean Error (m)": df["Adaptive Position Error (m)"].mean(),
        "Adaptive Std Error (m)": df["Adaptive Position Error (m)"].std(),
        "Baseline Mean Error (m)": df["Baseline Position Error (m)"].mean(),
        "Baseline Std Error (m)": df["Baseline Position Error (m)"].std(),
        "Frames": len(df),
        "Improvement (%)": (
            (df["Baseline Position Error (m)"].mean() - df["Adaptive Position Error (m)"].mean())
            / df["Baseline Position Error (m)"].mean() * 100.0
        ),
    }])

    table = pd.concat([per_drive, overall], ignore_index=True)
    _write_table(table, tables_dir, "fusion_error_summary_table")
    return table


def build_sensor_confidence_summary_table(merged_dir: Path, tables_dir: Path) -> pd.DataFrame | None:
    path = merged_dir / "merged_adaptive_sensor_weights.csv"
    if not path.exists():
        LOGGER.warning("merged_adaptive_sensor_weights.csv not found; skipping sensor-confidence summary table.")
        return None

    df = pd.read_csv(path)
    table = df.groupby("Drive").agg(
        **{
            "Mean Camera Confidence": ("Camera Confidence", "mean"),
            "Mean GPS Confidence": ("GPS Confidence", "mean"),
            "Mean IMU Confidence": ("IMU Confidence", "mean"),
            "Mean Camera Weight": ("Camera Weight", "mean"),
            "Mean GPS Weight": ("GPS Weight", "mean"),
            "Mean IMU Weight": ("IMU Weight", "mean"),
        }
    ).reset_index()
    _write_table(table, tables_dir, "sensor_confidence_summary_table")
    return table


# ---------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------

def plot_adaptive_vs_baseline(fusion_summary: pd.DataFrame, figures_dir: Path) -> Path | None:
    if fusion_summary is None:
        return None
    per_drive = fusion_summary[fusion_summary["Drive"] != "ALL DRIVES"]
    if per_drive.empty:
        return None

    x = range(len(per_drive))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], per_drive["Baseline Mean Error (m)"], width, label="Baseline", color="#B0BEC5")
    ax.bar([i + width / 2 for i in x], per_drive["Adaptive Mean Error (m)"], width, label="Adaptive Fusion", color="#2E7D32")
    ax.set_xticks(list(x))
    ax.set_xticklabels(per_drive["Drive"], rotation=45, ha="right")
    ax.set_ylabel("Mean Position Error (m)")
    ax.set_title("Adaptive Fusion vs. Baseline Position Error, per Drive")
    ax.legend()
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "fig_adaptive_vs_baseline_error.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Figure written -> %s", output_path)
    return output_path


def plot_confidence_distributions(merged_dir: Path, figures_dir: Path) -> Path | None:
    path = merged_dir / "merged_adaptive_sensor_weights.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    columns = ["Camera Confidence", "GPS Confidence", "IMU Confidence"]
    data = [df[c].dropna() for c in columns]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=columns, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#90CAF9"))
    ax.set_ylabel("Confidence Score")
    ax.set_title("Per-Sensor Confidence Distribution Across All Drives")
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "fig_confidence_distributions.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Figure written -> %s", output_path)
    return output_path


def plot_model_comparison(model_comparison: pd.DataFrame | None, figures_dir: Path) -> Path | None:
    if model_comparison is None:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2E7D32" if i == 0 else "#90A4AE" for i in range(len(model_comparison))]
    ax.bar(model_comparison["Model"], model_comparison["RMSE"], color=colors)
    ax.set_ylabel("Test RMSE (m)")
    ax.set_title("Model Comparison -- Localization Error Prediction")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "fig_model_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Figure written -> %s", output_path)
    return output_path


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def generate_paper_results(project_root: Path) -> dict[str, Path]:
    results_dir = project_root / "results"
    merged_dir = results_dir / "merged"
    tables_dir = results_dir / "paper"
    figures_dir = project_root / "figures" / "paper"

    written: dict[str, Path] = {}

    model_comparison = build_model_comparison_table(results_dir, tables_dir)
    fusion_summary = build_fusion_error_summary_table(merged_dir, tables_dir)
    build_sensor_confidence_summary_table(merged_dir, tables_dir)

    for label, path in (
        ("fig_adaptive_vs_baseline_error", plot_adaptive_vs_baseline(fusion_summary, figures_dir)),
        ("fig_confidence_distributions", plot_confidence_distributions(merged_dir, figures_dir)),
        ("fig_model_comparison", plot_model_comparison(model_comparison, figures_dir)),
    ):
        if path is not None:
            written[label] = path

    return written


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate publication tables and figures from the final merged data."
    )
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\Research\Papers\Paper_2"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        written = generate_paper_results(project_root=args.project_root)
    except PaperResultsError:
        LOGGER.exception("Paper-results generation failed.")
        return 1
    except Exception:
        LOGGER.exception("Unexpected paper-results failure.")
        return 1

    print("=" * 70)
    print("PAPER RESULTS GENERATED")
    print("=" * 70)
    print(f"Tables  -> {args.project_root / 'results' / 'paper'}")
    print(f"Figures -> {args.project_root / 'figures' / 'paper'}")
    for name, path in written.items():
        print(f"  - {name:35s} -> {path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
