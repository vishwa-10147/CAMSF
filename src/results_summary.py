"""
============================================================================
Honest Results Summary + Figures -- FIXED

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
============================================================================

Generates the canonical paper tables/figures from the corrected pipeline:

* ``results/summary/per_drive_comparison.csv`` -- per-drive RMSE/MAE for the
  adaptive and fixed-noise baseline EKFs plus the honest RMSE improvement %.
* ``results/summary/overall_summary.csv`` -- pooled (frame-weighted) metrics.
* ``figures/per_drive_rmse_improvement.png`` -- bar chart of per-drive
  adaptive-vs-baseline RMSE.
* ``figures/rmse_improvement_by_drive.png`` -- bar chart of improvement %.
* ``figures/trajectory_<drive>.png`` -- trajectory overlay for selected drives.

Everything is derived from ``results/fusion_summary_*.csv`` and
``results/fusion_trajectory_*.csv``; nothing is hand-tuned.
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class SummaryError(RuntimeError):
    """Raised when the summary cannot be produced."""


def _frame_weighted_rmse(df: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(np.mean(df[column] ** 2)))


def build_per_drive_table(results_dir: Path) -> pd.DataFrame:
    summary_files = sorted(results_dir.glob("fusion_summary_2011_09_26_*.csv"))
    if not summary_files:
        raise SummaryError(f"No fusion_summary_*.csv files found in {results_dir}")

    rows: list[dict[str, object]] = []
    for path in summary_files:
        drive = path.name.replace("fusion_summary_", "").replace(".csv", "")
        summary = pd.read_csv(path).set_index("Metric")["Value"]

        trajectory_path = path.parent / f"fusion_trajectory_{drive}.csv"
        frame_count = len(pd.read_csv(trajectory_path)) if trajectory_path.exists() else np.nan

        rows.append({
            "Drive": drive.replace("2011_09_26_", ""),
            "Frames": frame_count,
            "Adaptive RMSE (m)": summary["Adaptive RMSE (m)"],
            "Baseline RMSE (m)": summary["Baseline RMSE (m)"],
            "Constant Weight RMSE (m)": summary["Constant Weight RMSE (m)"],
            "Adaptive MAE (m)": summary["Adaptive MAE (m)"],
            "Baseline MAE (m)": summary["Baseline MAE (m)"],
            "Constant Weight MAE (m)": summary["Constant Weight MAE (m)"],
            "RMSE Improvement (%)": summary["RMSE Improvement (%)"],
            "RMSE Improvement vs Constant Weight (%)": summary["RMSE Improvement vs Constant Weight (%)"],
        })

    table = pd.DataFrame(rows)
    return table


def build_overall_summary(table: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    trajectory_files = sorted(results_dir.glob("fusion_trajectory_2011_09_26_*.csv"))
    pooled = pd.concat(
        [pd.read_csv(p) for p in trajectory_files], ignore_index=True
    )

    overall = pd.DataFrame([{
        "Adaptive RMSE (m)": _frame_weighted_rmse(pooled, "Adaptive Position Error (m)"),
        "Baseline RMSE (m)": _frame_weighted_rmse(pooled, "Baseline Position Error (m)"),
        "Constant Weight RMSE (m)": _frame_weighted_rmse(pooled, "Constant Weight Position Error (m)"),
        "Adaptive MAE (m)": float(np.mean(pooled["Adaptive Position Error (m)"].abs())),
        "Baseline MAE (m)": float(np.mean(pooled["Baseline Position Error (m)"].abs())),
        "Constant Weight MAE (m)": float(np.mean(pooled["Constant Weight Position Error (m)"].abs())),
        "Total Frames": len(pooled),
        "RMSE Improvement (%)": float(
            (1.0 - _frame_weighted_rmse(pooled, "Adaptive Position Error (m)")
             / _frame_weighted_rmse(pooled, "Baseline Position Error (m)")) * 100.0
        ),
        "RMSE Improvement vs Constant Weight (%)": float(
            (1.0 - _frame_weighted_rmse(pooled, "Adaptive Position Error (m)")
             / _frame_weighted_rmse(pooled, "Constant Weight Position Error (m)")) * 100.0
        ),
        "Drives Improved": int((table["RMSE Improvement (%)"] > 0).sum()),
        "Total Drives": len(table),
    }])
    return overall


def plot_per_drive_rmse(table: pd.DataFrame, output_path: Path) -> Path:
    drives = table["Drive"]
    x = np.arange(len(drives))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, table["Baseline RMSE (m)"], width,
           label="Fixed-noise baseline EKF", color="#90A4AE")
    ax.bar(x, table["Constant Weight RMSE (m)"], width,
           label="Constant-weight EKF (mean inflation)", color="#F9A825")
    ax.bar(x + width, table["Adaptive RMSE (m)"], width,
           label="Confidence-driven adaptive EKF", color="#2E7D32")

    for i, (ad, cw, bs) in enumerate(zip(table["Adaptive RMSE (m)"],
                                         table["Constant Weight RMSE (m)"],
                                         table["Baseline RMSE (m)"])):
        ax.text(i + width, ad + 0.05, f"{ad:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, cw + 0.05, f"{cw:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i - width, bs + 0.05, f"{bs:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("drive_", "").replace("_sync", "") for d in drives], rotation=45, ha="right")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Localization RMSE by Drive -- Adaptive vs Constant-Weight vs Fixed-Noise Baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Per-drive RMSE figure written -> %s", output_path)
    return output_path


def plot_improvement_by_drive(table: pd.DataFrame, output_path: Path) -> Path:
    drives = table["Drive"]
    x = np.arange(len(drives))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, table["RMSE Improvement (%)"], width,
           color="#2E7D32", label="Adaptive vs fixed-noise baseline")
    ax.bar(x + width / 2, table["RMSE Improvement vs Constant Weight (%)"], width,
           color="#F9A825", label="Adaptive vs constant-weight control")
    ax.axhline(0.0, color="black", linewidth=1)

    for (bar, v) in zip(
        list(ax.patches[:len(drives)]), table["RMSE Improvement (%)"].to_numpy()
    ):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.2 if v >= 0 else -0.2),
                f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    for (bar, v) in zip(
        list(ax.patches[len(drives):]), table["RMSE Improvement vs Constant Weight (%)"].to_numpy()
    ):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.2 if v >= 0 else -0.2),
                f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("drive_", "").replace("_sync", "") for d in drives], rotation=45, ha="right")
    ax.set_ylabel("RMSE Improvement (%)")
    ax.set_title("RMSE Improvement of Adaptive EKF -- vs Fixed-Noise Baseline and vs Constant-Weight Control")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Improvement-by-drive figure written -> %s", output_path)
    return output_path


def plot_trajectory(results_dir: Path, drive: str, output_path: Path) -> Path:
    trajectory_path = results_dir / f"fusion_trajectory_{drive}.csv"
    if not trajectory_path.exists():
        raise SummaryError(f"Trajectory file not found: {trajectory_path}")

    df = pd.read_csv(trajectory_path)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(df["X GT (m)"], df["Y GT (m)"], color="black", linewidth=1.5, label="Ground truth")
    ax.plot(df["X Adaptive"], df["Y Adaptive"], color="#2E7D32", linewidth=1.2, label="Adaptive EKF")
    ax.plot(df["X Constant"], df["Y Constant"], color="#F9A825", linewidth=1.2, label="Constant-weight EKF")
    ax.plot(df["X Baseline"], df["Y Baseline"], color="#90A4AE", linewidth=1.2, label="Fixed-noise EKF")
    ax.scatter(df["X GPS"], df["Y GPS"], s=6, alpha=0.35, color="#C62828", label="GPS fixes")

    outage = df[df["GPS Available"] == False]  # noqa: E712
    if not outage.empty:
        ax.scatter(outage["X GT (m)"], outage["Y GT (m)"], s=18, marker="x",
                   color="#FF6F00", label="Outage frames (GT)", alpha=0.8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Fused Trajectory -- {drive.replace('2011_09_26_', '')}")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Trajectory figure written -> %s", output_path)
    return output_path


def run_summary(
    project_root: Path,
    selected_drives: Iterable[str] | None = None,
) -> pd.DataFrame:
    project_root = Path(project_root)
    results_dir = project_root / "results"
    summary_dir = results_dir / "summary"
    figures_dir = project_root / "figures"

    table = build_per_drive_table(results_dir)
    overall = build_overall_summary(table, results_dir)

    summary_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_dir / "per_drive_comparison.csv", index=False)
    overall.to_csv(summary_dir / "overall_summary.csv", index=False)
    LOGGER.info("Per-drive table written -> %s", summary_dir / "per_drive_comparison.csv")
    LOGGER.info("Overall summary written -> %s", summary_dir / "overall_summary.csv")

    plot_per_drive_rmse(table, figures_dir / "per_drive_rmse.png")
    plot_improvement_by_drive(table, figures_dir / "rmse_improvement_by_drive.png")

    for drive in (selected_drives or ["2011_09_26_drive_0001_sync"]):
        plot_trajectory(results_dir, drive, figures_dir / f"trajectory_{drive}.png")

    return table


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate honest summary tables and figures.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--drive", action="append", dest="drives",
                        help="Drive to plot a trajectory for (repeatable).")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        table = run_summary(project_root=args.project_root, selected_drives=args.drives)
    except SummaryError:
        LOGGER.exception("Summary generation failed.")
        return 1

    print("=" * 88)
    print("HONEST RESULTS SUMMARY")
    print("=" * 88)
    print(table.to_string(index=False))
    print()
    overall_path = Path(args.project_root) / "results" / "summary" / "overall_summary.csv"
    print(pd.read_csv(overall_path).to_string(index=False))
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
