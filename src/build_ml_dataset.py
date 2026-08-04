"""
===============================================================
Build the Final ML Dataset from Merged Pipeline Outputs

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
===============================================================

``src/merge_results.py`` concatenates every per-drive artifact family
into dataset-wide tables under ``results/merged/``. This script joins
four of those tables -- one row per (Drive, Frame) -- into a single,
model-ready table:

    merged_camera_confidence.csv        (camera quality + confidence)
    merged_sensor_confidence_index.csv  (GPS/IMU raw signals + confidence)
    merged_adaptive_sensor_weights.csv  (per-sensor adaptive weights)
    merged_fusion_trajectory.csv        (fused trajectory + position error)

Output:

    results/merged/merged_localization_ml_dataset.csv

The regression target is ``Adaptive Position Error (m)`` from the
fusion-trajectory table -- the per-frame localization error of the
confidence-driven adaptive fusion estimate. This is the dataset
consumed by ``train_final_model.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT.parent))

LOGGER = logging.getLogger(__name__)


class BuildMLDatasetError(RuntimeError):
    """Raised when the ML dataset cannot be assembled safely."""


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

#: Join key shared by every merged table.
JOIN_KEYS: tuple[str, ...] = ("Drive", "Frame")

#: The column the model will learn to predict.
TARGET_COLUMN = "Adaptive Position Error (m)"

#: Columns that are identifiers, duplicated confidence labels, or would
#: leak the target (e.g. the baseline error is another error metric
#: derived from the same ground truth) -- excluded from the feature
#: matrix by default but kept in the output CSV for traceability.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "Drive",
    "Frame",
    "Confidence Level",
    "Confidence Level_camera",
    "Confidence Level_sensor",
    TARGET_COLUMN,
    "Baseline Position Error (m)",
)


@dataclass(frozen=True)
class SourceTable:
    """One merged CSV that contributes columns to the ML dataset."""

    name: str
    filename: str
    has_frame: bool  # False => Frame must be derived from row order per Drive
    suffix: str  # disambiguates duplicate column names on merge


SOURCE_TABLES: tuple[SourceTable, ...] = (
    SourceTable("camera_confidence", "merged_camera_confidence.csv", True, "_camera"),
    SourceTable("sensor_confidence_index", "merged_sensor_confidence_index.csv", False, "_sensor"),
    SourceTable("adaptive_sensor_weights", "merged_adaptive_sensor_weights.csv", True, "_weights"),
    SourceTable("fusion_trajectory", "merged_fusion_trajectory.csv", True, "_fusion"),
)


@dataclass(frozen=True)
class BuildReport:
    """Summary of the dataset-building run, printed at the end."""

    output_path: Path
    n_rows: int
    n_columns: int
    n_feature_columns: int
    drives: tuple[str, ...]
    dropped_missing_target: int
    duplicate_columns_collapsed: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def _load_source(table: SourceTable, merged_dir: Path) -> pd.DataFrame:
    """Load one merged CSV and make sure it carries a Drive/Frame key."""

    csv_path = merged_dir / table.filename
    if not csv_path.exists():
        raise BuildMLDatasetError(f"[{table.name}] missing required file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Drive" not in df.columns:
        raise BuildMLDatasetError(f"[{table.name}] has no 'Drive' column: {csv_path}")

    if not table.has_frame:
        # sensor_confidence_index has no explicit Frame column, but its
        # rows are written in original per-drive frame order (verified
        # against camera_confidence / adaptive_sensor_weights, which do
        # carry a Frame column and align 1:1 by row order within Drive).
        df = df.copy()
        df["Frame"] = df.groupby("Drive").cumcount()
        LOGGER.info("[%s] derived 'Frame' from per-drive row order.", table.name)

    LOGGER.info("[%s] loaded %s rows, %s columns from %s", table.name, len(df), df.shape[1], csv_path)
    return df


# ---------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------

def _merge_sources(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join every source table on (Drive, Frame)."""

    names = list(tables.keys())
    merged = tables[names[0]]
    for name in names[1:]:
        before = len(merged)
        merged = merged.merge(
            tables[name],
            on=list(JOIN_KEYS),
            how="inner",
            suffixes=("", f"_{name}"),
        )
        LOGGER.info(
            "Joined '%s': %s -> %s rows after merge on %s.",
            name, before, len(merged), JOIN_KEYS,
        )
    return merged


def _collapse_duplicate_confidence_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Camera/GPS/IMU confidence appear verbatim in more than one source
    table (they're passed through unchanged by later pipeline stages).
    Keep a single canonical copy of each and drop the rest.
    """

    canonical = ["Camera Confidence", "GPS Confidence", "IMU Confidence"]
    dropped: list[str] = []
    for base in canonical:
        duplicate_cols = [c for c in df.columns if c.startswith(base + "_")]
        for dup_col in duplicate_cols:
            if base in df.columns and dup_col in df.columns:
                mismatch = (df[base] - df[dup_col]).abs().max()
                if pd.notna(mismatch) and mismatch > 1e-6:
                    LOGGER.warning(
                        "'%s' and '%s' disagree by up to %.3g -- keeping '%s'.",
                        base, dup_col, mismatch, base,
                    )
                df.drop(columns=[dup_col], inplace=True)
                dropped.append(dup_col)

    # Drop any leftover duplicated "Confidence Level" text columns; the
    # numeric confidence scores above are what the model will use.
    level_cols = [c for c in df.columns if c.startswith("Confidence Level")]
    for col in level_cols[1:]:
        df.drop(columns=[col], inplace=True)
        dropped.append(col)

    return df, dropped


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_ml_dataset(
    project_root: Path | str,
    merged_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    drives: Sequence[str] | None = None,
) -> BuildReport:
    """
    Assemble ``merged_localization_ml_dataset.csv`` from the four merged
    pipeline tables.

    Returns
    -------
    BuildReport
        Summary of what was written, for logging / CLI reporting.
    """

    project_root = Path(project_root)
    merged_dir = Path(merged_dir) if merged_dir is not None else project_root / "results" / "merged"
    output_path = (
        Path(output_path)
        if output_path is not None
        else merged_dir / "merged_localization_ml_dataset.csv"
    )

    tables = {table.name: _load_source(table, merged_dir) for table in SOURCE_TABLES}

    if drives:
        for name, df in tables.items():
            tables[name] = df[df["Drive"].isin(drives)].reset_index(drop=True)

    merged = _merge_sources(tables)
    merged, dropped_cols = _collapse_duplicate_confidence_columns(merged)

    if TARGET_COLUMN not in merged.columns:
        raise BuildMLDatasetError(
            f"Target column '{TARGET_COLUMN}' not found after merging. "
            "Check that merged_fusion_trajectory.csv was included."
        )

    n_before = len(merged)
    merged = merged.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    dropped_missing_target = n_before - len(merged)
    if dropped_missing_target:
        LOGGER.warning("Dropped %s rows with missing target values.", dropped_missing_target)

    # Order columns: identifiers first, then features, then target last.
    feature_columns = [c for c in merged.columns if c not in NON_FEATURE_COLUMNS]
    ordered_columns = list(JOIN_KEYS) + feature_columns + [TARGET_COLUMN]
    ordered_columns = [c for c in ordered_columns if c in merged.columns]
    # Preserve any leftover columns (e.g. Baseline Position Error) at the end.
    leftover = [c for c in merged.columns if c not in ordered_columns]
    merged = merged[ordered_columns + leftover]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    report = BuildReport(
        output_path=output_path,
        n_rows=len(merged),
        n_columns=merged.shape[1],
        n_feature_columns=len(feature_columns),
        drives=tuple(sorted(merged["Drive"].unique())),
        dropped_missing_target=dropped_missing_target,
        duplicate_columns_collapsed=tuple(dropped_cols),
    )
    LOGGER.info(
        "Wrote ML dataset -> %s (%s rows, %s columns, %s features)",
        output_path, report.n_rows, report.n_columns, report.n_feature_columns,
    )
    return report


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build merged_localization_ml_dataset.csv from the merged pipeline outputs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"D:\Research\Papers\Paper_2"),
        help="Project root containing results/merged/.",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Directory holding the merged_*.csv files. Defaults to <project-root>/results/merged.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <merged-dir>/merged_localization_ml_dataset.csv.",
    )
    parser.add_argument(
        "--drive",
        action="append",
        dest="drives",
        help="Restrict to this drive. Repeat for multiple drives. Defaults to all drives present.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        report = build_ml_dataset(
            project_root=args.project_root,
            merged_dir=args.merged_dir,
            output_path=args.output,
            drives=args.drives,
        )
    except BuildMLDatasetError:
        LOGGER.exception("Failed to build ML dataset.")
        return 1
    except Exception:
        LOGGER.exception("Unexpected failure while building ML dataset.")
        return 1

    print("=" * 70)
    print("ML DATASET BUILD COMPLETE")
    print("=" * 70)
    print(f"Output file      : {report.output_path}")
    print(f"Rows             : {report.n_rows}")
    print(f"Columns (total)  : {report.n_columns}")
    print(f"Feature columns  : {report.n_feature_columns}")
    print(f"Target column    : {TARGET_COLUMN}")
    print(f"Drives included  : {', '.join(report.drives)}")
    if report.dropped_missing_target:
        print(f"Rows dropped (missing target): {report.dropped_missing_target}")
    if report.duplicate_columns_collapsed:
        print(f"Duplicate columns collapsed: {', '.join(report.duplicate_columns_collapsed)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
