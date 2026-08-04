"""
===============================================================
Merge Per-Drive Results Across the KITTI Raw Drive Set

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
===============================================================

Every stage in ``src/`` writes drive-scoped CSVs, e.g.
``results/adaptive_sensor_weights_2011_09_26_drive_0001_sync.csv``.
This script concatenates those per-drive files across the whole drive
set into single, dataset-wide tables under ``results/merged/`` -- the
tables used for the paper's cross-drive figures/statistics and for
training the final adaptive-weight model in ``train_final_model.py``.

Two kinds of tables are merged:

* "Summary" tables -- one row per metric per drive
  (``Metric``/``Value`` columns), e.g. ``adaptive_weight_summary_*``.
* "Full" tables -- one row per frame per drive, e.g.
  ``adaptive_sensor_weights_*`` or ``sensor_confidence_index_*``.

Both kinds get a ``Drive`` column prepended before concatenation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

from src.camera_quality import DEFAULT_DRIVES  # noqa: E402


LOGGER = logging.getLogger(__name__)


class MergeResultsError(RuntimeError):
    """Raised when result merging cannot complete safely."""


@dataclass(frozen=True)
class MergeSpec:
    """One artifact family to merge across drives."""

    output_name: str
    filename_template: str  # must contain a single "{drive}" placeholder
    required: bool = False  # if True, a drive missing this file raises


# "Summary" tables: one Metric/Value row pair per drive.
SUMMARY_SPECS: tuple[MergeSpec, ...] = (
    MergeSpec("camera_quality_statistics", "camera_quality_statistics_{drive}.csv"),
    MergeSpec("camera_confidence_statistics", "camera_confidence_statistics_{drive}.csv"),
    MergeSpec("gps_quality_summary", "gps_quality_summary_{drive}.csv"),
    MergeSpec("imu_quality_summary", "imu_quality_summary_{drive}.csv"),
    MergeSpec("gps_confidence_statistics", "gps_confidence_statistics_{drive}.csv"),
    MergeSpec("sensor_confidence_summary", "sensor_confidence_summary_{drive}.csv"),
    MergeSpec("adaptive_weight_summary", "adaptive_weight_summary_{drive}.csv"),
    MergeSpec("fusion_error_metrics", "fusion_error_metrics_{drive}.csv", required=True),
)

# "Full" tables: one row per frame per drive.
FULL_SPECS: tuple[MergeSpec, ...] = (
    MergeSpec("camera_confidence", "camera_confidence_{drive}.csv"),
    MergeSpec("sensor_confidence_index", "sensor_confidence_index_{drive}.csv"),
    MergeSpec("adaptive_sensor_weights", "adaptive_sensor_weights_{drive}.csv", required=True),
    MergeSpec("fusion_trajectory", "fusion_trajectory_{drive}.csv"),
)


def _merge_one(
    spec: MergeSpec,
    results_dir: Path,
    drives: Sequence[str],
) -> pd.DataFrame | None:
    """Concatenate one artifact family across drives, tagging each row with its drive."""

    frames: list[pd.DataFrame] = []
    for drive in drives:
        csv_path = results_dir / spec.filename_template.format(drive=drive)
        if not csv_path.exists():
            message = f"[{spec.output_name}] missing for drive '{drive}': {csv_path}"
            if spec.required:
                raise MergeResultsError(message)
            LOGGER.warning(message)
            continue

        drive_df = pd.read_csv(csv_path)
        drive_df.insert(0, "Drive", drive)
        frames.append(drive_df)

    if not frames:
        if spec.required:
            raise MergeResultsError(
                f"No drives produced the required '{spec.output_name}' artifact."
            )
        LOGGER.warning("Skipping '%s': no drives produced this artifact.", spec.output_name)
        return None

    return pd.concat(frames, axis=0, ignore_index=True)


def merge_results(
    project_root: Path | str,
    drives: Sequence[str] | None = None,
    results_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """
    Merge every configured per-drive artifact family into dataset-wide CSVs.

    Returns
    -------
    dict[str, Path]
        Mapping of merged-table name -> path of the written CSV, for
        every artifact family that had at least one drive present.
    """

    project_root = Path(project_root)
    results_dir = Path(results_dir) if results_dir is not None else project_root / "results"
    output_dir = Path(output_dir) if output_dir is not None else results_dir / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_drives = tuple(drives or DEFAULT_DRIVES)
    if not selected_drives:
        raise MergeResultsError("At least one drive must be provided.")

    written: dict[str, Path] = {}
    for spec in (*SUMMARY_SPECS, *FULL_SPECS):
        merged_df = _merge_one(spec, results_dir, selected_drives)
        if merged_df is None:
            continue
        output_path = output_dir / f"merged_{spec.output_name}.csv"
        merged_df.to_csv(output_path, index=False)
        written[spec.output_name] = output_path
        LOGGER.info(
            "Merged '%s': %s rows across %s drive(s) -> %s",
            spec.output_name,
            len(merged_df),
            merged_df["Drive"].nunique(),
            output_path,
        )

    _write_master_summary(written, output_dir)
    return written


def _write_master_summary(written: dict[str, Path], output_dir: Path) -> None:
    """
    Pivot every merged "summary" (Metric/Value) table into one wide
    table with a row per drive and a column per metric, for a quick
    at-a-glance comparison across drives.
    """

    summary_names = {spec.output_name for spec in SUMMARY_SPECS}
    pivoted_frames: list[pd.DataFrame] = []

    for name, path in written.items():
        if name not in summary_names:
            continue
        table = pd.read_csv(path)
        if not {"Drive", "Metric", "Value"}.issubset(table.columns):
            LOGGER.warning("Skipping '%s' in master summary: unexpected columns.", name)
            continue
        wide = table.pivot_table(index="Drive", columns="Metric", values="Value", aggfunc="first")
        wide.columns = [f"{name} | {column}" for column in wide.columns]
        pivoted_frames.append(wide)

    if not pivoted_frames:
        LOGGER.warning("No summary tables available; master summary was not written.")
        return

    master = pivoted_frames[0]
    for frame in pivoted_frames[1:]:
        master = master.join(frame, how="outer")

    master = master.reset_index()
    master_path = output_dir / "master_summary.csv"
    master.to_csv(master_path, index=False)
    LOGGER.info("Master summary written -> %s", master_path)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console logging for script execution."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the result-merging script."""

    parser = argparse.ArgumentParser(
        description="Merge per-drive result CSVs into dataset-wide tables."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"D:\Research\Papers\Paper_2"),
        help="Project root containing the results directory.",
    )
    parser.add_argument(
        "--drive",
        action="append",
        dest="drives",
        help="Drive name to include. Repeat for multiple drives. Defaults to all DEFAULT_DRIVES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write merged CSVs. Defaults to <project-root>/results/merged.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """Run result merging from the command line."""

    configure_logging()
    args = parse_args(argv)
    try:
        written = merge_results(
            project_root=args.project_root,
            drives=args.drives,
            output_dir=args.output_dir,
        )
    except MergeResultsError:
        LOGGER.exception("Result merging failed.")
        return 1
    except Exception:
        LOGGER.exception("Unexpected result-merging failure.")
        return 1

    print("=" * 70)
    print(f"Merged {len(written)} artifact famil{'y' if len(written) == 1 else 'ies'}.")
    for name, path in written.items():
        print(f"  - {name:30s} -> {path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())