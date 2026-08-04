"""
==============================================================
Paper 2 Pipeline Runner

Title:
Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa

==============================================================

Chains every processing stage for a single KITTI Raw drive:

    Camera Quality -> Camera Confidence
    GPS + IMU Quality -> GPS Confidence -> IMU Confidence (SCI)
    Adaptive Weights
    Fusion (CDA-EKF)

Each stage reuses the ``Processor``/``Config`` classes already defined
in the corresponding module, so this file contains no duplicated
metric logic -- it only sequences the stages and wires their
results/figures directories together.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .adaptive_fusion import AdaptiveFusionArtifacts, AdaptiveFusionConfig, AdaptiveFusionProcessor
    from .adaptive_weights import AdaptiveWeightArtifacts, AdaptiveWeightConfig, AdaptiveWeightProcessor
    from .camera_confidence import CameraConfidenceArtifacts, CameraConfidenceConfig, CameraConfidenceProcessor
    from .camera_quality import CameraQualityArtifacts, CameraQualityConfig, CameraQualityProcessor, DEFAULT_DRIVES
    from .gps_confidence import GPSConfidenceArtifacts, GPSConfidenceConfig, GPSConfidenceProcessor
    from .gps_quality import GPSQualityArtifacts, GPSQualityConfig, GPSQualityProcessor
    from .imu_confidence import IMUConfidenceArtifacts, IMUConfidenceConfig, IMUConfidenceProcessor
    from .imu_quality import IMUQualityArtifacts, IMUQualityConfig, IMUQualityProcessor
except ImportError:  # pragma: no cover - supports direct script execution.
    from adaptive_fusion import AdaptiveFusionArtifacts, AdaptiveFusionConfig, AdaptiveFusionProcessor
    from adaptive_weights import AdaptiveWeightArtifacts, AdaptiveWeightConfig, AdaptiveWeightProcessor
    from camera_confidence import CameraConfidenceArtifacts, CameraConfidenceConfig, CameraConfidenceProcessor
    from camera_quality import CameraQualityArtifacts, CameraQualityConfig, CameraQualityProcessor, DEFAULT_DRIVES
    from gps_confidence import GPSConfidenceArtifacts, GPSConfidenceConfig, GPSConfidenceProcessor
    from gps_quality import GPSQualityArtifacts, GPSQualityConfig, GPSQualityProcessor
    from imu_confidence import IMUConfidenceArtifacts, IMUConfidenceConfig, IMUConfidenceProcessor
    from imu_quality import IMUQualityArtifacts, IMUQualityConfig, IMUQualityProcessor


LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------
# KITTI Drives
# ----------------------------------------------------------

DRIVES = list(DEFAULT_DRIVES)

PROJECT_ROOT = Path(r"D:\Research\Papers\Paper_2")

DATASET_ROOT = Path(r"D:\Research\Datasets\KITTI")

RESULT_DIR = PROJECT_ROOT / "results"

FIGURE_DIR = PROJECT_ROOT / "figures"


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails for a drive."""


@dataclass(frozen=True)
class DrivePipelineArtifacts:
    """Artifacts produced by every stage for a single drive."""

    camera_quality: CameraQualityArtifacts
    camera_confidence: CameraConfidenceArtifacts
    gps_quality: GPSQualityArtifacts
    imu_quality: IMUQualityArtifacts
    gps_confidence: GPSConfidenceArtifacts
    imu_confidence: IMUConfidenceArtifacts
    adaptive_weights: AdaptiveWeightArtifacts
    adaptive_fusion: AdaptiveFusionArtifacts


def run_pipeline(
    dataset_root: Path | str,
    drive: str,
    project_root: Path | str,
    save_figures: bool = True,
    rolling_window: int = 5,
    logger: logging.Logger | None = None,
) -> DrivePipelineArtifacts:
    """
    Run every Paper 2 stage for a single KITTI Raw drive.

    Parameters
    ----------
    dataset_root:
        Path to the KITTI Raw dataset root.
    drive:
        KITTI Raw synchronized drive name, e.g. ``2011_09_26_drive_0001_sync``.
    project_root:
        Project root under which ``results/`` and ``figures/`` are created.
    save_figures:
        Whether each stage should export its diagnostic figures.
    rolling_window:
        Rolling-window size used by the adaptive-weight stability estimate.
    logger:
        Optional logger; defaults to this module's logger.

    Returns
    -------
    DrivePipelineArtifacts
        The artifact paths produced by every stage, in execution order.
    """

    logger = logger or LOGGER
    dataset_root = Path(dataset_root)
    project_root = Path(project_root)
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    def _stage(name: str, callable_):
        logger.info("-> %s: %s", drive, name)
        try:
            return callable_()
        except Exception as exc:
            raise PipelineError(f"Stage '{name}' failed for drive '{drive}': {exc}") from exc

    #
    # Camera Quality
    #
    camera_quality_config = CameraQualityConfig(
        dataset_root=dataset_root,
        results_dir=results_dir,
        figures_dir=figures_dir,
        save_figures=save_figures,
    )
    _, camera_quality_artifacts = _stage(
        "Camera Quality",
        lambda: CameraQualityProcessor(camera_quality_config, logger=logger).process_drive(drive),
    )

    #
    # Camera Confidence
    #
    camera_confidence_config = CameraConfidenceConfig(
        results_dir=results_dir,
        figures_dir=figures_dir,
        save_figures=save_figures,
    )
    _, camera_confidence_artifacts = _stage(
        "Camera Confidence",
        lambda: CameraConfidenceProcessor(camera_confidence_config, logger=logger).process_drive(drive),
    )

    #
    # GPS Quality
    #
    gps_quality_config = GPSQualityConfig(
        dataset_root=dataset_root,
        results_dir=results_dir,
        figures_dir=figures_dir,
        save_figures=save_figures,
    )
    _, gps_quality_artifacts = _stage(
        "GPS Quality",
        lambda: GPSQualityProcessor(gps_quality_config, logger=logger).process_drive(drive),
    )

    #
    # IMU Quality
    #
    imu_quality_config = IMUQualityConfig(
        dataset_root=dataset_root,
        results_dir=results_dir,
        figures_dir=figures_dir,
        save_figures=save_figures,
    )
    _, imu_quality_artifacts = _stage(
        "IMU Quality",
        lambda: IMUQualityProcessor(imu_quality_config, logger=logger).process_drive(drive),
    )

    #
    # GPS Confidence
    #
    gps_confidence_config = GPSConfidenceConfig(
        results_dir=results_dir,
        figures_dir=figures_dir,
        save_figures=save_figures,
    )
    _, gps_confidence_artifacts = _stage(
        "GPS Confidence",
        lambda: GPSConfidenceProcessor(gps_confidence_config, logger=logger).process_drive(drive),
    )

    #
    # SCI (IMU Confidence + combined Sensor Confidence Index)
    #
    imu_confidence_config = IMUConfidenceConfig(
        results_dir=results_dir,
        figures_dir=figures_dir,
        save_figures=save_figures,
    )
    _, imu_confidence_artifacts = _stage(
        "SCI (IMU Confidence)",
        lambda: IMUConfidenceProcessor(imu_confidence_config, logger=logger).process_drive(drive),
    )

    #
    # Adaptive Weights
    #
    adaptive_weight_config = AdaptiveWeightConfig(
        results_dir=results_dir,
        figures_dir=figures_dir,
        rolling_window=rolling_window,
        save_figures=save_figures,
    )
    _, adaptive_weight_artifacts = _stage(
        "Adaptive Weights",
        lambda: AdaptiveWeightProcessor(adaptive_weight_config, logger=logger).process_drive(drive),
    )

    #
    # Fusion / CDA-EKF
    #
    adaptive_fusion_config = AdaptiveFusionConfig(
        results_dir=results_dir,
        figures_dir=figures_dir,
        save_figures=save_figures,
    )
    _, adaptive_fusion_artifacts = _stage(
        "Fusion (CDA-EKF)",
        lambda: AdaptiveFusionProcessor(adaptive_fusion_config, logger=logger).process_drive(drive),
    )

    return DrivePipelineArtifacts(
        camera_quality=camera_quality_artifacts,
        camera_confidence=camera_confidence_artifacts,
        gps_quality=gps_quality_artifacts,
        imu_quality=imu_quality_artifacts,
        gps_confidence=gps_confidence_artifacts,
        imu_confidence=imu_confidence_artifacts,
        adaptive_weights=adaptive_weight_artifacts,
        adaptive_fusion=adaptive_fusion_artifacts,
    )


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console logging for script execution."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the single-process pipeline runner."""

    parser = argparse.ArgumentParser(
        description="Run the full Confidence-Driven Adaptive Fusion pipeline for KITTI Raw drives."
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--drive",
        action="append",
        dest="drives",
        help="Drive name to process. Repeat for multiple drives.",
    )
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure export.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """Run the pipeline for one or more drives from the command line."""

    configure_logging()
    args = parse_args(argv)
    drives = args.drives or DRIVES

    print("=" * 80)
    print("Confidence-Driven Pipeline")
    print("=" * 80)
    print()
    print("Total Drives :", len(drives))
    print()

    failures: list[str] = []
    for drive in drives:
        print("-" * 60)
        print("Processing :", drive)
        print("-" * 60)
        try:
            run_pipeline(
                dataset_root=args.dataset_root,
                drive=drive,
                project_root=args.project_root,
                save_figures=not args.no_figures,
                rolling_window=args.rolling_window,
            )
        except PipelineError:
            LOGGER.exception("Pipeline failed for drive: %s", drive)
            failures.append(drive)

    print()
    print("=" * 80)
    if failures:
        print(f"COMPLETED WITH {len(failures)} FAILURE(S): {failures}")
    else:
        print("ALL DRIVES COMPLETED")
    print("=" * 80)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())