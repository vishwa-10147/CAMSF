"""
============================================================================
GPS Degradation Simulator (honest-experiment fix)

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
============================================================================

Why this module exists
----------------------
The original pipeline fed the same pristine KITTI OXTS lat/lon into the EKF
both as the GPS *measurement* and as the *ground-truth reference* for error.
That is circular: the error then only measures how far the filter drifts from
its own input, and no sensor degradation ever actually occurs, so the paper's
central claim ("robust under degraded GPS") was never tested.

This module fixes that. It builds a controlled, reproducible degradation
experiment on top of the pristine OXTS reference:

    ground truth   = pristine OXTS ENU trajectory (cm-level, post-processed)
    GPS "fixes"    = pristine position + injected degradation
                     (multipath noise, and hard outages where the receiver
                      reports no fix)

Each frame is labelled healthy / degraded / outage. The degraded fix stream
is what the EKF receives as a measurement; error is always measured against
the pristine ground truth, which is independent of the measurement.

All randomness is seeded per drive so the experiment is fully reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .camera_quality import DEFAULT_DRIVES
    from .gps_quality import OXTS_COLUMNS
    from .kitti_raw_dataset import KITTIRawDataset
except ImportError:  # pragma: no cover - supports direct script execution.
    from camera_quality import DEFAULT_DRIVES
    from gps_quality import OXTS_COLUMNS
    from kitti_raw_dataset import KITTIRawDataset


LOGGER = logging.getLogger(__name__)

EARTH_RADIUS_METERS = 6_371_000.0

HEALTHY = "healthy"
DEGRADED = "degraded"
OUTAGE = "outage"
STATES = (HEALTHY, DEGRADED, OUTAGE)


@dataclass(frozen=True)
class GPSDegradationProcessorConfig:
    """Parameters controlling the injected degradation schedule and noise."""

    seed: int = 42
    healthy_noise_std_m: float = 0.5        # typical consumer-grade GPS noise
    degraded_noise_std_m: float = 8.0       # multipath / urban canyon level
    outage_probability: float = 0.15
    degraded_probability: float = 0.30
    min_segment_length: int = 6
    max_segment_length: int = 25
    healthy_numsats: tuple[int, int] = (8, 11)
    degraded_numsats: tuple[int, int] = (3, 5)
    healthy_pos_accuracy: tuple[float, float] = (1.0, 3.0)
    degraded_pos_accuracy: tuple[float, float] = (6.0, 15.0)


@dataclass(frozen=True)
class DegradedGPSResult:
    """Degraded OXTS dataframe plus per-frame state labels and ENU fix stream."""

    oxts_degraded: pd.DataFrame
    states: np.ndarray                      # one of HEALTHY / DEGRADED / OUTAGE
    x_fix_m: np.ndarray                     # degraded ENU x offered as measurement
    y_fix_m: np.ndarray                     # degraded ENU y offered as measurement
    gps_available: np.ndarray               # False => outage, no fix to feed
    gt_x_m: np.ndarray                      # pristine ENU x (independent reference)
    gt_y_m: np.ndarray                      # pristine ENU y (independent reference)
    seed: int


class GPSDegradationError(RuntimeError):
    """Raised when GPS degradation simulation cannot complete safely."""


def _latlon_to_local_xy(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    x = EARTH_RADIUS_METERS * (lon - lon0) * np.cos(lat0)
    y = EARTH_RADIUS_METERS * (lat - lat0)
    return x, y


def _local_xy_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    lat_deg = np.radians(float(lat0_deg)) + np.asarray(y, dtype=float) / EARTH_RADIUS_METERS
    lon_deg = lon0 + np.asarray(x, dtype=float) / (EARTH_RADIUS_METERS * np.cos(lat0))
    return np.degrees(lat_deg), np.degrees(lon_deg)


class GPSDegradationProcessor:
    """Inject a reproducible GPS degradation profile into a pristine OXTS frame."""

    def __init__(
        self,
        config: GPSDegradationProcessorConfig | None = None,
    ) -> None:
        self.config = config or GPSDegradationProcessorConfig()

    def degrade_drive(
        self,
        drive: str,
        oxts_pristine: pd.DataFrame,
    ) -> DegradedGPSResult:
        """Degrade one drive's OXTS reference into a noisy fix stream."""

        self._validate(oxts_pristine)
        rng = np.random.default_rng(self.config.seed + sum(bytearray(drive.encode("utf-8"))))

        n = len(oxts_pristine)
        lat0 = float(oxts_pristine["lat"].iloc[0])
        lon0 = float(oxts_pristine["lon"].iloc[0])

        gt_x, gt_y = _latlon_to_local_xy(
            oxts_pristine["lat"].to_numpy(dtype=float),
            oxts_pristine["lon"].to_numpy(dtype=float),
            lat0,
            lon0,
        )

        states = self._build_state_schedule(n, rng)
        available = states != OUTAGE

        x_fix = gt_x.copy()
        y_fix = gt_y.copy()

        degraded = oxts_pristine.copy()
        degraded["lat"] = degraded["lat"].astype(float)
        degraded["lon"] = degraded["lon"].astype(float)
        degraded["numsats"] = degraded["numsats"].astype(float)
        degraded["pos_accuracy"] = degraded["pos_accuracy"].astype(float)
        degraded["navstat"] = degraded["navstat"].astype(float)
        degraded["posmode"] = degraded["posmode"].astype(float)

        for i, state in enumerate(states):
            if state == HEALTHY:
                noise = rng.normal(0.0, self.config.healthy_noise_std_m, 2)
                x_fix[i] += noise[0]
                y_fix[i] += noise[1]
                degraded.loc[i, "numsats"] = rng.integers(*self.config.healthy_numsats)
                degraded.loc[i, "pos_accuracy"] = rng.uniform(*self.config.healthy_pos_accuracy)
                degraded.loc[i, "navstat"] = 5.0   # 3D fix
                degraded.loc[i, "posmode"] = 4.0   # auto
            elif state == DEGRADED:
                noise = rng.normal(0.0, self.config.degraded_noise_std_m, 2)
                x_fix[i] += noise[0]
                y_fix[i] += noise[1]
                degraded.loc[i, "numsats"] = rng.integers(*self.config.degraded_numsats)
                degraded.loc[i, "pos_accuracy"] = rng.uniform(*self.config.degraded_pos_accuracy)
                degraded.loc[i, "navstat"] = 4.0   # 2D fix
                degraded.loc[i, "posmode"] = 2.0   # 2D
            else:  # OUTAGE: no fix at all; hold previous position so the
                # quality metrics still see a discontinuity, but the EKF will
                # not be offered a measurement (gps_available=False).
                degraded.loc[i, "numsats"] = 0.0
                degraded.loc[i, "pos_accuracy"] = 999.0
                degraded.loc[i, "navstat"] = 1.0   # no fix
                degraded.loc[i, "posmode"] = 0.0

        lat_degraded, lon_degraded = _local_xy_to_latlon(x_fix, y_fix, lat0, lon0)
        degraded["lat"] = lat_degraded
        degraded["lon"] = lon_degraded

        return DegradedGPSResult(
            oxts_degraded=degraded,
            states=states,
            x_fix_m=x_fix,
            y_fix_m=y_fix,
            gps_available=available,
            gt_x_m=gt_x,
            gt_y_m=gt_y,
            seed=self.config.seed,
        )

    def _build_state_schedule(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Alternate healthy / degraded / outage runs with a minimum run length."""

        if n == 0:
            return np.array([], dtype=object)

        states: list[str] = []
        pos = 0
        while pos < n:
            roll = rng.random()
            if roll < self.config.outage_probability:
                state = OUTAGE
            elif roll < self.config.outage_probability + self.config.degraded_probability:
                state = DEGRADED
            else:
                state = HEALTHY
            run = int(rng.integers(self.config.min_segment_length, self.config.max_segment_length + 1))
            run = min(run, n - pos)
            states.extend([state] * run)
            pos += run
        return np.asarray(states[:n], dtype=object)

    def _validate(self, oxts_df: pd.DataFrame) -> None:
        missing = [column for column in OXTS_COLUMNS if column not in oxts_df.columns]
        if missing:
            raise GPSDegradationError(f"OXTS dataframe is missing columns: {missing}")
        if oxts_df.empty:
            raise GPSDegradationError("Cannot degrade an empty OXTS dataframe.")


def degrade_drive_to_csv(
    drive: str,
    oxts_pristine: pd.DataFrame,
    output_dir: Path | str,
    config: GPSDegradationProcessorConfig | None = None,
) -> DegradedGPSResult:
    """Degrade a drive and persist the degraded OXTS frame + labels to CSV."""

    processor = GPSDegradationProcessor(config)
    result = processor.degrade_drive(drive, oxts_pristine)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.oxts_degraded.to_csv(output_dir / f"degraded_oxts_{drive}.csv", index=False)

    labels = pd.DataFrame(
        {
            "Frame": np.arange(len(result.states)),
            "GPS State": result.states,
            "GPS Available": result.gps_available,
            "X Fix (m)": result.x_fix_m,
            "Y Fix (m)": result.y_fix_m,
            "X GT (m)": result.gt_x_m,
            "Y GT (m)": result.gt_y_m,
        }
    )
    labels.to_csv(output_dir / f"gps_degradation_schedule_{drive}.csv", index=False)
    return result


@dataclass(frozen=True)
class GPSDegradationConfig:
    """Configuration for a GPS-degradation processing run."""

    dataset_root: Path
    results_dir: Path
    figures_dir: Path
    seed: int = 42
    save_figures: bool = True
    degradation: GPSDegradationProcessor | None = None


@dataclass(frozen=True)
class GPSDegradationArtifacts:
    """Paths generated by a GPS-degradation processing run."""

    degraded_oxts_csv: Path
    schedule_csv: Path
    figure: Path | None


class GPSDegradationStage:
    """Load pristine OXTS, inject a seeded degradation, and persist artifacts."""

    def __init__(
        self,
        config: GPSDegradationConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or LOGGER

    def process_drive(self, drive: str) -> tuple[DegradedGPSResult, GPSDegradationArtifacts]:
        self.logger.info("Starting GPS degradation for %s", drive)
        dataset = KITTIRawDataset(self.config.dataset_root, drive)
        if not dataset.drive_path.exists():
            raise GPSDegradationError(f"Drive folder not found: {dataset.drive_path}")

        oxts_pristine = self._load_pristine_oxts(dataset, drive)
        processor = self.config.degradation or GPSDegradationProcessor(
            GPSDegradationProcessorConfig(seed=self.config.seed)
        )
        result = processor.degrade_drive(drive, oxts_pristine)
        artifacts = self._export(drive, result)
        self.logger.info(
            "Completed GPS degradation for %s: %s frames -> %s",
            drive,
            len(oxts_pristine),
            artifacts.schedule_csv,
        )
        return result, artifacts

    def _load_pristine_oxts(self, dataset: KITTIRawDataset, drive: str) -> pd.DataFrame:
        frame_count = dataset.get_frame_count()
        if frame_count <= 0:
            raise GPSDegradationError(f"No synchronized frames found for {drive}")
        rows = []
        for frame in range(frame_count):
            try:
                values = np.asarray(dataset.load_oxts(frame), dtype=float)
            except Exception as exc:
                raise GPSDegradationError(
                    f"Failed to load pristine OXTS for {drive}, frame {frame}."
                ) from exc
            if values.shape[0] != len(OXTS_COLUMNS):
                raise GPSDegradationError(
                    f"Unexpected OXTS column count for {drive}, frame {frame}: "
                    f"{values.shape[0]} != {len(OXTS_COLUMNS)}"
                )
            rows.append(values)
        return pd.DataFrame(rows, columns=OXTS_COLUMNS)

    def _export(self, drive: str, result: DegradedGPSResult) -> GPSDegradationArtifacts:
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)

        degraded_oxts_csv = self.config.results_dir / f"degraded_oxts_{drive}.csv"
        schedule_csv = self.config.results_dir / f"gps_degradation_schedule_{drive}.csv"
        result.oxts_degraded.to_csv(degraded_oxts_csv, index=False)

        labels = pd.DataFrame(
            {
                "Frame": np.arange(len(result.states)),
                "GPS State": result.states,
                "GPS Available": result.gps_available,
                "X Fix (m)": result.x_fix_m,
                "Y Fix (m)": result.y_fix_m,
                "X GT (m)": result.gt_x_m,
                "Y GT (m)": result.gt_y_m,
            }
        )
        labels.to_csv(schedule_csv, index=False)

        figure: Path | None = None
        if self.config.save_figures:
            figure = self.config.figures_dir / f"gps_degradation_{drive}.png"
            self._save_figure(labels, figure, drive)

        return GPSDegradationArtifacts(
            degraded_oxts_csv=degraded_oxts_csv,
            schedule_csv=schedule_csv,
            figure=figure,
        )

    def _save_figure(self, labels: pd.DataFrame, output_path: Path, drive: str) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        state_colors = {"healthy": "tab:green", "degraded": "tab:orange", "outage": "tab:red"}
        axes[0].scatter(
            labels["Frame"],
            labels["GPS State"],
            c=[state_colors[s] for s in labels["GPS State"]],
            s=4,
        )
        axes[0].set_title(f"GPS Degradation Schedule - {drive}")
        axes[0].set_ylabel("GPS State")
        axes[0].grid(True, alpha=0.3)

        error = np.sqrt(
            (labels["X Fix (m)"] - labels["X GT (m)"]) ** 2
            + (labels["Y Fix (m)"] - labels["Y GT (m)"]) ** 2
        )
        axes[1].plot(labels["Frame"], error, linewidth=1.2, color="tab:blue")
        axes[1].set_ylabel("Fix Error (m)")
        axes[1].set_xlabel("Frame")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def process_gps_degradation_for_drive(
    dataset_root: Path | str,
    drive: str,
    results_dir: Path | str,
    figures_dir: Path | str,
    seed: int = 42,
    save_figures: bool = True,
) -> DegradedGPSResult:
    """Convenience API used by orchestration code for one drive."""

    config = GPSDegradationConfig(
        dataset_root=Path(dataset_root),
        results_dir=Path(results_dir),
        figures_dir=Path(figures_dir),
        seed=seed,
        save_figures=save_figures,
    )
    stage = GPSDegradationStage(config)
    result, _ = stage.process_drive(drive)
    return result
