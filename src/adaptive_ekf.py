"""
============================================================================
Confidence-Driven Adaptive Extended Kalman Filter (CDA-EKF) -- FIXED

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
============================================================================

Fixes vs. the original ``adaptive_ekf.py``
------------------------------------------
1. **Baseline is a true fixed-noise EKF.**  The original code computed the
   "baseline" by running the *same* adaptive path with uniform 1/3 weights,
   which -- through ``noise_multiplier`` -- actually *inflated* the noise
   covariances instead of leaving them fixed.  Here ``adaptive=False`` uses
   the nominal ``R`` and ``Q`` directly (multiplier == 1.0), which is what a
   conventional non-adaptive EKF does.

2. **GPS outages are handled explicitly.**  When ``gps_available`` is False
   (or the GPS weight is zero) the correction step is skipped and the filter
   dead-reckons on the IMU motion model instead of receiving a garbage fix.

3. **No circular ground truth.**  The measurement fed to the filter comes from
   the degraded GPS fix stream; accuracy is always assessed against a separate
   pristine reference trajectory by the fusion stage, never against the
   measurement itself.

State vector (4,):  [x, y, yaw, v]
Control (from IMU / OXTS, not degraded):
    vf (forward velocity, m/s)   -- IMU/odometry speed, available in outages
    wu (yaw rate, rad/s)         -- IMU angular rate
Measurement (from degraded GPS): [x_fix, y_fix, v_fix]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .gps_quality import OXTS_COLUMNS
except ImportError:  # pragma: no cover - supports direct script execution.
    from gps_quality import OXTS_COLUMNS

LOGGER = logging.getLogger(__name__)

STATE_COLUMNS: tuple[str, ...] = ("X", "Y", "Yaw", "V")


@dataclass(frozen=True)
class AdaptiveEKFConfig:
    """Tunable noise parameters for the CDA-EKF."""

    dt: float = 0.1  # KITTI Raw OXTS packets are logged at 10 Hz.

    # Nominal process-noise variances (used directly by the fixed baseline).
    process_var_position: float = 0.05
    process_var_yaw: float = 0.02
    process_var_speed: float = 0.15

    # Nominal measurement-noise variances.  Position is set to a realistic
    # consumer-grade GPS level (std ~2.4 m); the adaptive stage inflates it
    # when confidence drops, while the fixed baseline keeps it constant.
    measurement_var_position: float = 6.0
    measurement_var_speed: float = 0.5

    # Initial state covariance (diagonal).
    initial_covariance: tuple[float, ...] = (10.0, 10.0, 0.5, 2.0)

    # How strongly a confidence/weight deficit inflates the associated noise.
    # multiplier = 1 + adaptive_gain * (1 - weight)
    adaptive_gain: float = 6.0

    min_multiplier: float = 1.0
    max_multiplier: float = 40.0

    # If True, GPS position/speed innovations never change the heading state:
    # yaw is propagated purely by the IMU yaw-rate input.  This reflects that
    # a single-antenna GPS position fix does not directly observe heading, so
    # noisy degraded fixes cannot corrupt the heading used for dead-reckoning.
    heading_from_imu_only: bool = True

    def noise_multiplier(self, weight: float) -> float:
        """Map a confidence/weight in [0, 1] to a noise-covariance multiplier."""

        weight = float(np.clip(weight, 0.0, 1.0))
        multiplier = 1.0 + self.adaptive_gain * (1.0 - weight)
        return float(np.clip(multiplier, self.min_multiplier, self.max_multiplier))


class AdaptiveEKFError(RuntimeError):
    """Raised when the CDA-EKF cannot process a drive safely."""


class AdaptiveEKF:
    """Confidence-driven adaptive EKF for the [x, y, yaw, v] bicycle model."""

    def __init__(
        self,
        initial_state: Sequence[float],
        config: AdaptiveEKFConfig | None = None,
    ) -> None:
        self.config = config or AdaptiveEKFConfig()
        self.state = np.asarray(initial_state, dtype=float).reshape(4)
        self.covariance = np.diag(
            np.asarray(self.config.initial_covariance, dtype=float)
        )

    def predict(
        self,
        vf: float,
        wu: float,
        dt: float | None = None,
        imu_weight: float = 1.0,
        camera_weight: float = 1.0,
    ) -> None:
        """Propagate state and covariance using the IMU velocity/yaw-rate input."""

        dt = self.config.dt if dt is None else float(dt)
        x, y, yaw, _v = self.state

        new_x = x + vf * np.cos(yaw) * dt
        new_y = y + vf * np.sin(yaw) * dt
        new_yaw = _wrap_angle(yaw + wu * dt)
        new_v = vf
        self.state = np.array([new_x, new_y, new_yaw, new_v], dtype=float)

        F = np.eye(4)
        F[0, 2] = -vf * np.sin(yaw) * dt
        F[0, 3] = np.cos(yaw) * dt
        F[1, 2] = vf * np.cos(yaw) * dt
        F[1, 3] = np.sin(yaw) * dt

        imu_multiplier = self.config.noise_multiplier(imu_weight)
        camera_multiplier = self.config.noise_multiplier(camera_weight)

        Q = np.diag(
            [
                self.config.process_var_position * imu_multiplier,
                self.config.process_var_position * imu_multiplier,
                self.config.process_var_yaw * camera_multiplier,
                self.config.process_var_speed * imu_multiplier,
            ]
        )
        self.covariance = F @ self.covariance @ F.T + Q

    def update(
        self,
        z: Sequence[float],
        gps_weight: float = 1.0,
    ) -> None:
        """Correct the state estimate using a GPS position/speed fix."""

        z = np.asarray(z, dtype=float).reshape(3)

        H = np.zeros((3, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 3] = 1.0

        gps_multiplier = self.config.noise_multiplier(gps_weight)
        R = np.diag(
            [
                self.config.measurement_var_position * gps_multiplier,
                self.config.measurement_var_position * gps_multiplier,
                self.config.measurement_var_speed * gps_multiplier,
            ]
        )

        predicted_z = H @ self.state
        innovation = z - predicted_z

        S = H @ self.covariance @ H.T + R
        kalman_gain = self.covariance @ H.T @ np.linalg.inv(S)

        if self.config.heading_from_imu_only:
            # GPS position/speed measurements cannot observe heading with a
            # single antenna; keep yaw driven only by the IMU yaw-rate.
            kalman_gain = kalman_gain.copy()
            kalman_gain[2, :] = 0.0

        self.state = self.state + kalman_gain @ innovation
        self.state[2] = _wrap_angle(self.state[2])

        identity = np.eye(4)
        # Joseph form keeps the covariance symmetric and positive-definite
        # even when the Kalman gain is modified (e.g. the yaw row is zeroed
        # for heading-from-IMU-only operation).
        imkh = identity - kalman_gain @ H
        self.covariance = imkh @ self.covariance @ imkh.T + kalman_gain @ R @ kalman_gain.T

    def step(
        self,
        vf: float,
        wu: float,
        z: Sequence[float],
        dt: float | None = None,
        imu_weight: float = 1.0,
        camera_weight: float = 1.0,
        gps_weight: float = 1.0,
        gps_available: bool = True,
    ) -> np.ndarray:
        """Run one predict/update cycle; skip correction when GPS is unavailable."""

        self.predict(
            vf=vf,
            wu=wu,
            dt=dt,
            imu_weight=imu_weight,
            camera_weight=camera_weight,
        )
        if gps_available and gps_weight > 0.0:
            self.update(z=z, gps_weight=gps_weight)
        return self.state.copy()


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class WeightSeries:
    """Per-frame adaptive weights fed to the CDA-EKF."""

    camera: np.ndarray
    gps: np.ndarray
    imu: np.ndarray


def _build_weight_series(
    frame_count: int,
    weights_df: pd.DataFrame | None,
) -> WeightSeries:
    if weights_df is None:
        return WeightSeries(
            camera=np.ones(frame_count),
            gps=np.ones(frame_count),
            imu=np.ones(frame_count),
        )

    required = {"Camera Weight", "GPS Weight", "IMU Weight"}
    missing = required.difference(weights_df.columns)
    if missing:
        raise AdaptiveEKFError(f"Adaptive weights dataframe is missing columns: {missing}")
    if len(weights_df) != frame_count:
        raise AdaptiveEKFError(
            f"Adaptive weights row count ({len(weights_df)}) does not match "
            f"frame count ({frame_count})."
        )

    ordered = weights_df.sort_values("Frame") if "Frame" in weights_df.columns else weights_df
    return WeightSeries(
        camera=ordered["Camera Weight"].to_numpy(dtype=float),
        gps=ordered["GPS Weight"].to_numpy(dtype=float),
        imu=ordered["IMU Weight"].to_numpy(dtype=float),
    )


def run_adaptive_ekf(
    oxts_df: pd.DataFrame,
    weights_df: pd.DataFrame | None = None,
    config: AdaptiveEKFConfig | None = None,
    adaptive: bool = True,
    x_fix_m: np.ndarray | None = None,
    y_fix_m: np.ndarray | None = None,
    v_fix_mps: np.ndarray | None = None,
    gps_available: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Run the CDA-EKF over an entire drive's OXTS sequence.

    Parameters
    ----------
    oxts_df:
        Per-frame OXTS packets (degraded GPS stream). Must contain at least
        ``OXTS_COLUMNS``.
    weights_df:
        Adaptive sensor weights. If ``None`` or ``adaptive=False``, nominal
        fixed noise covariances are used (true non-adaptive EKF baseline).
    adaptive:
        Whether to apply the supplied per-frame weights.
    x_fix_m / y_fix_m / v_fix_mps:
        Measurement fix stream (degraded). Defaults to deriving them from
        the OXTS lat/lon/vn/ve columns.
    gps_available:
        Per-frame boolean mask; ``False`` frames skip the correction step.
        Defaults to all True.

    Returns
    -------
    pandas.DataFrame
        One row per frame: ``Frame``, ``X``, ``Y``, ``Yaw``, ``V``,
        ``X GPS``, ``Y GPS``, ``V GPS`` (the degraded fix stream).
    """

    missing = [column for column in OXTS_COLUMNS if column not in oxts_df.columns]
    if missing:
        raise AdaptiveEKFError(f"OXTS dataframe is missing columns: {missing}")
    if oxts_df.empty:
        raise AdaptiveEKFError("OXTS dataframe contains no rows.")

    config = config or AdaptiveEKFConfig()
    frame_count = len(oxts_df)

    if x_fix_m is None or y_fix_m is None:
        x_gps, y_gps = _latlon_to_local_xy(
            oxts_df["lat"].to_numpy(dtype=float),
            oxts_df["lon"].to_numpy(dtype=float),
        )
    else:
        x_gps = np.asarray(x_fix_m, dtype=float)
        y_gps = np.asarray(y_fix_m, dtype=float)
    if len(x_gps) != frame_count:
        raise AdaptiveEKFError("x_fix_m length does not match frame count.")

    if v_fix_mps is None:
        v_gps = np.sqrt(
            oxts_df["vn"].to_numpy(dtype=float) ** 2
            + oxts_df["ve"].to_numpy(dtype=float) ** 2
        )
    else:
        v_gps = np.asarray(v_fix_mps, dtype=float)
    if len(v_gps) != frame_count:
        raise AdaptiveEKFError("v_fix_mps length does not match frame count.")

    if gps_available is None:
        gps_available = np.ones(frame_count, dtype=bool)
    else:
        gps_available = np.asarray(gps_available, dtype=bool)
    if len(gps_available) != frame_count:
        raise AdaptiveEKFError("gps_available length does not match frame count.")

    yaw_gps = oxts_df["yaw"].to_numpy(dtype=float)
    vf = oxts_df["vf"].to_numpy(dtype=float)
    wu = oxts_df["wu"].to_numpy(dtype=float)

    weight_series = _build_weight_series(frame_count, weights_df if adaptive else None)

    initial_state = np.array([x_gps[0], y_gps[0], yaw_gps[0], vf[0]], dtype=float)
    ekf = AdaptiveEKF(initial_state=initial_state, config=config)

    records: list[dict[str, float]] = [
        {
            "Frame": 0,
            "X": initial_state[0],
            "Y": initial_state[1],
            "Yaw": initial_state[2],
            "V": initial_state[3],
            "X GPS": x_gps[0],
            "Y GPS": y_gps[0],
            "V GPS": v_gps[0],
        }
    ]

    for frame in range(1, frame_count):
        z = (x_gps[frame], y_gps[frame], v_gps[frame])
        state = ekf.step(
            vf=vf[frame],
            wu=wu[frame],
            z=z,
            dt=config.dt,
            imu_weight=weight_series.imu[frame],
            camera_weight=weight_series.camera[frame],
            gps_weight=weight_series.gps[frame],
            gps_available=bool(gps_available[frame]),
        )
        records.append(
            {
                "Frame": frame,
                "X": state[0],
                "Y": state[1],
                "Yaw": state[2],
                "V": state[3],
                "X GPS": x_gps[frame],
                "Y GPS": y_gps[frame],
                "V GPS": v_gps[frame],
            }
        )

    return pd.DataFrame.from_records(records)


def _latlon_to_local_xy(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from .gps_quality import EARTH_RADIUS_METERS
    lat0 = float(lat_deg[0])
    lon0 = float(lon_deg[0])
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = EARTH_RADIUS_METERS * (lon - np.radians(lon0)) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_METERS * (lat - np.radians(lat0))
    return x, y
