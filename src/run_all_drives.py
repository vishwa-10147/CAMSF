"""
===============================================================
Run Complete Pipeline for All KITTI Drives (CLI entry point)

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
===============================================================

Thin, script-friendly wrapper around ``src.run_pipeline.run_pipeline``.
Unlike ``src/run_all_drives.py`` (a fixed, hard-coded loop used from
inside notebooks/IDE runs), this script is meant to be invoked from a
terminal at the project root:

    python scripts/run_all_drives.py --dataset-root D:\\Research\\Datasets\\KITTI ^
        --project-root D:\\Research\\Papers\\Paper_2

It processes every drive independently, keeps going if one drive
fails, and writes a JSON run manifest summarizing successes/failures
so the batch can be re-run for just the failed drives.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

# scripts/ is a sibling of src/, so make the project root importable
# when this file is executed directly (``python scripts/run_all_drives.py``).
PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

from src.run_pipeline import DATASET_ROOT, DRIVES, PROJECT_ROOT, PipelineError, run_pipeline  # noqa: E402


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console logging for script execution."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the batch driver."""

    parser = argparse.ArgumentParser(
        description="Run the full Confidence-Driven Adaptive Fusion pipeline for every KITTI Raw drive."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Path to the KITTI Raw dataset root.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing/receiving results and figures directories.",
    )
    parser.add_argument(
        "--drive",
        action="append",
        dest="drives",
        help="Drive name to process. Repeat for multiple drives. Defaults to all DRIVES.",
    )
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure export.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch on the first drive failure instead of continuing.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="run_manifest.json",
        help="Filename (under <project-root>/results) for the JSON run manifest.",
    )
    return parser.parse_args(argv)


def run_all_drives(
    dataset_root: Path,
    project_root: Path,
    drives: list[str],
    save_figures: bool = True,
    rolling_window: int = 5,
    stop_on_error: bool = False,
) -> dict:
    """Run the pipeline for every drive and return a JSON-serializable manifest."""

    manifest: dict = {
        "dataset_root": str(dataset_root),
        "project_root": str(project_root),
        "total_drives": len(drives),
        "drives": {},
    }

    started_at = time.time()
    successes: list[str] = []
    failures: list[str] = []

    for drive in drives:
        print()
        print("=" * 70)
        print(f"Processing : {drive}")
        print("=" * 70)

        drive_started_at = time.time()
        try:
            run_pipeline(
                dataset_root=dataset_root,
                drive=drive,
                project_root=project_root,
                save_figures=save_figures,
                rolling_window=rolling_window,
            )
            elapsed = time.time() - drive_started_at
            manifest["drives"][drive] = {"status": "success", "elapsed_seconds": elapsed}
            successes.append(drive)
            print(f"[OK] {drive} completed in {elapsed:.1f}s")
        except PipelineError as exc:
            elapsed = time.time() - drive_started_at
            LOGGER.exception("Pipeline failed for drive: %s", drive)
            manifest["drives"][drive] = {
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": str(exc),
            }
            failures.append(drive)
            if stop_on_error:
                print(f"[FAIL] {drive} failed; stopping (--stop-on-error).")
                break
            print(f"[FAIL] {drive} failed; continuing with remaining drives.")

    manifest["successes"] = successes
    manifest["failures"] = failures
    manifest["total_elapsed_seconds"] = time.time() - started_at
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    """Run the batch driver from the command line."""

    configure_logging()
    args = parse_args(argv)
    drives = args.drives or DRIVES

    print("=" * 70)
    print("RUNNING COMPLETE KITTI PIPELINE")
    print("=" * 70)

    manifest = run_all_drives(
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        drives=drives,
        save_figures=not args.no_figures,
        rolling_window=args.rolling_window,
        stop_on_error=args.stop_on_error,
    )

    results_dir = args.project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print("=" * 70)
    print(f"Successes : {len(manifest['successes'])}/{manifest['total_drives']}")
    print(f"Failures  : {len(manifest['failures'])}")
    if manifest["failures"]:
        print(f"Failed drives: {manifest['failures']}")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 70)

    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())