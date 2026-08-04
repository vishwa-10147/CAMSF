"""
====================================================================
KITTI Odometry Dataset API

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
====================================================================
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd


class KITTIOdometryDataset:
    """
    KITTI Odometry Dataset API

    This class provides methods for:

    - Dataset validation
    - Dataset summary
    - Image loading
    - Calibration loading
    - Ground truth loading
    - Sequence information
    """

    # ==========================================================
    # Constructor
    # ==========================================================

    def __init__(self, dataset_root):

        self.root = Path(dataset_root)

        self.color_dir = (
            self.root
            / "data_odometry_color"
            / "dataset"
            / "sequences"
        )

        self.calib_dir = (
            self.root
            / "data_odometry_calib"
            / "dataset"
            / "sequences"
        )

        self.pose_dir = (
            self.root
            / "data_odometry_poses"
            / "dataset"
            / "poses"
        )

    # ==========================================================
    # Dataset Validation
    # ==========================================================

    def validate(self):

        print("=" * 60)

        paths = {
            "Color Dataset": self.color_dir,
            "Calibration": self.calib_dir,
            "Ground Truth": self.pose_dir,
        }

        for name, path in paths.items():

            status = "FOUND" if path.exists() else "MISSING"

            print(f"{name:<20}: {status}")

        print("=" * 60)

    # ==========================================================
    # Dataset Information
    # ==========================================================

    def list_sequences(self):

        return sorted(
            [
                folder.name
                for folder in self.color_dir.iterdir()
                if folder.is_dir()
            ]
        )

    def get_frame_count(self, sequence):

        image_dir = (
            self.color_dir
            / sequence
            / "image_2"
        )

        return len(list(image_dir.glob("*.png")))

    def get_image_size(self, sequence="00"):

        image_path = (
            self.color_dir
            / sequence
            / "image_2"
            / "000000.png"
        )

        image = cv2.imread(str(image_path))

        height, width = image.shape[:2]

        return width, height

    def summary(self):

        sequences = self.list_sequences()

        training = 0
        testing = 0
        total_frames = 0

        for seq in sequences:

            total_frames += self.get_frame_count(seq)

            if (self.pose_dir / f"{seq}.txt").exists():

                training += 1

            else:

                testing += 1

        width, height = self.get_image_size()

        summary = pd.DataFrame({

            "Property": [

                "Total Sequences",

                "Training Sequences",

                "Testing Sequences",

                "Total Frames",

                "Image Width",

                "Image Height"

            ],

            "Value": [

                len(sequences),

                training,

                testing,

                total_frames,

                width,

                height

            ]

        })

        return summary

    # ==========================================================
    # Image API
    # ==========================================================

    def load_image(
        self,
        sequence,
        frame,
        camera="image_2"
    ):

        image_path = (

            self.color_dir
            / sequence
            / camera
            / f"{frame:06d}.png"

        )

        if not image_path.exists():

            raise FileNotFoundError(image_path)

        image = cv2.imread(str(image_path))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return image

    # ==========================================================
    # Calibration API
    # ==========================================================

    def load_calibration(self, sequence):

        calib_file = (

            self.calib_dir
            / sequence
            / "calib.txt"

        )

        calibration = {}

        with open(calib_file) as file:

            for line in file:

                key, value = line.strip().split(":", 1)

                calibration[key] = np.array(

                    list(map(float, value.split()))

                )

        return calibration

    # ==========================================================
    # Ground Truth API
    # ==========================================================

    def load_pose(self, sequence, frame):

        pose_file = self.pose_dir / f"{sequence}.txt"

        if not pose_file.exists():

            return None

        poses = np.loadtxt(pose_file)

        return poses[frame]

    # ==========================================================
    # Complete Sample
    # ==========================================================

    def load_sample(
        self,
        sequence,
        frame
    ):

        sample = {

            "sequence": sequence,

            "frame": frame,

            "image": self.load_image(
                sequence,
                frame
            ),

            "calibration": self.load_calibration(
                sequence
            ),

            "pose": self.load_pose(
                sequence,
                frame
            )

        }

        return sample