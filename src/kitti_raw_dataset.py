"""
====================================================================
KITTI Raw Dataset API

Paper:
Sensor Confidence Index (SCI): An Intelligent Framework for
Dynamic Sensor Quality Assessment in Autonomous Vehicle Perception

Author : Vishwa
====================================================================
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd


class KITTIRawDataset:

    """
    KITTI Raw Dataset API

    Supports

    - Dataset validation
    - Dataset summary
    - Image loading
    - OXTS loading
    - Calibration loading
    - Complete synchronized sample
    """

    def __init__(self, dataset_root, drive):

        self.root = Path(dataset_root)

        self.drive = drive

        # ----------------------------------------------------------
        # Dataset Structure
        #
        # KITTI/
        #   └── 2011_09_26/
        #         └── 2011_09_26_drive_0001_sync/
        # ----------------------------------------------------------

        self.date = "2011_09_26"

        self.drive_path = (

            self.root /

            self.date /

            drive

        )

        self.image_dir = (

            self.drive_path /

            "image_02" /

            "data"

        )

        self.oxts_dir = (

            self.drive_path /

            "oxts" /

            "data"

        )

        # ----------------------------------------------------------
        # Calibration Files
        # ----------------------------------------------------------

        self.calib_root = (

            self.root /

            self.date

        )

        self.calib_cam = (

            self.calib_root /

            "calib_cam_to_cam.txt"

        )

        self.calib_imu = (

            self.calib_root /

            "calib_imu_to_velo.txt"

        )

        self.calib_velo = (

            self.calib_root /

            "calib_velo_to_cam.txt"

        )

    # ==========================================================
    # Validation
    # ==========================================================

    def validate(self):

        print("=" * 60)

        paths = {

            "Drive" : self.drive_path,

            "Images" : self.image_dir,

            "OXTS" : self.oxts_dir,

            "Camera Calibration" : self.calib_cam,

            "IMU Calibration" : self.calib_imu,

            "Velodyne Calibration" : self.calib_velo

        }

        for name, path in paths.items():

            status = "FOUND" if path.exists() else "MISSING"

            print(f"{name:<25}: {status}")

        print("=" * 60)

    # ==========================================================
    # Dataset Information
    # ==========================================================

    def get_frame_count(self):

        return len(
            sorted(
                self.image_dir.glob("*.png")
            )
        )

    def get_image_size(self):

        image = cv2.imread(
            str(
                self.image_dir /
                "0000000000.png"
            )
        )

        h, w = image.shape[:2]

        return w, h

    def summary(self):

        width, height = self.get_image_size()

        summary = pd.DataFrame({

            "Property":[

                "Drive",

                "Total Frames",

                "Image Width",

                "Image Height"

            ],

            "Value":[

                self.drive,

                self.get_frame_count(),

                width,

                height

            ]

        })

        return summary

    # ==========================================================
    # Image
    # ==========================================================

    def load_image(self, frame):

        image_path = (

            self.image_dir /

            f"{frame:010d}.png"

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
    # OXTS
    # ==========================================================

    def load_oxts(self, frame):

        file = (

            self.oxts_dir /

            f"{frame:010d}.txt"

        )

        if not file.exists():

            raise FileNotFoundError(file)

        values = np.loadtxt(file)

        return values

    # ==========================================================
    # Calibration
    # ==========================================================

    def load_calibration(self):

        calibration = {}

        for file in [

            self.calib_cam,

            self.calib_imu,

            self.calib_velo

        ]:

            with open(file) as f:

                calibration[file.stem] = f.readlines()

        return calibration

    # ==========================================================
    # Complete Sample
    # ==========================================================

    def load_sample(self, frame):

        sample = {

            "frame": frame,

            "image": self.load_image(frame),

            "oxts": self.load_oxts(frame),

            "calibration": self.load_calibration()

        }

        return sample