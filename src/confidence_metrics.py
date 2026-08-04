"""
===============================================================
Confidence Metrics

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

Author : Vishwa
===============================================================
"""

import pandas as pd


class ConfidenceMetrics:
    """
    Computes Camera Confidence from image quality metrics.
    """

    def __init__(self):

        self.weights = {

            "Brightness": 0.15,

            "Contrast": 0.15,

            "Blur": 0.25,

            "Entropy": 0.20,

            "Edge Density": 0.15,

            "Noise": 0.10

        }

    # ==========================================================
    # Min-Max Normalization
    # ==========================================================

    def normalize(self, quality_df):

        normalized = quality_df.copy()

        columns = [

            "Brightness",

            "Contrast",

            "Blur",

            "Entropy",

            "Edge Density"

        ]

        for column in columns:

            minimum = normalized[column].min()

            maximum = normalized[column].max()

            normalized[column] = (

                normalized[column] - minimum

            ) / (

                maximum - minimum + 1e-8

            )

        # --------------------------------------------
        # Noise is inversely proportional to confidence
        # --------------------------------------------

        minimum = quality_df["Noise"].min()

        maximum = quality_df["Noise"].max()

        normalized["Noise"] = 1 - (

            (

                quality_df["Noise"] - minimum

            )

            /

            (

                maximum - minimum + 1e-8

            )

        )

        return normalized

    # ==========================================================
    # Camera Confidence
    # ==========================================================

    def camera_confidence(self, normalized):

        normalized = normalized.copy()

        normalized["Camera Confidence"] = (

              self.weights["Brightness"]

            * normalized["Brightness"]

            +

              self.weights["Contrast"]

            * normalized["Contrast"]

            +

              self.weights["Blur"]

            * normalized["Blur"]

            +

              self.weights["Entropy"]

            * normalized["Entropy"]

            +

              self.weights["Edge Density"]

            * normalized["Edge Density"]

            +

              self.weights["Noise"]

            * normalized["Noise"]

        )

        return normalized

    # ==========================================================
    # Confidence Level
    # ==========================================================

    @staticmethod
    def classify(score):

        if score >= 0.90:

            return "Excellent"

        elif score >= 0.75:

            return "High"

        elif score >= 0.60:

            return "Moderate"

        elif score >= 0.40:

            return "Low"

        else:

            return "Poor"

    # ==========================================================
    # Complete Pipeline
    # ==========================================================

    def compute(self, quality_df):

        result = self.normalize(quality_df)

        result = self.camera_confidence(result)

        result["Confidence Level"] = (

            result["Camera Confidence"]

            .apply(self.classify)

        )

        return result