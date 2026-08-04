"""
===============================================================
Camera Quality Metrics

Paper:
A Confidence-Driven Adaptive Multi-Sensor Fusion Framework
for Robust Autonomous Vehicle Localization

===============================================================
"""

import cv2
import numpy as np


class CameraQualityMetrics:
    """
    Computes image quality metrics for camera confidence estimation.
    """

    def __init__(self):
        pass

    # ==========================================================
    # Brightness
    # ==========================================================

    def brightness(self, image):

        return float(np.mean(image))

    # ==========================================================
    # Contrast
    # ==========================================================

    def contrast(self, image):

        return float(np.std(image))

    # ==========================================================
    # Blur Score
    # ==========================================================

    def blur_score(self, image):

        return float(
            cv2.Laplacian(
                image,
                cv2.CV_64F
            ).var()
        )

    # ==========================================================
    # Entropy
    # ==========================================================

    def entropy(self, image):

        histogram = cv2.calcHist(
            [image],
            [0],
            None,
            [256],
            [0,256]
        )

        histogram = histogram.ravel()

        probability = histogram / histogram.sum()

        probability = probability[
            probability > 0
        ]

        entropy = -np.sum(
            probability * np.log2(probability)
        )

        return float(entropy)

    # ==========================================================
    # Dynamic Range
    # ==========================================================

    def dynamic_range(self, image):

        return int(
            image.max() - image.min()
        )

    # ==========================================================
    # Edge Density
    # ==========================================================

    def edge_density(self, image):

        edges = cv2.Canny(
            image,
            100,
            200
        )

        density = np.sum(edges > 0) / edges.size

        return float(density)

    # ==========================================================
    # Noise Estimate
    # ==========================================================

    def noise_estimate(self, image):

        blur = cv2.GaussianBlur(
            image,
            (3,3),
            0
        )

        noise = image.astype(np.float32) - blur.astype(np.float32)

        return float(
            np.std(noise)
        )

    # ==========================================================
    # Complete Metrics
    # ==========================================================

    def compute(self, image):

        return {

            "Brightness":
                self.brightness(image),

            "Contrast":
                self.contrast(image),

            "Blur":
                self.blur_score(image),

            "Entropy":
                self.entropy(image),

            "Dynamic Range":
                self.dynamic_range(image),

            "Edge Density":
                self.edge_density(image),

            "Noise":
                self.noise_estimate(image)

        }