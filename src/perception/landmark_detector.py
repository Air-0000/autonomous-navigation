"""Visual Landmark Detector for Bio-Inspired Navigation.

Detects salient landmark pillars (Red, Blue, Green, Yellow) from monocular camera frames.
Extracts:
- Landmark ID / Color
- Bearing angle relative to robot forward heading (rad)
- Estimated distance based on pinhole geometry (m)
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class LandmarkDetector:
    """HSV color-space landmark detector with pinhole geometry estimation."""

    # Color ranges in HSV (Hue: 0-180, Sat: 0-255, Val: 0-255)
    COLOR_RANGES = {
        "landmark_red": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))
        ],
        "landmark_blue": [
            (np.array([100, 150, 70]), np.array([130, 255, 255]))
        ],
        "landmark_green": [
            (np.array([40, 120, 70]), np.array([80, 255, 255]))
        ],
        "landmark_yellow": [
            (np.array([20, 120, 70]), np.array([35, 255, 255]))
        ]
    }

    # Physical pillar dimensions (from SDF model: height 1.5m, radius 0.25m)
    REAL_HEIGHT_M = 1.5
    # Camera horizontal FOV: 1.047 rad (60 deg), default image width 640
    DEFAULT_HFOV = 1.04719755

    def __init__(self, image_width: int = 640, image_height: int = 480):
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = (image_width / 2.0) / math.tan(self.DEFAULT_HFOV / 2.0)

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect landmark pillars from RGB image.

        Args:
            image: HxWx3 RGB or BGR numpy array

        Returns:
            List of detected landmark dicts:
            [{
                "id": str,
                "bbox": (x, y, w, h),
                "bearing": float,  # radians relative to center (+ is right, - is left)
                "distance": float, # meters
                "confidence": float
            }]
        """
        if image is None or image.size == 0:
            return []

        # Convert to BGR if input is RGB
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        detections = []

        for landmark_id, ranges in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                sub_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, sub_mask)

            # Morphological noise removal
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 120:  # Minimum pixel threshold
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(h) / max(1, w)

                # Pillar visual geometry check: pillars are taller than wide
                if aspect_ratio < 1.1:
                    continue

                # Center offset
                cx = x + w / 2.0
                offset_x = cx - (self.image_width / 2.0)
                bearing = math.atan2(offset_x, self.focal_length)

                # Distance estimation via pinhole model: Z = f * H / h_pixel
                distance = max(0.3, (self.focal_length * self.REAL_HEIGHT_M) / max(1, h))

                detections.append({
                    "id": landmark_id,
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "bearing": float(bearing),
                    "distance": float(distance),
                    "confidence": min(1.0, float(area) / 2500.0)
                })

        return detections
