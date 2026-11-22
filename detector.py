"""
detector.py
Loads Haar Cascade XMLs (with bundled OpenCV fallback), detects frontal faces,
and slices the upper 65% face Region of Interest (ROI) for ocular detection.
"""

import os
import cv2


class HaarFaceEyeDetector:
    """Encapsulates face and eye cascade classifiers with parameter validation."""

    def __init__(self, face_xml_path: str = None, eye_xml_path: str = None):
        default_dir = cv2.data.haarcascades
        self.face_xml = (
            face_xml_path
            if face_xml_path and os.path.exists(face_xml_path)
            else os.path.join(default_dir, "haarcascade_frontalface_default.xml")
        )
        self.eye_xml = (
            eye_xml_path
            if eye_xml_path and os.path.exists(eye_xml_path)
            else os.path.join(default_dir, "haarcascade_eye.xml")
        )

        self.face_cascade = cv2.CascadeClassifier(self.face_xml)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_xml)

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise IOError("Failed to load Haar Cascade XML classifiers.")

    def detect_faces_and_eyes(
        self,
        gray_image,
        scale_factor: float = 1.10,
        min_neighbors: int = 5,
        min_size: tuple = (30, 30),
    ):
        """Detect frontal faces and localize eyes strictly inside upper 65% face ROI."""
        if scale_factor <= 1.0:
            raise ValueError("scaleFactor must be strictly greater than 1.0 (e.g., 1.10).")
        if min_neighbors < 1:
            raise ValueError("minNeighbors must be >= 1.")

        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detections = []
        for (x, y, w, h) in faces:
            # Slice strictly the upper 65% of the face bounding box to avoid mouth/collar false positives
            roi_height = max(1, int(0.65 * h))
            roi_gray = gray_image[y : y + roi_height, x : x + w]

            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.10,
                minNeighbors=6,
                minSize=(15, 15),
            )

            eye_boxes = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
            detections.append({
                "face_box": (int(x), int(y), int(w), int(h)),
                "eye_boxes": eye_boxes,
            })

        return detections