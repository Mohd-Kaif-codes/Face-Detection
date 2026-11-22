"""
preprocessor.py
Handles path/extension validation, proportional resizing for large images,
and 3-channel BGR to single-channel Grayscale conversion.
"""

import os
import cv2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_and_load_image(image_path: str):
    """Validate file existence, extension, and decode image via OpenCV."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    _, ext = os.path.splitext(image_path)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Use {SUPPORTED_EXTENSIONS}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Corrupted or unreadable image file: {image_path}")

    return image


def preprocess_frame(image, max_width: int = 1000):
    """Resize large images proportionally (width > 1000px) and convert to Grayscale."""
    if image is None or image.size == 0:
        raise ValueError("Empty image buffer passed to preprocessor.")

    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        new_dim = (max_width, int(height * scale))
        image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray