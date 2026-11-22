"""
visualizer.py
Draws green face bounding boxes, blue eye bounding boxes, a top summary banner,
and saves annotated output images to disk.
"""

import os
import cv2


def draw_detections(image, detections: list, runtime_ms: float = 0.0):
    """Annotate image with green face rectangles, blue eye rectangles, and HUD banner."""
    annotated = image.copy()
    total_faces = len(detections)
    total_eyes = sum(len(d["eye_boxes"]) for d in detections)

    for idx, item in enumerate(detections, start=1):
        fx, fy, fw, fh = item["face_box"]
        # Green rectangle for frontal face
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Face #{idx}",
            (fx, max(20, fy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        # Blue rectangles for eyes inside upper 65% ROI
        for (ex, ey, ew, eh) in item["eye_boxes"]:
            cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # Top Summary Banner
    banner = f"Faces: {total_faces} | Eyes: {total_eyes} | Time: {runtime_ms:.2f} ms"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (15, 23, 42), -1)
    cv2.putText(
        annotated,
        banner,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return annotated


def save_output_image(annotated_image, source_path: str, output_dir: str = "output") -> str:
    """Save annotated image to output directory and return saved file path."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
    cv2.imwrite(out_path, annotated_image)
    return out_path