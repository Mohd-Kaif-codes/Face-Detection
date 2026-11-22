"""
analysis.py
Benchmarks 4 parameter presets on a target image, computes Count Error or
Consensus Deviation, ranks presets, and summarizes detection_log.csv history.
"""

import statistics
import time
from data_manager import read_all_records

PRESETS = [
    {"name": "Strict Precision", "scaleFactor": 1.20, "minNeighbors": 7, "minSize": (40, 40)},
    {"name": "Fast Coarse Scan", "scaleFactor": 1.30, "minNeighbors": 5, "minSize": (50, 50)},
    {"name": "Balanced Default", "scaleFactor": 1.10, "minNeighbors": 5, "minSize": (30, 30)},
    {"name": "High Recall",      "scaleFactor": 1.05, "minNeighbors": 3, "minSize": (20, 20)},
]


def benchmark_presets(detector, gray_image, expected_faces: int = None) -> list:
    """Run all 4 parameter presets and rank by lowest error, highest eyes, fastest runtime."""
    results = []
    for preset in PRESETS:
        t0 = time.perf_counter()
        detections = detector.detect_faces_and_eyes(
            gray_image,
            scale_factor=preset["scaleFactor"],
            min_neighbors=preset["minNeighbors"],
            min_size=preset["minSize"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        face_cnt = len(detections)
        eye_cnt = sum(len(d["eye_boxes"]) for d in detections)

        results.append({
            "preset": preset["name"],
            "scaleFactor": preset["scaleFactor"],
            "minNeighbors": preset["minNeighbors"],
            "faces": face_cnt,
            "eyes": eye_cnt,
            "runtime_ms": round(elapsed_ms, 2),
        })

    # Reference count: ground truth if provided, else median consensus across presets
    reference = (
        expected_faces
        if expected_faces is not None
        else statistics.median([r["faces"] for r in results])
    )

    for r in results:
        r["error"] = abs(r["faces"] - reference)

    # Multi-criteria sort: (1) Lowest Count Error, (2) Highest Confirmed Eyes, (3) Fastest Runtime
    ranked = sorted(results, key=lambda x: (x["error"], -x["eyes"], x["runtime_ms"]))
    return ranked


def summarize_csv_history(csv_path: str = "data/detection_log.csv") -> dict:
    """Aggregate total runs, faces, eyes, and average latency from CSV history."""
    records = read_all_records(csv_path)
    if not records:
        return {"total_runs": 0, "total_faces": 0, "total_eyes": 0, "avg_faces": 0.0, "avg_ms": 0.0}

    total_runs = len(records)
    total_faces = sum(int(r["faces_detected"]) for r in records)
    total_eyes = sum(int(r["eyes_detected"]) for r in records)
    avg_ms = sum(float(r["runtime_ms"]) for r in records) / total_runs

    return {
        "total_runs": total_runs,
        "total_faces": total_faces,
        "total_eyes": total_eyes,
        "avg_faces": round(total_faces / total_runs, 2),
        "avg_ms": round(avg_ms, 2),
    }