"""
data_manager.py
Initializes data/detection_log.csv, appends run metrics, and reads history records.
"""

import csv
import os
from datetime import datetime

LOG_HEADERS = [
    "timestamp",
    "source_file",
    "faces_detected",
    "eyes_detected",
    "scale_factor",
    "min_neighbors",
    "runtime_ms",
]


def init_csv_log(csv_path: str = "data/detection_log.csv") -> str:
    """Create parent directory and CSV header row if not already present."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)
    return csv_path


def append_detection_record(
    source_file: str,
    faces_count: int,
    eyes_count: int,
    scale_factor: float,
    min_neighbors: int,
    runtime_ms: float,
    csv_path: str = "data/detection_log.csv",
):
    """Append a single detection execution record to detection_log.csv."""
    init_csv_log(csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            os.path.basename(source_file),
            faces_count,
            eyes_count,
            scale_factor,
            min_neighbors,
            round(runtime_ms, 2),
        ])


def read_all_records(csv_path: str = "data/detection_log.csv") -> list:
    """Read all logged detection rows as a list of dictionaries."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))