 AUTOMATED-FACE-DETECTION-AND-FACIAL-FEATURE-
# Automated Face Detection & Facial Feature Localization System

A modular, CPU-friendly Python computer vision pipeline built using **OpenCV** and **Viola–Jones Haar Cascade Classifiers** with **Upper-65% Face Region of Interest (ROI)** slicing for real-time frontal face and ocular localization.

---

## Academic & Team Details
* **University:** Bennett University, Greater Noida, Uttar Pradesh
* **Program:** B.Tech in Computer Science and Engineering (Batch 40 / S20)
* **Course:** Computational Thinking and Programming (`2026CSAE100`)
* **Faculty Mentor:** Dr. Jayant Vyas (Assistant Professor, SCSET)

### Team Members
| S. No. | Student Name | Enrollment No. | Assigned Modules & Role |
| :---: | :--- | :--- | :--- |
| 1 | **Shivendra Sharma** | `S26CSEU1403` | Project Lead, CLI Integration & Parameter Analysis (`main.py`, `analysis.py`) |
| 2 | **Mohd Kaif** | `S26CSEU1395` | Core Haar Cascade Face & Upper-65% Eye ROI Engine (`detector.py`) |
| 3 | **Deepak Rathi** | `S26CSEU1414` | Image Validation, Resizing & Grayscale Pipeline (`preprocessor.py`) |
| 4 | **Malya Sharma** | `S26CSEU1393` | Bounding-Box Annotation & HUD Rendering (`visualizer.py`) |
| 5 | **Anshika** | `S26CSEU1388` | CSV Audit Logging, Unit Testing & Documentation (`data_manager.py`, `tests/`) |

---

## System Architecture (6-Module Pipeline)
1. **`preprocessor.py`**: Validates file paths/extensions, normalizes large images (`max_width = 1000 px`) while preserving aspect ratio, and converts BGR frames to 1-channel grayscale.
2. **`detector.py`**: Encapsulates `HaarFaceEyeDetector` with bundled OpenCV XML fallback paths and restricts ocular cascade search strictly to the **upper 65% ROI** (`gray[y : y + int(0.65 * h), x : x + w]`) of each detected face.
3. **`visualizer.py`**: Draws green face bounding boxes, blue eye bounding boxes, and a top HUD status banner (`Faces | Eyes | Time ms`).
4. **`data_manager.py`**: Initializes `data/detection_log.csv`, appends timestamped execution records, and reads historical run metrics.
5. **`analysis.py`**: Benchmarks 4 hyperparameter presets (`Strict Precision`, `Fast Coarse Scan`, `Balanced Default`, `High Recall`) and ranks them by Count Error (`|Detected - Expected|`), Eye Confirmations, and Execution Latency (ms).
6. **`main.py`**: Interactive 7-option Command-Line Interface (CLI) supporting single-image detection, folder batch processing, runtime parameter tuning, preset benchmarking, CSV history summary, and live webcam tracking.

---

## Key Benchmark & Performance Highlights
* **Live Webcam Speed:** `55–70 FPS` (`14.13–18.10 ms/frame`) on standard laptop CPU hardware without GPU dependency.
* **Single Portrait Localization:** Localized `1` frontal face and `2` eyes in `138.28 ms` (`portrait_test.jpg`).
* **Multi-Face Crowd Detection:** Localized all `10` faces in `66.44 ms` with zero background false positives (`Screenshot (221).png`).
* **Preset Optimization:** `Strict Precision` (`scaleFactor=1.20`, `minNeighbors=7`, `minSize=(40, 40)`) achieved `0` Count Error in `87.44 ms` on `group_test.jpg`.

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/Mohd-Kaif-codes/AUTOMATED-FACE-DETECTION-AND-FACIAL-FEATURE-.git](https://github.com/Mohd-Kaif-codes/AUTOMATED-FACE-DETECTION-AND-FACIAL-FEATURE-.git)
cd AUTOMATED-FACE-DETECTION-AND-FACIAL-FEATURE-
