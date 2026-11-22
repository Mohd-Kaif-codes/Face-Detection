"""
main.py
CLI Menu driver integrating preprocessing, Haar Cascade face/eye ROI detection,
visual rendering, CSV logging, preset benchmarking, and live webcam streaming.
"""

import os
import time
import cv2

from preprocessor import validate_and_load_image, preprocess_frame, SUPPORTED_EXTENSIONS
from detector import HaarFaceEyeDetector
from visualizer import draw_detections, save_output_image
from data_manager import init_csv_log, append_detection_record
from analysis import benchmark_presets, summarize_csv_history


def process_single_image(detector, image_path: str, params: dict, show_window: bool = True):
    """Run full pipeline on one image, save output, and log metrics to CSV."""
    raw_img = validate_and_load_image(image_path)
    resized_img, gray = preprocess_frame(raw_img)

    t0 = time.perf_counter()
    detections = detector.detect_faces_and_eyes(
        gray,
        scale_factor=params["scaleFactor"],
        min_neighbors=params["minNeighbors"],
        min_size=params["minSize"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    face_cnt = len(detections)
    eye_cnt = sum(len(d["eye_boxes"]) for d in detections)

    annotated = draw_detections(resized_img, detections, elapsed_ms)
    out_path = save_output_image(annotated, image_path)

    append_detection_record(
        image_path,
        face_cnt,
        eye_cnt,
        params["scaleFactor"],
        params["minNeighbors"],
        elapsed_ms,
    )

    print(f"[OK] {os.path.basename(image_path)} -> Faces: {face_cnt}, Eyes: {eye_cnt} ({elapsed_ms:.2f} ms)")
    print(f"     Saved to: {out_path}")

    if show_window:
        cv2.imshow("Detected Faces & Eyes (Press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(detector, params: dict):
    """Live webcam detection loop with real-time face & upper-65% eye ROI boxes."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam device (index 0).")

    print("Live webcam started. Press 'q' to return to menu...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized, gray = preprocess_frame(frame)
        t0 = time.perf_counter()
        detections = detector.detect_faces_and_eyes(
            gray,
            scale_factor=params["scaleFactor"],
            min_neighbors=params["minNeighbors"],
            min_size=params["minSize"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        annotated = draw_detections(resized, detections, elapsed_ms)

        cv2.imshow("Live Webcam Face & Eye Detection (Press 'q' to exit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    init_csv_log()
    detector = HaarFaceEyeDetector()
    params = {"scaleFactor": 1.10, "minNeighbors": 5, "minSize": (30, 30)}

    while True:
        print("\n=== Automated Face & Eye Localization System ===")
        print("1. Single Image Detection")
        print("2. Folder Batch Processing")
        print("3. Update Detection Parameters")
        print("4. Compare & Rank Presets (Benchmark)")
        print("5. View CSV Detection History Summary")
        print("6. Live Webcam Detection")
        print("7. Exit")

        choice = input("Select option (1-7): ").strip()

        try:
            if choice == "1":
                path = input("Enter image path (e.g., portrait_test.jpg): ").strip()
                process_single_image(detector, path, params, show_window=True)

            elif choice == "2":
                folder = input("Enter folder path: ").strip()
                if not os.path.isdir(folder):
                    raise FileNotFoundError(f"Directory not found: {folder}")
                files = [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                ]
                for fpath in files:
                    process_single_image(detector, fpath, params, show_window=False)

            elif choice == "3":
                sf = float(input("Enter scaleFactor (e.g., 1.10): "))
                mn = int(input("Enter minNeighbors (e.g., 5): "))
                ms = int(input("Enter minSize px (e.g., 30): "))
                if sf <= 1.0 or mn < 1 or ms < 10:
                    raise ValueError("Invalid parameter values.")
                params = {"scaleFactor": sf, "minNeighbors": mn, "minSize": (ms, ms)}
                print(f"[UPDATED] Parameters set to {params}")

            elif choice == "4":
                path = input("Enter benchmark image path (e.g., group_test.jpg): ").strip()
                exp_str = input("Expected face count (press Enter if unknown): ").strip()
                expected = int(exp_str) if exp_str else None
                raw_img = validate_and_load_image(path)
                _, gray = preprocess_frame(raw_img)
                ranked = benchmark_presets(detector, gray, expected)
                print("\nRank | Preset           | sf   | mn | Faces | Eyes | Err | Runtime")
                print("-" * 68)
                for rank, r in enumerate(ranked, start=1):
                    print(
                        f"#{rank:<3} | {r['preset']:<16} | {r['scaleFactor']:<4} | "
                        f"{r['minNeighbors']:<2} | {r['faces']:<5} | {r['eyes']:<4} | "
                        f"{r['error']:<3} | {r['runtime_ms']} ms"
                    )

            elif choice == "5":
                summary = summarize_csv_history()
                print("\n--- CSV Detection Log Summary ---")
                for k, v in summary.items():
                    print(f"{k}: {v}")

            elif choice == "6":
                run_webcam(detector, params)

            elif choice == "7":
                print("Exiting application. Goodbye!")
                break

            else:
                print("Invalid option. Choose 1 to 7.")

        except Exception as exc:
            print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()