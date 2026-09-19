"""
PHASE 3 - WEBCAM TEST
----------------------
Bare-minimum check that OpenCV can open your webcam and show live video.
No face detection or emotion prediction yet - just proves the camera works.

Run with:
    python test_2_webcam.py

Press 'q' in the video window to quit.
"""

import cv2
import sys


def main():
    # --- Open the default webcam (index 0) ---
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Check that:")
        print("  - Your webcam is connected and not used by another app")
        print("  - You granted camera permission to this terminal/IDE")
        print("  - Try changing VideoCapture(0) to VideoCapture(1) if you")
        print("    have multiple cameras")
        sys.exit(1)

    print("Webcam opened. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to read frame from webcam.")
            break

        cv2.imshow("Webcam Test - press q to quit", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released, windows closed.")


if __name__ == "__main__":
    main()
