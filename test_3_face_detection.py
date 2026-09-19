"""
PHASE 4 - FACE DETECTION TEST
------------------------------
Opens the webcam and draws a bounding box around the largest detected
face using OpenCV's built-in Haar Cascade. Still no emotion prediction.

Run with:
    python test_3_face_detection.py

Press 'q' in the video window to quit.
"""

import cv2
import sys


def load_face_detector():
    # OpenCV ships pretrained Haar Cascade XML files; this locates the
    # frontal-face one inside the installed opencv-python package.
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        print(f"ERROR: Could not load face detector from {cascade_path}")
        sys.exit(1)

    return detector


def get_largest_face(faces):
    """Given a list of (x, y, w, h) boxes, return the one with the largest area.
    We only care about one primary student for this prototype."""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def main():
    face_detector = load_face_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    print("Webcam opened. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam.")
            break

        # Haar Cascade face detection works on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        largest_face = get_largest_face(faces)

        if largest_face is not None:
            (x, y, w, h) = largest_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Face detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Face Detection Test - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
