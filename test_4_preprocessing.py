"""
PHASE 5 - PREPROCESSING TEST
-----------------------------
Captures one frame from the webcam, detects the largest face, and runs
it through the exact preprocessing pipeline the CNN expects. Prints the
final shape and dtype so you can confirm it is (1, 48, 48, 1) float32
before wiring up the real model.

Run with:
    python test_4_preprocessing.py

Press 'q' in the video window to quit. Press any other key after a face
is detected to print the shape info to the console.
"""

import cv2
import numpy as np
import sys

IMG_SIZE = 48


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print(f"ERROR: Could not load face detector from {cascade_path}")
        sys.exit(1)
    return detector


def get_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def preprocess_face(gray_frame, box):
    """
    Turns a detected face region into the (1, 48, 48, 1) float32 array
    the CNN expects.

    Steps and why each is needed:
      1. Crop to just the face   - removes background the CNN wasn't trained on
      2. Already grayscale here  - matches the model's 1-channel input
      3. Resize to 48x48         - matches the model's fixed input size
      4. Convert to float32      - required numeric type for the model
      5. Divide by 255.0         - scales pixels to 0-1, matching training
      6. Add batch dimension     - Keras expects a batch axis, even for 1 image
      7. Add channel dimension   - Keras expects an explicit channels axis
    """
    (x, y, w, h) = box
    face = gray_frame[y:y + h, x:x + w]                 # 1. crop
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))         # 3. resize
    face = face.astype("float32")                         # 4. float32
    face = face / 255.0                                    # 5. normalize
    face = np.expand_dims(face, axis=-1)                   # 7. channel dim -> (48,48,1)
    face = np.expand_dims(face, axis=0)                    # 6. batch dim   -> (1,48,48,1)
    return face


def main():
    face_detector = load_face_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    print("Webcam opened. Press 'q' to quit.")
    print("When a face is detected, its preprocessed shape prints below.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        largest_face = get_largest_face(faces)

        if largest_face is not None:
            (x, y, w, h) = largest_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            processed = preprocess_face(gray, largest_face)
            print(
                f"Preprocessed face -> shape: {processed.shape}, "
                f"dtype: {processed.dtype}, "
                f"min: {processed.min():.3f}, max: {processed.max():.3f}"
            )
        else:
            cv2.putText(
                frame, "No face detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )

        cv2.imshow("Preprocessing Test - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
