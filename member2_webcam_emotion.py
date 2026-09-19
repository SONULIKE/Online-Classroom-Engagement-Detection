"""
member2_webcam_emotion.py
--------------------------
ML-Based Emotion Detection in Online Education System
Member 2 module: OpenCV real-time webcam facial-expression detection.

Pipeline:
    Webcam -> Face detection (Haar Cascade) -> Preprocessing (48x48 grayscale)
    -> Trained CNN (facial_expression_model.keras) -> Emotion + confidence
    -> Displayed on screen

This module produces facial-expression predictions (Member 2's scope) and
now also drives the engagement estimate via engagement.py (Member 3's
scope, imported here rather than reimplemented). See engagement.py for
the emotion -> engagement heuristic, confidence handling, and smoothing.

IMPORTANT LIMITATION:
The CNN classifies facial EXPRESSIONS, not a student's true internal
emotional state. A confident "happy" prediction means the model saw a
smiling-type expression - it is not proof of how the student actually
feels. Treat the confidence percentage as a model probability, not as
psychological certainty.

Run with:
    python member2_webcam_emotion.py

Press 'q' in the video window to quit.
"""

import os
import sys
import time

import cv2
import numpy as np

from engagement import EngagementEstimator  # Member 3's engagement module

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------

MODEL_PATH = "facial_expression_model.keras"
IMG_SIZE = 48

# 2. CLASS NAMES - must match the order the model was trained with.
#    This order was supplied by Member 1; it has NOT been independently
#    verified against the training notebook, since a saved .keras file
#    does not store label names. Confirm with Member 1 if predictions
#    look systematically wrong (e.g. "happy" is always shown for sad faces).
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Below this confidence, the prediction is shown as "Uncertain" instead of
# a specific emotion. 40% is a simple, generous threshold appropriate for
# a prototype demo - low enough to rarely hide a real prediction, high
# enough to flag genuinely ambiguous frames (e.g. partially turned face).
CONFIDENCE_THRESHOLD = 40.0


# --------------------------------------------------------------------------
# 3. MODEL LOADING
# --------------------------------------------------------------------------

def load_emotion_model(path):
    if not os.path.exists(path):
        print(f"ERROR: Model file not found at '{path}'.")
        print("Make sure facial_expression_model.keras is in this folder.")
        sys.exit(1)

    print("Loading emotion recognition model...")
    from tensorflow import keras  # imported here so the error above is fast

    try:
        model = keras.models.load_model(path)
    except Exception as e:
        print("ERROR: Model failed to load.")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("Model loaded successfully.")
    return model


# --------------------------------------------------------------------------
# 4. FACE DETECTION
# --------------------------------------------------------------------------

def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        print(f"ERROR: Could not load face detector from {cascade_path}")
        sys.exit(1)

    return detector


def get_largest_face(faces):
    """Select the largest detected face. This prototype demonstrates a
    single student; multi-person tracking is out of scope for Member 2."""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


# --------------------------------------------------------------------------
# 5. FACE PREPROCESSING
# --------------------------------------------------------------------------

def preprocess_face(gray_frame, box):
    """
    Crop -> grayscale (already gray) -> resize 48x48 -> float32 -> /255
    -> add channel dim -> add batch dim. Result shape: (1, 48, 48, 1).
    """
    (x, y, w, h) = box
    face = gray_frame[y:y + h, x:x + w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32")
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)   # (48, 48, 1)
    face = np.expand_dims(face, axis=0)    # (1, 48, 48, 1)
    return face


# --------------------------------------------------------------------------
# 6. PREDICTION
# --------------------------------------------------------------------------

def predict_emotion(model, processed_face):
    predictions = model.predict(processed_face, verbose=0)[0]  # shape (7,)
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index]) * 100.0
    emotion = CLASS_NAMES[class_index]
    return emotion, confidence


# --------------------------------------------------------------------------
# 7. DISPLAY HELPERS
# --------------------------------------------------------------------------

def draw_prediction(frame, box, emotion, confidence, engagement_result):
    (x, y, w, h) = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if confidence < CONFIDENCE_THRESHOLD:
        label = f"Uncertain ({confidence:.1f}%)"
        color = (0, 165, 255)  # orange
    else:
        label = f"{emotion.capitalize()} ({confidence:.1f}%)"
        color = (0, 255, 0)  # green

    cv2.putText(
        frame, label, (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
    )

    # --- Member 3: engagement estimate, drawn below the box ---
    if engagement_result is not None:
        eng_label = f"Engagement: {engagement_result['engagement']}"
        score_label = f"Score: {engagement_result['engagement_score']}"
        if not engagement_result["reliable"]:
            eng_label += " (low reliability)"

        eng_color = (0, 255, 0) if engagement_result["reliable"] else (0, 165, 255)
        cv2.putText(
            frame, eng_label, (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, eng_color, 2,
        )
        cv2.putText(
            frame, score_label, (x, y + h + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, eng_color, 2,
        )


def draw_no_face(frame):
    cv2.putText(
        frame, "No face detected", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )


# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------

def main():
    model = load_emotion_model(MODEL_PATH)
    face_detector = load_face_detector()
    engagement_estimator = EngagementEstimator()  # Member 3's smoothing/state

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Check that no other application is using the camera, and that")
        print("camera permission has been granted to this terminal/IDE.")
        sys.exit(1)

    print("Webcam opened. Press 'q' in the video window to quit.\n")

    # Optional: latest result, kept here so Member 3 knows where to hook in.
    # This is the conceptual output Member 3 will eventually consume:
    #   {"emotion": "happy", "confidence": 0.823}
    latest_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        largest_face = get_largest_face(faces)

        if largest_face is not None:
            processed = preprocess_face(gray, largest_face)
            emotion, confidence = predict_emotion(model, processed)

            # Member 3: convert the facial-expression prediction into a
            # heuristic engagement estimate (Interested / Confused / Bored).
            engagement_result = engagement_estimator.update(emotion, confidence)

            draw_prediction(frame, largest_face, emotion, confidence, engagement_result)

            # Structured result Member 4's dashboard can eventually consume.
            latest_result = engagement_result
        else:
            draw_no_face(frame)
            # Engagement state is intentionally left as-is (not reset) so it
            # doesn't flicker to nothing during a brief face-detection gap.
            latest_result = engagement_estimator.current_result()

        cv2.imshow("Emotion Detection - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released, windows closed.")


if __name__ == "__main__":
    main()
