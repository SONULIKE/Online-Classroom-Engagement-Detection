"""
app.py
------
Member 4 module: Flask backend + teacher dashboard integration.

This file does NOT reimplement webcam access, face detection,
preprocessing, CNN inference, or engagement estimation - it imports
those directly from member2_webcam_emotion.py and engagement.py
(Members 2 and 3's existing, unmodified code) and runs the same
per-frame logic inside a background thread instead of a cv2.imshow()
loop. The browser dashboard receives only small JSON results, never
raw camera frames (see PRIVACY note below).

Run with:
    python app.py

Then open:
    http://127.0.0.1:5000

Click "Start Session" to begin the webcam pipeline, "Stop Session" to
release the camera. The dashboard polls this server for updates - it
does not talk to the webcam directly.

DEMO MODE
---------
If the live webcam/model can't be relied on for a presentation, set
DEMO_MODE = True below (or set the environment variable
ML_DEMO_MODE=1 before running). This feeds the SAME EngagementEstimator
a stream of simulated (emotion, confidence) pairs instead of real CNN
predictions, so the rest of the pipeline (smoothing, scoring, charts)
is exercised identically. The dashboard clearly labels this as
"DEMO MODE - simulated data" - it is never presented as a real
prediction.

PRIVACY
-------
This is a local prototype. No webcam frames or facial images are sent
anywhere - not to the browser, not to any external service. Only the
small JSON prediction result (emotion name, confidence number,
engagement label, score) crosses from the Python backend to the page.
"""

import os
import random
import threading
import time
from collections import deque

import cv2
from flask import Flask, jsonify, render_template, request

from engagement import EngagementEstimator, EMOTION_TO_ENGAGEMENT
from member2_webcam_emotion import (
    CLASS_NAMES,
    MODEL_PATH,
    get_largest_face,
    load_emotion_model,
    load_face_detector,
    predict_emotion,
    preprocess_face,
)

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# Simulated data instead of the real webcam/CNN - for demo-day safety net
# only. See module docstring. Environment variable overrides the constant
# so you can flip it without editing code: ML_DEMO_MODE=1 python app.py
DEMO_MODE = os.environ.get("ML_DEMO_MODE", "0") == "1"

# Show the classic OpenCV preview window locally, in addition to the
# dashboard. Off by default: the dashboard is the point now, and a second
# GUI window running from a background thread is unreliable on some
# platforms (notably macOS, which wants GUI calls on the main thread).
SHOW_LOCAL_PREVIEW = False

STUDENT_ID = "Student 1"  # single-student prototype (see README for how
                            # this would extend to multiple students)

HISTORY_MAXLEN = 300          # cap on stored session samples
HISTORY_SAMPLE_INTERVAL = 1.0  # seconds between samples added to history
STALE_SECONDS = 5.0            # no update in this long -> treat as stale

# --------------------------------------------------------------------------
# SHARED STATE (written by the background thread, read by Flask routes)
# --------------------------------------------------------------------------

_lock = threading.Lock()
_history = deque(maxlen=HISTORY_MAXLEN)

_state = {
    "status": "stopped",       # stopped | starting | running | error
    "face_detected": False,
    "emotion": None,
    "emotion_confidence": None,
    "engagement": None,
    "engagement_score": None,
    "reliable": None,
    "error": None,
    "last_update": None,       # epoch seconds
    "session_start": None,     # epoch seconds
    "frames_processed": 0,
}

_worker_thread = None
_stop_event = threading.Event()

app = Flask(__name__)


# --------------------------------------------------------------------------
# BACKGROUND WORKER - REAL MODE
# --------------------------------------------------------------------------

def _real_camera_worker(stop_event):
    """Runs the same pipeline as member2_webcam_emotion.main(), but writes
    results into the shared state instead of calling cv2.imshow()."""
    try:
        model = load_emotion_model(MODEL_PATH)
        face_detector = load_face_detector()
    except SystemExit:
        # load_emotion_model / load_face_detector call sys.exit() on
        # failure in the original script (fine for a standalone CLI tool);
        # here we convert that into an error state for the dashboard
        # instead of silently killing the thread.
        with _lock:
            _state["status"] = "error"
            _state["error"] = "Model or face detector failed to load. Check the terminal log."
        return
    except Exception as e:
        with _lock:
            _state["status"] = "error"
            _state["error"] = f"{type(e).__name__}: {e}"
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with _lock:
            _state["status"] = "error"
            _state["error"] = "Could not open webcam (in use elsewhere, or no camera permission)."
        return

    estimator = EngagementEstimator()
    last_history_time = 0.0

    with _lock:
        _state["status"] = "running"
        _state["error"] = None
        _state["session_start"] = time.time()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                with _lock:
                    _state["status"] = "error"
                    _state["error"] = "Failed to read frame from webcam."
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            largest_face = get_largest_face(faces)

            now = time.time()

            if largest_face is not None:
                processed = preprocess_face(gray, largest_face)
                emotion, confidence = predict_emotion(model, processed)
                result = estimator.update(emotion, confidence)

                with _lock:
                    _state["face_detected"] = True
                    _state["emotion"] = result["emotion"]
                    _state["emotion_confidence"] = result["emotion_confidence"]
                    _state["engagement"] = result["engagement"]
                    _state["engagement_score"] = result["engagement_score"]
                    _state["reliable"] = result["reliable"]
                    _state["last_update"] = now
                    _state["frames_processed"] += 1

                if now - last_history_time >= HISTORY_SAMPLE_INTERVAL:
                    _history.append({"timestamp": now, **result})
                    last_history_time = now
            else:
                with _lock:
                    _state["face_detected"] = False
                    _state["last_update"] = now
                    _state["frames_processed"] += 1
                    # Engagement fields intentionally left as-is here too
                    # (engagement.py already holds state through brief
                    # no-face gaps) - the dashboard uses face_detected to
                    # decide what to show, not a wipe of these fields.

            if SHOW_LOCAL_PREVIEW:
                cv2.imshow("Emotion Detection (local preview)", frame)
                cv2.waitKey(1)
    finally:
        cap.release()
        if SHOW_LOCAL_PREVIEW:
            cv2.destroyAllWindows()
        with _lock:
            _state["status"] = "stopped"


# --------------------------------------------------------------------------
# BACKGROUND WORKER - DEMO MODE
# --------------------------------------------------------------------------

def _demo_worker(stop_event):
    """Feeds the real EngagementEstimator a plausible simulated stream of
    (emotion, confidence) pairs, so smoothing/scoring/charts behave exactly
    like real mode. Clearly flagged as demo via the 'demo' state field."""
    estimator = EngagementEstimator()
    last_history_time = 0.0

    with _lock:
        _state["status"] = "running"
        _state["error"] = None
        _state["session_start"] = time.time()

    # Weighted so "happy"/"neutral" dominate, like a plausible classroom feed
    weights = {
        "happy": 0.28, "neutral": 0.28, "surprise": 0.10, "sad": 0.12,
        "fear": 0.08, "angry": 0.08, "disgust": 0.06,
    }
    current_emotion = "neutral"

    try:
        while not stop_event.is_set():
            now = time.time()

            # Occasionally simulate a brief no-face gap, to exercise that
            # path in the demo too.
            if random.random() < 0.04:
                with _lock:
                    _state["face_detected"] = False
                    _state["last_update"] = now
                    _state["frames_processed"] += 1
                time.sleep(0.3)
                continue

            # Mostly keep the same expression frame-to-frame (a real face
            # doesn't teleport between expressions), occasionally switch.
            if random.random() < 0.15:
                current_emotion = random.choices(
                    list(weights.keys()), weights=list(weights.values())
                )[0]
            confidence = random.uniform(45, 95)

            result = estimator.update(current_emotion, confidence)

            with _lock:
                _state["face_detected"] = True
                _state["emotion"] = result["emotion"]
                _state["emotion_confidence"] = result["emotion_confidence"]
                _state["engagement"] = result["engagement"]
                _state["engagement_score"] = result["engagement_score"]
                _state["reliable"] = result["reliable"]
                _state["last_update"] = now
                _state["frames_processed"] += 1

            if now - last_history_time >= HISTORY_SAMPLE_INTERVAL:
                _history.append({"timestamp": now, **result})
                last_history_time = now

            time.sleep(0.2)  # ~5 simulated "frames" per second
    finally:
        with _lock:
            _state["status"] = "stopped"


# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        student_id=STUDENT_ID,
        demo_mode=DEMO_MODE,
        class_names=CLASS_NAMES,
        engagement_categories=sorted(set(EMOTION_TO_ENGAGEMENT.values())),
    )


@app.route("/api/current")
def api_current():
    with _lock:
        state = dict(_state)

    now = time.time()
    stale = (
        state["status"] == "running"
        and state["last_update"] is not None
        and (now - state["last_update"]) > STALE_SECONDS
    )
    duration = (
        (now - state["session_start"]) if state["session_start"] else 0.0
    )

    return jsonify({
        "student_id": STUDENT_ID,
        "demo_mode": DEMO_MODE,
        "status": state["status"],
        "stale": stale,
        "face_detected": state["face_detected"],
        "emotion": state["emotion"],
        "emotion_confidence": state["emotion_confidence"],
        "engagement": state["engagement"],
        "engagement_score": state["engagement_score"],
        "reliable": state["reliable"],
        "error": state["error"],
        "session": {
            "duration_seconds": round(duration, 1),
            "frames_processed": state["frames_processed"],
            "samples_recorded": len(_history),
        },
    })


@app.route("/api/history")
def api_history():
    with _lock:
        history_copy = list(_history)
    return jsonify({"history": history_copy, "count": len(history_copy)})


@app.route("/api/start", methods=["POST"])
def api_start():
    global _worker_thread, _stop_event

    with _lock:
        already_running = _worker_thread is not None and _worker_thread.is_alive()

    if already_running:
        return jsonify({"ok": False, "message": "Session already running."})

    # Reset session state/history for a fresh run
    with _lock:
        _history.clear()
        _state.update({
            "status": "starting",
            "face_detected": False,
            "emotion": None,
            "emotion_confidence": None,
            "engagement": None,
            "engagement_score": None,
            "reliable": None,
            "error": None,
            "last_update": None,
            "session_start": None,
            "frames_processed": 0,
        })

    _stop_event = threading.Event()
    target = _demo_worker if DEMO_MODE else _real_camera_worker
    _worker_thread = threading.Thread(target=target, args=(_stop_event,), daemon=True)
    _worker_thread.start()

    return jsonify({"ok": True, "message": "Session starting.", "demo_mode": DEMO_MODE})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _worker_thread

    with _lock:
        running = _worker_thread is not None and _worker_thread.is_alive()

    if not running:
        return jsonify({"ok": False, "message": "No session is running."})

    _stop_event.set()
    _worker_thread.join(timeout=5)

    return jsonify({"ok": True, "message": "Session stopped."})


if __name__ == "__main__":
    if DEMO_MODE:
        print("Starting in DEMO MODE (simulated data, no real webcam/CNN).")
    else:
        print("Starting in REAL MODE (live webcam + CNN).")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
