# Member 2 — OpenCV / Real-Time Webcam Emotion Detection

Part of **ML-Based Emotion Detection in Online Education System**.

This folder covers Member 2's scope only: taking a webcam feed, finding a
face, and running Member 1's trained CNN on it to get a facial-expression
prediction and confidence score. It does **not** cover engagement
estimation (Member 3) or the teacher dashboard (Member 4).

## Pipeline

```
Webcam frame → grayscale → Haar Cascade face detection → crop face
→ resize to 48x48 → normalize (0-1) → add batch/channel dims
→ CNN prediction → argmax → emotion + confidence → drawn on screen
```

## Folder contents

| File | Purpose |
|---|---|
| `facial_expression_model.keras` | Member 1's trained CNN (not modified) |
| `requirements.txt` | Python dependencies |
| `test_1_model_loading.py` | Phase 2 — verifies the model loads and has the expected input/output shape |
| `test_2_webcam.py` | Phase 3 — confirms the webcam opens and streams video |
| `test_3_face_detection.py` | Phase 4 — confirms Haar Cascade face detection works |
| `test_4_preprocessing.py` | Phase 5 — confirms the preprocessing pipeline produces a `(1, 48, 48, 1)` float32 array |
| `member2_webcam_emotion.py` | **Final program** — combines everything into the real-time emotion detector |

## Model verification (already run)

The supplied `facial_expression_model.keras` was inspected before writing
any of this code:

- Loads successfully with TensorFlow 2.21 / Keras 3.15
- Input shape: `(None, 48, 48, 1)` — matches the expected grayscale 48×48 input
- Output shape: `(None, 7)` — matches the expected 7 emotion classes
- No discrepancy from the stated spec, so no preprocessing changes were needed

One caveat: a saved `.keras` file does not store class label names, so the
class order below is **taken as given from Member 1**, not independently
verified. If predictions look systematically off (e.g. every happy face is
labeled "sad"), double-check the label order used during training.

```python
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```

## Setup

1. Check your Python version:
   ```
   python3 --version
   ```
   Any Python 3.9–3.12 should work with current TensorFlow builds.

2. (Recommended) create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Confirm installs:
   ```
   python -c "import cv2, numpy, tensorflow; print(cv2.__version__, numpy.__version__, tensorflow.__version__)"
   ```

## Running, in order

Run these in order the first time, so any problem is caught at the
smallest possible step rather than inside the full app.

```
python test_1_model_loading.py     # confirms the model loads correctly
python test_2_webcam.py            # confirms the webcam works
python test_3_face_detection.py    # confirms face detection works
python test_4_preprocessing.py     # confirms the image pipeline shape is correct
python member2_webcam_emotion.py   # the real thing
```

In each of the webcam scripts, press **q** in the video window to quit.

## Expected output

- A window opens showing your webcam feed.
- A green box is drawn around your face.
- Above the box: the predicted emotion and confidence, e.g. `Happy (82.3%)`.
- If confidence is below 40%, it shows `Uncertain (31.2%)` instead of a
  specific emotion — this threshold exists because the CNN is not perfectly
  accurate, and a low-confidence guess is more honestly reported as
  "uncertain" than as a false-sounding definite emotion.
- If no face is visible, it shows `No face detected` and the CNN is not
  called at all (no prediction is made on an empty/invalid image).

## Testing checklist

| # | Scenario | Expected behavior |
|---|---|---|
| 1 | Face directly in front of camera | Bounding box + emotion + confidence shown |
| 2 | No face in frame | "No face detected", no CNN call |
| 3 | Move face left/right | Box tracks the face; prediction updates each frame |
| 4 | Different facial expressions | Predicted label changes accordingly (not always accurately — it's a prototype model) |
| 5 | Different lighting | Detection may degrade in very low light; Haar Cascade is lighting-sensitive |
| 6 | Multiple faces in frame | Only the largest face is boxed and predicted (by design, for this prototype) |
| 7 | Webcam unavailable/in use elsewhere | Clear error message, program exits instead of crashing |
| 8 | Model file missing/renamed | Clear error message naming the missing file, program exits |
| 9 | Low-confidence prediction | Shown as "Uncertain (X%)" instead of a specific emotion |

## Testing checklist — engagement (Member 3)

| # | Scenario | Expected behavior |
|---|---|---|
| 1 | Happy face | Engagement: Interested, score 85, reliable if confidence ≥60% |
| 2 | Sad face | Engagement: Bored, score 15 |
| 3 | Neutral face | Engagement: Bored, score 15 |
| 4 | Angry face | Engagement: Bored, score 15 |
| 5 | Surprise face | Engagement: Interested, score 85 |
| 6 | Fear face | Engagement: Confused, score 50 |
| 7 | Disgust face | Engagement: Confused, score 50 |
| 8 | No face | Engagement text unchanged from last known state (no flicker) |
| 9 | Multiple faces | Only the largest face's expression drives engagement (existing Member 2 behavior) |
| 10 | Low-confidence prediction (<40%) | On-screen shows "Uncertain"; engagement holds previous state, `reliable: False` |
| 11 | Rapid expression changes | Displayed engagement changes only when the new state wins majority vote over the last 15 frames — no frame-to-frame flicker |

## Troubleshooting

- **`ERROR: Could not open webcam.`**
  Close any other app using the camera (Zoom, Teams, browser tabs). Try
  `cv2.VideoCapture(1)` instead of `0` if you have more than one camera.

- **`ERROR: Model file not found`**
  Confirm `facial_expression_model.keras` is in the same folder as the
  script, or update `MODEL_PATH` at the top of the script.

- **Model loads but predictions look random/wrong**
  Re-run `test_1_model_loading.py` and confirm the shapes match
  `(None, 48, 48, 1)` → `(None, 7)`. If they don't, stop and check with
  Member 1 rather than changing the preprocessing to compensate.

- **Face detection is flaky / misses faces**
  Haar Cascade is a lightweight, prototype-grade detector — it's sensitive
  to lighting and angle. This is expected and acceptable for this stage;
  a more robust detector is future work, not part of the current scope.

- **Window doesn't close / program hangs on quit**
  Make sure the video window (not the terminal) is focused when pressing
  `q`.

## Important scientific limitation

This CNN predicts **facial expressions** (angry, disgust, fear, happy,
neutral, sad, surprise) — it does **not** measure a student's actual
internal emotional state. A "happy" prediction means the model detected a
smiling-type expression pattern, not that the student is provably feeling
happy. Confidence percentages are model probabilities, not psychological
certainty. This limitation should be stated explicitly in the project
report.

## Member 3 — Engagement estimation

`engagement.py` converts Member 2's facial-expression prediction into a
project-defined engagement estimate: **Interested / Confused / Bored**.
It's imported into `member2_webcam_emotion.py` and called once per frame
that has a detected face — no separate script to run.

**What it does:** `Facial expression → heuristic engagement estimate`.
**What it does NOT do:** claim to measure a student's actual mental
state. There's no labeled engagement dataset for this project, so this
is a transparent rule-based lookup, not a trained model.

**The heuristic mapping:**

| Expression | Engagement | Reason |
|---|---|---|
| happy, surprise | Interested | Positive affect / active attention |
| fear, disgust | Confused | Apprehensive or averse reaction |
| neutral, sad, angry | Bored | Low-arousal / disengaged affect |

**Confidence handling** (reuses Member 2's existing `CONFIDENCE_THRESHOLD = 40`):
- ≥60% confidence → engagement is marked reliable
- 40–59% → still mapped, but flagged `reliable: False`
- <40% → this frame's guess is ignored; the previous engagement state is
  held over instead of reacting to a shaky prediction

**Temporal smoothing:** the last 15 engagement labels are kept in a
rolling window; the displayed engagement is whichever label is most
common in that window (majority vote). This stops the on-screen label
from flickering frame-to-frame when a face briefly registers as, say,
neutral in the middle of a run of happy frames.

**Engagement score:** a fixed, simple number per bucket
(`Interested=85, Confused=50, Bored=15`) — deliberately not a formula
tied to confidence, so it's easy to explain as "a project-defined
number for the dashboard" rather than something that looks scientifically
precise.

**Limitations to state in the report/viva:**
- The 7-class → 3-bucket mapping is a design choice, not a validated model
- No ground-truth engagement labels exist to check this against
- One student, one face at a time (see multi-student note below)
- Facial expression ≠ true internal emotional or cognitive state

**Extending to multiple students (future work, not built now):** give each
student's video stream its own `EngagementEstimator()` instance — the
class already keeps all of its state (history, last result) internally
per instance, so this requires no changes to `engagement.py` itself,
just one estimator object per tracked face instead of one shared one.

## Member 4 — Teacher dashboard

`app.py` (Flask) + `templates/dashboard.html` + `static/style.css` +
`static/script.js` add a browser dashboard on top of the existing
Member 2/3 pipeline. **Nothing in `member2_webcam_emotion.py` or
`engagement.py` was changed** — `app.py` imports their functions
directly and runs the same per-frame logic in a background thread
instead of `cv2.imshow()`.

### Architecture

```
Webcam → OpenCV → Face Detection → CNN → Emotion+Confidence
                                              ↓
                        EngagementEstimator (engagement.py, unchanged)
                                              ↓
              Background thread → lock-protected shared state + history
                                              ↓
        Flask (app.py): GET /, /api/current, /api/history, POST /api/start, /api/stop
                                              ↓
              Browser polls every 1s (current) / 4s (history) → updates
              summary cards + 3 Chart.js charts
```

Only small JSON results cross into the browser — never raw webcam
frames. Nothing is uploaded externally; everything runs on localhost.

### Running the dashboard

```
python app.py
```

Then open **http://127.0.0.1:5000**. Click **Start Session** to begin
the webcam pipeline, **Stop Session** to release the camera. The
dashboard polls the server — it never talks to the webcam directly.

**Why a Start/Stop button instead of auto-starting the camera:**
loading TensorFlow and opening the webcam takes a few seconds and can
fail (camera busy, missing model). Blocking Flask's own startup on
that would make the dashboard itself unreliable to open. Start/Stop
lets you confirm the page is alive first, then start the pipeline
explicitly, and stop it cleanly without killing the server.

### API

| Route | Method | Returns |
|---|---|---|
| `/` | GET | The dashboard page |
| `/api/current` | GET | Current emotion, confidence, engagement, score, session stats, `stale`/`status` flags |
| `/api/history` | GET | Up to the last 300 recorded samples (one per second), used to build all three charts |
| `/api/start` | POST | Starts the background webcam/demo thread |
| `/api/stop` | POST | Stops it and releases the camera |

### No-face / stale-data handling

- Face briefly missing: `face_detected: false` for that frame; the
  dashboard shows "No face detected" without wiping the last known
  engagement number (matches `engagement.py`'s own hold-over behavior).
- No update at all for 5+ seconds while `status` is `"running"`
  (thread stalled or camera frozen): the API marks `stale: true` and
  the dashboard switches to "Waiting for student..." rather than
  silently continuing to show old numbers as if they were live.
- Session not started / stopped: dashboard shows "Waiting for
  student..." and disables anything implying a live reading.

### Demo mode

Set `DEMO_MODE = True` in `app.py`, or run:

```
ML_DEMO_MODE=1 python app.py
```

This feeds the **same, real** `EngagementEstimator` a simulated stream
of (emotion, confidence) pairs — so smoothing, scoring, and every chart
work exactly as in real mode — instead of reading the actual webcam/CNN.
The dashboard shows a clearly labeled purple "DEMO MODE — simulated
data" banner whenever this is active; every API response also includes
`"demo_mode": true` so the frontend can never present simulated data as
a real prediction. Use this only as a presentation backup, never as the
primary system.

### Session data shape (what Member 4's charts, or any future consumer, read)

```json
{
    "timestamp": 1737500000.12,
    "emotion": "happy",
    "emotion_confidence": 0.823,
    "engagement": "Interested",
    "engagement_score": 85,
    "reliable": true
}
```

The standalone `member2_webcam_emotion.py` script (run without `app.py`)
still exposes this same shape via its own `latest_result` variable, for
anyone testing the CNN/engagement pipeline without the dashboard running.

## Full-system test procedure (Member 4)

**Terminal:**
```
python app.py
```

**Browser:** open `http://127.0.0.1:5000`

1. Confirm the page loads and shows "Idle" / a disabled Stop button.
2. Click **Start Session**.
3. Sit in front of the webcam — confirm "Current Emotion" updates within ~1s.
4. Confirm "Emotion Confidence" and "Estimated Engagement" update alongside it.
5. Check the engagement badge color changes appropriately (green/orange/red).
6. Change your facial expression — confirm the dashboard changes within a few seconds (not instantly, due to the 15-frame smoothing window — that's expected).
7. Wait ~15–20 seconds — confirm the Emotion Distribution and Engagement Summary charts start showing non-zero bars/slices.
8. Confirm the "Engagement Score Over Time" line chart is plotting points.
9. Move out of frame — confirm "No face detected" appears without the engagement number vanishing entirely.
10. Click **Stop Session** — confirm the status changes and the webcam light turns off.
11. Restart with `ML_DEMO_MODE=1 python app.py` — confirm the purple demo banner appears and data still flows without a webcam.

## Final demo checklist

- [ ] Model loads (`test_1_model_loading.py` passes)
- [ ] Webcam opens (`test_2_webcam.py` passes)
- [ ] Face detected (`test_3_face_detection.py` passes)
- [ ] Emotion predicted, confidence displayed (`member2_webcam_emotion.py` standalone run)
- [ ] Engagement estimated (badge + score visible)
- [ ] Dashboard opens at `http://127.0.0.1:5000`
- [ ] Dashboard receives real data after Start Session
- [ ] Current emotion/engagement update automatically
- [ ] Emotion distribution chart updates
- [ ] Engagement distribution chart updates
- [ ] Engagement-over-time chart updates
- [ ] No-face handling works without crashing or freezing
- [ ] Stale-data timeout shows "Waiting for student..." if the pipeline stalls
- [ ] Demo mode works and is clearly labeled
- [ ] Start/Stop buttons both function correctly
- [ ] README is complete

## Limitations (full list, for the report)

1. Facial expressions are not direct measurements of internal emotions.
2. Engagement is a heuristic estimate, not a validated measurement.
3. The CNN's accuracy is limited by its training data and architecture.
4. Lighting conditions affect face/expression detection quality.
5. Camera angle affects detection quality.
6. Occlusion (masks, hands, hair) can block detection entirely.
7. Individual differences in how people express emotion aren't modeled.
8. The current prototype supports one student at a time.
9. The engagement mapping is a hand-designed rule table, not trained on any labeled engagement dataset.
10. This system should not be used for high-stakes decisions about any individual student.

## Future improvements (out of current scope)

- Multi-student tracking (one `EngagementEstimator` per tracked face)
- A learned engagement model, if a labeled engagement dataset becomes available
- Persisting session history to a file/database for later analysis
- Authenticated, multi-class teacher view

## Privacy

This is a local, offline prototype. No webcam frames or facial images
leave the machine — the browser only ever receives small JSON
prediction results (an emotion label, a confidence number, an
engagement label, a score). No cloud facial-recognition APIs are used
anywhere in this project.
