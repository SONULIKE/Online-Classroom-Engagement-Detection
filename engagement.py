"""
engagement.py
--------------
Member 3 module: converts Member 2's facial-expression prediction into a
project-defined ENGAGEMENT ESTIMATE (Interested / Confused / Bored).

IMPORTANT SCIENTIFIC LIMITATION
--------------------------------
This is a transparent, rule-based heuristic - NOT a scientifically
validated measure of student engagement. We do not have a labeled
engagement dataset, so this module does not use machine learning; it is
a simple lookup table plus a confidence check and a smoothing window.

    Facial expression -> heuristic engagement estimate

is what this module does. It is NOT:

    Facial expression -> actual mental state

Treat "engagement_score" the same way: it is a project-defined number
for the dashboard, not a scientifically calibrated engagement metric.

HOW TO USE
----------
    from engagement import EngagementEstimator

    estimator = EngagementEstimator()

    # once per frame, after Member 2 produces (emotion, confidence):
    result = estimator.update(emotion, confidence)
    # result = {
    #     "emotion": "happy",
    #     "emotion_confidence": 0.823,
    #     "engagement": "Interested",
    #     "engagement_score": 85,
    #     "reliable": True,
    # }

When no face is detected, simply don't call update() for that frame -
the estimator keeps showing its last known state instead of flickering
to nothing.
"""

from collections import deque, Counter

# --------------------------------------------------------------------------
# 1. ENGAGEMENT MAPPING
# --------------------------------------------------------------------------
# See README / chat for the full reasoning table. Summary:
#   Interested = happy, surprise      (positive affect / active attention)
#   Confused   = fear, disgust        (apprehensive / averse reaction)
#   Bored      = neutral, sad, angry  (low-arousal / disengaged affect)
#
# This mapping is a design choice, not a proven fact - it is easy to
# defend in a viva ("we grouped the 7 classes into 3 buckets by arousal
# and affect valence") but should always be presented as a heuristic.

EMOTION_TO_ENGAGEMENT = {
    "happy": "Interested",
    "surprise": "Interested",
    "fear": "Confused",
    "disgust": "Confused",
    "neutral": "Bored",
    "sad": "Bored",
    "angry": "Bored",
}

# --------------------------------------------------------------------------
# 2. ENGAGEMENT SCORE (project-defined, NOT scientifically validated)
# --------------------------------------------------------------------------
# One fixed score per bucket. Deliberately simple - a single flat number
# per category is easy to explain and easy to defend as "not pretending
# to be more precise than it is."

ENGAGEMENT_BASE_SCORE = {
    "Interested": 85,
    "Confused": 50,
    "Bored": 15,
}

# --------------------------------------------------------------------------
# 3. CONFIDENCE HANDLING
# --------------------------------------------------------------------------
# Reuses the exact same threshold Member 2's code already uses to decide
# whether to show "Uncertain" on screen (CONFIDENCE_THRESHOLD = 40 in
# member2_webcam_emotion.py), so the two modules stay consistent.

LOW_CONFIDENCE_THRESHOLD = 40.0   # below this: don't trust this frame at all
HIGH_CONFIDENCE_THRESHOLD = 60.0  # at/above this: treat as fully reliable

# --------------------------------------------------------------------------
# 4. TEMPORAL SMOOTHING
# --------------------------------------------------------------------------
# A webcam prediction can flip every frame (Happy -> Neutral -> Happy ->
# Sad -> Happy). Instead of displaying that raw flicker, we keep a short
# rolling history of the last HISTORY_SIZE reliable-enough engagement
# labels and display whichever one is most common in that window. This
# is a simple majority vote, not a moving average - easy to explain:
# "we show whatever the last N frames mostly agreed on."

HISTORY_SIZE = 15  # roughly half a second to a second of frames, depending on FPS


class EngagementEstimator:
    """
    Stateful engagement estimator. Create ONE instance per video stream
    (one per student, in a future multi-student version - see note in
    README) and call update() once per frame that has a face.
    """

    def __init__(self, history_size=HISTORY_SIZE):
        self._history = deque(maxlen=history_size)
        self._last_result = None  # holds the last returned dict

    def update(self, emotion, confidence):
        """
        emotion: one of the 7 class names from Member 2's CLASS_NAMES
        confidence: 0-100 (percentage), matching Member 2's predict_emotion()

        Returns a structured dict - see module docstring for its shape.
        """
        if emotion not in EMOTION_TO_ENGAGEMENT:
            # Defensive check: an unexpected label should never silently
            # produce a wrong engagement bucket.
            raise ValueError(
                f"Unknown emotion label '{emotion}'. Expected one of: "
                f"{list(EMOTION_TO_ENGAGEMENT.keys())}"
            )

        reliable = confidence >= HIGH_CONFIDENCE_THRESHOLD

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            # Too uncertain to trust this frame's emotion for engagement.
            # Hold the previous engagement state instead of reacting to
            # what might be a bad guess. If we have no previous state yet
            # (very first frames), fall back to this frame's raw mapping
            # so the dashboard isn't left with nothing to show.
            if self._last_result is not None:
                held = dict(self._last_result)
                held["emotion"] = emotion
                held["emotion_confidence"] = round(confidence / 100.0, 3)
                held["reliable"] = False
                self._last_result = held
                return held
            # else: fall through and compute normally for the first frame

        raw_engagement = EMOTION_TO_ENGAGEMENT[emotion]
        self._history.append(raw_engagement)

        # Majority vote over the recent history smooths frame-to-frame jitter.
        smoothed_engagement = Counter(self._history).most_common(1)[0][0]
        score = ENGAGEMENT_BASE_SCORE[smoothed_engagement]

        result = {
            "emotion": emotion,
            "emotion_confidence": round(confidence / 100.0, 3),
            "engagement": smoothed_engagement,
            "engagement_score": score,
            "reliable": reliable,
        }
        self._last_result = result
        return result

    def current_result(self):
        """Return the last computed result without updating (e.g. for a
        frame where no face was detected, so the display doesn't flicker)."""
        return self._last_result

    def reset(self):
        """Clear history - useful if a new student session begins."""
        self._history.clear()
        self._last_result = None


# --------------------------------------------------------------------------
# Convenience function form, if a function-style API is preferred over
# the class (e.g. for a quick script or a notebook cell).
# --------------------------------------------------------------------------
_default_estimator = EngagementEstimator()


def get_engagement_result(emotion, confidence):
    """Function-style wrapper around a shared default EngagementEstimator.
    For multi-student use, create separate EngagementEstimator() instances
    instead of using this shared one."""
    return _default_estimator.update(emotion, confidence)
