"""
PHASE 2 - MODEL VERIFICATION
-----------------------------
Loads facial_expression_model.keras and reports its input shape,
output shape, and whether it loads successfully.

Run this BEFORE building the webcam app. It only needs to be run once
to confirm the model matches what member2_webcam_emotion.py expects.

Run with:
    python test_1_model_loading.py
"""

import os
import sys

MODEL_PATH = "facial_expression_model.keras"

EXPECTED_INPUT_SHAPE = (None, 48, 48, 1)
EXPECTED_OUTPUT_SHAPE = (None, 7)


def main():
    # --- Check the file exists before even importing TensorFlow ---
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Make sure facial_expression_model.keras is in this folder.")
        sys.exit(1)

    print("Loading TensorFlow/Keras (this can take a few seconds)...")
    from tensorflow import keras

    # --- Attempt to load the model ---
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print("ERROR: Model failed to load.")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("Model loaded successfully.\n")

    input_shape = model.input_shape
    output_shape = model.output_shape

    print("=== Model Info ===")
    print(f"Input shape : {input_shape}")
    print(f"Output shape: {output_shape}")
    print(f"Total params: {model.count_params():,}")
    print()

    # --- Compare against what member2_webcam_emotion.py assumes ---
    problems = []
    if input_shape != EXPECTED_INPUT_SHAPE:
        problems.append(
            f"Input shape is {input_shape}, expected {EXPECTED_INPUT_SHAPE}."
        )
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        problems.append(
            f"Output shape is {output_shape}, expected {EXPECTED_OUTPUT_SHAPE}."
        )

    if problems:
        print("DISCREPANCY DETECTED - do not proceed until this is resolved:")
        for p in problems:
            print(f"  - {p}")
        print()
        print("The preprocessing pipeline in member2_webcam_emotion.py assumes")
        print("48x48 grayscale input and 7 output classes. If the shapes above")
        print("differ, that pipeline needs to be changed before use.")
        sys.exit(1)
    else:
        print("Model shape matches expectations:")
        print("  - Input : 48x48 grayscale (1 channel)")
        print("  - Output: 7 class probabilities")
        print()
        print("You can proceed to Phase 3 (webcam test).")


if __name__ == "__main__":
    main()
