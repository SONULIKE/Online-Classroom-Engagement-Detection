# Member 1 — Facial Expression Recognition Using CNN

## 1. Project Overview

This module is responsible for developing and training the **facial expression recognition model** for the ML-Based Student Engagement and Emotion Detection project.

The goal of Member 1's work is to train a Convolutional Neural Network (CNN) that can recognize seven facial expressions from face images:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The trained model is saved and provided to Member 2 for integration with the OpenCV real-time webcam module.

---

## 2. Member 1 Responsibilities

Member 1 completed the following tasks:

1. Dataset preparation and organization
2. Dataset extraction in Google Colab
3. Dataset folder verification
4. Emotion-class verification
5. Image preprocessing
6. Training/validation split
7. Data augmentation
8. Dataset loading using Keras generators
9. Class-index verification
10. CNN model design
11. CNN model training
12. Training and validation performance visualization
13. Test-set evaluation
14. Classification report
15. Confusion matrix
16. Individual image prediction
17. Model saving
18. Verification of the saved model
19. Providing the trained model to Member 2

---

## 3. Dataset Structure

The dataset used for this version contains separate `train` and `test` folders.

```text
face_expression/
│
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Since there is no separate validation folder, **20% of the training data is used for validation**.

```text
Training folder
      │
      ├── 80% → Training
      │
      └── 20% → Validation

Test folder
      │
      └── Final Test Evaluation
```

The test set is kept separate from training and validation.

---

## 4. Technologies Used

- Python
- Google Colab
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

---

## 5. Image Preprocessing

The CNN expects:

```text
Image Size: 48 × 48 pixels
Color Mode: Grayscale
Channels: 1
```

Therefore, each image is converted to:

```text
(48, 48, 1)
```

Pixel values are normalized from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
pixel / 255.0
```

---

## 6. Data Augmentation

The training data uses augmentation to improve the model's ability to generalize.

The following transformations are used:

- Small rotations
- Width shifting
- Height shifting
- Zooming
- Horizontal flipping

Validation and test images are not augmented.

---

## 7. Emotion Class Mapping

Keras generated the following class indices:

```python
{
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}
```

Therefore, the CNN output mapping is:

| Index | Emotion |
|---:|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

The mapping must remain the same when Member 2 integrates the model with OpenCV.

---

## 8. CNN Architecture

The facial expression model uses a Convolutional Neural Network.

Architecture:

```text
Input
48 × 48 × 1
     ↓
Conv2D (32 filters)
     ↓
Batch Normalization
     ↓
Max Pooling
     ↓
Dropout
     ↓
Conv2D (64 filters)
     ↓
Batch Normalization
     ↓
Max Pooling
     ↓
Dropout
     ↓
Conv2D (128 filters)
     ↓
Batch Normalization
     ↓
Max Pooling
     ↓
Dropout
     ↓
Flatten
     ↓
Dense (128)
     ↓
Dropout
     ↓
Dense (7)
     ↓
Softmax
```

The final `Dense(7, activation="softmax")` layer produces probabilities for the seven emotion classes.

---

## 9. Model Compilation

The model is compiled using:

```python
optimizer = "adam"
loss = "categorical_crossentropy"
metrics = ["accuracy"]
```

---

## 10. Model Training

The model is trained using:

- Training dataset
- Validation dataset
- Up to 30 epochs
- Early stopping

Early stopping monitors validation loss and restores the best model weights.

Example:

```python
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

---

## 11. Model Evaluation

After training, the model is evaluated on the separate test dataset.

The following evaluation methods are used:

### Test Accuracy

Measures the overall percentage of correctly classified test images.

### Classification Report

Provides:

- Precision
- Recall
- F1-score
- Support

for each emotion.

### Confusion Matrix

Shows which emotions are correctly classified and which emotions are confused with one another.

---

## 12. Individual Image Testing

A random image from the test dataset is also passed through the trained model.

The system displays:

```text
Actual Emotion
Predicted Emotion
Confidence
```

Example:

```text
Actual Emotion: happy
Predicted Emotion: happy
Confidence: 82.50%
```

This provides a simple visual demonstration of individual prediction.

---

## 13. Saved Model

The final trained model is saved as:

```text
facial_expression_model.keras
```

The model can be loaded using:

```python
from tensorflow.keras.models import load_model

model = load_model("facial_expression_model.keras")
```

This saved model is the main output of Member 1's work.

---

## 14. Handoff to Member 2

Member 2 uses the saved CNN model in the OpenCV real-time detection module.

The expected pipeline is:

```text
Webcam
   ↓
OpenCV
   ↓
Face Detection
   ↓
Face Crop
   ↓
Grayscale
   ↓
Resize to 48 × 48
   ↓
Normalize /255
   ↓
CNN Model
   ↓
7 Emotion Probabilities
   ↓
Highest Probability
   ↓
Predicted Emotion
```

Member 2 must use the same preprocessing and class mapping as Member 1.

---

## 15. Handoff Information

Member 2 should receive:

### Model

```text
facial_expression_model.keras
```

### Input format

```text
48 × 48 × 1
```

### Normalization

```python
image = image / 255.0
```

### Emotion mapping

```python
class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]
```

### Prediction

```python
prediction = model.predict(face)
predicted_index = np.argmax(prediction[0])
predicted_emotion = class_names[predicted_index]
```

---

## 16. Important Project Note

The CNN predicts **facial expressions**, not a person's internal emotional state with certainty.

The later project stages use these facial-expression predictions as an input for estimating student engagement categories such as:

- Interested
- Confused
- Bored

Therefore, these engagement categories should be described as **estimated engagement states**, not direct measurements of a student's actual mental state.

---

## 17. Limitations

The model may be affected by:

- Poor lighting
- Camera angle
- Face occlusion
- Image quality
- Facial pose
- Similar-looking expressions
- Class imbalance
- Limited training data
- Differences between training images and real webcam images

Real-time webcam performance may therefore differ from the test-set performance.

---

## 18. Future Improvements

Possible improvements include:

- Using a larger and more diverse dataset
- Improving class balance
- Using a stronger CNN architecture
- Transfer learning
- Better face detection
- Real-time multi-face detection
- Better temporal analysis of video
- More reliable engagement estimation
- Integration with online classroom/video-call systems

---

## 19. Files Produced by Member 1

Recommended Member 1 folder:

```text
Member1/
│
├── Member_1_Facial_Expression_Detection.ipynb
├── facial_expression_model.keras
└── README.md
```

---

## 20. Final Output of Member 1

The primary deliverable is:

**`facial_expression_model.keras`**

This trained CNN model will be integrated by Member 2 into the OpenCV real-time facial expression detection system.

---

## 21. Project Flow

```text
Dataset
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
Train / Validation Split
   ↓
CNN Training
   ↓
Model Evaluation
   ↓
Classification Report
   ↓
Confusion Matrix
   ↓
Individual Image Testing
   ↓
Save CNN Model
   ↓
Member 2: OpenCV Integration
```

---

## 22. Conclusion

Member 1 developed the machine learning component of the project by preparing the facial-expression dataset, training a CNN model for seven facial expressions, evaluating its performance, and saving the trained model.

The saved model serves as the machine learning component that will later be connected to the OpenCV webcam system and the teacher dashboard.
