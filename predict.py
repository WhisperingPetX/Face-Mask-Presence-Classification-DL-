import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "best_mask_model.h5"

IMAGE_PATH = "test_image.jpg"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

model = load_model(MODEL_PATH)

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("Unable to read image. Check the image file.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

prediction = float(model.predict(img, verbose=0)[0][0])

if prediction < 0.5:
    label = "with_mask"
    confidence = 1 - prediction
else:
    label = "without_mask"
    confidence = prediction

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2%}")
