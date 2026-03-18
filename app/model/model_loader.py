import tensorflow as tf
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "new_tomato_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "training", "class_indices.json")

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → class name
index_to_class = {v: k for k, v in class_indices.items()}
