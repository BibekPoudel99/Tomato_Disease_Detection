import tensorflow as tf
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "new_tomato_disease_model.h5")
GATE_MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "tomato_gate_model.h5")
GATE_CONFIG_PATH = os.path.join(BASE_DIR, "trained_model", "tomato_gate_config.json")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "training", "class_indices.json")

# Load disease model once
disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)

# Gate model is optional during transition. If missing, fallback rejection logic is used.
gate_model = None
if os.path.exists(GATE_MODEL_PATH):
    gate_model = tf.keras.models.load_model(GATE_MODEL_PATH)

gate_config = {
    "tomato_threshold": 0.6,
    "tomato_index": 1,
    "fallback_confidence_threshold": 0.55,
    "fallback_margin_threshold": 0.12,
    "fallback_entropy_threshold": 1.9,
}
if os.path.exists(GATE_CONFIG_PATH):
    with open(GATE_CONFIG_PATH, "r") as f:
        loaded_config = json.load(f)
    if isinstance(loaded_config, dict):
        gate_config.update(loaded_config)

# Load class indices
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → class name
index_to_class = {v: k for k, v in class_indices.items()}
