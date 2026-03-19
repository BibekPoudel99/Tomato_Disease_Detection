import numpy as np
import cv2
from app.model.model_loader import (
    disease_model,
    index_to_class,
    gate_model,
    gate_config,
)

IMG_SIZE = 224


def _softmax_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    safe_probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(safe_probs * np.log(safe_probs)))


def _parse_gate_output(raw_gate_output: np.ndarray) -> float:
    """
    Returns the probability that an image is a tomato leaf.
    Supports sigmoid (shape: [1, 1]) and 2-class softmax outputs (shape: [1, 2]).
    """
    values = np.array(raw_gate_output).flatten()
    if values.size == 1:
        return float(values[0])

    if values.size >= 2:
        tomato_index = int(gate_config.get("tomato_index", 1))
        if tomato_index < 0 or tomato_index >= values.size:
            tomato_index = int(np.argmax(values))
        return float(values[tomato_index])

    return 0.0

def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_disease(image_bytes):
    processed_img = preprocess_image(image_bytes)

    if gate_model is not None:
        raw_gate_output = gate_model.predict(processed_img, verbose=0)
        tomato_probability = _parse_gate_output(raw_gate_output)
        gate_threshold = float(gate_config.get("tomato_threshold", 0.6))
        if tomato_probability < gate_threshold:
            return {
                "disease": "NotTomato",
                "confidence": round((1.0 - tomato_probability) * 100, 2),
                "severity": "None",
                "reason": "gate_rejected_non_tomato",
                "tomato_probability": round(tomato_probability * 100, 2),
            }

    predictions = disease_model.predict(processed_img, verbose=0)
    probs = predictions[0]
    sorted_indices = np.argsort(probs)[::-1]
    predicted_index = int(sorted_indices[0])
    top1_conf = float(probs[sorted_indices[0]])
    top2_conf = float(probs[sorted_indices[1]]) if probs.shape[0] > 1 else 0.0
    confidence_margin = top1_conf - top2_conf
    entropy = _softmax_entropy(probs)

    fallback_conf_threshold = float(gate_config.get("fallback_confidence_threshold", 0.55))
    fallback_margin_threshold = float(gate_config.get("fallback_margin_threshold", 0.12))
    fallback_entropy_threshold = float(gate_config.get("fallback_entropy_threshold", 1.9))

    if (
        gate_model is None
        and (
            top1_conf < fallback_conf_threshold
            or confidence_margin < fallback_margin_threshold
            or entropy > fallback_entropy_threshold
        )
    ):
        return {
            "disease": "NotTomato",
            "confidence": round((1.0 - top1_conf) * 100, 2),
            "severity": "None",
            "reason": "fallback_uncertainty_rejection",
            "top1_confidence": round(top1_conf * 100, 2),
            "margin": round(confidence_margin * 100, 2),
            "entropy": round(entropy, 4),
        }

    disease_name = index_to_class[predicted_index]

    return {
        "disease": disease_name,
        "confidence": round(top1_conf * 100, 2),
        "top1_confidence": round(top1_conf * 100, 2),
        "top2_confidence": round(top2_conf * 100, 2),
        "margin": round(confidence_margin * 100, 2),
        "entropy": round(entropy, 4),
        "reason": "disease_classification",
    }
