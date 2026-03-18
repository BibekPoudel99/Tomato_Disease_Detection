import numpy as np
import cv2
from app.model.model_loader import model, index_to_class

IMG_SIZE = 224

def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(image_bytes):
    processed_img = preprocess_image(image_bytes)
    
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    disease_name = index_to_class[predicted_index]

    return {
        "disease": disease_name,
        "confidence": round(confidence * 100, 2)
    }
