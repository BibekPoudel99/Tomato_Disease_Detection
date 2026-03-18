import os
import io
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.utils.predict import predict_disease
from app.utils.severity import calculate_severity
from app.services.gemini_llm import GeminiLLM
from app.services.mock_llm import MockLLM
from app.services.ollama_llm import OllamaLLM
from PIL import Image

load_dotenv()

app = FastAPI()
CONFIDENCE_THRESHOLD = 50

LLM_TYPE = os.getenv("LLM_TYPE", "mock").lower()

if LLM_TYPE == "ollama":
    llm = OllamaLLM()
elif LLM_TYPE == "gemini":
    llm = GeminiLLM()
else:
    llm = MockLLM()

# Define /predict BEFORE mounting static files
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )
    
     #  image dimension check
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width < 100 or img.height < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too small (min 100x100 pixels).",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image: {str(e)}",
        )

    try:
        pred = predict_disease(image_bytes)
        if isinstance(pred, dict):
            disease = pred.get("disease")
            confidence = float(pred.get("confidence"))
        else:
            disease, confidence = pred
            confidence = float(confidence)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "disease": "Uncertain",
            "confidence": round(confidence, 2),
            "severity": "Unknown",
            "explanation": "The model is not confident about the prediction. Please upload a clearer image or consult an agricultural expert.",
        }

    severity = calculate_severity(confidence)

    try:
        explanation = llm.generate_explanation(disease, confidence, severity)
    except Exception:
        explanation = "Prediction was successful, but explanation generation failed. Please try again."

    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "severity": severity,
        "explanation": explanation,
    }

# Mount static files AFTER API routes
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")