import os
from dotenv import load_dotenv
from google import genai
from app.services.base_llm import BaseLLM

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiLLM(BaseLLM):
    
    def generate_explanation(
        self,
        disease: str,
        confidence: float,
        severity: str,
        context: str | None = None,
    ) -> str:

        if disease == "NotTomato":
            return (
                "The uploaded image does not appear to be a tomato leaf, so disease analysis "
                "was not performed. Please upload a clear tomato leaf image (single leaf, good "
                "lighting, close-up, minimal background clutter)."
            )
 
        prompt = f"""
    You are an expert agricultural plant pathologist.

    A tomato leaf image was analyzed by a CNN model.

    Prediction details:
    - Disease: {disease}
    - Model Confidence: {confidence:.2f}%
    - Estimated Severity: {severity}
    - System Context: {context or 'disease_classification'}

    Return output in this exact structure and order only:
    Causes:
    - ...

    Treatment:
    - ...

    Prevention:
    - ...

    Rules:
    - Use 1-2 short bullet points per section.
    - Total response must be under 120 words.
    - Keep language simple and farmer-friendly.
    - Do NOT provide chemical dosages.
    - Do NOT provide unsafe pesticide instructions.
    - Do not add extra headings, intro, or conclusion.
    """

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt)
            return response.text.strip()

        except Exception as e:
            return (
                "Explanation service is temporarily unavailable. "
                "Please consult an agricultural expert."
            )
