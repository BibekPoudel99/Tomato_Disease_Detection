import os
from dotenv import load_dotenv
from google import genai
from app.services.base_llm import BaseLLM

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiLLM(BaseLLM):
    
    def generate_explanation(self, disease: str, confidence: float, severity: str) -> str:
 
        prompt = f"""
    You are an expert agricultural plant pathologist.

    A tomato leaf image was analyzed by a CNN model.

    Prediction details:
    - Disease: {disease}
    - Model Confidence: {confidence:.2f}%
    - Estimated Severity: {severity}

    Provide a structured but easy-to-understand explanation including:

    1. What this disease is
    2. Why it occurs
    3. Symptoms farmers should observe
    4. Recommended treatment methods (safe and general advice only)
    5. Prevention measures
    6. Practical farmer-friendly tips

    Important:
    - Do NOT provide chemical dosage quantities.
    - Do NOT provide unsafe pesticide instructions.
    - Keep language simple and clear.
    - Keep response under 400 words.
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
