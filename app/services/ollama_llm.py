import requests
import os
from app.services.base_llm import BaseLLM
from dotenv import load_dotenv

load_dotenv()


class OllamaLLM(BaseLLM):
    def __init__(self, model=None, base_url=None):
        env_model = os.getenv("OLLAMA_MODEL")
        env_base_url = os.getenv("OLLAMA_BASE_URL")
        env_timeout = os.getenv("OLLAMA_TIMEOUT", "120")

        try:
            request_timeout = int(env_timeout)
        except ValueError:
            request_timeout = 120

        if request_timeout <= 0:
            request_timeout = 120

        self.model = model or env_model or "llama3:3b"
        self.base_url = base_url or env_base_url or "http://localhost:11434"
        self.timeout = request_timeout

    def generate_explanation(
        self,
        disease: str,
        confidence: float,
        severity: str,
        context: str | None = None,
    ) -> str:
        if disease == "NotTomato":
            return (
                "The image appears to be non-tomato. Disease diagnosis is skipped. "
                "Please upload a clear tomato leaf image for analysis."
            )

        prompt = f"""
        Explain {disease} in tomatoes.
        Detection confidence: {confidence}%
        Severity: {severity}
        System context: {context or 'disease_classification'}

        Return output using this exact format only:
        Causes:
        - ...

        Treatment:
        - ...

        Prevention:
        - ...

        Rules:
        - Use 1-2 short bullet points per section.
        - Keep total output under 120 words.
        - Keep language simple and practical.
        - Do not include chemical dosage instructions.
        - Do not include any extra heading or conclusion.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error generating explanation: {str(e)}"