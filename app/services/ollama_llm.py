import requests
from app.services.base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model="llama3:3b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate_explanation(self, disease: str, confidence: float, severity: str) -> str:
        prompt = f"""
        Explain {disease} in tomatoes.
        Detection confidence: {confidence}%
        Severity: {severity}
        
        Include causes, treatment, and prevention.
        Keep it simple and practical.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error generating explanation: {str(e)}"