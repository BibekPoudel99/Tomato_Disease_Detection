class BaseLLM:
    def generate_explanation(self, disease: str, confidence: float, severity: str) -> str:
        raise NotImplementedError
