class BaseLLM:
    def generate_explanation(
        self,
        disease: str,
        confidence: float,
        severity: str,
        context: str | None = None,
    ) -> str:
        raise NotImplementedError
