from app.services.base_llm import BaseLLM

class MockLLM(BaseLLM):
    def generate_explanation(self, disease, confidence, severity):
        return (
            f"[MOCK RESPONSE]\n\n"
            f"Disease: {disease}\n"
            f"confidence: {confidence}\n"
            f"Severity: {severity}\n\n"
            "This is a simulated explanation for development purposes.\n"
            "Treatment: Apply general fungicide.\n"
            "Prevention: Maintain dry foliage."
        )
