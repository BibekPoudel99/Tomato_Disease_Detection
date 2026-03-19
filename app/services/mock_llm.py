from app.services.base_llm import BaseLLM


class MockLLM(BaseLLM):
    def generate_explanation(self, disease, confidence, severity, context=None):
        if disease == "NotTomato":
            return (
                "[MOCK RESPONSE]\n\n"
                "The uploaded image does not appear to be a tomato leaf.\n"
                f"Tomato confidence signal: {confidence:.2f}%\n"
                f"Reason: {context or 'non_tomato_detected'}\n\n"
                "Please upload a clear tomato leaf image for disease diagnosis."
            )

        return (
            f"[MOCK RESPONSE]\n\n"
            f"Disease: {disease}\n"
            f"confidence: {confidence}\n"
            f"Severity: {severity}\n\n"
            "This is a simulated explanation for development purposes.\n"
            "Treatment: Apply general fungicide.\n"
            "Prevention: Maintain dry foliage."
        )
