def calculate_severity(disease: str, confidence: float) -> str:
    """
    Determines severity based on disease label and prediction confidence.
    This remains a proxy and can later be replaced by lesion segmentation.
    """

    if disease in {"NotTomato", "Tomato___healthy"}:
        return "None"

    if confidence < 50:
        return "Unknown"
    elif 50 <= confidence < 70:
        return "Mild"
    elif 70 <= confidence < 85:
        return "Moderate"
    else:
        return "Severe"
