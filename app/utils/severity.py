def calculate_severity(confidence: float) -> str:
    """
    Determines disease severity level based on prediction confidence.
    This is a proxy severity estimation and can later be replaced
    with lesion area segmentation or disease progression modeling.
    """

    if confidence < 50:
        return "Unknown"
    elif 50 <= confidence < 70:
        return "Mild"
    elif 70 <= confidence < 85:
        return "Moderate"
    else:
        return "Severe"
