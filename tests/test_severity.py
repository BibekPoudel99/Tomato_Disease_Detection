from app.utils.severity import calculate_severity


def test_not_tomato_severity_none():
    assert calculate_severity("NotTomato", 90.0) == "None"


def test_healthy_severity_none():
    assert calculate_severity("Tomato___healthy", 95.0) == "None"


def test_low_confidence_unknown():
    assert calculate_severity("Tomato___Early_blight", 40.0) == "Unknown"


def test_high_confidence_disease_severe():
    assert calculate_severity("Tomato___Late_blight", 88.0) == "Severe"
