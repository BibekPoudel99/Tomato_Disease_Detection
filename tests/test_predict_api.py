import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_home():
    """Test home endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


@patch("app.main.predict_disease")
def test_predict_high_confidence(mock_predict):
    """Test /predict with high confidence"""
    mock_predict.return_value = {"disease": "Tomato___Septoria_leaf_spot", "confidence": 85.5}
    
    with open("tests/sample_image.jpg", "rb") as f:
        response = client.post("/predict", files={"file": f})
    
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "Tomato___Septoria_leaf_spot"
    assert data["confidence"] == 85.5
    assert data["severity"] in ["Mild", "Moderate", "Severe"]
    assert "explanation" in data


@patch("app.main.predict_disease")
def test_predict_low_confidence(mock_predict):
    """Test /predict with low confidence (< 50%)"""
    mock_predict.return_value = {"disease": "Unknown", "confidence": 30.0}
    
    with open("tests/sample_image.jpg", "rb") as f:
        response = client.post("/predict", files={"file": f})
    
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "Uncertain"
    assert data["severity"] == "Unknown"
    assert "not confident" in data["explanation"].lower()


def test_predict_no_file():
    """Test /predict without file"""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_invalid_file_type():
    """Test /predict with non-image file"""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_predict_empty_file():
    """Test /predict with empty file"""
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Empty file" in response.json()["detail"]