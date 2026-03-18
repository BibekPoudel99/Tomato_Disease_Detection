import pytest
from PIL import Image
import io


@pytest.fixture(scope="session", autouse=True)
def create_test_image():
    """Create a dummy test image once per test session"""
    img = Image.new("RGB", (100, 100), color="red")
    img.save("tests/sample_image.jpg")
    yield
    # Optional: cleanup after tests
    # os.remove("tests/sample_image.jpg")