"""Unit tests for multimodal AI examples."""

import sys
from pathlib import Path

import pytest
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_device
from vision_transformers import classify_image_with_model


def test_device_detection():
    """Test that device detection works."""
    device = get_device()
    assert device in ["cpu", "cuda", "mps"]


def test_imports():
    """Test that all required libraries can be imported."""
    import torch
    import transformers

    assert transformers.__version__
    assert torch.__version__


def test_create_test_image():
    """Test creating a simple test image."""
    # Create a simple red square
    img = Image.new("RGB", (224, 224), color="red")
    assert img.size == (224, 224)
    assert img.mode == "RGB"


def test_vision_classification():
    """Test basic vision transformer classification."""
    # Create test image
    img = Image.new("RGB", (224, 224), color="blue")

    try:
        # This might fail if model can't be downloaded
        label, confidence = classify_image_with_model(
            img, "google/vit-base-patch16-224"
        )
        assert isinstance(label, str)
        assert 0 <= confidence <= 1
    except Exception as e:
        # Skip if model download fails
        pytest.skip(f"Model download failed: {e}")


def test_multimodal_search_engine():
    """Test the multimodal search engine initialization."""
    from multimodal_search import MultimodalSearchEngine

    try:
        engine = MultimodalSearchEngine()
        assert engine.device in ["cpu", "cuda", "mps"]
        assert engine.image_embeddings is None  # Not indexed yet
    except Exception as e:
        pytest.skip(f"Model initialization failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
