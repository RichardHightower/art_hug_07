"""Vision Transformer examples using ViT, DeiT, and Swin."""

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests
from config import VISION_MODEL, get_device, SAMPLE_IMAGE_URL, OUTPUT_DIR
import numpy as np
from pathlib import Path

def create_sample_image():
    """Create a sample image for testing."""
    # Create a simple gradient image
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            img_array[i, j] = [i, j, (i+j)//2]
    return Image.fromarray(img_array)

def classify_image_with_model(image_path, model_name=VISION_MODEL):
    """
    Classify an image using a vision transformer model.
    
    Args:
        image_path: Path to image or URL
        model_name: Model to use (ViT, DeiT, Swin, etc.)
    
    Returns:
        Predicted class and confidence
    """
    device = get_device()
    
    # Load image
    if isinstance(image_path, str) and image_path.startswith('http'):
        try:
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        except Exception as e:
            print(f"Failed to download image from URL: {e}")
            # Create a sample image as fallback
            image = Image.new('RGB', (224, 224), color='red')
    else:
        image = Image.open(image_path)
    
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    if device != "cpu":
        model = model.to(device)
    
    # Preprocess and predict
    inputs = processor(images=image, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction
    logits = outputs.logits
    predicted_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_idx]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0, predicted_idx].item()
    
    return predicted_label, confidence

def compare_vision_models(image_path):
    """Compare different vision transformer architectures."""
    models = {
        "ViT": "google/vit-base-patch16-224",
        "DeiT": "facebook/deit-base-patch16-224",
        "Swin": "microsoft/swin-tiny-patch4-window7-224",
    }
    
    results = {}
    for name, model_id in models.items():
        try:
            label, confidence = classify_image_with_model(image_path, model_id)
            results[name] = (label, confidence)
            print(f"{name}: {label} ({confidence:.2%})")
        except Exception as e:
            print(f"{name}: Error - {e}")
            results[name] = ("Error", 0.0)
    
    return results

def demonstrate_vision_transformers():
    """Run vision transformer demonstrations."""
    
    # Try to use URL or create sample image
    try:
        print("Attempting to download sample image...")
        # Test if URL works
        response = requests.head(SAMPLE_IMAGE_URL, timeout=5)
        if response.status_code == 200:
            image_source = SAMPLE_IMAGE_URL
        else:
            raise Exception("URL not accessible")
    except:
        print("Using generated sample image instead...")
        # Save sample image
        sample_img = create_sample_image()
        sample_path = OUTPUT_DIR / "sample_image.png"
        sample_img.save(sample_path)
        image_source = str(sample_path)
    
    # Example 1: Basic classification
    print("\n1. Basic Image Classification with ViT:")
    label, confidence = classify_image_with_model(image_source)
    print(f"   Predicted: {label} (confidence: {confidence:.2%})")
    
    # Example 2: Compare architectures
    print("\n2. Comparing Vision Transformer Architectures:")
    print("   (This may take a moment as models are downloaded...)")
    compare_vision_models(SAMPLE_IMAGE_URL)
    
    # Example 3: Patch visualization concept
    print("\n3. Understanding Vision Transformers:")
    print("   - Images are divided into patches (e.g., 16x16 pixels)")
    print("   - Each patch is treated like a 'word' in a sentence")
    print("   - Self-attention learns relationships between patches")
    print("   - This enables understanding of both local and global features")
    
    print("\nKey advantages of modern vision transformers:")
    print("✓ DeiT: More data-efficient training")
    print("✓ Swin: Hierarchical features for better scalability")
    print("✓ All models use the same Hugging Face API!")

if __name__ == "__main__":
    print("=== Vision Transformer Examples ===\n")
    demonstrate_vision_transformers()
