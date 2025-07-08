"""Simplified main entry point for quick demonstrations."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from vision_transformers import classify_image_with_model, create_sample_image
from multimodal_models import compute_clip_similarity
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from PIL import Image
import numpy as np

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def main():
    """Run simplified multimodal AI demonstrations."""
    print_section("MULTIMODAL AI: QUICK DEMONSTRATIONS")
    print("This demonstrates key concepts without downloading large models.\n")
    
    # Vision Transformers
    print_section("1. VISION TRANSFORMERS")
    print("Creating a sample image and classifying it...")
    
    # Create and save sample image
    sample_img = create_sample_image()
    sample_path = Path("sample_image.png")
    sample_img.save(sample_path)
    
    print(f"Created sample image: {sample_path}")
    print("\nClassifying with Vision Transformer (ViT)...")
    
    try:
        label, confidence = classify_image_with_model(str(sample_path))
        print(f"Predicted: {label} (confidence: {confidence:.2%})")
    except Exception as e:
        print(f"Note: Model download required. Error: {e}")
    
    # CLIP Multimodal Demo
    print_section("2. MULTIMODAL MODELS (CLIP)")
    print("Demonstrating image-text similarity with CLIP...")
    
    # Create sample images
    images = []
    for i, color in enumerate(['red', 'green', 'blue']):
        img = Image.new('RGB', (224, 224), color=color)
        images.append(img)
    
    texts = ["a red square", "a green square", "a blue square"]
    
    print("\nComputing similarities between colored squares and text descriptions:")
    try:
        similarities = compute_clip_similarity(images, texts)
        print("\nSimilarity matrix (images x texts):")
        print("              'red square'  'green square'  'blue square'")
        for i, color in enumerate(['Red image:', 'Green image:', 'Blue image:']):
            print(f"{color:12} {similarities[i,0]:.3f}        {similarities[i,1]:.3f}          {similarities[i,2]:.3f}")
        
        print("\n✓ Notice how each colored image best matches its text description!")
    except Exception as e:
        print(f"Note: CLIP model download required. Error: {e}")
    
    # Key Concepts Summary
    print_section("KEY CONCEPTS DEMONSTRATED")
    
    print("1. Vision Transformers:")
    print("   • Process images as sequences of patches")
    print("   • Use self-attention to understand spatial relationships")
    print("   • Same API for different architectures (ViT, DeiT, Swin)")
    
    print("\n2. Multimodal Models:")
    print("   • CLIP creates shared embeddings for images and text")
    print("   • Enables zero-shot classification and search")
    print("   • Foundation for advanced applications")
    
    print("\n3. Practical Applications:")
    print("   • Image classification and search")
    print("   • Audio transcription (Whisper)")
    print("   • Image generation (Stable Diffusion)")
    print("   • Cross-modal understanding")
    
    print("\n" + "=" * 70)
    print("For full demonstrations with all models, run: task run")
    print("For interactive exploration, run: task notebook")
    
    # Clean up
    if sample_path.exists():
        sample_path.unlink()

if __name__ == "__main__":
    main()