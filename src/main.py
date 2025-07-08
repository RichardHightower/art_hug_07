"""Main entry point demonstrating all multimodal AI examples."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from vision_transformers import demonstrate_vision_transformers
from audio_processing import demonstrate_audio_processing
from diffusion_models import demonstrate_diffusion_models
from multimodal_models import demonstrate_multimodal_models
from multimodal_search import demonstrate_multimodal_search
from modern_vision_transformers import demonstrate_modern_vision_transformers
from audio_classification import demonstrate_audio_classification
from advanced_multimodal import demonstrate_advanced_multimodal
from sglang_multimodal_pipeline import demonstrate_sglang_pipeline

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def main():
    """Run all multimodal AI examples."""
    print_section("MULTIMODAL AI: VISION, AUDIO & GENERATIVE MODELS")
    print("This script demonstrates how transformers have expanded beyond text")
    print("to process images, audio, and multimodal data.\n")
    
    # Vision Transformers
    print_section("1. VISION TRANSFORMERS")
    print("Classifying images with ViT, DeiT, and Swin Transformer...")
    demonstrate_vision_transformers()
    
    # Audio Processing
    print_section("2. AUDIO PROCESSING")
    print("Speech recognition and audio analysis with Whisper...")
    demonstrate_audio_processing()
    
    # Diffusion Models
    print_section("3. DIFFUSION MODELS")
    print("Generating images from text with Stable Diffusion XL...")
    demonstrate_diffusion_models()
    
    # Multimodal Models
    print_section("4. MULTIMODAL MODELS")
    print("Cross-modal understanding with CLIP and BLIP...")
    demonstrate_multimodal_models()
    
    # Multimodal Search
    print_section("5. MULTIMODAL SEARCH ENGINE")
    print("Building a practical image search application...")
    demonstrate_multimodal_search()
    
    # Modern Vision Transformers
    print_section("6. MODERN VISION TRANSFORMERS")
    print("Comparing DeiT, Swin, and other architectures...")
    demonstrate_modern_vision_transformers()
    
    # Audio Classification
    print_section("7. AUDIO CLASSIFICATION")
    print("Classifying audio events and sounds...")
    demonstrate_audio_classification()
    
    # Advanced Multimodal
    print_section("8. ADVANCED MULTIMODAL MODELS")
    print("BLIP-2, LLaVA, and unified models...")
    demonstrate_advanced_multimodal()
    
    # SGLang Pipeline
    print_section("9. SGLANG PRODUCTION PIPELINE")
    print("Building scalable multimodal pipelines...")
    demonstrate_sglang_pipeline()
    
    # Conclusion
    print_section("CONCLUSION")
    print("You've seen how transformers now power vision, audio, and multimodal AI!")
    print("These models enable new applications across industries:")
    print("- Automated visual inspection and content moderation")
    print("- Speech analytics and accessibility tools")
    print("- Creative AI for design and content generation")
    print("- Intelligent search across different data types")
    print("\nExplore the individual modules and notebooks to dive deeper!")

if __name__ == "__main__":
    main()
