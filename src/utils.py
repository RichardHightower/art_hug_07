"""Utility functions for multimodal examples."""

import requests
from PIL import Image
from pathlib import Path
from config import IMAGES_DIR, AUDIO_DIR

def download_samples():
    """Download sample images and audio files."""
    
    # Sample images
    image_urls = {
        "parrot.jpg": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification_parrots.png",
        "cat.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
        "dog.jpg": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
        "sunset.jpg": "https://images.unsplash.com/photo-1495616811223-4d98c6e9c869?w=400",
        "city.jpg": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400"
    }
    
    print("Downloading sample images...")
    for filename, url in image_urls.items():
        filepath = IMAGES_DIR / filename
        if not filepath.exists():
            try:
                response = requests.get(url, stream=True)
                img = Image.open(response.raw)
                img.save(filepath)
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  ✗ Error downloading {filename}: {e}")
        else:
            print(f"  - {filename} already exists")
    
    # Note: Audio samples would be downloaded similarly
    print("\nSample download complete!")

if __name__ == "__main__":
    download_samples()
