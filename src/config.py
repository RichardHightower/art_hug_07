"""Configuration module for multimodal AI examples."""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# Create directories if they don't exist
for dir_path in [DATA_DIR, IMAGES_DIR, AUDIO_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
VISION_MODEL = os.getenv("VISION_MODEL", "google/vit-base-patch16-224")
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "openai/whisper-base")
DIFFUSION_MODEL = os.getenv(
    "DIFFUSION_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
)
MULTIMODAL_MODEL = os.getenv("MULTIMODAL_MODEL", "openai/clip-vit-base-patch16")

# Performance settings
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

# Hugging Face token (optional)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Cache directory
CACHE_DIR = Path(os.getenv("CACHE_DIR", "~/.cache/huggingface")).expanduser()


def get_device():
    """Get the best available device."""
    if FORCE_CPU:
        return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


DEVICE = get_device()

# Sample URLs for downloading test data
SAMPLE_IMAGE_URL = (
    "https://images.unsplash.com/photo-1594736797933-d0501ba2fe65?w=400"  # Parrot image
)
SAMPLE_AUDIO_URL = (
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
)
