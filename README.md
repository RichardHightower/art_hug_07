# Beyond Language: Transformers for Vision, Audio, and Multimodal AI

This project contains working examples for Chapter 7 of the Hugging Face Transformers book, demonstrating how transformers have expanded beyond text to vision, audio, and multimodal applications.

## Overview

Learn how to implement and understand:

- Vision Transformers (ViT, DeiT, Swin) for image classification and analysis
- Audio processing with Wav2Vec 2.0 and Whisper for speech recognition
- Generative AI with Stable Diffusion XL for text-to-image generation
- Multimodal models like CLIP and BLIP for cross-modal search and understanding
- Building multimodal search engines and applications
- Production deployment with SGLang

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- CUDA-capable GPU recommended (but CPU mode supported)
- (Optional) Hugging Face account for accessing gated models

## Setup

1. Clone this repository
2. Run the setup task:
   ```bash
   task setup
   ```
3. Copy `.env.example` to `.env` and configure as needed
4. Download sample data:
   ```bash
   task download-samples
   ```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration and utilities
│   ├── main.py                        # Entry point with all examples
│   ├── vision_transformers.py         # ViT, DeiT, Swin implementations
│   ├── audio_processing.py            # Wav2Vec2, Whisper examples
│   ├── diffusion_models.py            # Stable Diffusion XL generation
│   ├── multimodal_models.py           # CLIP, BLIP cross-modal search
│   ├── multimodal_search.py           # Building search applications
│   ├── sglang_deployment.py           # Production deployment examples
│   └── gradio_app.py                  # Interactive web interface
├── tests/
│   └── test_multimodal.py             # Unit tests
├── notebooks/
│   ├── vision_exploration.ipynb       # Interactive vision examples
│   └── multimodal_search.ipynb        # Search engine tutorial
├── data/
│   ├── images/                        # Sample images
│   └── audio/                         # Sample audio files
├── outputs/                           # Generated images and results
├── .env.example                       # Environment template
├── Taskfile.yml                       # Task automation
└── pyproject.toml                     # Poetry configuration
```

## Running Examples

Run all examples:
```bash
task run
```

Or run individual modules:
```bash
task run-vision         # Vision transformer examples
task run-audio          # Audio processing examples
task run-diffusion      # Image generation with SDXL
task run-multimodal     # CLIP/BLIP multimodal examples
task run-search         # Multimodal search engine
task run-sglang         # SGLang deployment demo
```

Launch interactive web app:
```bash
task gradio
```

## Key Concepts Demonstrated

1. **Vision Transformers**: How ViT, DeiT, and Swin process images as patches
2. **Audio Transformers**: End-to-end speech recognition with Whisper
3. **Diffusion Models**: Generate images from text with SDXL
4. **Cross-Modal Understanding**: CLIP and BLIP for connecting text and images
5. **Production Deployment**: Using SGLang for scalable multimodal pipelines

## Example Output

The examples demonstrate:

1. **Image Classification**: Classify images using state-of-the-art vision transformers
2. **Speech-to-Text**: Transcribe audio in multiple languages
3. **Text-to-Image**: Generate creative images from prompts
4. **Multimodal Search**: Find images using natural language queries
5. **Production Pipeline**: Deploy chained models with SGLang

## Models Used

- **Vision**: ViT, DeiT, Swin Transformer
- **Audio**: Wav2Vec 2.0, Whisper
- **Generation**: Stable Diffusion XL
- **Multimodal**: CLIP, BLIP, BLIP-2, LLaVA

## Available Tasks

- `task setup` - Set up Python environment and install dependencies
- `task run` - Run all examples
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task clean` - Clean up generated files and outputs
- `task download-samples` - Download sample images and audio
- `task gradio` - Launch interactive web interface
- `task notebook` - Launch Jupyter notebook server

## GPU vs CPU Mode

The code automatically detects available hardware:
- CUDA GPU: Fastest performance
- MPS (Apple Silicon): Good performance on Mac
- CPU: Slower but functional

To force CPU mode, set `FORCE_CPU=true` in your `.env` file.

## Troubleshooting

- **Out of Memory**: Try smaller models or enable CPU offloading
- **Slow Generation**: Use GPU or reduce image resolution
- **Model Download**: First run downloads several GB of models
- **Audio Issues**: Ensure audio files are 16kHz mono WAV

## Learn More

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [Vision Transformer Papers](https://arxiv.org/abs/2010.11929)
