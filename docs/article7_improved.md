# Beyond Language: Transformers for Vision, Audio, and Multimodal AI

## Executive Summary (2 minutes)

**What**: Transformers now excel at processing images, audio, and multiple modalities—not just text.

**Why It Matters**: Enable new applications like visual search, automated transcription, and content generation.

**Key Technologies**:
- Vision: ViT, DeiT, Swin Transformer
- Audio: Whisper, Wav2Vec 2.0
- Multimodal: CLIP, BLIP-2
- Generation: Stable Diffusion XL

**Quick Win**: Implement CLIP-based image search in under 50 lines of code (see Quick Start).

**Investment Required**: 
- Development: 1-2 weeks for POC
- Infrastructure: GPU with 8-16GB VRAM
- Scaling: $500-2000/month for production

## Introduction

Transformers have revolutionized natural language processing. Now they're transforming how AI processes images, audio, and multiple data types simultaneously. This article explores practical applications of multimodal transformers and shows you how to implement them.

You'll learn to:
- Build image classification systems using vision transformers
- Create audio transcription and classification tools
- Implement cross-modal search connecting text and images
- Deploy production-ready multimodal pipelines

## Environment Setup

Set up your development environment once using your preferred package manager:

### Poetry (Recommended)

```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create project
poetry new multimodal-ai-project
cd multimodal-ai-project

# Add dependencies
poetry add "transformers>=4.40.0,<5.0.0" torch torchvision torchaudio
poetry add diffusers accelerate sentencepiece pillow soundfile
poetry add --group dev jupyter ipykernel matplotlib

# Activate environment
poetry shell
```

### Alternative: Conda

```bash
# Create environment
conda create -n multimodal-ai python=3.12.9
conda activate multimodal-ai

# Install packages
conda install -c pytorch -c huggingface transformers torch torchvision torchaudio
conda install -c conda-forge diffusers accelerate pillow soundfile matplotlib
pip install sentencepiece
```

## Vision Transformers: Images as Sequences

Vision Transformer (ViT) adapts the transformer architecture to process images. Instead of word tokens, ViT divides images into patches and treats them as a sequence. This approach enables transformers to understand visual content with the same mechanisms they use for text.

### How Vision Transformers Work

1. **Image Patching**: Divide a 224×224 image into 16×16 patches (196 total)
2. **Patch Embedding**: Convert each patch to a vector representation
3. **Position Encoding**: Add spatial information to maintain patch locations
4. **Transformer Processing**: Apply self-attention across all patches

Modern architectures improve on ViT in specific ways:
- **DeiT**: Trains efficiently with less data using knowledge distillation
- **Swin**: Handles large images through hierarchical processing  
- **MaxViT**: Combines local and global attention for balanced performance

Choose based on your constraints: limited data (DeiT), high-resolution images (Swin), or balanced needs (MaxViT).

### Image Classification Example

Let's classify an image using Vision Transformer:

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# Load image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification_parrots.png"
image = Image.open(requests.get(url, stream=True).raw)

# Load model and processor
model_id = "google/vit-base-patch16-224"  # Can swap: facebook/deit-base-patch16-224
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

# Process and predict
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Get result
predicted_class = outputs.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class])
```

This code:
1. Downloads a sample image
2. Loads a pre-trained model with its processor
3. Converts the image to model-ready tensors
4. Runs inference and decodes the result

The same API works across all vision transformer variants. Change the model ID to experiment with different architectures.

### Comparing Vision Architectures

Different models excel at different tasks. Here's a benchmarking tool:

```python
import time
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODELS = {
    "vit": "google/vit-base-patch16-224",
    "deit": "facebook/deit-base-patch16-224", 
    "swin": "microsoft/swin-tiny-patch4-window7-224"
}

def benchmark_models(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(image_path)
    results = {}
    
    for name, model_id in MODELS.items():
        start = time.time()
        
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        
        if device == "cuda":
            model = model.to(device)
        
        inputs = processor(images=image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
            
        inference_time = time.time() - start
        
        # Get top prediction
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_idx = torch.max(probs[0], 0)
        
        results[name] = {
            "time": inference_time,
            "prediction": model.config.id2label[top_idx.item()],
            "confidence": top_prob.item()
        }
    
    return results
```

## Audio Processing with Transformers

Transformers excel at audio tasks through models like Whisper and Wav2Vec 2.0. These models process raw audio waveforms end-to-end, eliminating complex preprocessing pipelines.

### Speech Recognition with Whisper

Whisper provides robust multilingual transcription:

```python
from transformers import pipeline

# Create transcription pipeline
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Transcribe audio
result = transcriber("meeting_audio.wav")
print("Transcription:", result["text"])
```

Whisper handles:
- 99 languages with automatic detection
- Background noise and accents
- Long-form audio through chunking
- Timestamps for subtitles

### Audio Classification

Beyond transcription, transformers can classify audio events:

```python
from transformers import pipeline

def classify_audio(audio_path, model="superb/wav2vec2-base-superb-ks"):
    # Create classifier
    classifier = pipeline(
        "audio-classification",
        model=model,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Classify audio
    results = classifier(audio_path)
    
    # Show top predictions
    for result in results[:3]:
        print(f"{result['label']}: {result['score']:.3f}")
    
    return results

# Example usage
classify_audio("alarm_sound.wav")
```

Applications include:
- Security systems detecting glass breaking or alarms
- Industrial monitoring for equipment failures
- Healthcare devices identifying coughs or breathing patterns
- Smart home automation responding to specific sounds

## Multimodal Models: Connecting Vision and Language

Multimodal models understand relationships between different data types. CLIP pioneered this by creating a shared embedding space for images and text, enabling powerful applications like visual search.

### How CLIP Works

CLIP (Contrastive Language-Image Pretraining) uses two encoders:
1. **Image Encoder**: Converts images to vectors
2. **Text Encoder**: Converts text to vectors

During training, CLIP learns to place matching image-text pairs close together in the embedding space. This enables zero-shot classification and cross-modal search.

### Image-Text Search Implementation

```python
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image

# Load CLIP
model = AutoModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Prepare images and texts
images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
texts = ["a photo of a cat", "a photo of a dog"]

# Process inputs
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

# Compute similarities
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

print("Image-text similarity scores:")
print(probs)
```

### Advanced Multimodal: BLIP-2 for Generation

While CLIP matches images and text, BLIP-2 generates descriptions and answers questions:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# Image captioning
def generate_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# Visual question answering
def answer_question(image_path, question):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
    
    image = Image.open(image_path)
    inputs = processor(image, question, return_tensors="pt")
    
    out = model.generate(**inputs, max_length=30)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return answer
```

## Building a Production Multimodal Search Engine

Let's build a complete image search system using CLIP:

```python
class MultimodalSearch:
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            
        self.image_features = None
        self.image_files = []
    
    def index_images(self, image_folder):
        """Index all images in a folder."""
        from pathlib import Path
        
        # Find all images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(Path(image_folder).glob(ext))
        
        # Process in batches
        batch_size = 8
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features /= features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu())
        
        self.image_features = torch.cat(all_features, dim=0)
        self.image_files = [str(p) for p in image_paths]
        
    def search(self, query, top_k=5):
        """Search images using text query."""
        # Encode text
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu()
        
        # Compute similarities
        similarities = (self.image_features @ text_features.T).squeeze(1)
        values, indices = similarities.topk(min(top_k, len(self.image_files)))
        
        results = [(self.image_files[idx], score.item()) 
                  for idx, score in zip(indices, values)]
        
        return results
```

For production scale:
- Use vector databases (FAISS, Milvus, Pinecone) for millions of images
- Cache embeddings to avoid recomputation
- Build REST APIs for search operations
- Monitor query latency and relevance

## Image Generation with Diffusion Models

Diffusion models represent a breakthrough in generative AI. They create images by learning to reverse a noise-adding process, guided by text descriptions.

### Stable Diffusion XL Example

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)

# Use GPU if available
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

# Generate image
prompt = "A serene mountain landscape at sunset, photorealistic"
negative_prompt = "blurry, low quality, oversaturated"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("generated_landscape.png")
```

Key parameters:
- `num_inference_steps`: Quality vs speed tradeoff (20-50 typical)
- `guidance_scale`: How closely to follow prompt (7-12 typical)
- `negative_prompt`: What to avoid in generation

## Production Deployment with SGLang

SGLang (Serving Graph Language) enables complex multimodal pipelines. Here's a customer support system that processes screenshots and audio:

```python
import sglang as sgl

@sgl.function
def classify_screenshot(s, image):
    s += sgl.image(image)
    s += "Classify this support issue as: bug, feature_request, or question.\n"
    s += "Category: " + sgl.gen("category", max_tokens=10)

@sgl.function
def transcribe_message(s, audio):
    s += "Transcribing customer audio message..."
    # In production, integrate with Whisper
    s += "Transcription: Customer reports login error 403"

@sgl.function
def generate_ticket(s, category, transcription):
    s += f"Category: {category}\n"
    s += f"Description: {transcription}\n"
    s += "Generate support ticket summary:\n"
    s += sgl.gen("summary", max_tokens=100)

@sgl.function
def support_pipeline(s, screenshot, audio):
    # Process inputs
    s_img = classify_screenshot.run(image=screenshot)
    s_audio = transcribe_message.run(audio=audio)
    
    # Generate ticket
    s = generate_ticket(s, 
                       s_img["category"], 
                       s_audio["transcription"])
    return s

# Deploy with quantization for efficiency
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    quantization="awq",  # 4x memory reduction
    tp_size=1
)
```

SGLang features for production:
- **Quantization**: AWQ/GPTQ reduces memory 4x
- **Speculative Decoding**: 2-3x faster inference
- **Multi-LoRA**: Serve multiple model variants
- **Auto-scaling**: Handle variable load

## Interactive Demos with Gradio

Create web interfaces for your models:

```python
import gradio as gr

def create_demo():
    with gr.Blocks(title="Multimodal AI Demo") as demo:
        
        with gr.Tab("Image Classification"):
            image_input = gr.Image(type="pil")
            model_dropdown = gr.Dropdown(
                choices=["vit", "deit", "swin"],
                value="vit",
                label="Model"
            )
            classify_btn = gr.Button("Classify")
            output = gr.Textbox(label="Result")
            
            def classify(img, model_choice):
                # Your classification logic
                return f"Predicted: [result] with {model_choice}"
            
            classify_btn.click(
                classify,
                inputs=[image_input, model_dropdown],
                outputs=output
            )
        
        with gr.Tab("Text-to-Image Search"):
            query = gr.Textbox(label="Search query")
            search_btn = gr.Button("Search")
            results = gr.Gallery(label="Results")
            
            # Add search logic
            
    return demo

# Launch
demo = create_demo()
demo.launch(share=True)
```

## Key Takeaways

- **Vision Transformers** process images as sequences of patches, enabling powerful visual understanding
- **Audio Transformers** handle speech and sound end-to-end without complex preprocessing
- **Multimodal Models** connect different data types, enabling cross-modal search and generation
- **Hugging Face** provides consistent APIs across all modalities
- **Production deployment** requires optimization (quantization, caching) and proper infrastructure

## Next Steps

1. **Start Small**: Implement image classification or audio transcription
2. **Experiment**: Try different models and architectures
3. **Optimize**: Use quantization and efficient serving
4. **Scale**: Deploy with proper monitoring and infrastructure
5. **Iterate**: Fine-tune models for your specific domain

The transformer architecture continues to unify AI across modalities. Master these tools to build the next generation of intelligent applications.