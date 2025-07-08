"""All code examples from Article 7 in one file for reference."""

from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    AutoProcessor, AutoModel, WhisperForConditionalGeneration,
    pipeline, BlipProcessor, BlipForConditionalGeneration
)
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch
import requests
import soundfile as sf
import os

# Environment setup examples from article
"""
# Poetry setup
poetry new multimodal-ai-project
cd multimodal-ai-project
poetry add "transformers>=4.40.0,<5.0.0" torch torchvision torchaudio
poetry add diffusers accelerate sentencepiece pillow soundfile

# Conda setup
conda create -n multimodal-ai python=3.12.9
conda activate multimodal-ai
conda install -c pytorch -c huggingface transformers torch torchvision torchaudio

# Pip with pyenv
pyenv install 3.12.9
pyenv local 3.12.9
python -m venv venv
source venv/bin/activate
pip install "transformers>=4.40.0,<5.0.0" torch torchvision torchaudio
"""

def example_1_basic_vit_classification():
    """Example from 'Classifying an Image with a Vision Transformer (ViT)'"""
    
    # Download an example image (parrots)
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification_parrots.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # Load the processor and model
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Preprocess the image and make a prediction
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Get the predicted class label
    predicted_class = outputs.logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class])

def example_2_modern_vision_transformers():
    """Example from 'Classifying Images with Modern Vision Transformers'"""
    
    # 1. Load an image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification_parrots.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # 2. Choose a modern vision transformer (e.g., DeiT, Swin, MaxViT)
    # Examples: 'facebook/deit-base-patch16-224', 'microsoft/swin-tiny-patch4-window7-224'
    model_id = "facebook/deit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)

    # 3. Preprocess: resize, normalize, split into patches, and convert to tensor
    inputs = processor(images=image, return_tensors="pt")

    # 4. Predict: model outputs raw prediction scores (logits)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()

    # 5. Decode to label
    print("Predicted class:", model.config.id2label[predicted_class])

def example_3_whisper_pipeline():
    """Example from 'Transcribing Audio with Whisper and Hugging Face Pipeline'"""
    
    # Create an automatic speech recognition pipeline with Whisper
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")

    # Transcribe your audio file (WAV, MP3, FLAC, etc.)
    # result = asr("sample.wav")
    # print("Transcription:", result["text"])
    
    # Demo output
    print("Transcription: [Would transcribe audio file here]")

def example_4_whisper_manual():
    """Example from 'Transcribing Audio with AutoProcessor and Whisper'"""
    
    # This is a conceptual example - requires actual audio file
    """
    # Load audio file
    audio, rate = sf.read("sample.wav")

    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    # Preprocess audio
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

    # Model inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode output
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print("Transcription:", transcription[0])
    """
    print("Manual transcription example (requires audio file)")

def example_5_audio_classification():
    """Example from 'Classifying Audio Events with Hugging Face Pipeline'"""
    
    # Create an audio classification pipeline
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")

    # Classify your audio file
    # result = classifier("dog_bark.wav")
    # print("Predicted label:", result[0]["label"])
    
    print("Audio classification example (requires audio file)")

def example_6_stable_diffusion():
    """Example from 'Generating Art with Stable Diffusion XL'"""
    
    # Note: This is resource intensive
    """
    # Load the SDXL pipeline from the Hugging Face Hub
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )

    # Enable memory-efficient inference
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()

    # Your text prompt and negative prompt
    prompt = "A futuristic city skyline at sunset, digital art"
    negative_prompt = "blurry, low quality, distorted"

    # Generate the image
    result = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=7.0)
    image = result.images[0]

    # Save the image
    image.save("generated_city.png")
    print("Image saved as generated_city.png")
    """
    print("Stable Diffusion example (requires GPU and significant memory)")

def example_7_clip_similarity():
    """Example from 'Searching Images by Text with CLIP (Modern API)'"""
    
    # Load model and processor using the recommended Auto* interfaces
    model_id = "openai/clip-vit-base-patch16"
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # Prepare images and texts (demo with dummy data)
    # images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
    texts = ["a photo of a cat", "a photo of a dog"]
    
    # For demo, create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    images = [dummy_image, dummy_image]

    # Preprocess inputs
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Compute similarity scores
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    print("Probabilities:", probs)

def example_8_multimodal_search_step1():
    """Example from 'Step 1: Embed a Collection of Images (Modern API)'"""
    
    # Load model and processor
    model_id = "openai/clip-vit-base-patch16"
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # Demo with dummy data
    image_folder = "./images"
    # image_files = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
    
    # For demo, create dummy images
    images = [Image.new('RGB', (224, 224), color='blue') for _ in range(3)]
    
    # Preprocess images
    inputs = processor(images=images, return_tensors="pt", padding=True)

    # Compute image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    print("Image embeddings shape:", image_features.shape)

def example_9_multimodal_search_step2():
    """Example from 'Step 2: Query with Text and Retrieve'"""
    
    # Continuing from step 1...
    model_id = "openai/clip-vit-base-patch16"
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # User provides a text query
    text_query = "a smiling person wearing sunglasses on the beach"

    # Preprocess and embed the text
    text_inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print("Text embedding shape:", text_features.shape)
    
    # In practice, you would compute similarities with image_features
    # similarities = (image_features @ text_features.T).squeeze(1)
    # best_idx = similarities.argmax().item()

def example_10_sglang_pipeline():
    """Example from 'Defining a Multimodal Serving Pipeline in SGLang'"""
    
    # This is the conceptual SGLang code from the article
    sglang_code = '''
import sglang as sgl

@sgl.function
def classify_image(s, image):
    s += sgl.image(image)
    s += "What type of customer support issue is shown in this image? "
    s += "Classify as: error, feature_request, or other.\\n"
    s += "Classification: " + sgl.gen("classification", max_tokens=10)

@sgl.function
def transcribe_audio(s, audio):
    s += "Transcribed audio: Customer reporting login issues with error code 403"

@sgl.function
def summarize_support_request(s, image_class, audio_text):
    s += f"Image classification: {image_class}\\n"
    s += f"Audio transcription: {audio_text}\\n"
    s += "Please provide a brief summary of this support request:\\n"
    s += sgl.gen("summary", max_tokens=100)

@sgl.function
def support_pipeline(s, image, audio):
    s_img = classify_image.run(image=image)
    image_class = s_img["classification"]
    
    s_audio = transcribe_audio.run(audio=audio)
    audio_text = "Customer reporting login issues with error code 403"
    
    s = summarize_support_request(s, image_class, audio_text)
    return s

# Runtime configuration with quantization
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    quantization="awq",
    tp_size=1
)

sgl.set_default_backend(runtime)
'''
    
    print("SGLang pipeline example (requires SGLang installation)")
    print("Code structure shown above demonstrates the pattern")

def example_11_summary_vit_with_auto():
    """Example from Summary: 'Classifying an Image with Vision Transformer'"""
    
    # 1. Load an image from the web
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification_parrots.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # 2. Load pre-trained ViT model and processor using Auto classes
    processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224')
    model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # 3. Preprocess the image and predict the class
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class])

def example_12_summary_clip_search():
    """Example from Summary: 'Text-to-Image Search with CLIP'"""
    
    # 1. Load CLIP model and processor using Auto classes
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # 2. Prepare images and text queries
    # images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
    texts = ["a photo of a cat", "a photo of a dog"]
    
    # Demo with dummy images
    images = [Image.new('RGB', (224, 224), color=c) for c in ['red', 'blue']]

    # 3. Compute similarity between images and text
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print("Probabilities:", probs)

def main():
    """Run all examples (those that don't require external resources)."""
    
    print("Article 7: All Code Examples")
    print("=" * 70)
    
    print("\n1. Basic ViT Classification")
    example_1_basic_vit_classification()
    
    print("\n2. Modern Vision Transformers (DeiT)")
    example_2_modern_vision_transformers()
    
    print("\n3. Whisper Pipeline")
    example_3_whisper_pipeline()
    
    print("\n4. Whisper Manual")
    example_4_whisper_manual()
    
    print("\n5. Audio Classification")
    example_5_audio_classification()
    
    print("\n6. Stable Diffusion")
    example_6_stable_diffusion()
    
    print("\n7. CLIP Similarity")
    example_7_clip_similarity()
    
    print("\n8. Multimodal Search - Embedding")
    example_8_multimodal_search_step1()
    
    print("\n9. Multimodal Search - Query")
    example_9_multimodal_search_step2()
    
    print("\n10. SGLang Pipeline")
    example_10_sglang_pipeline()
    
    print("\n11. Summary ViT Example")
    example_11_summary_vit_with_auto()
    
    print("\n12. Summary CLIP Search")
    example_12_summary_clip_search()
    
    print("\n" + "=" * 70)
    print("Note: Some examples require audio files, images, or GPU resources")
    print("See individual module files for complete implementations")

if __name__ == "__main__":
    main()