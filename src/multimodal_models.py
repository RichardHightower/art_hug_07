"""Cross-modal understanding with CLIP and BLIP models."""

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config import MULTIMODAL_MODEL, SAMPLE_IMAGE_URL, get_device


def compute_clip_similarity(images, texts, model_name=MULTIMODAL_MODEL):
    """
    Compute similarity between images and texts using CLIP.

    Args:
        images: List of PIL Images
        texts: List of text descriptions
        model_name: CLIP model to use

    Returns:
        Similarity matrix (images x texts)
    """
    device = get_device()

    # Load model and processor
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    if device != "cpu":
        model = model.to(device)

    # Process inputs
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute features
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    return probs.cpu().numpy()


def find_best_match(image, text_queries, model_name=MULTIMODAL_MODEL):
    """
    Find the best matching text for an image.

    Args:
        image: PIL Image
        text_queries: List of text descriptions
        model_name: Model to use

    Returns:
        Best matching text and confidence
    """
    probs = compute_clip_similarity([image], text_queries, model_name)
    best_idx = probs[0].argmax()
    best_text = text_queries[best_idx]
    confidence = probs[0, best_idx]

    return best_text, confidence


def demonstrate_multimodal_models():
    """Run multimodal model demonstrations."""

    print("Demonstrating cross-modal understanding with CLIP...")

    # Load sample image
    print("\n1. Loading sample image...")
    image = Image.open(requests.get(SAMPLE_IMAGE_URL, stream=True).raw)

    # Example 1: Zero-shot classification
    print("\n2. Zero-shot Image Classification:")
    labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a car",
        "a photo of a person",
    ]

    best_label, confidence = find_best_match(image, labels)
    print(f"   Best match: '{best_label}' (confidence: {confidence:.2%})")

    # Example 2: More detailed queries
    print("\n3. Detailed Description Matching:")
    detailed_queries = [
        "a colorful parrot in nature",
        "a red and blue tropical bird",
        "a small songbird on a branch",
        "an eagle soaring in the sky",
        "a group of penguins",
    ]

    probs = compute_clip_similarity([image], detailed_queries)
    print("   Similarity scores:")
    for i, query in enumerate(detailed_queries):
        print(f"   - '{query}': {probs[0, i]:.2%}")

    # Example 3: Understanding CLIP's embedding space
    print("\n4. How CLIP Works:")
    print("   - Dual encoders: one for images, one for text")
    print("   - Both map to the same embedding space")
    print("   - Contrastive learning aligns matching pairs")
    print("   - Enables zero-shot classification and search")

    # Example 4: BLIP and beyond
    print("\n5. Beyond CLIP:")
    print("   BLIP-2: Better captioning and VQA")
    print("   LLaVA: Multimodal conversation")
    print("   ImageBind: Connects 6+ modalities")
    print("   All available through Hugging Face!")

    print("\nPractical applications:")
    print("✓ Image search with natural language")
    print("✓ Content moderation and filtering")
    print("✓ Automatic image tagging")
    print("✓ Visual question answering")


if __name__ == "__main__":
    print("=== Multimodal Model Examples ===\n")
    demonstrate_multimodal_models()
