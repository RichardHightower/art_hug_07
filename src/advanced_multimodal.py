"""Advanced multimodal models: BLIP-2, LLaVA, and unified models."""

import requests
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
)

from config import SAMPLE_IMAGE_URL, get_device


def image_captioning_with_blip(
    image_path, model_name="Salesforce/blip-image-captioning-base"
):
    """
    Generate image captions using BLIP model.

    Args:
        image_path: Path to image or URL
        model_name: BLIP model to use

    Returns:
        Generated caption
    """
    device = get_device()

    # Load image
    if isinstance(image_path, str) and image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    # Load processor and model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    if device != "cpu":
        model = model.to(device)

    # Generate caption
    inputs = processor(image, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


def visual_question_answering(
    image_path, question, model_name="Salesforce/blip-vqa-base"
):
    """
    Answer questions about images using BLIP VQA model.

    Args:
        image_path: Path to image or URL
        question: Question about the image
        model_name: BLIP VQA model to use

    Returns:
        Answer to the question
    """
    device = get_device()

    # Load image
    if isinstance(image_path, str) and image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    # Load processor and model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    if device != "cpu":
        model = model.to(device)

    # Process question and image
    inputs = processor(image, question, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_length=50)
    answer = processor.decode(out[0], skip_special_tokens=True)

    return answer


def demonstrate_advanced_multimodal():
    """Demonstrate advanced multimodal capabilities."""

    print("Advanced Multimodal Models Demonstration")
    print("=" * 60)

    # Test image
    test_image = SAMPLE_IMAGE_URL

    print("\n1. Image Captioning with BLIP")
    print("-" * 40)
    print(f"Image: {test_image}")

    try:
        caption = image_captioning_with_blip(test_image)
        print(f"Generated caption: {caption}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: BLIP models require significant memory")

    print("\n2. Visual Question Answering (VQA)")
    print("-" * 40)

    questions = [
        "What animal is in the image?",
        "What color is the main subject?",
        "How many objects are visible?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        try:
            answer = visual_question_answering(test_image, question)
            print(f"A: {answer}")
        except Exception as e:
            print(f"A: Error - {e}")

    print("\n3. Modern Multimodal Models Overview")
    print("-" * 40)

    models_info = {
        "BLIP-2": {
            "strengths": "Efficient, uses frozen image encoder + LLM",
            "use_cases": "Captioning, VQA, image-text matching",
            "model_id": "Salesforce/blip2-flan-t5-xl",
        },
        "LLaVA": {
            "strengths": "Strong reasoning, instruction following",
            "use_cases": "Complex visual reasoning, detailed descriptions",
            "model_id": "llava-hf/llava-1.5-7b-hf",
        },
        "IDEFICS": {
            "strengths": "Open reproduction of Flamingo, multi-image",
            "use_cases": "Multi-image reasoning, interleaved text-image",
            "model_id": "HuggingFaceM4/idefics-9b",
        },
        "Unified Models (GPT-4V, Gemini)": {
            "strengths": "Native multimodal, supports text/image/audio/video",
            "use_cases": "Complex reasoning across all modalities",
            "model_id": "Via API only",
        },
    }

    for model_name, info in models_info.items():
        print(f"\n{model_name}:")
        print(f"  • Strengths: {info['strengths']}")
        print(f"  • Use cases: {info['use_cases']}")
        print(f"  • Model ID: {info['model_id']}")

    print("\n4. Practical Applications")
    print("-" * 40)
    print("• E-commerce: Auto-generate product descriptions")
    print("• Accessibility: Describe images for visually impaired")
    print("• Content moderation: Understand image context")
    print("• Education: Interactive visual learning assistants")
    print("• Healthcare: Medical image analysis with explanations")

    print("\n5. Implementation Tips")
    print("-" * 40)
    print("• Start with BLIP for basic tasks (captioning, VQA)")
    print("• Use BLIP-2 or LLaVA for complex reasoning")
    print("• Consider unified models for production systems")
    print("• Always validate outputs for your use case")
    print("• Use appropriate safety filters for generated content")


def demonstrate_zero_shot_classification():
    """Demonstrate zero-shot image classification with CLIP."""

    print("\n\nZero-Shot Image Classification")
    print("=" * 60)
    print("CLIP enables classification without training on specific classes")

    # This would use the CLIP model already in multimodal_models.py
    print("\nExample: Classifying images into custom categories")
    print("Categories: ['a photo of a cat', 'a photo of a dog', 'a photo of a bird']")
    print("\nCLIP computes similarity between image and each text description")
    print("The highest similarity indicates the predicted class")

    print("\nAdvantages of zero-shot classification:")
    print("• No training required for new categories")
    print("• Works with natural language descriptions")
    print("• Easily adaptable to new domains")
    print("• Can handle fine-grained distinctions")


if __name__ == "__main__":
    demonstrate_advanced_multimodal()
    demonstrate_zero_shot_classification()
