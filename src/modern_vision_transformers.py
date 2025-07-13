"""Modern Vision Transformer examples: DeiT, Swin, and MaxViT."""

import time

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import SAMPLE_IMAGE_URL, get_device

# Modern vision transformer models as mentioned in the article
VISION_MODELS = {
    "vit": "google/vit-base-patch16-224",
    "deit": "facebook/deit-base-patch16-224",
    "swin": "microsoft/swin-tiny-patch4-window7-224",
    # MaxViT would be: "google/maxvit-tiny-224" if available
}


def compare_vision_transformers(image_path=SAMPLE_IMAGE_URL):
    """
    Compare different vision transformer architectures on the same image.

    Args:
        image_path: Path to image or URL

    Returns:
        Dictionary of results for each model
    """
    device = get_device()
    results = {}

    # Load image once
    if isinstance(image_path, str) and image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    print("Comparing Vision Transformer Architectures")
    print("=" * 60)

    for model_name, model_id in VISION_MODELS.items():
        print(f"\n{model_name.upper()} - {model_id}")
        print("-" * 40)

        try:
            start_time = time.time()

            # Load processor and model
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)

            if device != "cpu":
                model = model.to(device)

            # Preprocess and predict
            inputs = processor(images=image, return_tensors="pt")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Get top 3 predictions
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top3_probs, top3_indices = torch.topk(probs[0], 3)

            inference_time = time.time() - start_time

            results[model_name] = {
                "model_id": model_id,
                "inference_time": inference_time,
                "predictions": [],
            }

            print(f"Inference time: {inference_time:.3f}s")
            print("Top 3 predictions:")

            for i, (prob, idx) in enumerate(
                zip(top3_probs, top3_indices, strict=False)
            ):
                label = model.config.id2label[idx.item()]
                results[model_name]["predictions"].append(
                    {"label": label, "probability": prob.item()}
                )
                print(f"  {i+1}. {label}: {prob.item():.3f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = {"error": str(e)}

    return results


def explain_architectures():
    """Explain the key differences between vision transformer architectures."""

    print("\n" + "=" * 60)
    print("Vision Transformer Architecture Comparison")
    print("=" * 60)

    architectures = {
        "ViT (Vision Transformer)": {
            "key_innovation": "First pure transformer for images",
            "how_it_works": "Splits image into fixed patches, treats as sequence",
            "strengths": "Simple, scalable, good with large datasets",
            "use_cases": "General image classification, foundation for other models",
        },
        "DeiT (Data-efficient Image Transformer)": {
            "key_innovation": "Knowledge distillation from CNNs",
            "how_it_works": "Uses teacher-student training for efficiency",
            "strengths": "Works well with smaller datasets, faster training",
            "use_cases": "When you have limited training data or compute",
        },
        "Swin Transformer": {
            "key_innovation": "Hierarchical + shifted window attention",
            "how_it_works": "Processes image at multiple scales with local windows",
            "strengths": "Efficient for high-res images, good for dense tasks",
            "use_cases": "Object detection, segmentation, any multi-scale task",
        },
        "MaxViT": {
            "key_innovation": "Hybrid local-global attention",
            "how_it_works": "Combines window attention with grid attention",
            "strengths": "Best of both worlds - efficiency + global context",
            "use_cases": "State-of-the-art results on various vision tasks",
        },
    }

    for name, details in architectures.items():
        print(f"\n{name}:")
        print(f"  • Key Innovation: {details['key_innovation']}")
        print(f"  • How it Works: {details['how_it_works']}")
        print(f"  • Strengths: {details['strengths']}")
        print(f"  • Use Cases: {details['use_cases']}")

    print("\n" + "-" * 60)
    print("\nProduction Recommendations:")
    print("• For general use: Start with DeiT or Swin Transformer")
    print("• For efficiency: Use quantized models (INT8/FP16)")
    print("• For accuracy: Ensemble multiple architectures")
    print("• For deployment: Export to ONNX or TensorRT")


def demonstrate_modern_vision_transformers():
    """Run the full demonstration of modern vision transformers."""

    print("Modern Vision Transformers Demonstration")
    print("=" * 60)
    print("\nThis example shows how different vision transformer")
    print("architectures process the same image.\n")

    # Compare models on sample image
    results = compare_vision_transformers()

    # Explain architectures
    explain_architectures()

    # Summary
    print("\n" + "=" * 60)
    print("Summary of Results:")
    print("-" * 60)

    for model_name, result in results.items():
        if "error" not in result:
            print(f"\n{model_name.upper()}:")
            print(f"  Inference time: {result['inference_time']:.3f}s")
            if result["predictions"]:
                top_pred = result["predictions"][0]
                print(
                    f"  Top prediction: {top_pred['label']} "
                    f"({top_pred['probability']:.3f})"
                )


if __name__ == "__main__":
    demonstrate_modern_vision_transformers()
