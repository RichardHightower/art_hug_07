"""Text-to-image generation with Stable Diffusion XL."""

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

from config import DIFFUSION_MODEL, OUTPUT_DIR, get_device


def generate_image(
    prompt,
    negative_prompt="blurry, low quality, distorted",
    model_name=DIFFUSION_MODEL,
    guidance_scale=7.5,
    num_inference_steps=25,
):
    """
    Generate an image from a text prompt using SDXL.

    Args:
        prompt: Text description of desired image
        negative_prompt: What to avoid in the image
        model_name: Diffusion model to use
        guidance_scale: How closely to follow the prompt
        num_inference_steps: Number of denoising steps

    Returns:
        Generated PIL Image
    """
    device = get_device()

    print(f"Loading {model_name}...")
    print(f"Using device: {device}")

    # Load pipeline with optimizations
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device != "cpu" else None,
    )

    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Move to device with memory optimizations
    if device == "cuda":
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
    elif device == "mps":
        pipe = pipe.to("mps")
    else:
        pipe = pipe.to("cpu")

    # Generate image
    print(f"Generating image: '{prompt}'...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    return result.images[0]


def demonstrate_diffusion_models():
    """Run diffusion model demonstrations."""

    print("Demonstrating text-to-image generation with Stable Diffusion XL...")
    print("Note: First run will download several GB of model weights.")

    # Example prompts
    prompts = [
        {
            "prompt": "A serene Japanese garden with cherry blossoms and a wooden "
            "bridge, highly detailed digital art",
            "style": "peaceful landscape",
        },
        {
            "prompt": "A friendly robot teaching mathematics to children in a "
            "colorful classroom, cartoon style",
            "style": "educational illustration",
        },
        {
            "prompt": "An astronaut riding a horse on Mars, photorealistic, "
            "dramatic lighting",
            "style": "surreal concept",
        },
    ]

    print("\nGenerating example images:")

    for i, example in enumerate(prompts):
        print(f"\n{i+1}. {example['style'].title()}:")
        print(f"   Prompt: {example['prompt']}")

        try:
            # Generate image
            image = generate_image(
                example["prompt"], num_inference_steps=25  # Fewer steps for demo
            )

            # Save image
            filename = (
                OUTPUT_DIR / f"generated_{i+1}_{example['style'].replace(' ', '_')}.png"
            )
            image.save(filename)
            print(f"   Saved to: {filename}")

        except Exception as e:
            print(f"   Error: {e}")
            print("   (This often happens on CPU or with limited memory)")

    # Explain the process
    print("\n" + "-" * 50)
    print("\nHow Diffusion Models Work:")
    print("1. Start with random noise")
    print("2. Text prompt is encoded into embeddings")
    print("3. Model gradually 'denoises' the image")
    print("4. Each step refines details guided by the prompt")
    print("5. Result: AI-generated image matching your description")

    print("\nTips for better results:")
    print("✓ Be specific and descriptive in prompts")
    print("✓ Use negative prompts to avoid unwanted features")
    print("✓ Experiment with guidance_scale (7-12 usually works well)")
    print("✓ Try different artistic styles and modifiers")
    print("✓ Use SDXL for highest quality, SD 1.5 for speed")


if __name__ == "__main__":
    print("=== Diffusion Model Examples ===\n")
    demonstrate_diffusion_models()
