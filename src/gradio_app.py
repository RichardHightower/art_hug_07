"""Interactive Gradio app for multimodal AI demos."""

import gradio as gr
from PIL import Image

from config import get_device
from diffusion_models import generate_image
from multimodal_models import find_best_match
from vision_transformers import classify_image_with_model


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="Multimodal AI Demo") as demo:
        gr.Markdown(
            """
            # 🤖 Multimodal AI: Vision, Audio & Generation
            
            Explore how transformers work with images, audio, and 
            cross-modal understanding!
            """
        )

        with gr.Tab("Vision Transformers"):
            gr.Markdown(
                "### Classify images using state-of-the-art vision transformers"
            )

            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Image")
                model_choice = gr.Dropdown(
                    choices=[
                        "google/vit-base-patch16-224",
                        "facebook/deit-base-patch16-224",
                        "microsoft/swin-tiny-patch4-window7-224",
                    ],
                    value="google/vit-base-patch16-224",
                    label="Choose Model",
                )

            classify_btn = gr.Button("Classify Image", variant="primary")
            classification_output = gr.Textbox(label="Classification Result")

            def classify_wrapper(img, model):
                if img is None:
                    return "Please upload an image"
                try:
                    label, conf = classify_image_with_model(img, model)
                    return f"Predicted: {label} (confidence: {conf:.2%})"
                except Exception as e:
                    return f"Error: {str(e)}"

            classify_btn.click(
                classify_wrapper,
                inputs=[image_input, model_choice],
                outputs=classification_output,
            )

        with gr.Tab("CLIP Search"):
            gr.Markdown("### Find best matching description for an image")

            clip_image = gr.Image(type="pil", label="Upload Image")
            clip_queries = gr.Textbox(
                label="Enter descriptions (one per line)",
                placeholder="a happy dog\na sad cat\na beautiful sunset\na city street",
                lines=4,
            )

            clip_btn = gr.Button("Find Best Match", variant="primary")
            clip_output = gr.Textbox(label="Results")

            def clip_search(img, queries):
                if img is None or not queries:
                    return "Please provide both image and descriptions"
                try:
                    query_list = [q.strip() for q in queries.split("\n") if q.strip()]
                    best, conf = find_best_match(img, query_list)
                    return f"Best match: '{best}' (confidence: {conf:.2%})"
                except Exception as e:
                    return f"Error: {str(e)}"

            clip_btn.click(
                clip_search, inputs=[clip_image, clip_queries], outputs=clip_output
            )

        with gr.Tab("Image Generation"):
            gr.Markdown("### Generate images from text descriptions")
            gr.Markdown("⚠️ Note: This requires significant GPU memory")

            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="A serene mountain landscape at sunset, digital art",
                lines=2,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt", value="blurry, low quality, distorted", lines=2
            )

            generate_btn = gr.Button("Generate Image", variant="primary")
            generated_image = gr.Image(label="Generated Image")

            def generate_wrapper(prompt, neg_prompt):
                if not prompt:
                    return None
                try:
                    # Use fewer steps for demo
                    img = generate_image(prompt, neg_prompt, num_inference_steps=20)
                    return img
                except Exception:
                    # Return error as image
                    error_img = Image.new("RGB", (512, 512), color="black")
                    return error_img

            generate_btn.click(
                generate_wrapper,
                inputs=[prompt_input, negative_prompt],
                outputs=generated_image,
            )

        gr.Markdown(
            f"""
            ---
            ### About This Demo
            
            This interactive demo showcases:
            - **Vision Transformers**: ViT, DeiT, and Swin for image classification
            - **CLIP**: Cross-modal understanding between text and images
            - **Stable Diffusion**: Text-to-image generation
            
            Device: {get_device().upper()}
            
            Learn more in the [Hugging Face documentation](https://huggingface.co/docs).
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
