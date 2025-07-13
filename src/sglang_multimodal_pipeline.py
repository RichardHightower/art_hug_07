"""Complete SGLang multimodal pipeline example from the article."""

import json

# Note: This is a demonstration of the SGLang concepts from the article.
# For actual SGLang usage, install: pip install "sglang[all]>=0.2.0"


def demonstrate_sglang_pipeline():
    """
    Demonstrate the complete multimodal customer support pipeline from the article.
    """

    print("SGLang Multimodal Pipeline: Customer Support Automation")
    print("=" * 70)

    print("\nThis example shows how to build a production multimodal pipeline")
    print("that processes screenshots and voice messages for support tickets.\n")

    # Show the conceptual pipeline code from the article
    pipeline_code = """
import sglang as sgl

# Define model functions with quantization enabled
@sgl.function
def classify_image(s, image):
    s += sgl.image(image)
    s += "What type of customer support issue is shown in this image? "
    s += "Classify as: error, feature_request, or other.\\n"
    s += "Classification: " + sgl.gen("classification", max_tokens=10)

@sgl.function
def transcribe_audio(s, audio):
    # In practice, you'd use a speech-to-text model here
    # For demo, we'll simulate transcription
    s += "Transcribed audio: Customer reporting login issues with error code 403"

@sgl.function
def summarize_support_request(s, image_class, audio_text):
    s += f"Image classification: {image_class}\\n"
    s += f"Audio transcription: {audio_text}\\n"
    s += "Please provide a brief summary of this support request:\\n"
    s += sgl.gen("summary", max_tokens=100)

# Create the pipeline
@sgl.function
def support_pipeline(s, image, audio):
    # Process image
    s_img = classify_image.run(image=image)
    image_class = s_img["classification"]
    
    # Process audio  
    s_audio = transcribe_audio.run(audio=audio)
    audio_text = "Customer reporting login issues with error code 403"
    
    # Combine and summarize
    s = summarize_support_request(s, image_class, audio_text)
    return s

# Runtime configuration with quantization
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    quantization="awq",  # Enable AWQ quantization
    tp_size=1  # Tensor parallelism size
)

# Set global runtime
sgl.set_default_backend(runtime)

# Example usage
# result = support_pipeline.run(image=user_image, audio=user_audio)
# print(result["summary"])
"""

    print("Pipeline Code:")
    print("-" * 70)
    print(pipeline_code)
    print("-" * 70)

    # Simulate pipeline execution
    print("\n\nSimulated Pipeline Execution:")
    print("=" * 70)

    # Simulated inputs
    print("\n1. Input Processing:")
    print("   • Screenshot: error_screenshot.png (showing login error)")
    print("   • Audio: customer_message.wav (reporting access issues)")

    # Simulated processing steps
    print("\n2. Pipeline Steps:")
    print("   a) Image Classification Node:")
    print("      - Input: Screenshot of error dialog")
    print("      - Processing: Vision transformer analyzes image")
    print("      - Output: Classification = 'error'")

    print("\n   b) Audio Transcription Node:")
    print("      - Input: Customer voice message")
    print("      - Processing: Whisper model transcribes audio")
    print("      - Output: 'Customer reporting login issues with error code 403'")

    print("\n   c) Summary Generation Node:")
    print("      - Inputs: Image class + Audio transcript")
    print("      - Processing: LLM generates unified summary")
    print("      - Output: Support ticket summary")

    # Simulated output
    print("\n3. Final Output:")
    print("-" * 70)
    support_summary = {
        "ticket_type": "error",
        "summary": "Customer experiencing login authentication failure (Error 403). "
        "Screenshot shows error dialog when attempting to access account. "
        "Issue appears to be related to authentication permissions.",
        "priority": "high",
        "suggested_actions": [
            "Check user account permissions",
            "Verify authentication service status",
            "Review recent security policy changes",
        ],
    }

    print(json.dumps(support_summary, indent=2))

    # Advanced features
    print("\n\n4. SGLang Advanced Features:")
    print("-" * 70)

    features = {
        "Quantization": {
            "description": "Reduce model memory usage by 4x",
            "options": ["AWQ", "GPTQ", "INT8", "FP8"],
            "benefit": "Run larger models on smaller GPUs",
        },
        "Speculative Decoding": {
            "description": "Use draft model for faster generation",
            "speedup": "2-3x faster inference",
            "benefit": "Lower latency for real-time applications",
        },
        "RadixAttention": {
            "description": "Efficient KV cache sharing",
            "benefit": "Higher throughput for batch processing",
        },
        "Multi-LoRA": {
            "description": "Serve multiple LoRA adapters",
            "benefit": "A/B testing and personalization",
        },
        "Streaming": {
            "description": "Progressive token generation",
            "benefit": "Better user experience",
        },
    }

    for feature, details in features.items():
        print(f"\n{feature}:")
        for key, value in details.items():
            print(f"  • {key.title()}: {value}")

    # Deployment options
    print("\n\n5. Deployment Options:")
    print("-" * 70)

    deployment_commands = """
# Local development
python -m sglang.launch_server \\
    --model-path meta-llama/Llama-2-7b-chat-hf \\
    --port 8080 \\
    --quantization awq

# Docker deployment
docker run -p 8080:8080 \\
    -v ~/.cache:/root/.cache \\
    sglang/sglang:latest \\
    --model-path meta-llama/Llama-2-7b-chat-hf \\
    --quantization awq

# Kubernetes deployment
kubectl apply -f sglang-deployment.yaml
"""

    print(deployment_commands)

    # Production architecture
    print("\n6. Production Architecture:")
    print("-" * 70)
    print(
        """
    Load Balancer
         |
    +---------+
    | SGLang  |
    | Server  |
    +---------+
         |
    +----+----+----+
    |    |    |    |
  Model Model Model Cache
  Node  Node  Node  Layer
    |    |    |      |
    +----+----+------+
         |
    Storage Layer
    """
    )

    print("\n7. Integration Example:")
    print("-" * 70)

    integration_code = '''
# Client code to call SGLang pipeline
import requests

def process_support_ticket(image_path, audio_path):
    """Send support data to SGLang pipeline."""
    
    with open(image_path, 'rb') as img, open(audio_path, 'rb') as aud:
        files = {
            'image': img,
            'audio': aud
        }
        
        response = requests.post(
            'http://localhost:8080/support_pipeline',
            files=files
        )
        
    return response.json()

# Use in your application
result = process_support_ticket(
    'screenshot.png',
    'voice_message.wav'
)
print(result['summary'])
'''

    print(integration_code)

    print("\n\nKey Takeaways:")
    print("-" * 70)
    print("✓ SGLang enables complex multimodal pipelines with simple decorators")
    print("✓ Built-in optimizations (quantization, batching) for production")
    print("✓ Supports text, vision, audio, and custom functions in one graph")
    print("✓ Scales from local development to cloud deployment")
    print("✓ Integrates with existing infrastructure via REST APIs")

    print("\n\nLearn more: https://github.com/sgl-project/sglang")


if __name__ == "__main__":
    demonstrate_sglang_pipeline()
