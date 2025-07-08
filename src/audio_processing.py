"""Audio processing examples with Wav2Vec2 and Whisper."""

from transformers import pipeline, AutoProcessor, WhisperForConditionalGeneration, Wav2Vec2ForCTC
import torch
import numpy as np
from config import AUDIO_MODEL, get_device, AUDIO_DIR
import soundfile as sf

def transcribe_audio_pipeline(audio_path, model_name=AUDIO_MODEL):
    """
    Transcribe audio using the pipeline API (recommended).
    
    Args:
        audio_path: Path to audio file
        model_name: Model to use (Whisper, Wav2Vec2, etc.)
    
    Returns:
        Transcribed text
    """
    device = get_device()
    
    # Create pipeline
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=0 if device == "cuda" else -1
    )
    
    # Transcribe
    result = asr(audio_path)
    return result["text"]

def transcribe_audio_manual(audio_path, model_name=AUDIO_MODEL):
    """
    Transcribe audio using AutoProcessor and AutoModel (for learning).
    
    Args:
        audio_path: Path to audio file
        model_name: Model to use
    
    Returns:
        Transcribed text
    """
    device = get_device()
    
    # Load audio
    audio, sampling_rate = sf.read(audio_path)
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Choose the right model class based on model name
    if "whisper" in model_name.lower():
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    if device != "cpu":
        model = model.to(device)
    
    # Process audio
    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )
    
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate transcription
    with torch.no_grad():
        if "whisper" in model_name.lower():
            # Whisper uses generate method
            predicted_ids = model.generate(**inputs)
        else:
            # Wav2Vec2 uses argmax on logits
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

def create_sample_audio():
    """Create a simple test audio file."""
    # Generate a simple sine wave
    duration = 3  # seconds
    sample_rate = 16000
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add some variation
    audio += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)
    
    # Save
    audio_path = AUDIO_DIR / "test_tone.wav"
    sf.write(audio_path, audio, sample_rate)
    
    return audio_path

def demonstrate_audio_processing():
    """Run audio processing demonstrations."""
    
    print("Preparing audio examples...")
    
    # Create sample audio
    test_audio = create_sample_audio()
    print(f"Created test audio: {test_audio}")
    
    # Example 1: Pipeline API (recommended)
    print("\n1. Transcription with Pipeline API:")
    print("   (Note: Test audio is just a tone, expect no meaningful transcription)")
    
    try:
        # For real transcription, use actual speech audio
        print("   Using Whisper model for robust transcription...")
        # transcription = transcribe_audio_pipeline(test_audio, "openai/whisper-base")
        # print(f"   Transcribed: '{transcription}'")
        print("   [Skipping tone transcription - use real speech audio]")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 2: Understanding audio transformers
    print("\n2. How Audio Transformers Work:")
    print("   - Raw audio waveform is input (no manual feature extraction)")
    print("   - Model learns to extract features automatically")
    print("   - Self-attention captures temporal dependencies")
    print("   - End-to-end learning from audio to text")
    
    # Example 3: Model comparison
    print("\n3. Audio Model Comparison:")
    print("   Whisper:")
    print("   - Multilingual, robust to noise")
    print("   - Trained on diverse web audio")
    print("   - Best for general transcription")
    print("\n   Wav2Vec 2.0:")
    print("   - Self-supervised pretraining")
    print("   - Good for fine-tuning on specific domains")
    print("   - Efficient for real-time applications")
    
    print("\nTip: For production use, consider:")
    print("- Streaming inference for real-time transcription")
    print("- Model quantization for faster inference")
    print("- Language-specific models for better accuracy")

if __name__ == "__main__":
    print("=== Audio Processing Examples ===\n")
    demonstrate_audio_processing()
