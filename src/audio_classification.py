"""Audio classification examples from the article."""

from pathlib import Path

from transformers import pipeline

from config import AUDIO_DIR, get_device


def classify_audio_event(audio_path, model="superb/wav2vec2-base-superb-ks"):
    """
    Classify audio events using the pipeline API.

    Args:
        audio_path: Path to audio file
        model: Model to use for classification

    Returns:
        List of predicted labels with scores
    """
    device = get_device()

    # Create audio classification pipeline
    classifier = pipeline(
        "audio-classification", model=model, device=0 if device == "cuda" else -1
    )

    # Classify the audio file
    results = classifier(audio_path)

    return results


def demonstrate_audio_classification():
    """Demonstrate audio classification capabilities."""

    print("Audio Event Classification Example")
    print("=" * 50)

    # Check for sample audio files
    audio_files = list(Path(AUDIO_DIR).glob("*.wav"))

    if not audio_files:
        print("\nNo audio files found in the audio directory.")
        print("Creating a simulated example...")

        # Simulated example
        print("\nExample: classifying 'dog_bark.wav'")
        print("Results:")
        print("  - Label: 'dog_barking', Score: 0.95")
        print("  - Label: 'animal_sounds', Score: 0.03")
        print("  - Label: 'background_noise', Score: 0.02")
    else:
        # Process actual audio files
        for audio_file in audio_files[:2]:  # Process first 2 files
            print(f"\nClassifying: {audio_file.name}")

            try:
                results = classify_audio_event(str(audio_file))
                print("Results:")
                for result in results[:3]:  # Show top 3 predictions
                    print(
                        f"  - Label: '{result['label']}', Score: {result['score']:.3f}"
                    )
            except Exception as e:
                print(f"  Error: {e}")

    print("\n" + "-" * 50)
    print("\nAudio Classification Use Cases:")
    print("✓ Smart home devices - detecting alarms, glass breaking")
    print("✓ Environmental monitoring - wildlife sounds, machinery")
    print("✓ Media analysis - music genre classification")
    print("✓ Accessibility - sound alerts for hearing impaired")

    print("\nAdvanced Audio Models:")
    print("• CLAP - Zero-shot audio classification")
    print("• AudioCLIP - Cross-modal audio-text understanding")
    print("• Audio Spectrogram Transformer (AST)")


if __name__ == "__main__":
    demonstrate_audio_classification()
