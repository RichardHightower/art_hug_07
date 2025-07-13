"""Simplified CLIP demonstration without model downloads."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def simulate_clip_embeddings():
    """Simulate CLIP embeddings to demonstrate the concept."""

    # Create sample images
    images = []
    colors = ["red", "green", "blue", "yellow"]
    for color in colors:
        img = Image.new("RGB", (224, 224), color=color)
        images.append(img)

    # Text descriptions
    texts = [
        "a red square",
        "a green square",
        "a blue square",
        "a yellow square",
        "a circle",
        "a triangle",
    ]

    # Simulate embeddings (in reality, CLIP would compute these)
    # Image embeddings - each color gets a distinct embedding
    image_embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # red
            [0.0, 1.0, 0.0, 0.0],  # green
            [0.0, 0.0, 1.0, 0.0],  # blue
            [0.0, 0.0, 0.0, 1.0],  # yellow
        ]
    )

    # Text embeddings - matching texts get similar embeddings
    text_embeddings = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],  # "a red square"
            [0.1, 0.9, 0.0, 0.0],  # "a green square"
            [0.0, 0.1, 0.9, 0.0],  # "a blue square"
            [0.0, 0.0, 0.1, 0.9],  # "a yellow square"
            [0.25, 0.25, 0.25, 0.25],  # "a circle" - unrelated
            [0.25, 0.25, 0.25, 0.25],  # "a triangle" - unrelated
        ]
    )

    # Normalize embeddings (CLIP uses cosine similarity)
    image_embeddings = image_embeddings / np.linalg.norm(
        image_embeddings, axis=1, keepdims=True
    )
    text_embeddings = text_embeddings / np.linalg.norm(
        text_embeddings, axis=1, keepdims=True
    )

    # Compute similarity matrix
    similarity_matrix = np.dot(image_embeddings, text_embeddings.T)

    return images, texts, similarity_matrix, colors


def visualize_clip_concept():
    """Visualize how CLIP works conceptually."""

    images, texts, similarity_matrix, colors = simulate_clip_embeddings()

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Show sample images
    ax1.axis("off")
    ax1.set_title("Sample Images", fontsize=14, fontweight="bold")

    # Create a grid of colored squares
    grid_img = np.zeros((224 * 2, 224 * 2, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        row = i // 2
        col = i % 2
        if color == "red":
            grid_img[row * 224 : (row + 1) * 224, col * 224 : (col + 1) * 224] = [
                255,
                0,
                0,
            ]
        elif color == "green":
            grid_img[row * 224 : (row + 1) * 224, col * 224 : (col + 1) * 224] = [
                0,
                255,
                0,
            ]
        elif color == "blue":
            grid_img[row * 224 : (row + 1) * 224, col * 224 : (col + 1) * 224] = [
                0,
                0,
                255,
            ]
        elif color == "yellow":
            grid_img[row * 224 : (row + 1) * 224, col * 224 : (col + 1) * 224] = [
                255,
                255,
                0,
            ]

    ax1.imshow(grid_img)

    # Show similarity heatmap
    im = ax2.imshow(similarity_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax2.set_title("Image-Text Similarity Matrix", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Text Descriptions")
    ax2.set_ylabel("Images")

    # Set tick labels
    ax2.set_xticks(range(len(texts)))
    ax2.set_xticklabels(texts, rotation=45, ha="right")
    ax2.set_yticks(range(len(colors)))
    ax2.set_yticklabels([f"{c} square" for c in colors])

    # Add colorbar
    plt.colorbar(im, ax=ax2, label="Similarity Score")

    # Add values to heatmap
    for i in range(len(colors)):
        for j in range(len(texts)):
            ax2.text(
                j,
                i,
                f"{similarity_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if similarity_matrix[i, j] < 0.5 else "white",
            )

    plt.tight_layout()

    # Save the plot
    output_path = Path("clip_concept_demo.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path, similarity_matrix


def main():
    """Run the CLIP concept demonstration."""

    print("\n" + "=" * 70)
    print("  CLIP: Contrastive Language-Image Pre-training")
    print("=" * 70 + "\n")

    print("This demonstration shows how CLIP creates shared embeddings")
    print("for images and text, enabling zero-shot classification.\n")

    print("Creating visualization...")
    output_path, similarity_matrix = visualize_clip_concept()

    print(f"\nVisualization saved to: {output_path}")

    print("\nKey Insights:")
    print("• CLIP maps images and text to the same embedding space")
    print("• Similar concepts have high cosine similarity")
    print("• Notice how 'red square' text matches the red image (0.94)")
    print("• Unrelated concepts like 'circle' have low similarity (0.50)")

    print("\nSimilarity Matrix Analysis:")
    print("• Diagonal values are highest (correct matches)")
    print("• Off-diagonal values are lower (incorrect matches)")
    print("• This enables zero-shot image classification!")

    print("\nReal CLIP models:")
    print("• Use vision transformers for image encoding")
    print("• Use text transformers for text encoding")
    print("• Trained on 400M image-text pairs from the internet")
    print("• Enable amazing applications like DALL-E and Stable Diffusion")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
