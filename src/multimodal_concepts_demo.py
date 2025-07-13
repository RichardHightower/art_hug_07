"""Comprehensive demonstration of multimodal AI concepts without large downloads."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def create_vision_transformer_visualization():
    """Visualize how Vision Transformers process images."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original Image
    ax1 = axes[0]
    img = np.zeros((224, 224, 3))
    # Create a simple pattern
    img[50:100, 50:100] = [1, 0, 0]  # Red square
    img[120:170, 120:170] = [0, 1, 0]  # Green square
    img[50:100, 120:170] = [0, 0, 1]  # Blue square

    ax1.imshow(img)
    ax1.set_title("1. Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. Patch Division
    ax2 = axes[1]
    ax2.imshow(img)
    ax2.set_title("2. Divided into 16x16 Patches", fontsize=12, fontweight="bold")

    # Draw patch grid
    patch_size = 16
    for i in range(0, 224, patch_size):
        ax2.axhline(y=i, color="white", linewidth=0.5)
        ax2.axvline(x=i, color="white", linewidth=0.5)
    ax2.axis("off")

    # 3. Attention Visualization
    ax3 = axes[2]
    attention_map = np.random.rand(14, 14)
    # Make certain patches have higher attention
    attention_map[3:6, 3:6] = 0.9  # High attention on red square
    attention_map[7:10, 7:10] = 0.8  # High attention on green square

    im = ax3.imshow(attention_map, cmap="hot", interpolation="nearest")
    ax3.set_title("3. Self-Attention Map", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Patch Position")
    ax3.set_ylabel("Patch Position")

    plt.colorbar(im, ax=ax3, label="Attention Weight")

    plt.tight_layout()
    plt.savefig("vision_transformer_concept.png", dpi=150, bbox_inches="tight")
    plt.close()

    return "vision_transformer_concept.png"


def create_multimodal_pipeline_visualization():
    """Visualize a multimodal AI pipeline."""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        5, 9.5, "Multimodal AI Pipeline", fontsize=16, fontweight="bold", ha="center"
    )

    # Input modalities
    inputs = [
        {"name": "Image\nInput", "pos": (2, 7), "color": "lightblue"},
        {"name": "Text\nInput", "pos": (5, 7), "color": "lightgreen"},
        {"name": "Audio\nInput", "pos": (8, 7), "color": "lightyellow"},
    ]

    for inp in inputs:
        rect = patches.FancyBboxPatch(
            (inp["pos"][0] - 0.8, inp["pos"][1] - 0.4),
            1.6,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor=inp["color"],
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            inp["pos"][0],
            inp["pos"][1],
            inp["name"],
            ha="center",
            va="center",
            fontsize=10,
        )

    # Encoders
    encoders = [
        {"name": "Vision\nEncoder\n(ViT)", "pos": (2, 5), "color": "lightcoral"},
        {"name": "Text\nEncoder\n(BERT)", "pos": (5, 5), "color": "lightcoral"},
        {"name": "Audio\nEncoder\n(Whisper)", "pos": (8, 5), "color": "lightcoral"},
    ]

    for enc in encoders:
        rect = patches.FancyBboxPatch(
            (enc["pos"][0] - 0.8, enc["pos"][1] - 0.4),
            1.6,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor=enc["color"],
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            enc["pos"][0],
            enc["pos"][1],
            enc["name"],
            ha="center",
            va="center",
            fontsize=9,
        )

    # Shared embedding space
    rect = patches.FancyBboxPatch(
        (2, 2.5),
        6,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor="lavender",
        edgecolor="purple",
        linewidth=3,
    )
    ax.add_patch(rect)
    ax.text(
        5,
        3.1,
        "Shared Embedding Space",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Applications
    apps = [
        {"name": "Image-Text\nSearch", "pos": (1.5, 1)},
        {"name": "Visual Q&A", "pos": (3.5, 1)},
        {"name": "Caption\nGeneration", "pos": (5.5, 1)},
        {"name": "Cross-Modal\nRetrieval", "pos": (7.5, 1)},
        {"name": "Zero-Shot\nClassification", "pos": (9, 1)},
    ]

    for app in apps:
        rect = patches.FancyBboxPatch(
            (app["pos"][0] - 0.6, app["pos"][1] - 0.3),
            1.2,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor="lightgreen",
            edgecolor="darkgreen",
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(
            app["pos"][0],
            app["pos"][1],
            app["name"],
            ha="center",
            va="center",
            fontsize=8,
        )

    # Draw arrows
    # Inputs to encoders
    for _, (inp, _) in enumerate(zip(inputs, encoders, strict=False)):
        ax.arrow(
            inp["pos"][0],
            inp["pos"][1] - 0.4,
            0,
            -0.8,
            head_width=0.1,
            head_length=0.1,
            fc="black",
            ec="black",
        )

    # Encoders to embedding space
    for enc in encoders:
        ax.arrow(
            enc["pos"][0],
            enc["pos"][1] - 0.4,
            0,
            -0.8,
            head_width=0.1,
            head_length=0.1,
            fc="purple",
            ec="purple",
        )

    # Embedding space to applications
    for app in apps:
        ax.arrow(
            app["pos"][0],
            2.5,
            0,
            -1.0,
            head_width=0.05,
            head_length=0.05,
            fc="darkgreen",
            ec="darkgreen",
            linestyle="--",
            linewidth=1,
        )

    plt.savefig("multimodal_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()

    return "multimodal_pipeline.png"


def create_diffusion_process_visualization():
    """Visualize the diffusion process for image generation."""

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    # Simulate diffusion process
    steps = [1000, 750, 500, 250, 0]

    for _, (ax, step) in enumerate(zip(axes, steps, strict=False)):
        # Create progressively less noisy images
        if step == 1000:
            # Pure noise
            img = np.random.randn(64, 64, 3) * 0.5 + 0.5
        elif step == 0:
            # Final image (simple geometric shapes)
            img = np.ones((64, 64, 3)) * 0.9
            img[20:40, 20:40] = [0.2, 0.3, 0.8]  # Blue square
            img[15:25, 40:50] = [0.8, 0.2, 0.2]  # Red rectangle
        else:
            # Intermediate steps
            noise_level = step / 1000
            clean = np.ones((64, 64, 3)) * 0.9
            clean[20:40, 20:40] = [0.2, 0.3, 0.8]
            clean[15:25, 40:50] = [0.8, 0.2, 0.2]
            noise = np.random.randn(64, 64, 3) * 0.3
            img = clean * (1 - noise_level) + noise * noise_level

        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Step {step}", fontsize=10)
        ax.axis("off")

    fig.suptitle(
        "Diffusion Process: From Noise to Image", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("diffusion_process.png", dpi=150, bbox_inches="tight")
    plt.close()

    return "diffusion_process.png"


def create_model_comparison_chart():
    """Create a comparison chart of different multimodal models."""

    models = ["CLIP", "BLIP", "BLIP-2", "LLaVA", "Flamingo"]
    capabilities = [
        "Image-Text\nMatching",
        "Image\nCaptioning",
        "Visual\nQ&A",
        "Few-Shot\nLearning",
        "Instruction\nFollowing",
    ]

    # Capability matrix (1 = full support, 0.5 = partial, 0 = none)
    capability_matrix = np.array(
        [
            [1.0, 0.5, 0.3, 0.0, 0.0],  # CLIP
            [1.0, 1.0, 0.8, 0.0, 0.5],  # BLIP
            [1.0, 1.0, 1.0, 0.3, 0.7],  # BLIP-2
            [0.8, 1.0, 1.0, 0.5, 1.0],  # LLaVA
            [1.0, 1.0, 1.0, 1.0, 0.8],  # Flamingo
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(capability_matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(capabilities)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(capabilities)
    ax.set_yticklabels(models)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Capability Level", rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(capabilities)):
            ax.text(
                j,
                i,
                f"{capability_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black" if capability_matrix[i, j] < 0.5 else "white",
            )

    ax.set_title(
        "Multimodal Model Capabilities Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    return "model_comparison.png"


def main():
    """Run all concept demonstrations."""

    print("\n" + "=" * 70)
    print("  MULTIMODAL AI: COMPREHENSIVE CONCEPT DEMONSTRATION")
    print("=" * 70 + "\n")

    print("This demonstration visualizes key concepts in multimodal AI")
    print("without requiring large model downloads.\n")

    # Vision Transformers
    print("1. Creating Vision Transformer visualization...")
    vit_path = create_vision_transformer_visualization()
    print(f"   ✓ Saved to: {vit_path}")

    # Multimodal Pipeline
    print("\n2. Creating Multimodal Pipeline visualization...")
    pipeline_path = create_multimodal_pipeline_visualization()
    print(f"   ✓ Saved to: {pipeline_path}")

    # Diffusion Process
    print("\n3. Creating Diffusion Process visualization...")
    diffusion_path = create_diffusion_process_visualization()
    print(f"   ✓ Saved to: {diffusion_path}")

    # Model Comparison
    print("\n4. Creating Model Comparison chart...")
    comparison_path = create_model_comparison_chart()
    print(f"   ✓ Saved to: {comparison_path}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)

    print("\n1. Vision Transformers:")
    print("   • Split images into patches (typically 16x16)")
    print("   • Process patches as sequences like NLP transformers")
    print("   • Self-attention learns relationships between patches")

    print("\n2. Multimodal Models:")
    print("   • Encode different modalities into shared space")
    print("   • Enable cross-modal tasks (search, Q&A, generation)")
    print("   • Foundation for many AI applications")

    print("\n3. Diffusion Models:")
    print("   • Generate images by denoising random noise")
    print("   • Iterative refinement process")
    print("   • Can be conditioned on text, images, etc.")

    print("\n4. Model Evolution:")
    print("   • CLIP: Pioneered image-text alignment")
    print("   • BLIP/BLIP-2: Added generation capabilities")
    print("   • LLaVA: Instruction-following visual assistant")
    print("   • Each builds on previous innovations")

    print("\n" + "=" * 70)
    print("All visualizations have been saved to the current directory.")
    print("For hands-on experimentation, see the Jupyter notebook!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
