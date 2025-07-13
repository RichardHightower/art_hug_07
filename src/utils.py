"""Utility functions for multimodal examples."""

import requests
from PIL import Image

from config import IMAGES_DIR


def download_samples():
    """Download sample images and audio files."""

    # Sample images
    image_urls = {
        "parrot.jpg": "https://datasets-server.huggingface.co/assets/huggingface/documentation-images/--/c5576004ed72e995f79afd8dcc2c6238a9a95f03/--/default/train/1/image/image.jpg?Expires=1752428913&Signature=p1r-anShI1wdBenn3qsmZBgF1n8U3HdWQ8m4Z2DWDi08-FN73cAINSIghNzIf7Iv2WPBpp~PkFy5PbLbKqoIq5jqIw-JVxZ7EClHUR24SdzVwXfLiPMkx8QTkFdPsCTYZnn0PVXrAjbX7yyNoHOoxoIt9TJr-A61ZcZZ809BCnM7pTQantyCfH6NpKQkqxKXQh-da7sFtjrcTZlCAMlvJ7FsodJan8JJcOkYU8uRKBdIhjP-2jPjnWPXaQzP5swZC9MBO4eJR4aMHFkqTd7hy-BYRdCzvI1wUlYZVsavbVYeYLN7w-rfLsiAjvYVGEFmwj~~CQAn6I499FGE3Z4D3A__&Key-Pair-Id=K3EI6M078Z3AC3",
        "cat.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
        "dog.jpg": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
        "sunset.jpg": "https://images.unsplash.com/photo-1495616811223-4d98c6e9c869?w=400",
        "city.jpg": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400",
    }

    print("Downloading sample images...")
    for filename, url in image_urls.items():
        filepath = IMAGES_DIR / filename
        if not filepath.exists():
            try:
                response = requests.get(url, stream=True)
                img = Image.open(response.raw)
                img.save(filepath)
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  ✗ Error downloading {filename}: {e}")
        else:
            print(f"  - {filename} already exists")

    # Note: Audio samples would be downloaded similarly
    print("\nSample download complete!")


if __name__ == "__main__":
    download_samples()
