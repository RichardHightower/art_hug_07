"""Building a multimodal search engine with CLIP."""

from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import requests
from config import MULTIMODAL_MODEL, get_device, IMAGES_DIR

class MultimodalSearchEngine:
    """A simple multimodal search engine using CLIP."""
    
    def __init__(self, model_name=MULTIMODAL_MODEL):
        self.device = get_device()
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        
        self.image_embeddings = None
        self.image_paths = []
    
    def index_images(self, image_folder):
        """
        Index all images in a folder.
        
        Args:
            image_folder: Path to folder containing images
        """
        image_folder = Path(image_folder)
        self.image_paths = list(image_folder.glob("*.jpg")) + \
                          list(image_folder.glob("*.png"))
        
        if not self.image_paths:
            print(f"No images found in {image_folder}")
            return
        
        print(f"Indexing {len(self.image_paths)} images...")
        
        # Load and process images
        images = []
        valid_paths = []
        
        for path in self.image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        self.image_paths = valid_paths
        
        # Compute embeddings
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.image_embeddings = self.model.get_image_features(**inputs)
            # Normalize for cosine similarity
            self.image_embeddings /= self.image_embeddings.norm(dim=-1, keepdim=True)
        
        print(f"Indexed {len(self.image_paths)} images successfully!")
    
    def search(self, query, top_k=5):
        """
        Search for images using a text query.
        
        Args:
            query: Text description
            top_k: Number of results to return
        
        Returns:
            List of (image_path, similarity_score) tuples
        """
        if self.image_embeddings is None:
            raise ValueError("No images indexed. Call index_images() first.")
        
        # Encode text query
        text_inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        if self.device != "cpu":
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (self.image_embeddings @ text_features.T).squeeze(1)
        
        # Get top results
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            path = self.image_paths[idx.item()]
            score = similarities[idx].item()
            results.append((path, score))
        
        return results

def download_sample_images():
    """Download sample images for search demo."""
    sample_urls = {
        "parrot.jpg": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification_parrots.png",
        "cat.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
        "dog.jpg": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
        "sunset.jpg": "https://images.unsplash.com/photo-1495616811223-4d98c6e9c869?w=400",
    }
    
    for filename, url in sample_urls.items():
        filepath = IMAGES_DIR / filename
        if not filepath.exists():
            try:
                img = Image.open(requests.get(url, stream=True).raw)
                img.save(filepath)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

def demonstrate_multimodal_search():
    """Run multimodal search demonstrations."""
    
    print("Building a multimodal search engine...")
    
    # Download sample images
    print("\n1. Preparing sample images...")
    download_sample_images()
    
    # Create search engine
    print("\n2. Creating search engine...")
    search_engine = MultimodalSearchEngine()
    
    # Index images
    print("\n3. Indexing images...")
    search_engine.index_images(IMAGES_DIR)
    
    # Run searches
    print("\n4. Running example searches:")
    
    queries = [
        "a colorful bird",
        "a cute pet",
        "beautiful sunset",
        "animal with feathers",
        "orange and warm colors"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = search_engine.search(query, top_k=3)
        
        for i, (path, score) in enumerate(results, 1):
            print(f"  {i}. {path.name} (similarity: {score:.3f})")
    
    # Explain the technology
    print("\n" + "-" * 50)
    print("\nHow Multimodal Search Works:")
    print("1. Index: Compute embeddings for all images")
    print("2. Query: Convert text to embedding")
    print("3. Search: Find images with similar embeddings")
    print("4. Rank: Return top matches by similarity")
    
    print("\nScaling to production:")
    print("✓ Use vector databases (FAISS, Pinecone, Qdrant)")
    print("✓ Implement caching and batch processing")
    print("✓ Add metadata filtering")
    print("✓ Consider approximate nearest neighbor search")

if __name__ == "__main__":
    print("=== Multimodal Search Engine ===\n")
    demonstrate_multimodal_search()
