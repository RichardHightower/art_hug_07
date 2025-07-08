"""Advanced multimodal search implementation from the article."""

from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import os
from config import get_device, IMAGES_DIR

class AdvancedMultimodalSearch:
    """
    Complete multimodal search engine implementation following the article examples.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        self.device = get_device()
        self.model_name = model_name
        
        print(f"Initializing multimodal search with {model_name}")
        print(f"Using device: {self.device}")
        
        # Load model and processor using Auto classes as recommended
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        
        # Storage for embeddings
        self.image_features = None
        self.image_files = []
        
    def embed_images_batch(self, image_folder):
        """
        Step 1 from article: Embed a collection of images
        
        Args:
            image_folder: Path to folder containing images
        """
        image_folder = Path(image_folder)
        
        # Collect image file paths
        self.image_files = [
            os.path.join(image_folder, fname) 
            for fname in os.listdir(image_folder) 
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not self.image_files:
            print(f"No images found in {image_folder}")
            return
        
        print(f"\nEmbedding {len(self.image_files)} images...")
        
        # Load and preprocess images
        images = []
        valid_files = []
        
        for f in self.image_files:
            try:
                img = Image.open(f).convert("RGB")
                images.append(img)
                valid_files.append(f)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        self.image_files = valid_files
        
        if not images:
            print("No valid images to process")
            return
        
        # Process in batches for efficiency
        batch_size = 8
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Compute image embeddings
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features /= features.norm(dim=-1, keepdim=True)  # Normalize
                all_features.append(features.cpu())
        
        # Combine all features
        self.image_features = torch.cat(all_features, dim=0)
        print(f"Successfully embedded {len(self.image_files)} images")
        
    def search_by_text(self, text_query, top_k=5):
        """
        Step 2 from article: Query with text and retrieve relevant images
        
        Args:
            text_query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of (filename, similarity_score) tuples
        """
        if self.image_features is None:
            print("No images indexed. Please run embed_images_batch first.")
            return []
        
        print(f"\nSearching for: '{text_query}'")
        
        # Preprocess and embed the text
        text_inputs = self.processor(
            text=[text_query], 
            return_tensors="pt", 
            padding=True
        )
        
        if self.device != "cpu":
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize
            text_features = text_features.cpu()
        
        # Compute cosine similarities
        similarities = (self.image_features @ text_features.T).squeeze(1)
        
        # Get top-k results
        top_k = min(top_k, len(self.image_files))
        values, indices = similarities.topk(top_k)
        
        results = []
        for idx, score in zip(indices, values):
            results.append((self.image_files[idx], score.item()))
        
        return results
    
    def search_by_image(self, query_image_path, top_k=5):
        """
        Bonus: Search for similar images using an image query
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
            
        Returns:
            List of (filename, similarity_score) tuples
        """
        if self.image_features is None:
            print("No images indexed. Please run embed_images_batch first.")
            return []
        
        # Load and process query image
        query_image = Image.open(query_image_path).convert("RGB")
        inputs = self.processor(images=[query_image], return_tensors="pt")
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            query_features = self.model.get_image_features(**inputs)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_features = query_features.cpu()
        
        # Compute similarities
        similarities = (self.image_features @ query_features.T).squeeze(1)
        
        # Get top-k results (excluding the query image if it's in the index)
        values, indices = similarities.topk(min(top_k + 1, len(self.image_files)))
        
        results = []
        for idx, score in zip(indices, values):
            if self.image_files[idx] != query_image_path:
                results.append((self.image_files[idx], score.item()))
                if len(results) >= top_k:
                    break
        
        return results

def demonstrate_multimodal_search_engine():
    """Complete demonstration of building a multimodal search engine."""
    
    print("Building a Multimodal Search Engine with CLIP")
    print("=" * 70)
    
    # Initialize search engine
    search_engine = AdvancedMultimodalSearch()
    
    # Create sample data if needed
    sample_dir = Path(IMAGES_DIR)
    if not sample_dir.exists():
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated image directory: {sample_dir}")
        print("Add some images to test the search engine!")
    
    # Index images
    print(f"\nStep 1: Indexing images from {sample_dir}")
    search_engine.embed_images_batch(sample_dir)
    
    if search_engine.image_features is not None:
        # Example text queries
        print("\nStep 2: Text-based search examples")
        print("-" * 50)
        
        example_queries = [
            "a smiling person wearing sunglasses on the beach",
            "a red car on a city street",
            "nature landscape with mountains",
            "food on a plate",
            "computer or technology"
        ]
        
        for query in example_queries:
            results = search_engine.search_by_text(query, top_k=3)
            print(f"\nQuery: '{query}'")
            if results:
                print("Top matches:")
                for i, (filename, score) in enumerate(results, 1):
                    print(f"  {i}. {Path(filename).name} (score: {score:.3f})")
            else:
                print("  No results found")
    
    # Scaling considerations
    print("\n\nScaling to Production")
    print("=" * 70)
    
    print("\nFor small collections (< 10,000 images):")
    print("• In-memory search (as shown above) works well")
    print("• Consider caching embeddings to disk")
    
    print("\nFor large collections (> 10,000 images):")
    print("• Use a vector database:")
    print("  - FAISS: Facebook's similarity search library")
    print("  - Milvus: Purpose-built vector database")
    print("  - Pinecone: Managed vector database service")
    print("  - Qdrant: Open-source vector search engine")
    
    print("\nExample: Integrating with FAISS")
    print("-" * 50)
    
    faiss_example = '''
import faiss

# After computing image_features
dimension = image_features.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product index

# Add embeddings to index
index.add(image_features.numpy())

# Search
text_embedding = compute_text_embedding(query)
distances, indices = index.search(text_embedding.numpy(), k=5)

# Retrieve corresponding filenames
results = [(image_files[idx], dist) for idx, dist in zip(indices[0], distances[0])]
'''
    
    print(faiss_example)
    
    print("\nBusiness Applications")
    print("=" * 70)
    
    applications = {
        "E-commerce": [
            "Natural language product search",
            "Visual similarity recommendations",
            "Inventory management"
        ],
        "Digital Asset Management": [
            "Search large media libraries",
            "Auto-tag and organize content",
            "Find duplicates or similar assets"
        ],
        "Content Moderation": [
            "Flag inappropriate content",
            "Policy compliance checking",
            "Brand safety monitoring"
        ],
        "Creative Industries": [
            "Mood board creation",
            "Style matching",
            "Inspiration discovery"
        ]
    }
    
    for industry, uses in applications.items():
        print(f"\n{industry}:")
        for use in uses:
            print(f"  • {use}")
    
    print("\n\nNext Steps")
    print("=" * 70)
    print("1. Add more images to your search index")
    print("2. Experiment with different CLIP models (larger = better)")
    print("3. Implement filtering (by date, category, etc.)")
    print("4. Add a web interface with Gradio or Streamlit")
    print("5. Deploy with vector database for scale")

if __name__ == "__main__":
    demonstrate_multimodal_search_engine()