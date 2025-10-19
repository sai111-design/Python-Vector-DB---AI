#!/usr/bin/env python3
"""
Day 4: Serialization & Storage - Complete Executable Program
Task: Save embeddings to .npy and reload them for search
"""

import numpy as np
import pickle
import json
import time
from typing import List, Tuple
import os
import pandas as pd

class EmbeddingStorage:
    """Production-ready embedding storage and search system"""

    def __init__(self, storage_dir="embeddings_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.embeddings_file = os.path.join(storage_dir, "embeddings.npy")
        self.metadata_file = os.path.join(storage_dir, "metadata.pkl")

    def save_embeddings(self, embeddings, texts, metadata=None):
        """Save embeddings and associated data"""
        embeddings = np.array(embeddings)
        np.save(self.embeddings_file, embeddings)

        full_metadata = {
            'texts': texts,
            'num_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        if metadata:
            full_metadata.update(metadata)

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(full_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✓ Saved {len(embeddings)} embeddings to {self.storage_dir}")
        return True

    def load_embeddings(self):
        """Load embeddings and metadata"""
        if not os.path.exists(self.embeddings_file) or not os.path.exists(self.metadata_file):
            raise FileNotFoundError("Embedding files not found. Please save embeddings first.")

        embeddings = np.load(self.embeddings_file)
        with open(self.metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        print(f"✓ Loaded {len(embeddings)} embeddings from {self.storage_dir}")
        return embeddings, metadata

    def search(self, query_embedding, top_k=5):
        """Search for similar embeddings"""
        embeddings, metadata = self.load_embeddings()

        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Compute similarities
        similarities = np.dot(embeddings, query_norm)

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'text': metadata['texts'][idx],
                'similarity': float(similarities[idx])
            })

        return results

    def get_info(self):
        """Get information about stored embeddings"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        return None

def create_sample_data():
    """Create sample embeddings for demonstration"""
    print("Creating sample embeddings...")
    np.random.seed(42)  # For reproducibility

    num_embeddings = 100
    embedding_dim = 384

    # Generate normalized random embeddings
    embeddings = np.random.randn(num_embeddings, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create corresponding text samples
    texts = [f"Sample document {i}: This is a sample text for embedding {i}" 
             for i in range(num_embeddings)]

    print(f"Created {num_embeddings} embeddings with dimension {embedding_dim}")
    return embeddings, texts

def demo_serialization():
    """Demonstrate different serialization approaches"""
    print("\n=== Serialization Methods Comparison ===")

    embeddings, texts = create_sample_data()

    # Method 1: NumPy + Pickle
    print("\nTesting NumPy + Pickle approach...")
    start_time = time.time()
    np.save("demo_embeddings.npy", embeddings)
    metadata = {'texts': texts, 'count': len(embeddings)}
    with open("demo_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    save_time = time.time() - start_time

    start_time = time.time()
    loaded_emb = np.load("demo_embeddings.npy")
    with open("demo_metadata.pkl", 'rb') as f:
        loaded_meta = pickle.load(f)
    load_time = time.time() - start_time

    file_size = os.path.getsize("demo_embeddings.npy") + os.path.getsize("demo_metadata.pkl")

    print(f"✓ NumPy method - Save: {save_time:.4f}s, Load: {load_time:.4f}s, Size: {file_size/1024:.2f} KB")

    # Method 2: Pure Pickle
    print("\nTesting Pure Pickle approach...")
    start_time = time.time()
    all_data = {'embeddings': embeddings, 'texts': texts}
    with open("demo_all.pkl", 'wb') as f:
        pickle.dump(all_data, f)
    pickle_save_time = time.time() - start_time

    start_time = time.time()
    with open("demo_all.pkl", 'rb') as f:
        loaded_all = pickle.load(f)
    pickle_load_time = time.time() - start_time

    pickle_size = os.path.getsize("demo_all.pkl")

    print(f"✓ Pickle method - Save: {pickle_save_time:.4f}s, Load: {pickle_load_time:.4f}s, Size: {pickle_size/1024:.2f} KB")

    return embeddings, texts

def demo_search():
    """Demonstrate search functionality"""
    print("\n=== Search Functionality Demo ===")

    # Create storage system
    storage = EmbeddingStorage("demo_storage")

    # Get sample data
    embeddings, texts = create_sample_data()

    # Save embeddings
    storage.save_embeddings(embeddings, texts, {'model': 'demo_model', 'version': '1.0'})

    # Test search
    print("\nPerforming search tests...")

    # Test 1: Search with slight variation of existing embedding
    query_embedding = embeddings[25] + np.random.normal(0, 0.01, embeddings.shape[1])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    results = storage.search(query_embedding, top_k=5)

    print("\nSearch Results (Top 5):")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. [Index: {result['index']:2d}] Similarity: {result['similarity']:.4f}")
        print(f"   Text: {result['text']}")
        print()

    # Display storage info
    info = storage.get_info()
    print(f"Storage Info: {info['num_embeddings']} embeddings, created {info['created_at']}")

    return storage

def cleanup_demo_files():
    """Clean up demonstration files"""
    demo_files = [
        "demo_embeddings.npy", "demo_metadata.pkl", "demo_all.pkl"
    ]

    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)

    # Remove demo storage directory
    import shutil
    if os.path.exists("demo_storage"):
        shutil.rmtree("demo_storage")

def main():
    """Main execution function"""
    print("=" * 60)
    print("        Day 4: Serialization & Storage Demo")
    print("        Task: Save embeddings to .npy and reload for search")
    print("=" * 60)

    try:
        # Run serialization comparison
        embeddings, texts = demo_serialization()

        # Run search demonstration
        storage = demo_search()

        print("\n" + "=" * 60)
        print("✅ Day 4 Task Completed Successfully!")
        print("✅ Embeddings saved and loaded using .npy format")
        print("✅ Metadata handled with pickle serialization")
        print("✅ Search functionality working with cosine similarity")
        print("✅ Production-ready EmbeddingStorage class created")
        print("=" * 60)

        # Ask user if they want to keep demo files
        keep_files = input("\nKeep demo files? (y/n): ").lower().strip()
        if keep_files != 'y':
            cleanup_demo_files()
            print("✓ Demo files cleaned up")
        else:
            print("✓ Demo files preserved in current directory")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
