#!/usr/bin/env python3
"""
Day 5: Complete Implementation - Embeddings Basics with SentenceTransformers
Task: Generate embeddings for 20 sentences, compare with cosine similarity
"""

import numpy as np
import pandas as pd
import time
from typing import List, Tuple
import json

def create_sample_sentences():
    """Create 20 diverse sentences for embedding generation"""
    sentences = [
        # Technology and Programming
        "Python is a powerful programming language for data science",
        "Machine learning algorithms can process vast amounts of data",
        "Artificial intelligence is transforming modern society",

        # Weather and Nature
        "The weather is beautiful and sunny today",
        "It's raining heavily outside with thunder and lightning",
        "Spring brings colorful flowers and fresh green leaves",

        # Food and Cooking
        "I love cooking Italian pasta with fresh tomatoes",
        "The chocolate cake tastes incredibly sweet and delicious",
        "Healthy vegetables provide essential vitamins and nutrients",

        # Sports and Activities
        "Playing soccer requires teamwork and physical fitness",
        "Running in the morning gives me energy for the day",
        "Swimming is an excellent full-body cardiovascular exercise",

        # Travel and Places
        "Paris is known for its beautiful architecture and museums",
        "The mountains offer breathtaking views and fresh air",
        "Tropical beaches have crystal clear water and white sand",

        # Education and Learning
        "Reading books expands knowledge and improves vocabulary",
        "Students learn best through hands-on practical experience",
        "Online education makes learning accessible to everyone",

        # Music and Arts
        "Classical music creates a peaceful and relaxing atmosphere",
        "Modern art expresses creativity through bold colors and shapes"
    ]

    return sentences

def analyze_embeddings(embeddings, sentences):
    """Analyze the properties of generated embeddings"""
    print("\n📊 Embedding Analysis:")
    print("-" * 50)

    print(f"Number of sentences: {len(sentences)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Data type: {embeddings.dtype}")

    # Statistical analysis
    print(f"\nStatistical Properties:")
    print(f"Mean value: {np.mean(embeddings):.6f}")
    print(f"Standard deviation: {np.std(embeddings):.6f}")
    print(f"Min value: {np.min(embeddings):.6f}")
    print(f"Max value: {np.max(embeddings):.6f}")

    # Check if embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms (should be ~1 if normalized):")
    print(f"Mean norm: {np.mean(norms):.6f}")
    print(f"Std norm: {np.std(norms):.6f}")
    print(f"Min norm: {np.min(norms):.6f}")
    print(f"Max norm: {np.max(norms):.6f}")

def compute_similarity_metrics(embeddings, model=None):
    """Compute various similarity metrics"""
    print("\n🔍 Computing Similarity Metrics:")
    print("-" * 50)

    # Method 1: Using model.similarity (recommended)
    if model:
        start_time = time.time()
        cosine_similarities = model.similarity(embeddings, embeddings)
        cosine_time = time.time() - start_time
        print(f"✅ Model.similarity(): {cosine_time:.4f} seconds")

    # Method 2: Manual cosine similarity
    start_time = time.time()
    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Compute cosine similarity (dot product of normalized vectors)
    manual_cosine = np.dot(normalized_embeddings, normalized_embeddings.T)
    manual_time = time.time() - start_time
    print(f"✅ Manual cosine similarity: {manual_time:.4f} seconds")

    # Method 3: Dot product (for normalized embeddings, same as cosine)
    start_time = time.time()
    dot_similarities = np.dot(embeddings, embeddings.T)
    dot_time = time.time() - start_time
    print(f"✅ Dot product similarity: {dot_time:.4f} seconds")

    # Method 4: Euclidean distance (convert to similarity)
    start_time = time.time()
    from scipy.spatial.distance import pdist, squareform
    euclidean_distances = squareform(pdist(embeddings, metric='euclidean'))
    # Convert distances to similarities (higher distance = lower similarity)
    max_distance = np.max(euclidean_distances)
    euclidean_similarities = 1 - (euclidean_distances / max_distance)
    euclidean_time = time.time() - start_time
    print(f"✅ Euclidean similarity: {euclidean_time:.4f} seconds")

    return {
        'cosine': cosine_similarities if model else manual_cosine,
        'manual_cosine': manual_cosine,
        'dot_product': dot_similarities,
        'euclidean': euclidean_similarities
    }

def find_similar_sentences(similarities, sentences, metric_name="cosine", top_k=5):
    """Find and display most similar sentence pairs"""
    print(f"\n🔗 Top {top_k} Most Similar Sentence Pairs ({metric_name}):")
    print("-" * 70)

    # Get all pairwise similarities (excluding diagonal)
    pairs = []
    n = len(sentences)

    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle to avoid duplicates
            similarity = similarities[i][j].item() if hasattr(similarities[i][j], 'item') else similarities[i][j]
            pairs.append((similarity, i, j))

    # Sort by similarity score
    pairs.sort(reverse=True)

    # Display top pairs
    for rank, (score, i, j) in enumerate(pairs[:top_k], 1):
        print(f"{rank}. Similarity: {score:.4f}")
        print(f"   Sentence {i}: {sentences[i]}")
        print(f"   Sentence {j}: {sentences[j]}")
        print()

    return pairs[:top_k]

def create_similarity_heatmap_data(similarities, sentences):
    """Create data for similarity heatmap visualization"""
    # Convert to numpy if tensor
    if hasattr(similarities, 'cpu'):
        similarities_np = similarities.cpu().numpy()
    else:
        similarities_np = np.array(similarities)

    # Create DataFrame for better visualization
    df = pd.DataFrame(
        similarities_np,
        index=[f"S{i}" for i in range(len(sentences))],
        columns=[f"S{i}" for i in range(len(sentences))]
    )

    return df

def semantic_clustering_analysis(embeddings, sentences):
    """Analyze semantic clusters in the embeddings"""
    print("\n🎯 Semantic Clustering Analysis:")
    print("-" * 50)

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Try different numbers of clusters
        best_k = 2
        best_score = -1

        for k in range(2, min(8, len(sentences) // 2)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)

            if score > best_score:
                best_score = score
                best_k = k

        # Apply best clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        print(f"Optimal number of clusters: {best_k}")
        print(f"Silhouette score: {best_score:.4f}")

        # Group sentences by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, sentences[i]))

        print("\nClusters:")
        for cluster_id, sentence_list in clusters.items():
            print(f"\nCluster {cluster_id}:")
            for idx, sentence in sentence_list:
                print(f"  {idx}: {sentence}")

        return labels, clusters

    except ImportError:
        print("sklearn not available - skipping clustering analysis")
        return None, None

def save_results(embeddings, similarities, sentences, analysis_results):
    """Save analysis results to files"""
    print("\n💾 Saving Results:")
    print("-" * 30)

    # Save embeddings
    np.save('day5_embeddings.npy', embeddings)
    print("✅ Saved embeddings to day5_embeddings.npy")

    # Save sentences and metadata
    metadata = {
        'sentences': sentences,
        'num_sentences': len(sentences),
        'embedding_dimension': embeddings.shape[1],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_used': 'all-MiniLM-L6-v2'
    }

    with open('day5_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved metadata to day5_metadata.json")

    # Save similarity matrix
    similarities_np = similarities.cpu().numpy() if hasattr(similarities, 'cpu') else np.array(similarities)
    df_similarity = pd.DataFrame(
        similarities_np,
        index=[f"Sentence_{i}" for i in range(len(sentences))],
        columns=[f"Sentence_{i}" for i in range(len(sentences))]
    )
    df_similarity.to_csv('day5_similarity_matrix.csv')
    print("✅ Saved similarity matrix to day5_similarity_matrix.csv")

    # Save sentence pairs analysis
    pairs_data = []
    n = len(sentences)
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarities_np[i][j]
            pairs_data.append({
                'sentence_1_id': i,
                'sentence_2_id': j,
                'sentence_1': sentences[i],
                'sentence_2': sentences[j],
                'cosine_similarity': similarity
            })

    df_pairs = pd.DataFrame(pairs_data)
    df_pairs = df_pairs.sort_values('cosine_similarity', ascending=False)
    df_pairs.to_csv('day5_sentence_pairs.csv', index=False)
    print("✅ Saved sentence pairs to day5_sentence_pairs.csv")

def main():
    """Main execution function for Day 5"""
    print("=" * 70)
    print("          Day 5: Embeddings Basics with SentenceTransformers")
    print("          Task: Generate embeddings for 20 sentences")
    print("=" * 70)

    try:
        # Import required libraries
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers imported successfully")

        # Load model
        print("\n🤖 Loading SentenceTransformer model...")
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        print(f"✅ Model '{model_name}' loaded successfully")
        print(f"✅ Model device: {model.device}")

        # Create sample sentences
        sentences = create_sample_sentences()
        print(f"\n📝 Created {len(sentences)} sample sentences")

        # Generate embeddings
        print("\n🔄 Generating embeddings...")
        start_time = time.time()
        embeddings = model.encode(sentences)
        embedding_time = time.time() - start_time

        print(f"✅ Embeddings generated in {embedding_time:.4f} seconds")
        print(f"✅ Embedding shape: {embeddings.shape}")

        # Analyze embeddings
        analyze_embeddings(embeddings, sentences)

        # Compute similarity metrics
        similarities_dict = compute_similarity_metrics(embeddings, model)
        main_similarities = similarities_dict['cosine']

        # Find similar sentences
        top_pairs = find_similar_sentences(main_similarities, sentences, "cosine", top_k=5)

        # Create heatmap data
        heatmap_df = create_similarity_heatmap_data(main_similarities, sentences)
        print(f"\n📈 Similarity matrix created: {heatmap_df.shape}")

        # Semantic clustering
        labels, clusters = semantic_clustering_analysis(embeddings, sentences)

        # Save all results
        save_results(embeddings, main_similarities, sentences, {
            'top_pairs': top_pairs,
            'clusters': clusters
        })

        print("\n" + "=" * 70)
        print("✅ Day 5 Task Completed Successfully!")
        print("=" * 70)
        print("🎯 What we accomplished:")
        print("✅ Generated embeddings for 20 diverse sentences")
        print("✅ Computed cosine similarity between all sentence pairs")
        print("✅ Analyzed embedding properties and statistics")
        print("✅ Identified most semantically similar sentence pairs")
        print("✅ Performed clustering analysis on embeddings")
        print("✅ Saved all results to files for further analysis")

        print("\n📊 Key Insights:")
        print(f"• Embedding dimension: {embeddings.shape[1]}")
        print(f"• Processing time: {embedding_time:.4f} seconds")
        print(f"• Highest similarity: {top_pairs[0][0]:.4f}")
        print(f"• Model used: {model_name}")

        print("\n🎓 Skills Learned:")
        print("• Loading and using SentenceTransformer models")
        print("• Generating high-quality sentence embeddings")
        print("• Computing and interpreting cosine similarity")
        print("• Analyzing semantic relationships in vector space")
        print("• Saving embeddings for downstream tasks")

        print("\n➡️  Ready for Day 6: ANN Libraries (FAISS)")

        # Ask to clean up files
        cleanup = input("\nKeep generated files? (y/n): ").lower().strip()
        if cleanup != 'y':
            import os
            files_to_remove = [
                'day5_embeddings.npy', 'day5_metadata.json',
                'day5_similarity_matrix.csv', 'day5_sentence_pairs.csv'
            ]
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
            print("✅ Temporary files cleaned up")
        else:
            print("✅ Files preserved for further analysis")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 To fix:")
        print("1. Run: python setup_day5.py")
        print("2. Or manually: pip install sentence-transformers")
        return False

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("\n🔧 Try:")
        print("1. Check internet connection (for model download)")
        print("2. Ensure sufficient disk space (~500MB)")
        print("3. Run: python test_day5.py")
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
