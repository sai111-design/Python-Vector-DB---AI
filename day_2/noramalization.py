# Day 2 Workflow: Load CSV → Normalize vectors → Compute cosine similarity manually with NumPy

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

print("=" * 60)
print("DAY 2 WORKFLOW: CSV LOADING → VECTOR NORMALIZATION → COSINE SIMILARITY")
print("=" * 60)

# STEP 1: Load CSV with proper error handling
print("\n1. LOADING CSV DATA")
print("-" * 30)

try:
    # Load the CSV file
    df = pd.read_csv(r'S:/daily_prep/day_2/sample_vectors.csv')

    print(f"✓ Successfully loaded CSV with shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"✓ Missing values: {missing_count}")
    
    # Display first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
except FileNotFoundError:
    print("❌ Error: CSV file not found!")
    exit()
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# STEP 2: Extract vector columns and handle missing data
print("\n2. EXTRACTING VECTOR DATA")
print("-" * 30)

# Identify vector columns (assuming they start with 'dim_')
vector_columns = [col for col in df.columns if col.startswith('dim_')]
print(f"✓ Vector columns identified: {vector_columns}")

# Extract vectors as numpy array
vectors = df[vector_columns].values
print(f"✓ Extracted vectors shape: {vectors.shape}")
print(f"✓ Data type: {vectors.dtype}")

# Handle any missing values in vectors (if present)
if np.isnan(vectors).any():
    print("⚠️  Missing values detected in vectors")
    # Option 1: Remove rows with missing values
    mask = ~np.isnan(vectors).any(axis=1)
    vectors = vectors[mask]
    df_clean = df[mask].reset_index(drop=True)
    print(f"✓ After removing NaN rows: {vectors.shape}")
else:
    print("✓ No missing values in vector data")
    df_clean = df

print(f"Final vector data shape: {vectors.shape}")

# STEP 3: Vector Normalization
print("\n3. VECTOR NORMALIZATION")
print("-" * 30)

def normalize_vectors_manual(vectors):
    """Manually normalize vectors using NumPy operations"""
    # Calculate L2 norm (magnitude) for each vector
    magnitudes = np.sqrt(np.sum(vectors**2, axis=1, keepdims=True))
    
    # Handle zero vectors to avoid division by zero
    magnitudes = np.where(magnitudes == 0, 1, magnitudes)
    
    # Normalize by dividing by magnitude
    normalized = vectors / magnitudes
    
    return normalized, magnitudes.flatten()

# Original vectors info
print("Original vectors:")
print(f"✓ Mean magnitude: {np.mean(np.linalg.norm(vectors, axis=1)):.4f}")
print(f"✓ Min magnitude: {np.min(np.linalg.norm(vectors, axis=1)):.4f}")
print(f"✓ Max magnitude: {np.max(np.linalg.norm(vectors, axis=1)):.4f}")

# Normalize vectors
normalized_vectors, original_magnitudes = normalize_vectors_manual(vectors)

print("\nNormalized vectors:")
print(f"✓ Mean magnitude: {np.mean(np.linalg.norm(normalized_vectors, axis=1)):.4f}")
print(f"✓ Min magnitude: {np.min(np.linalg.norm(normalized_vectors, axis=1)):.4f}")
print(f"✓ Max magnitude: {np.max(np.linalg.norm(normalized_vectors, axis=1)):.4f}")

# Verify normalization worked
assert np.allclose(np.linalg.norm(normalized_vectors, axis=1), 1.0), "Normalization failed!"
print("✓ All vectors successfully normalized to unit length")

# STEP 4: Cosine Similarity Computation
print("\n4. COSINE SIMILARITY COMPUTATION")
print("-" * 30)

def cosine_similarity_manual(vec1, vec2):
    """Manually compute cosine similarity using NumPy"""
    # Dot product
    dot_product = np.dot(vec1, vec2)
    
    # Magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Handle zero vectors
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return cosine_sim

def cosine_similarity_matrix_manual(vectors):
    """Compute pairwise cosine similarity matrix manually"""
    n = vectors.shape[0]
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_similarity_manual(vectors[i], vectors[j])
    
    return similarity_matrix

# Example: Compute similarity between first two vectors
vec1 = vectors[0]
vec2 = vectors[1]

print("Computing similarity between first two vectors:")
print(f"Vector 1: {vec1}")
print(f"Vector 2: {vec2}")

# Manual calculation step by step
dot_prod = np.dot(vec1, vec2)
mag1 = np.linalg.norm(vec1)
mag2 = np.linalg.norm(vec2)
cosine_sim = dot_prod / (mag1 * mag2)

print(f"\nStep-by-step calculation:")
print(f"✓ Dot product: {dot_prod:.4f}")
print(f"✓ Magnitude 1: {mag1:.4f}")
print(f"✓ Magnitude 2: {mag2:.4f}")
print(f"✓ Cosine similarity: {cosine_sim:.4f}")

# Verify with our manual function
manual_sim = cosine_similarity_manual(vec1, vec2)
print(f"✓ Manual function result: {manual_sim:.4f}")

# Verify with scipy (for validation)
scipy_sim = 1 - cosine(vec1, vec2)
print(f"✓ SciPy validation: {scipy_sim:.4f}")

assert np.isclose(cosine_sim, manual_sim, atol=1e-10), "Manual calculation mismatch!"
assert np.isclose(cosine_sim, scipy_sim, atol=1e-10), "SciPy validation failed!"

print("✅ All similarity calculations match!")

# STEP 5: Compute similarity matrix for all vectors
print("\n5. FULL SIMILARITY MATRIX")
print("-" * 30)

# Limit to first 5 vectors for demonstration
n_demo = min(5, vectors.shape[0])
demo_vectors = vectors[:n_demo]

print(f"Computing {n_demo}x{n_demo} similarity matrix...")

# Using vectorized approach for efficiency
def cosine_similarity_vectorized(vectors):
    """Efficient vectorized cosine similarity computation"""
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    
    # Compute similarity matrix as dot product of normalized vectors
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix

# Compute similarity matrix
sim_matrix = cosine_similarity_vectorized(demo_vectors)

print("✓ Similarity matrix computed")
print(f"Matrix shape: {sim_matrix.shape}")
print("\nSimilarity Matrix:")
print(sim_matrix.round(4))

# Verify diagonal is all 1s (each vector similar to itself)
diagonal_check = np.allclose(np.diag(sim_matrix), 1.0)
print(f"✓ Diagonal check (should be 1.0): {diagonal_check}")

# Find most similar pair (excluding diagonal)
np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarity
max_sim_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
max_sim_value = sim_matrix[max_sim_idx]

print(f"✓ Most similar pair: Vector {max_sim_idx[0]} and Vector {max_sim_idx[1]}")
print(f"✓ Similarity score: {max_sim_value:.4f}")

print("\n" + "=" * 60)
print("WORKFLOW COMPLETED SUCCESSFULLY! ✅")
print("=" * 60)