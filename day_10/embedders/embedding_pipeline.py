from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, some features will be disabled")

@dataclass
class EmbeddingMetadata:
    """Metadata for embeddings"""
    model_name: str
    embedding_dimension: int
    created_at: str
    chunk_count: int
    total_tokens: Optional[int] = None
    processing_time_seconds: Optional[float] = None

class EmbeddingPipeline:
    """Pipeline for generating and managing embeddings from text chunks"""

    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str = "auto",
                 batch_size: int = 32,
                 cache_dir: Optional[str] = None):
        """
        Initialize embedding pipeline

        Args:
            model_name: SentenceTransformer model name
            device: Device to use ('cpu', 'cuda', or 'auto')
            batch_size: Batch size for embedding generation
            cache_dir: Directory to cache embeddings
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for embedding pipeline")

        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Initialize model
        try:
            self.model = SentenceTransformer(model_name)
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()

            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized {model_name} on {self.device} (dim: {self.embedding_dimension})")

        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise

        # Setup cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_embeddings(self, chunks: List[Dict[str, Any]], 
                          use_cache: bool = True,
                          save_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            use_cache: Whether to use cached embeddings if available
            save_cache: Whether to save embeddings to cache

        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            return []

        start_time = datetime.now()

        # Check cache first
        if use_cache and self.cache_dir:
            cached_chunks = self._load_from_cache(chunks)
            if cached_chunks:
                logger.info(f"Loaded {len(cached_chunks)} embeddings from cache")
                return cached_chunks

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} chunks using {self.model_name}")

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    show_progress_bar=True if i == 0 else False,
                    device=self.device
                )
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//self.batch_size}: {e}")
                # Create zero embeddings as fallback
                batch_embeddings = np.zeros((len(batch_texts), self.embedding_dimension))
                all_embeddings.extend(batch_embeddings)

        # Add embeddings to chunks
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = chunk.copy()
            enriched_chunk["embedding"] = all_embeddings[i].tolist()
            enriched_chunk["embedding_metadata"] = {
                "model_name": self.model_name,
                "dimension": self.embedding_dimension,
                "generated_at": datetime.now().isoformat()
            }
            enriched_chunks.append(enriched_chunk)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Save to cache
        if save_cache and self.cache_dir:
            self._save_to_cache(enriched_chunks, processing_time)

        logger.info(f"Generated {len(enriched_chunks)} embeddings in {processing_time:.2f}s")

        return enriched_chunks

    def embed_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False, device=self.device)
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to generate single embedding: {e}")
            return np.zeros(self.embedding_dimension)

    def compute_similarity(self, embedding1: Union[np.ndarray, List[float]], 
                          embedding2: Union[np.ndarray, List[float]], 
                          metric: str = "cosine") -> float:
        """
        Compute similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine', 'dot', 'euclidean')

        Returns:
            Similarity score
        """
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)

        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        elif metric == "dot":
            # Dot product
            return np.dot(embedding1, embedding2)

        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)

        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def find_similar_chunks(self, query_embedding: Union[np.ndarray, List[float]],
                           chunk_embeddings: List[Dict[str, Any]],
                           top_k: int = 5,
                           similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find most similar chunks to a query embedding

        Args:
            query_embedding: Query embedding
            chunk_embeddings: List of chunks with embeddings
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of similar chunks with similarity scores
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        similarities = []

        for i, chunk in enumerate(chunk_embeddings):
            if "embedding" not in chunk:
                continue

            chunk_embedding = np.array(chunk["embedding"])
            similarity = self.compute_similarity(query_embedding, chunk_embedding)

            if similarity >= similarity_threshold:
                similarities.append({
                    "chunk": chunk,
                    "similarity": similarity,
                    "index": i
                })

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Return top_k results
        return similarities[:top_k]

    def _generate_cache_key(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate cache key based on chunks and model"""
        # Create hash of chunk texts and metadata
        import hashlib

        content_hash = hashlib.md5()
        for chunk in chunks:
            content_hash.update(chunk["text"].encode('utf-8'))
            if "metadata" in chunk:
                content_hash.update(str(chunk["metadata"]).encode('utf-8'))

        cache_key = f"{self.model_name}_{content_hash.hexdigest()[:16]}"
        return cache_key

    def _load_from_cache(self, chunks: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Load embeddings from cache if available"""
        if not self.cache_dir:
            return None

        cache_key = self._generate_cache_key(chunks)
        cache_file = self.cache_dir / f"{cache_key}_embeddings.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

        if not cache_file.exists() or not metadata_file.exists():
            return None

        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if model and dimensions match
            if (metadata.get("model_name") != self.model_name or 
                metadata.get("embedding_dimension") != self.embedding_dimension):
                logger.warning("Cache metadata mismatch, ignoring cache")
                return None

            # Load embeddings
            with open(cache_file, 'rb') as f:
                cached_chunks = pickle.load(f)

            return cached_chunks

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, chunks: List[Dict[str, Any]], processing_time: float):
        """Save embeddings to cache"""
        if not self.cache_dir:
            return

        cache_key = self._generate_cache_key(chunks)
        cache_file = self.cache_dir / f"{cache_key}_embeddings.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

        try:
            # Save embeddings
            with open(cache_file, 'wb') as f:
                pickle.dump(chunks, f)

            # Save metadata
            metadata = EmbeddingMetadata(
                model_name=self.model_name,
                embedding_dimension=self.embedding_dimension,
                created_at=datetime.now().isoformat(),
                chunk_count=len(chunks),
                processing_time_seconds=processing_time
            )

            with open(metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)

            logger.info(f"Saved embeddings to cache: {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def export_embeddings(self, chunks: List[Dict[str, Any]], 
                         output_path: str, format: str = "npz"):
        """
        Export embeddings to file

        Args:
            chunks: Chunks with embeddings
            output_path: Output file path
            format: Export format ('npz', 'json', 'csv')
        """
        output_path = Path(output_path)

        if format == "npz":
            # Export as NumPy archive
            embeddings = np.array([chunk["embedding"] for chunk in chunks if "embedding" in chunk])
            texts = [chunk["text"] for chunk in chunks if "embedding" in chunk]
            metadata = [chunk.get("metadata", {}) for chunk in chunks if "embedding" in chunk]

            np.savez_compressed(
                output_path,
                embeddings=embeddings,
                texts=texts,
                metadata=metadata,
                model_name=self.model_name,
                embedding_dimension=self.embedding_dimension
            )

        elif format == "json":
            # Export as JSON
            export_data = {
                "model_name": self.model_name,
                "embedding_dimension": self.embedding_dimension,
                "created_at": datetime.now().isoformat(),
                "chunks": chunks
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

        elif format == "csv":
            # Export as CSV (flattened embeddings)
            import pandas as pd

            rows = []
            for i, chunk in enumerate(chunks):
                if "embedding" not in chunk:
                    continue

                row = {
                    "chunk_id": chunk.get("metadata", {}).get("chunk_id", f"chunk_{i}"),
                    "text": chunk["text"],
                    "embedding_dimension": len(chunk["embedding"])
                }

                # Add embedding dimensions as columns
                for j, value in enumerate(chunk["embedding"]):
                    row[f"embed_{j}"] = value

                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unknown export format: {format}")

        logger.info(f"Exported {len(chunks)} embeddings to {output_path} (format: {format})")

    def load_embeddings(self, input_path: str, format: str = "npz") -> List[Dict[str, Any]]:
        """
        Load embeddings from file

        Args:
            input_path: Input file path
            format: File format ('npz', 'json')

        Returns:
            List of chunks with embeddings
        """
        input_path = Path(input_path)

        if format == "npz":
            # Load from NumPy archive
            data = np.load(input_path, allow_pickle=True)

            chunks = []
            for i in range(len(data["texts"])):
                chunk = {
                    "text": str(data["texts"][i]),
                    "embedding": data["embeddings"][i].tolist(),
                    "metadata": data["metadata"][i].item() if data["metadata"][i] else {},
                    "embedding_metadata": {
                        "model_name": str(data["model_name"]),
                        "dimension": int(data["embedding_dimension"]),
                        "loaded_from": str(input_path)
                    }
                }
                chunks.append(chunk)

            return chunks

        elif format == "json":
            # Load from JSON
            with open(input_path, 'r') as f:
                data = json.load(f)

            return data.get("chunks", [])

        else:
            raise ValueError(f"Unknown import format: {format}")

    def get_embedding_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about embeddings"""
        chunks_with_embeddings = [c for c in chunks if "embedding" in c]

        if not chunks_with_embeddings:
            return {"error": "No chunks with embeddings found"}

        embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])

        # Calculate statistics
        norms = np.linalg.norm(embeddings, axis=1)
        mean_embedding = np.mean(embeddings, axis=0)

        return {
            "total_chunks": len(chunks_with_embeddings),
            "embedding_dimension": embeddings.shape[1],
            "model_name": self.model_name,
            "norm_statistics": {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "min": float(np.min(norms)),
                "max": float(np.max(norms))
            },
            "embedding_statistics": {
                "mean_values": {
                    "mean": float(np.mean(mean_embedding)),
                    "std": float(np.std(mean_embedding)),
                    "min": float(np.min(mean_embedding)),
                    "max": float(np.max(mean_embedding))
                }
            }
        }

class MockEmbeddingPipeline:
    """Mock embedding pipeline for testing when transformers is not available"""

    def __init__(self, embedding_dimension: int = 384):
        self.model_name = "mock-model"
        self.embedding_dimension = embedding_dimension
        logger.info(f"Initialized mock embedding pipeline (dim: {embedding_dimension})")

    def generate_embeddings(self, chunks: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Generate mock embeddings"""
        enriched_chunks = []
        for chunk in chunks:
            # Generate random embedding
            embedding = np.random.rand(self.embedding_dimension).tolist()

            enriched_chunk = chunk.copy()
            enriched_chunk["embedding"] = embedding
            enriched_chunk["embedding_metadata"] = {
                "model_name": self.model_name,
                "dimension": self.embedding_dimension,
                "generated_at": datetime.now().isoformat()
            }
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks

    def embed_single_text(self, text: str) -> np.ndarray:
        """Generate mock embedding for single text"""
        return np.random.rand(self.embedding_dimension)

    def compute_similarity(self, embedding1, embedding2, metric="cosine") -> float:
        """Compute mock similarity"""
        return np.random.rand()  # Random similarity score

def create_embedding_pipeline(**kwargs) -> Union[EmbeddingPipeline, MockEmbeddingPipeline]:
    """Factory function to create embedding pipeline"""
    if TRANSFORMERS_AVAILABLE:
        return EmbeddingPipeline(**kwargs)
    else:
        logger.warning("Using mock embedding pipeline - install sentence-transformers for real embeddings")
        return MockEmbeddingPipeline()
