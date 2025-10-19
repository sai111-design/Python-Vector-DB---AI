import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from chunkers.token_chunker import create_chunker as create_token_chunker
from chunkers.sentence_chunker import SentenceBasedChunker, ParagraphChunker
from chunkers.hybrid_chunker import HybridChunker
from embedders.embedding_pipeline import create_embedding_pipeline
from utils.text_processor import TextProcessor, DocumentLoader
from utils.evaluation import ChunkingEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGBasicsManager:
    """Main class for managing RAG basics - text chunking and embeddings"""

    def __init__(self, 
                 chunker_type: str = "hybrid",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 output_dir: str = "data/processed_chunks",
                 cache_embeddings: bool = True):
        """
        Initialize RAG basics manager

        Args:
            chunker_type: Type of chunker ('token', 'sentence', 'hybrid')
            embedding_model: Embedding model name
            output_dir: Directory to save processed chunks
            cache_embeddings: Whether to cache embeddings
        """
        self.chunker_type = chunker_type
        self.embedding_model = embedding_model
        self.output_dir = Path(output_dir)
        self.cache_embeddings = cache_embeddings

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.text_processor = TextProcessor()
        self.document_loader = DocumentLoader()
        self.embedding_pipeline = create_embedding_pipeline(
            model_name=embedding_model,
            cache_dir=str(self.output_dir / "embeddings_cache") if cache_embeddings else None
        )
        self.evaluator = ChunkingEvaluator()

        # Initialize chunker
        self.chunker = self._create_chunker(chunker_type)

        logger.info(f"Initialized RAG basics manager with {chunker_type} chunker and {embedding_model} embeddings")

    def _create_chunker(self, chunker_type: str):
        """Create chunker based on type"""
        if chunker_type == "token":
            return create_token_chunker("token", max_tokens=1000, overlap_tokens=100)
        elif chunker_type == "sentence":
            return SentenceBasedChunker(max_sentences=5, overlap_sentences=1)
        elif chunker_type == "paragraph":
            return ParagraphChunker(max_paragraphs=3, overlap_paragraphs=1)
        elif chunker_type == "hybrid":
            return HybridChunker(
                primary_strategy="token",
                fallback_strategy="sentence",
                max_tokens=1000,
                max_sentences=5,
                overlap_tokens=100
            )
        else:
            raise ValueError(f"Unknown chunker type: {chunker_type}")

    def process_text(self, text: str, source_document: str = None) -> Dict[str, Any]:
        """
        Process text through the complete RAG pipeline

        Args:
            text: Text to process
            source_document: Source document identifier

        Returns:
            Dictionary with chunks, embeddings, and statistics
        """
        start_time = datetime.now()

        logger.info(f"Processing text: {len(text)} characters")

        # Step 1: Clean and preprocess text
        cleaned_text = self.text_processor.clean_text(text)
        logger.info(f"Text cleaned: {len(cleaned_text)} characters after cleaning")

        # Step 2: Chunk the text
        chunks = self.chunker.chunk_text(cleaned_text, source_document)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Generate embeddings
        enriched_chunks = self.embedding_pipeline.generate_embeddings(chunks)
        logger.info(f"Generated embeddings for {len(enriched_chunks)} chunks")

        # Step 4: Calculate statistics
        chunk_stats = self.chunker.get_chunk_statistics(chunks)
        embedding_stats = self.embedding_pipeline.get_embedding_statistics(enriched_chunks)

        processing_time = (datetime.now() - start_time).total_seconds()

        result = {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "chunks": enriched_chunks,
            "statistics": {
                "processing_time_seconds": processing_time,
                "original_length": len(text),
                "cleaned_length": len(cleaned_text),
                "chunking": chunk_stats,
                "embeddings": embedding_stats
            },
            "metadata": {
                "chunker_type": self.chunker_type,
                "embedding_model": self.embedding_model,
                "processed_at": datetime.now().isoformat(),
                "source_document": source_document
            }
        }

        logger.info(f"Text processing completed in {processing_time:.2f}s")
        return result

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document file

        Args:
            file_path: Path to document file

        Returns:
            Processing results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Load document
        text = self.document_loader.load_document(str(file_path))

        # Process with document name as source
        return self.process_text(text, source_document=file_path.name)

    def process_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents"""
        results = []

        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    "error": str(e),
                    "file_path": file_path
                })

        return results

    def save_processed_chunks(self, result: Dict[str, Any], 
                            filename: str = None) -> str:
        """Save processed chunks to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_chunks_{timestamp}.json"

        output_file = self.output_dir / filename

        # Save result to JSON
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processed chunks to {output_file}")
        return str(output_file)

    def load_processed_chunks(self, filename: str) -> Dict[str, Any]:
        """Load processed chunks from file"""
        file_path = Path(filename)
        if not file_path.exists():
            file_path = self.output_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Processed chunks file not found: {filename}")

        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        logger.info(f"Loaded processed chunks from {file_path}")
        return result

    def search_similar_chunks(self, query: str, 
                            processed_result: Dict[str, Any],
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity

        Args:
            query: Search query
            processed_result: Previously processed text result
            top_k: Number of top results to return

        Returns:
            List of similar chunks with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_pipeline.embed_single_text(query)

        # Find similar chunks
        chunks = processed_result.get("chunks", [])
        similar_chunks = self.embedding_pipeline.find_similar_chunks(
            query_embedding, chunks, top_k=top_k
        )

        return similar_chunks

    def evaluate_chunking_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of chunking"""
        chunks = result.get("chunks", [])
        original_text = result.get("original_text", "")

        evaluation = self.evaluator.evaluate_chunks(chunks, original_text)

        return evaluation

    def compare_chunking_strategies(self, text: str, 
                                  strategies: List[str] = None) -> Dict[str, Any]:
        """
        Compare different chunking strategies

        Args:
            text: Text to chunk
            strategies: List of strategies to compare

        Returns:
            Comparison results
        """
        if strategies is None:
            strategies = ["token", "sentence", "hybrid"]

        comparison_results = {}

        for strategy in strategies:
            try:
                # Create temporary chunker
                temp_chunker = self._create_chunker(strategy)

                # Process with this strategy
                start_time = time.time()
                chunks = temp_chunker.chunk_text(text)
                processing_time = time.time() - start_time

                # Get statistics
                stats = temp_chunker.get_chunk_statistics(chunks)
                stats["processing_time"] = processing_time

                comparison_results[strategy] = {
                    "chunks": chunks,
                    "statistics": stats,
                    "evaluation": self.evaluator.evaluate_chunks(chunks, text)
                }

            except Exception as e:
                logger.error(f"Failed to test {strategy} strategy: {e}")
                comparison_results[strategy] = {"error": str(e)}

        return comparison_results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about processing"""
        return {
            "chunker_type": self.chunker_type,
            "embedding_model": self.embedding_model,
            "embedding_dimension": getattr(self.embedding_pipeline, 'embedding_dimension', 'unknown'),
            "output_directory": str(self.output_dir),
            "cache_enabled": self.cache_embeddings,
            "cache_directory": getattr(self.embedding_pipeline, 'cache_dir', None)
        }

def main():
    """Main function for demonstration"""
    # Initialize RAG manager
    rag_manager = RAGBasicsManager(
        chunker_type="hybrid",
        embedding_model="all-MiniLM-L6-v2",
        cache_embeddings=True
    )

    # Sample text for demonstration
    sample_text = """
    Artificial Intelligence has revolutionized many aspects of modern technology and society. 
    Machine learning, a subset of AI, enables computers to learn and make decisions without 
    explicit programming for each scenario.

    Natural Language Processing (NLP) is another crucial area of AI that focuses on the 
    interaction between computers and human language. It enables machines to read, understand, 
    and derive meaning from human language in a valuable way.

    Deep learning, inspired by the structure and function of the human brain, uses artificial 
    neural networks to process data and create patterns for decision making. This technology 
    has enabled breakthroughs in image recognition, speech recognition, and language translation.

    The applications of AI are vast and continue to expand. In healthcare, AI assists in 
    diagnostic imaging and drug discovery. In finance, it powers algorithmic trading and 
    fraud detection. In transportation, it enables autonomous vehicles and traffic optimization.

    However, the development and deployment of AI systems also raise important ethical 
    considerations. Issues such as bias in algorithms, privacy concerns, and the potential 
    for job displacement must be carefully addressed as AI technology continues to advance.
    """

    print("🚀 RAG Basics - Text Chunking and Embeddings Demo")
    print("=" * 60)

    # Process sample text
    print("\n📝 Processing sample text...")
    result = rag_manager.process_text(sample_text, source_document="ai_overview.txt")

    # Display results
    print(f"\n📊 Processing Results:")
    stats = result["statistics"]
    print(f"  Original length: {stats['original_length']} characters")
    print(f"  Cleaned length: {stats['cleaned_length']} characters")
    print(f"  Total chunks: {stats['chunking']['total_chunks']}")
    print(f"  Avg tokens per chunk: {stats['chunking']['avg_tokens_per_chunk']:.1f}")
    print(f"  Processing time: {stats['processing_time_seconds']:.2f}s")

    # Display chunks
    print(f"\n📄 Generated Chunks:")
    for i, chunk in enumerate(result["chunks"][:3]):  # Show first 3 chunks
        print(f"\n  Chunk {i+1}:")
        print(f"    Text: {chunk['text'][:100]}...")
        print(f"    Tokens: {chunk['metadata']['token_count']}")
        print(f"    Type: {chunk['metadata']['chunk_type']}")

    # Demonstrate similarity search
    print(f"\n🔍 Similarity Search Demo:")
    queries = ["machine learning", "healthcare AI", "ethical considerations"]

    for query in queries:
        print(f"\n  Query: '{query}'")
        similar_chunks = rag_manager.search_similar_chunks(query, result, top_k=2)

        for j, sim_chunk in enumerate(similar_chunks):
            similarity = sim_chunk["similarity"]
            chunk_text = sim_chunk["chunk"]["text"][:80]
            print(f"    {j+1}. Similarity: {similarity:.3f}")
            print(f"       Text: {chunk_text}...")

    # Compare chunking strategies
    print(f"\n⚖️ Chunking Strategy Comparison:")
    comparison = rag_manager.compare_chunking_strategies(sample_text)

    for strategy, results in comparison.items():
        if "error" in results:
            print(f"  {strategy}: ERROR - {results['error']}")
        else:
            stats = results["statistics"]
            print(f"  {strategy}:")
            print(f"    Chunks: {stats['total_chunks']}")
            print(f"    Avg tokens: {stats.get('avg_tokens_per_chunk', 'N/A')}")
            print(f"    Processing time: {stats['processing_time']:.3f}s")

    # Save results
    output_file = rag_manager.save_processed_chunks(result, "demo_results.json")
    print(f"\n💾 Results saved to: {output_file}")

    # Display summary
    print(f"\n📋 System Summary:")
    summary = rag_manager.get_summary_statistics()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n✅ RAG Basics demo completed successfully!")

if __name__ == "__main__":
    main()
