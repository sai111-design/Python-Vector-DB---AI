#!/usr/bin/env python3

"""
Day 10 - RAG Basics Demo Script
Demonstrates text chunking strategies and embedding generation
"""

import os
import sys
from pathlib import Path
import time
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import box

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import RAGBasicsManager
from utils.text_processor import create_sample_documents
from utils.evaluation import create_evaluation_report

console = Console()

def display_header():
    """Display demo header"""
    console.print()
    console.print("🚀 RAG Basics - Text Chunking & Embeddings Demo", style="bold blue", justify="center")
    console.print("=" * 60, style="blue", justify="center")
    console.print()
    console.print("Day 10 Task: Implement text chunking (by tokens/sentences) + embeddings", style="cyan")
    console.print()

def demo_sample_documents():
    """Create and demonstrate with sample documents"""
    console.print("📝 Creating Sample Documents...", style="bold yellow")

    try:
        sample_dir = create_sample_documents()
        console.print(f"✅ Sample documents created in: {sample_dir}", style="green")

        # List created files
        sample_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.md"))

        table = Table(title="Sample Documents", box=box.ROUNDED)
        table.add_column("File", style="cyan")
        table.add_column("Size (chars)", style="magenta")
        table.add_column("Content Preview", style="green")

        for file_path in sample_files:
            content = file_path.read_text(encoding='utf-8')
            preview = content.replace('\n', ' ')[:80] + "..." if len(content) > 80 else content
            table.add_row(file_path.name, str(len(content)), preview)

        console.print(table)
        return sample_files

    except Exception as e:
        console.print(f"❌ Failed to create sample documents: {e}", style="red")
        return []

def demo_chunking_strategies():
    """Demonstrate different chunking strategies"""
    console.print("\n⚙️ Chunking Strategy Comparison", style="bold yellow")

    sample_text = """
    Retrieval-Augmented Generation (RAG) is a powerful technique in natural language processing 
    that combines the strengths of retrieval systems and generative models. The approach works 
    by first retrieving relevant information from a knowledge base or document collection, 
    then using this retrieved context to generate more accurate and informative responses.

    The RAG pipeline typically consists of several key components. First, documents are 
    preprocessed and chunked into smaller, manageable pieces. These chunks are then converted 
    into vector embeddings using techniques like sentence transformers or other embedding models. 
    The embeddings are stored in a vector database for efficient similarity search.

    When a user query is received, it is also converted into an embedding. The system then 
    performs a similarity search to find the most relevant document chunks. These retrieved 
    chunks are combined with the original query and fed into a large language model to 
    generate a comprehensive and contextually relevant response.

    The benefits of RAG include improved factual accuracy, reduced hallucination, and the 
    ability to incorporate up-to-date information without retraining the entire model. 
    This makes RAG particularly valuable for applications like question-answering systems, 
    chatbots, and knowledge management tools.
    """

    strategies = ["token", "sentence", "paragraph", "hybrid"]
    results = {}

    with Progress() as progress:
        task = progress.add_task("Testing chunking strategies...", total=len(strategies))

        for strategy in strategies:
            try:
                manager = RAGBasicsManager(
                    chunker_type=strategy, 
                    cache_embeddings=False  # Disable for demo
                )

                start_time = time.time()
                result = manager.process_text(sample_text, source_document="demo_text")
                processing_time = time.time() - start_time

                results[strategy] = {
                    "result": result,
                    "processing_time": processing_time
                }

                progress.update(task, advance=1)

            except Exception as e:
                console.print(f"❌ Failed to test {strategy} strategy: {e}", style="red")
                results[strategy] = {"error": str(e)}

    # Display comparison
    table = Table(title="Chunking Strategy Comparison", box=box.ROUNDED)
    table.add_column("Strategy", style="cyan")
    table.add_column("Chunks", style="magenta")
    table.add_column("Avg Tokens", style="yellow")
    table.add_column("Processing Time", style="green")
    table.add_column("Status", style="blue")

    for strategy, data in results.items():
        if "error" in data:
            table.add_row(strategy, "-", "-", "-", f"ERROR: {data['error']}")
        else:
            result = data["result"]
            stats = result["statistics"]["chunking"]
            chunks = len(result["chunks"])
            avg_tokens = f"{stats.get('avg_tokens_per_chunk', 0):.1f}"
            proc_time = f"{data['processing_time']:.2f}s"
            table.add_row(strategy, str(chunks), avg_tokens, proc_time, "✅")

    console.print(table)
    return results

def demo_chunk_analysis(results):
    """Analyze and display detailed chunk information"""
    console.print("\n🔍 Detailed Chunk Analysis", style="bold yellow")

    for strategy, data in results.items():
        if "error" in data:
            continue

        console.print(f"\n### {strategy.title()} Strategy", style="bold cyan")

        result = data["result"]
        chunks = result["chunks"]

        # Show first few chunks
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            console.print(f"\n**Chunk {i+1}:**", style="yellow")
            console.print(f"Text: {chunk['text'][:100]}...", style="white")

            metadata = chunk["metadata"]
            console.print(f"Tokens: {metadata.get('token_count', 'N/A')}", style="green")
            console.print(f"Type: {metadata.get('chunk_type', 'N/A')}", style="green")

            if "embedding" in chunk:
                embedding_dim = len(chunk["embedding"])
                console.print(f"Embedding: {embedding_dim}D vector", style="blue")

        if len(chunks) > 3:
            console.print(f"... and {len(chunks) - 3} more chunks", style="dim")

def demo_similarity_search(results):
    """Demonstrate semantic similarity search"""
    console.print("\n🔎 Semantic Similarity Search Demo", style="bold yellow")

    # Use hybrid strategy results for demo
    if "hybrid" in results and "error" not in results["hybrid"]:
        manager = RAGBasicsManager(chunker_type="hybrid", cache_embeddings=False)
        result = results["hybrid"]["result"]

        # Test queries
        test_queries = [
            "What is retrieval-augmented generation?",
            "How does vector similarity search work?",
            "What are the benefits of RAG systems?",
            "Tell me about document preprocessing"
        ]

        table = Table(title="Similarity Search Results", box=box.ROUNDED)
        table.add_column("Query", style="cyan", width=30)
        table.add_column("Top Match", style="yellow", width=40)
        table.add_column("Similarity", style="green")

        for query in test_queries:
            try:
                similar_chunks = manager.search_similar_chunks(query, result, top_k=1)

                if similar_chunks:
                    top_match = similar_chunks[0]
                    chunk_text = top_match["chunk"]["text"][:60] + "..."
                    similarity = f"{top_match['similarity']:.3f}"
                    table.add_row(query, chunk_text, similarity)
                else:
                    table.add_row(query, "No matches found", "0.000")

            except Exception as e:
                table.add_row(query, f"Error: {str(e)[:30]}", "N/A")

        console.print(table)
    else:
        console.print("❌ Cannot demo similarity search - hybrid strategy failed", style="red")

def demo_evaluation_report(results):
    """Generate and display evaluation reports"""
    console.print("\n📊 Chunking Quality Evaluation", style="bold yellow")

    for strategy, data in results.items():
        if "error" in data:
            continue

        result = data["result"]
        chunks = result["chunks"]
        original_text = result["original_text"]

        try:
            # Generate evaluation report
            report = create_evaluation_report(chunks, original_text)

            console.print(f"\n### {strategy.title()} Strategy Evaluation", style="bold cyan")

            # Extract key metrics from report
            lines = report.split('\n')
            for line in lines:
                if "Overall Score:" in line or "Grade:" in line or "Total chunks:" in line:
                    console.print(line, style="green")
                elif "Assessment:" in line or "Summary:" in line:
                    console.print(line, style="yellow")

        except Exception as e:
            console.print(f"❌ Evaluation failed for {strategy}: {e}", style="red")

def demo_performance_metrics(results):
    """Display performance metrics"""
    console.print("\n⚡ Performance Metrics", style="bold yellow")

    table = Table(title="Processing Performance", box=box.ROUNDED)
    table.add_column("Strategy", style="cyan")
    table.add_column("Total Time", style="yellow")
    table.add_column("Chunks/sec", style="green")
    table.add_column("Memory Est.", style="blue")

    for strategy, data in results.items():
        if "error" in data:
            continue

        result = data["result"]
        processing_time = data["processing_time"]
        chunks_count = len(result["chunks"])

        chunks_per_sec = chunks_count / processing_time if processing_time > 0 else 0

        # Estimate memory usage
        total_text_size = sum(len(chunk["text"]) for chunk in result["chunks"])
        embedding_size = sum(len(chunk.get("embedding", [])) * 4 for chunk in result["chunks"])  # 4 bytes per float
        total_memory = total_text_size + embedding_size

        memory_str = f"{total_memory / 1024:.1f} KB"

        table.add_row(
            strategy,
            f"{processing_time:.2f}s",
            f"{chunks_per_sec:.1f}",
            memory_str
        )

    console.print(table)

def demo_real_document_processing():
    """Process real sample documents"""
    console.print("\n📄 Real Document Processing Demo", style="bold yellow")

    sample_dir = Path("data/sample_documents")
    if not sample_dir.exists():
        console.print("❌ Sample documents not found, skipping real document demo", style="red")
        return

    manager = RAGBasicsManager(chunker_type="hybrid", cache_embeddings=False)

    # Process each sample document
    for file_path in sample_dir.glob("*.txt"):
        console.print(f"\n📖 Processing: {file_path.name}", style="bold cyan")

        try:
            result = manager.process_document(str(file_path))

            stats = result["statistics"]
            console.print(f"Original length: {stats['original_length']:,} characters", style="green")
            console.print(f"Chunks created: {stats['chunking']['total_chunks']}", style="green")
            console.print(f"Avg tokens per chunk: {stats['chunking']['avg_tokens_per_chunk']:.1f}", style="green")
            console.print(f"Processing time: {stats['processing_time_seconds']:.2f}s", style="green")

            # Save results for later inspection
            output_file = f"processed_{file_path.stem}.json"
            saved_path = manager.save_processed_chunks(result, output_file)
            console.print(f"Results saved to: {saved_path}", style="blue")

        except Exception as e:
            console.print(f"❌ Failed to process {file_path.name}: {e}", style="red")

def demo_export_capabilities():
    """Demonstrate export and import capabilities"""
    console.print("\n💾 Export/Import Capabilities Demo", style="bold yellow")

    try:
        manager = RAGBasicsManager(chunker_type="hybrid", cache_embeddings=False)

        # Create sample processing result
        sample_text = "This is a sample text for export demonstration. " * 10
        result = manager.process_text(sample_text)

        chunks = result["chunks"]

        # Export in different formats
        export_formats = ["json", "npz"]

        for fmt in export_formats:
            try:
                output_path = f"data/processed_chunks/demo_export.{fmt}"
                manager.embedding_pipeline.export_embeddings(chunks, output_path, format=fmt)
                console.print(f"✅ Exported to {fmt.upper()}: {output_path}", style="green")

                # Test import (for supported formats)
                if fmt in ["json", "npz"]:
                    imported_chunks = manager.embedding_pipeline.load_embeddings(output_path, format=fmt)
                    console.print(f"✅ Imported {len(imported_chunks)} chunks from {fmt.upper()}", style="green")

            except Exception as e:
                console.print(f"❌ Export/Import failed for {fmt}: {e}", style="red")

    except Exception as e:
        console.print(f"❌ Export demo failed: {e}", style="red")

def display_summary():
    """Display demo summary"""
    console.print("\n🎉 Demo Summary", style="bold green")
    console.print()
    console.print("✅ Key Features Demonstrated:", style="bold")

    features = [
        "Multiple chunking strategies (token, sentence, paragraph, hybrid)",
        "Embedding generation with sentence transformers",
        "Semantic similarity search",
        "Chunking quality evaluation",
        "Performance metrics and comparison",
        "Real document processing",
        "Export/import capabilities",
        "Comprehensive error handling"
    ]

    for feature in features:
        console.print(f"  • {feature}", style="green")

    console.print()
    console.print("🔗 Next Steps:", style="bold")
    console.print("  • Experiment with different chunking parameters", style="cyan")
    console.print("  • Try different embedding models", style="cyan")
    console.print("  • Process your own documents", style="cyan")
    console.print("  • Integrate with vector databases", style="cyan")
    console.print("  • Build retrieval-augmented generation systems", style="cyan")

    console.print()
    console.print("📁 Generated Files:", style="bold")
    console.print("  • data/sample_documents/ - Sample documents for testing", style="blue")
    console.print("  • data/processed_chunks/ - Processed results and exports", style="blue")
    console.print("  • Cached embeddings (if enabled)", style="blue")

def main():
    """Main demo function"""
    display_header()

    try:
        # Create sample documents
        sample_files = demo_sample_documents()

        # Demonstrate chunking strategies
        results = demo_chunking_strategies()

        # Analyze chunks in detail
        demo_chunk_analysis(results)

        # Demonstrate similarity search
        demo_similarity_search(results)

        # Generate evaluation reports
        demo_evaluation_report(results)

        # Show performance metrics
        demo_performance_metrics(results)

        # Process real documents
        if sample_files:
            demo_real_document_processing()

        # Demonstrate export/import
        demo_export_capabilities()

        # Display summary
        display_summary()

    except KeyboardInterrupt:
        console.print("\n⚠️ Demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Demo failed with error: {e}", style="red")
        console.print("Check dependencies and try again", style="red")

if __name__ == "__main__":
    main()
