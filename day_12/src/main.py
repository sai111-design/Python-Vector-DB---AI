#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 12 - Mini RAG Pipeline
==========================
Task: Connect Chroma → Retrieval → LLM → Answer

Complete RAG (Retrieval-Augmented Generation) pipeline that:
1. Stores documents in Chroma vector database
2. Retrieves relevant context for user queries
3. Uses LLM to generate contextual answers
4. Provides comprehensive logging and error handling
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import time

# Core dependencies
import numpy as np
from sentence_transformers import SentenceTransformer

# ChromaDB for vector storage
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

# OpenAI for LLM (fallback to mock if not available)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Using mock LLM responses.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/rag_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document in the knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RAGQuery:
    """Represents a user query to the RAG system"""
    id: str
    text: str
    max_results: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RAGResponse:
    """Represents the complete response from RAG system"""
    query: RAGQuery
    retrieved_docs: List[Document]
    generated_answer: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        return {
            "query_id": self.query.id,
            "query_text": self.query.text,
            "answer": self.generated_answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata
                }
                for doc in self.retrieved_docs
            ],
            "processing_time": self.processing_time,
            "timestamp": datetime.now().isoformat(),
            "metadata": self.metadata
        }

class VectorStore:
    """Vector storage interface using ChromaDB"""

    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "data/chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, using mock vector store")
            self._mock_documents = {}
            return

        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._mock_documents = {}

    def add_document(self, document: Document) -> bool:
        """Add a document to the vector store"""
        try:
            if not CHROMADB_AVAILABLE or self.collection is None:
                return self._mock_add_document(document)

            # Generate embedding if not provided
            if document.embedding is None:
                document.embedding = self.embedding_model.encode(document.content).tolist()

            # Add to ChromaDB
            self.collection.add(
                ids=[document.id],
                embeddings=[document.embedding],
                documents=[document.content],
                metadatas=[document.metadata]
            )

            logger.info(f"Added document: {document.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            return False

    def _mock_add_document(self, document: Document) -> bool:
        """Mock implementation for adding documents"""
        if not hasattr(self, '_mock_documents'):
            self._mock_documents = {}

        if document.embedding is None:
            document.embedding = self.embedding_model.encode(document.content).tolist()

        self._mock_documents[document.id] = document
        logger.info(f"Mock: Added document {document.id}")
        return True

    def search(self, query: str, max_results: int = 5) -> List[Document]:
        """Search for similar documents"""
        try:
            if not CHROMADB_AVAILABLE or self.collection is None:
                return self._mock_search(query, max_results)

            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )

            # Convert results to Document objects
            documents = []
            for i in range(len(results['ids'][0])):
                doc = Document(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] or {},
                    embedding=results['embeddings'][0][i] if results['embeddings'] and results['embeddings'][0] else None
                )
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents

        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []

    def _mock_search(self, query: str, max_results: int = 5) -> List[Document]:
        """Mock implementation for searching documents"""
        if not hasattr(self, '_mock_documents'):
            return []

        query_embedding = self.embedding_model.encode(query)
        similarities = []

        for doc_id, doc in self._mock_documents.items():
            if doc.embedding:
                similarity = np.dot(query_embedding, doc.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
                )
                similarities.append((similarity, doc))

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in similarities[:max_results]]

        logger.info(f"Mock: Retrieved {len(results)} documents for query")
        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not CHROMADB_AVAILABLE or self.collection is None:
            return {
                "type": "mock",
                "document_count": len(getattr(self, '_mock_documents', {})),
                "embedding_dimension": 384
            }

        try:
            count = self.collection.count()
            return {
                "type": "chromadb",
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dimension": 384
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"type": "error", "error": str(e)}

class LLMProvider:
    """Language model provider interface"""

    def __init__(self, provider: str = "openai", model_name: str = "gpt-3.5-turbo", api_key: str = None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if provider == "openai" and OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
            self.use_real_llm = True
            logger.info(f"Initialized OpenAI LLM: {model_name}")
        else:
            self.use_real_llm = False
            logger.info("Using mock LLM responses")

    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using LLM"""
        if self.use_real_llm:
            return self._generate_real_response(prompt, max_tokens)
        else:
            return self._generate_mock_response(prompt)

    def _generate_real_response(self, prompt: str, max_tokens: int) -> str:
        """Generate response using real LLM"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing"""
        # Simple rule-based mock responses
        prompt_lower = prompt.lower()

        if "machine learning" in prompt_lower or "ml" in prompt_lower:
            return ("Machine learning is a subset of artificial intelligence that enables "
                   "computers to learn and improve from experience without being explicitly programmed. "
                   "Based on the provided context, machine learning involves algorithms that can "
                   "identify patterns in data and make predictions or decisions.")

        elif "deep learning" in prompt_lower or "neural" in prompt_lower:
            return ("Deep learning is a specialized area of machine learning that uses neural "
                   "networks with multiple layers to model complex patterns in data. According to "
                   "the context, deep learning has achieved significant breakthroughs in areas "
                   "like image recognition and natural language processing.")

        elif "artificial intelligence" in prompt_lower or "ai" in prompt_lower:
            return ("Artificial intelligence (AI) refers to the simulation of human intelligence "
                   "in machines that are programmed to think and learn. Based on the provided "
                   "context, AI encompasses various technologies including machine learning, "
                   "natural language processing, and computer vision.")

        else:
            return ("Based on the provided context, I can help answer questions about the topics "
                   "covered in the retrieved documents. The information suggests several key concepts "
                   "that are relevant to your query. Would you like me to elaborate on any specific aspect?")

class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) pipeline"""

    def __init__(self, 
                 collection_name: str = "rag_knowledge_base",
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-3.5-turbo"):

        self.vector_store = VectorStore(collection_name=collection_name)
        self.llm_provider = LLMProvider(provider=llm_provider, model_name=llm_model)
        self.query_history: List[RAGResponse] = []

        logger.info("RAG Pipeline initialized successfully")

    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Add multiple documents to the knowledge base"""
        start_time = time.time()
        results = {"successful": 0, "failed": 0, "errors": []}

        for doc in documents:
            if self.vector_store.add_document(doc):
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(f"Failed to add document: {doc.id}")

        processing_time = time.time() - start_time
        results["processing_time"] = processing_time

        logger.info(f"Added {results['successful']}/{len(documents)} documents in {processing_time:.3f}s")
        return results

    def query(self, query_text: str, max_results: int = 5) -> RAGResponse:
        """Process a complete RAG query"""
        start_time = time.time()
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.query_history)}"

        # Create query object
        rag_query = RAGQuery(
            id=query_id,
            text=query_text,
            max_results=max_results
        )

        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Processing query: {query_text}")
            retrieved_docs = self.vector_store.search(query_text, max_results)

            if not retrieved_docs:
                logger.warning("No documents retrieved for query")
                response = RAGResponse(
                    query=rag_query,
                    retrieved_docs=[],
                    generated_answer="I couldn't find any relevant information to answer your question.",
                    processing_time=time.time() - start_time,
                    metadata={"status": "no_documents_found"}
                )
                self.query_history.append(response)
                return response

            # Step 2: Build context from retrieved documents
            context = self._build_context(retrieved_docs)

            # Step 3: Create prompt
            prompt = self._create_rag_prompt(query_text, context)

            # Step 4: Generate LLM response
            generated_answer = self.llm_provider.generate_response(prompt)

            # Create response object
            processing_time = time.time() - start_time
            response = RAGResponse(
                query=rag_query,
                retrieved_docs=retrieved_docs,
                generated_answer=generated_answer,
                processing_time=processing_time,
                metadata={
                    "status": "success",
                    "documents_retrieved": len(retrieved_docs),
                    "context_length": len(context)
                }
            )

            self.query_history.append(response)
            logger.info(f"Query processed successfully in {processing_time:.3f}s")
            return response

        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            error_response = RAGResponse(
                query=rag_query,
                retrieved_docs=[],
                generated_answer=f"I encountered an error while processing your question: {str(e)}",
                processing_time=time.time() - start_time,
                metadata={"status": "error", "error": str(e)}
            )
            self.query_history.append(error_response)
            return error_response

    def _build_context(self, documents: List[Document], max_length: int = 2000) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            doc_text = f"Document {i}:\n{doc.content}\n\n"

            if current_length + len(doc_text) > max_length:
                # Truncate the document to fit
                remaining_length = max_length - current_length - 50
                if remaining_length > 0:
                    truncated_text = f"Document {i}:\n{doc.content[:remaining_length]}...\n\n"
                    context_parts.append(truncated_text)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "".join(context_parts)

    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create the RAG prompt template"""
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
Be accurate and only use information from the provided context. If the context doesn't contain 
enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        collection_info = self.vector_store.get_collection_info()

        query_times = [r.processing_time for r in self.query_history]
        avg_query_time = sum(query_times) / len(query_times) if query_times else 0

        successful_queries = [r for r in self.query_history if r.metadata.get("status") == "success"]

        return {
            "vector_store": collection_info,
            "queries": {
                "total_queries": len(self.query_history),
                "successful_queries": len(successful_queries),
                "average_query_time": avg_query_time,
                "success_rate": len(successful_queries) / len(self.query_history) if self.query_history else 0
            },
            "llm": {
                "provider": self.llm_provider.provider,
                "model": self.llm_provider.model_name,
                "use_real_llm": self.llm_provider.use_real_llm
            }
        }

    def save_conversation_history(self, filepath: Path):
        """Save conversation history to file"""
        history_data = {
            "pipeline_stats": self.get_pipeline_stats(),
            "conversations": [response.to_dict() for response in self.query_history],
            "exported_at": datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Conversation history saved to {filepath}")

# Sample data and demo functions
def create_sample_knowledge_base() -> List[Document]:
    """Create sample documents for the knowledge base"""
    documents = [
        Document(
            id="ai_overview",
            content="""Artificial Intelligence (AI) is a broad field of computer science focused on creating 
            systems that can perform tasks that typically require human intelligence. This includes learning, 
            reasoning, problem-solving, perception, and language understanding. AI can be categorized into 
            narrow AI (designed for specific tasks) and general AI (human-level intelligence across domains).""",
            metadata={"topic": "artificial_intelligence", "category": "overview", "source": "knowledge_base"}
        ),
        Document(
            id="machine_learning",
            content="""Machine Learning (ML) is a subset of artificial intelligence that enables computers 
            to learn and improve from experience without being explicitly programmed. ML algorithms build 
            mathematical models based on training data to make predictions or decisions. Common types include 
            supervised learning, unsupervised learning, and reinforcement learning.""",
            metadata={"topic": "machine_learning", "category": "definition", "source": "knowledge_base"}
        ),
        Document(
            id="deep_learning",
            content="""Deep Learning is a specialized area of machine learning that uses neural networks 
            with multiple layers (hence "deep") to model and understand complex patterns in data. It has 
            achieved remarkable success in image recognition, natural language processing, and speech 
            recognition. Deep learning models can automatically learn hierarchical representations of data.""",
            metadata={"topic": "deep_learning", "category": "technical", "source": "knowledge_base"}
        ),
        Document(
            id="nlp_fundamentals",
            content="""Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
            interpret, and manipulate human language. NLP combines computational linguistics with statistical 
            and machine learning models. Applications include text analysis, machine translation, sentiment 
            analysis, and chatbots. Modern NLP relies heavily on transformer architectures.""",
            metadata={"topic": "nlp", "category": "technical", "source": "knowledge_base"}
        ),
        Document(
            id="computer_vision",
            content="""Computer Vision is a field of AI that trains computers to interpret and understand 
            the visual world. Using digital images from cameras and videos, computer vision systems can 
            identify and classify objects, track movement, and even understand scenes. Applications include 
            facial recognition, autonomous vehicles, medical imaging, and quality control in manufacturing.""",
            metadata={"topic": "computer_vision", "category": "application", "source": "knowledge_base"}
        ),
        Document(
            id="rag_systems",
            content="""Retrieval-Augmented Generation (RAG) is an AI technique that combines information 
            retrieval with text generation. RAG systems first retrieve relevant documents from a knowledge 
            base, then use this context to generate more accurate and informed responses. This approach 
            helps overcome the limitations of pure generative models by grounding responses in factual information.""",
            metadata={"topic": "rag", "category": "advanced", "source": "knowledge_base"}
        )
    ]

    return documents

def demo_queries() -> List[str]:
    """Sample queries for testing the RAG pipeline"""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?", 
        "What are the applications of computer vision?",
        "Explain deep learning and neural networks",
        "What is RAG and how does it work?",
        "What are the different types of machine learning?",
        "How is NLP used in modern applications?"
    ]

async def main():
    """Main demonstration function"""
    print("🤖 DAY 12 - MINI RAG PIPELINE")
    print("=" * 50)
    print("Task: Connect Chroma → Retrieval → LLM → Answer")
    print()

    # Initialize RAG pipeline
    print("🔧 Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        collection_name="day12_knowledge_base",
        llm_provider="openai",  # Will fallback to mock if no API key
        llm_model="gpt-3.5-turbo"
    )

    # Add sample documents to knowledge base
    print("📚 Building Knowledge Base...")
    sample_docs = create_sample_knowledge_base()
    add_results = pipeline.add_documents(sample_docs)

    print(f"✅ Knowledge Base Created:")
    print(f"  • Documents added: {add_results['successful']}/{len(sample_docs)}")
    print(f"  • Processing time: {add_results['processing_time']:.3f}s")

    if add_results['failed'] > 0:
        print(f"  • Failed documents: {add_results['failed']}")

    # Get pipeline stats
    stats = pipeline.get_pipeline_stats()
    print(f"\n📊 Pipeline Statistics:")
    print(f"  • Vector Store: {stats['vector_store']['type']}")
    print(f"  • Documents in collection: {stats['vector_store']['document_count']}")
    print(f"  • LLM Provider: {stats['llm']['provider']} ({stats['llm']['model']})")
    print(f"  • Using real LLM: {stats['llm']['use_real_llm']}")

    # Interactive query demonstration
    print("\n🔍 RAG QUERY DEMONSTRATION")
    print("-" * 30)

    sample_queries = demo_queries()

    for i, query_text in enumerate(sample_queries[:3], 1):  # Demo first 3 queries
        print(f"\nQuery {i}: {query_text}")
        print("-" * 40)

        # Process query
        response = pipeline.query(query_text, max_results=3)

        # Display results
        print(f"📋 Retrieved {len(response.retrieved_docs)} documents:")
        for j, doc in enumerate(response.retrieved_docs, 1):
            print(f"  {j}. {doc.id} (topic: {doc.metadata.get('topic', 'unknown')})")
            print(f"     {doc.content[:100]}...")

        print(f"\n🤖 Generated Answer:")
        print(f"   {response.generated_answer}")

        print(f"\n⏱️  Processing time: {response.processing_time:.3f}s")

        # Add small delay between queries for demo effect
        await asyncio.sleep(1)

    # Final statistics
    final_stats = pipeline.get_pipeline_stats()
    print(f"\n📈 FINAL PIPELINE STATISTICS")
    print("-" * 35)
    print(f"Total queries processed: {final_stats['queries']['total_queries']}")
    print(f"Successful queries: {final_stats['queries']['successful_queries']}")
    print(f"Success rate: {final_stats['queries']['success_rate']:.1%}")
    print(f"Average query time: {final_stats['queries']['average_query_time']:.3f}s")

    # Save conversation history
    print("\n💾 Saving conversation history...")
    history_file = Path("logs/conversation_history.json")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_conversation_history(history_file)
    print(f"✅ History saved to: {history_file}")

    print("\n🎉 Day 12 Mini RAG Pipeline demonstration completed!")
    print("\n🔗 Integration Points:")
    print("  • Vector Store: ChromaDB with persistence")
    print("  • Embeddings: SentenceTransformers (all-MiniLM-L6-v2)")
    print("  • LLM: OpenAI GPT-3.5-turbo (with fallback)")
    print("  • Pipeline: Full RAG with logging and stats")

    return pipeline

if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Run main demo
    asyncio.run(main())
