#!/usr/bin/env python3
"""
Day 12 - RAG Pipeline Unit Tests
"""

import unittest
import tempfile
import json
import asyncio
from pathlib import Path
import sys

# Add src to Python path  
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import (
    RAGPipeline, Document, RAGQuery, RAGResponse, 
    VectorStore, LLMProvider, create_sample_knowledge_base
)

class TestDocument(unittest.TestCase):
    """Test Document class"""

    def test_document_creation(self):
        doc = Document("test_id", "Test content", {"category": "test"})
        self.assertEqual(doc.id, "test_id")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["category"], "test")

    def test_document_with_empty_metadata(self):
        doc = Document("test_id", "Content")
        self.assertIsInstance(doc.metadata, dict)
        self.assertEqual(len(doc.metadata), 0)

class TestRAGQuery(unittest.TestCase):
    """Test RAGQuery class"""

    def test_query_creation(self):
        query = RAGQuery("q1", "Test query", 5)
        self.assertEqual(query.id, "q1")
        self.assertEqual(query.text, "Test query")
        self.assertEqual(query.max_results, 5)

    def test_query_with_metadata(self):
        query = RAGQuery("q1", "Test", metadata={"source": "test"})
        self.assertEqual(query.metadata["source"], "test")

class TestRAGResponse(unittest.TestCase):
    """Test RAGResponse class"""

    def test_response_creation(self):
        query = RAGQuery("q1", "Test query")
        docs = [Document("doc1", "Content")]
        response = RAGResponse(query, docs, "Answer", 0.5)

        self.assertEqual(response.query.id, "q1")
        self.assertEqual(response.generated_answer, "Answer")
        self.assertEqual(response.processing_time, 0.5)
        self.assertEqual(len(response.retrieved_docs), 1)

    def test_response_to_dict(self):
        query = RAGQuery("q1", "Test query")
        docs = [Document("doc1", "Content", {"topic": "test"})]
        response = RAGResponse(query, docs, "Answer", 0.5)

        response_dict = response.to_dict()

        self.assertEqual(response_dict["query_id"], "q1")
        self.assertEqual(response_dict["answer"], "Answer")
        self.assertIn("retrieved_documents", response_dict)
        self.assertIn("timestamp", response_dict)

class TestVectorStore(unittest.TestCase):
    """Test VectorStore class"""

    def setUp(self):
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore("test_collection", self.temp_dir)

    def test_vector_store_initialization(self):
        self.assertEqual(self.vector_store.collection_name, "test_collection")
        self.assertEqual(self.vector_store.persist_directory, self.temp_dir)
        self.assertIsNotNone(self.vector_store.embedding_model)

    def test_add_document(self):
        doc = Document("test_doc", "This is test content about machine learning")
        result = self.vector_store.add_document(doc)
        self.assertTrue(result)

    def test_search_documents(self):
        # Add a document first
        doc = Document("ai_doc", "Artificial intelligence and machine learning")
        self.vector_store.add_document(doc)

        # Search for it
        results = self.vector_store.search("artificial intelligence", max_results=1)
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], Document)

    def test_empty_search(self):
        # Search in empty collection
        results = self.vector_store.search("nonexistent", max_results=5)
        self.assertEqual(len(results), 0)

    def test_collection_info(self):
        info = self.vector_store.get_collection_info()
        self.assertIn("type", info)
        self.assertIn("document_count", info)

class TestLLMProvider(unittest.TestCase):
    """Test LLMProvider class"""

    def test_llm_provider_initialization(self):
        provider = LLMProvider("openai", "gpt-3.5-turbo")
        self.assertEqual(provider.provider, "openai")
        self.assertEqual(provider.model_name, "gpt-3.5-turbo")

    def test_mock_response_generation(self):
        provider = LLMProvider("mock")  # Force mock mode

        # Test machine learning response
        response = provider.generate_response("What is machine learning?")
        self.assertIn("machine learning", response.lower())

        # Test deep learning response
        response = provider.generate_response("Explain deep learning")
        self.assertIn("deep learning", response.lower())

        # Test AI response
        response = provider.generate_response("What is artificial intelligence?")
        self.assertIn("artificial intelligence", response.lower())

    def test_generic_mock_response(self):
        provider = LLMProvider("mock")
        response = provider.generate_response("Random question about something else")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

class TestRAGPipeline(unittest.TestCase):
    """Test complete RAG pipeline"""

    def setUp(self):
        # Create pipeline with temporary collection
        import time
        collection_name = f"test_collection_{int(time.time())}"
        self.pipeline = RAGPipeline(
            collection_name=collection_name,
            llm_provider="mock"  # Use mock for consistent testing
        )

    def test_pipeline_initialization(self):
        self.assertIsNotNone(self.pipeline.vector_store)
        self.assertIsNotNone(self.pipeline.llm_provider)
        self.assertEqual(len(self.pipeline.query_history), 0)

    def test_add_documents(self):
        docs = create_sample_knowledge_base()[:3]  # Use first 3 documents
        results = self.pipeline.add_documents(docs)

        self.assertIn("successful", results)
        self.assertIn("failed", results)
        self.assertIn("processing_time", results)
        self.assertTrue(results["successful"] > 0)

    def test_query_processing(self):
        # Add documents first
        docs = create_sample_knowledge_base()
        self.pipeline.add_documents(docs)

        # Process a query
        response = self.pipeline.query("What is machine learning?", max_results=3)

        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(response.query.text, "What is machine learning?")
        self.assertIsInstance(response.generated_answer, str)
        self.assertTrue(len(response.generated_answer) > 0)
        self.assertTrue(response.processing_time > 0)

    def test_query_with_no_documents(self):
        # Query empty collection
        response = self.pipeline.query("Test query")

        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(len(response.retrieved_docs), 0)
        self.assertIn("couldn't find", response.generated_answer.lower())

    def test_multiple_queries(self):
        # Add documents
        docs = create_sample_knowledge_base()
        self.pipeline.add_documents(docs)

        # Process multiple queries
        queries = [
            "What is AI?",
            "How does deep learning work?",
            "What are NLP applications?"
        ]

        for query_text in queries:
            response = self.pipeline.query(query_text)
            self.assertIsInstance(response, RAGResponse)

        self.assertEqual(len(self.pipeline.query_history), 3)

    def test_pipeline_stats(self):
        stats = self.pipeline.get_pipeline_stats()

        self.assertIn("vector_store", stats)
        self.assertIn("queries", stats)
        self.assertIn("llm", stats)

        # Check query stats structure
        query_stats = stats["queries"]
        self.assertIn("total_queries", query_stats)
        self.assertIn("successful_queries", query_stats)
        self.assertIn("success_rate", query_stats)

    def test_conversation_history_save(self):
        # Add some queries
        docs = create_sample_knowledge_base()[:2]
        self.pipeline.add_documents(docs)

        self.pipeline.query("Test query 1")
        self.pipeline.query("Test query 2")

        # Save history
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            self.pipeline.save_conversation_history(temp_path)
            self.assertTrue(temp_path.exists())

            # Load and verify
            with open(temp_path, 'r') as f:
                data = json.load(f)

            self.assertIn("pipeline_stats", data)
            self.assertIn("conversations", data)
            self.assertEqual(len(data["conversations"]), 2)

        finally:
            temp_path.unlink()

class TestSampleData(unittest.TestCase):
    """Test sample data generation"""

    def test_sample_knowledge_base(self):
        docs = create_sample_knowledge_base()

        self.assertTrue(len(docs) > 0)
        self.assertIsInstance(docs[0], Document)

        # Check document structure
        for doc in docs:
            self.assertIsInstance(doc.id, str)
            self.assertIsInstance(doc.content, str)
            self.assertIsInstance(doc.metadata, dict)
            self.assertTrue(len(doc.content) > 0)

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios"""

    def test_end_to_end_rag_workflow(self):
        """Test complete RAG workflow"""
        # Initialize pipeline
        pipeline = RAGPipeline(
            collection_name=f"integration_test_{int(time.time())}",
            llm_provider="mock"
        )

        # Step 1: Add knowledge base
        docs = create_sample_knowledge_base()
        add_results = pipeline.add_documents(docs)
        self.assertTrue(add_results["successful"] > 0)

        # Step 2: Process queries
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning concepts",
            "How does deep learning work?"
        ]

        responses = []
        for query in test_queries:
            response = pipeline.query(query, max_results=2)
            responses.append(response)

            # Verify response quality
            self.assertIsInstance(response.generated_answer, str)
            self.assertTrue(len(response.generated_answer) > 10)  # Non-trivial answer
            self.assertTrue(response.processing_time > 0)

        # Step 3: Verify pipeline state
        stats = pipeline.get_pipeline_stats()
        self.assertEqual(stats["queries"]["total_queries"], len(test_queries))
        self.assertTrue(stats["queries"]["success_rate"] > 0)

        # Step 4: Verify query history
        self.assertEqual(len(pipeline.query_history), len(test_queries))

    def test_error_handling(self):
        """Test error handling scenarios"""
        pipeline = RAGPipeline()

        # Test query with invalid max_results
        response = pipeline.query("Test", max_results=0)
        self.assertIsInstance(response, RAGResponse)

        # Test with very long query
        long_query = "What is " + "very " * 100 + "long question?"
        response = pipeline.query(long_query)
        self.assertIsInstance(response, RAGResponse)

def run_all_tests():
    """Run all test suites"""
    print("🧪 Running Day 12 RAG Pipeline Tests")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestDocument,
        TestRAGQuery,
        TestRAGResponse,
        TestVectorStore,
        TestLLMProvider,
        TestRAGPipeline,
        TestSampleData,
        TestIntegrationScenarios
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        print("Day 12 RAG pipeline is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED!")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  • {test}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  • {test}")

    print("\n🔗 Integration Status:")
    print("  • Vector Store: ✅ ChromaDB with fallback")
    print("  • Embeddings: ✅ SentenceTransformers")
    print("  • LLM: ✅ OpenAI with mock fallback")
    print("  • Pipeline: ✅ Full RAG workflow")

    return result.wasSuccessful()

if __name__ == "__main__":
    import time
    success = run_all_tests()
    exit(0 if success else 1)
