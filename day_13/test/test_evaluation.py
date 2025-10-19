#!/usr/bin/env python3
"""
Day 13 - Unit Tests for Retrieval Evaluation System
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import (
    RetrievalEvaluator, EvaluationMetrics, RetrievalResult, 
    Document, Query, create_sample_ground_truth, simulate_retrieval_results
)

class TestDocument(unittest.TestCase):
    """Test Document class"""

    def test_document_creation(self):
        doc = Document("doc_1", "Test content", {"source": "test"})
        self.assertEqual(doc.id, "doc_1")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["source"], "test")

    def test_document_with_empty_metadata(self):
        doc = Document("doc_1", "Content")
        self.assertIsInstance(doc.metadata, dict)
        self.assertEqual(len(doc.metadata), 0)

class TestQuery(unittest.TestCase):
    """Test Query class"""

    def test_query_creation(self):
        query = Query("q1", "Test query", {"doc_1", "doc_2"})
        self.assertEqual(query.id, "q1")
        self.assertEqual(query.text, "Test query")
        self.assertEqual(query.relevant_doc_ids, {"doc_1", "doc_2"})

    def test_query_with_list_relevant_docs(self):
        query = Query("q1", "Test", ["doc_1", "doc_2"])
        self.assertIsInstance(query.relevant_doc_ids, set)
        self.assertEqual(query.relevant_doc_ids, {"doc_1", "doc_2"})

class TestRetrievalResult(unittest.TestCase):
    """Test RetrievalResult class"""

    def test_retrieval_result_creation(self):
        result = RetrievalResult("q1", [("doc_1", 0.9), ("doc_2", 0.7)], 0.05)
        self.assertEqual(result.query_id, "q1")
        self.assertEqual(len(result.retrieved_docs), 2)
        self.assertEqual(result.retrieval_time, 0.05)

    def test_get_top_k(self):
        result = RetrievalResult("q1", [("doc_1", 0.9), ("doc_2", 0.7), ("doc_3", 0.5)])
        top_2 = result.get_top_k(2)
        self.assertEqual(top_2, ["doc_1", "doc_2"])

class TestEvaluationMetrics(unittest.TestCase):
    """Test EvaluationMetrics class"""

    def test_metrics_creation(self):
        metrics = EvaluationMetrics(
            precision=0.75, recall=0.60, f1=0.67, 
            map_score=0.72, ndcg=0.68, hit_rate=1.0
        )
        self.assertEqual(metrics.precision, 0.75)
        self.assertEqual(metrics.recall, 0.60)
        self.assertEqual(metrics.f1, 0.67)

    def test_to_dict(self):
        metrics = EvaluationMetrics(precision=0.8, recall=0.7, f1=0.75)
        metrics_dict = metrics.to_dict()

        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict["precision"], 0.8)
        self.assertEqual(metrics_dict["recall"], 0.7)
        self.assertEqual(metrics_dict["f1"], 0.75)

class TestRetrievalEvaluator(unittest.TestCase):
    """Test RetrievalEvaluator class"""

    def setUp(self):
        self.evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

        # Add test documents
        self.docs = [
            Document("doc_1", "AI and machine learning", {"topic": "ai"}),
            Document("doc_2", "Deep learning networks", {"topic": "dl"}),
            Document("doc_3", "Natural language processing", {"topic": "nlp"}),
            Document("doc_4", "Computer vision systems", {"topic": "cv"}),
            Document("doc_5", "Reinforcement learning agents", {"topic": "rl"})
        ]

        for doc in self.docs:
            self.evaluator.add_document(doc)

        # Add test queries
        self.queries = [
            Query("q1", "What is AI?", {"doc_1"}),
            Query("q2", "Deep learning explanation", {"doc_2"}),
            Query("q3", "Machine learning types", {"doc_1", "doc_2", "doc_5"})
        ]

        for query in self.queries:
            self.evaluator.add_query(query)

    def test_document_addition(self):
        self.assertEqual(len(self.evaluator.documents), 5)
        self.assertIn("doc_1", self.evaluator.documents)
        self.assertEqual(self.evaluator.documents["doc_1"].content, "AI and machine learning")

    def test_query_addition(self):
        self.assertEqual(len(self.evaluator.queries), 3)
        self.assertIn("q1", self.evaluator.queries)
        self.assertEqual(self.evaluator.queries["q1"].text, "What is AI?")

    def test_precision_calculation(self):
        # Perfect precision case
        retrieved = ["doc_1", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        precision = self.evaluator.calculate_precision_at_k(retrieved, relevant, 2)
        self.assertEqual(precision, 1.0)

        # Partial precision case
        retrieved = ["doc_1", "doc_3", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        precision = self.evaluator.calculate_precision_at_k(retrieved, relevant, 3)
        self.assertAlmostEqual(precision, 2/3, places=3)

    def test_recall_calculation(self):
        # Perfect recall case
        retrieved = ["doc_1", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        recall = self.evaluator.calculate_recall_at_k(retrieved, relevant, 2)
        self.assertEqual(recall, 1.0)

        # Partial recall case
        retrieved = ["doc_1", "doc_3"]
        relevant = {"doc_1", "doc_2"}
        recall = self.evaluator.calculate_recall_at_k(retrieved, relevant, 2)
        self.assertEqual(recall, 0.5)

    def test_f1_calculation(self):
        # Equal precision and recall
        f1 = self.evaluator.calculate_f1_at_k(0.8, 0.8)
        self.assertEqual(f1, 0.8)

        # Different precision and recall
        f1 = self.evaluator.calculate_f1_at_k(0.6, 0.8)
        expected_f1 = 2 * 0.6 * 0.8 / (0.6 + 0.8)
        self.assertAlmostEqual(f1, expected_f1, places=3)

        # Zero case
        f1 = self.evaluator.calculate_f1_at_k(0.0, 0.0)
        self.assertEqual(f1, 0.0)

    def test_average_precision(self):
        # Perfect ranking
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_2"}
        ap = self.evaluator.calculate_average_precision(retrieved, relevant)
        expected_ap = (1/1 + 2/2) / 2  # (1.0 + 1.0) / 2 = 1.0
        self.assertEqual(ap, expected_ap)

        # Imperfect ranking  
        retrieved = ["doc_1", "doc_3", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        ap = self.evaluator.calculate_average_precision(retrieved, relevant)
        expected_ap = (1/1 + 2/3) / 2  # (1.0 + 0.667) / 2 = 0.833
        self.assertAlmostEqual(ap, expected_ap, places=3)

    def test_reciprocal_rank(self):
        # First position
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1"}
        rr = self.evaluator.calculate_reciprocal_rank(retrieved, relevant)
        self.assertEqual(rr, 1.0)

        # Third position
        retrieved = ["doc_3", "doc_4", "doc_1"]
        relevant = {"doc_1"}
        rr = self.evaluator.calculate_reciprocal_rank(retrieved, relevant)
        self.assertAlmostEqual(rr, 1/3, places=3)

        # Not found
        retrieved = ["doc_3", "doc_4", "doc_5"]
        relevant = {"doc_1"}
        rr = self.evaluator.calculate_reciprocal_rank(retrieved, relevant)
        self.assertEqual(rr, 0.0)

    def test_ndcg_calculation(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_2"}
        ndcg = self.evaluator.calculate_ndcg_at_k(retrieved, relevant, 3)

        # NDCG should be between 0 and 1
        self.assertTrue(0 <= ndcg <= 1)

        # Perfect ranking should have high NDCG
        self.assertGreater(ndcg, 0.8)

    def test_hit_rate_calculation(self):
        # Hit case
        retrieved = ["doc_3", "doc_1", "doc_4"]
        relevant = {"doc_1"}
        hit_rate = self.evaluator.calculate_hit_rate_at_k(retrieved, relevant, 3)
        self.assertEqual(hit_rate, 1.0)

        # Miss case
        retrieved = ["doc_3", "doc_4", "doc_5"]
        relevant = {"doc_1"}
        hit_rate = self.evaluator.calculate_hit_rate_at_k(retrieved, relevant, 3)
        self.assertEqual(hit_rate, 0.0)

    def test_single_query_evaluation(self):
        retrieved_docs = [("doc_1", 0.9), ("doc_3", 0.7), ("doc_2", 0.6)]
        metrics = self.evaluator.evaluate_single_query("q1", retrieved_docs, k=3)

        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('ap', metrics)
        self.assertIn('rr', metrics)
        self.assertIn('ndcg', metrics)
        self.assertIn('hit_rate', metrics)

        # Check that metrics are reasonable
        self.assertTrue(0 <= metrics['precision'] <= 1)
        self.assertTrue(0 <= metrics['recall'] <= 1)
        self.assertTrue(0 <= metrics['f1'] <= 1)

    def test_batch_evaluation(self):
        results = [
            RetrievalResult("q1", [("doc_1", 0.9), ("doc_2", 0.8)]),
            RetrievalResult("q2", [("doc_2", 0.9), ("doc_1", 0.7)]),
        ]

        metrics = self.evaluator.evaluate_batch(results, k=2)

        self.assertIsInstance(metrics, EvaluationMetrics)
        self.assertTrue(0 <= metrics.precision <= 1)
        self.assertTrue(0 <= metrics.recall <= 1)
        self.assertTrue(0 <= metrics.f1 <= 1)

    def test_multiple_k_evaluation(self):
        results = [
            RetrievalResult("q1", [("doc_1", 0.9), ("doc_3", 0.7), ("doc_2", 0.6)]),
            RetrievalResult("q2", [("doc_2", 0.9), ("doc_1", 0.8), ("doc_4", 0.5)]),
        ]

        k_results = self.evaluator.evaluate_at_multiple_k(results)

        # Check that we get results for all K values
        self.assertEqual(len(k_results), len(self.evaluator.k_values))
        for k in self.evaluator.k_values:
            self.assertIn(k, k_results)
            self.assertIsInstance(k_results[k], EvaluationMetrics)

    def test_load_ground_truth(self):
        # Create temporary ground truth file
        ground_truth = {
            "documents": [
                {"id": "temp_doc_1", "content": "Test document", "metadata": {}}
            ],
            "queries": [
                {"id": "temp_q1", "text": "Test query", "relevant_doc_ids": ["temp_doc_1"], "metadata": {}}
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ground_truth, f)
            temp_path = f.name

        try:
            new_evaluator = RetrievalEvaluator()
            new_evaluator.load_ground_truth(Path(temp_path))

            self.assertEqual(len(new_evaluator.documents), 1)
            self.assertEqual(len(new_evaluator.queries), 1)
            self.assertIn("temp_doc_1", new_evaluator.documents)
            self.assertIn("temp_q1", new_evaluator.queries)

        finally:
            Path(temp_path).unlink()

class TestSampleDataGeneration(unittest.TestCase):
    """Test sample data generation functions"""

    def test_create_sample_ground_truth(self):
        gt = create_sample_ground_truth()

        self.assertIn("documents", gt)
        self.assertIn("queries", gt)
        self.assertIn("metadata", gt)

        self.assertTrue(len(gt["documents"]) > 0)
        self.assertTrue(len(gt["queries"]) > 0)

        # Check document structure
        doc = gt["documents"][0]
        self.assertIn("id", doc)
        self.assertIn("content", doc)
        self.assertIn("metadata", doc)

        # Check query structure
        query = gt["queries"][0]
        self.assertIn("id", query)
        self.assertIn("text", query)
        self.assertIn("relevant_doc_ids", query)
        self.assertIn("metadata", query)

    def test_simulate_retrieval_results(self):
        results = simulate_retrieval_results()

        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], RetrievalResult)

        # Check result structure
        result = results[0]
        self.assertIsInstance(result.query_id, str)
        self.assertIsInstance(result.retrieved_docs, list)
        self.assertTrue(len(result.retrieved_docs) > 0)
        self.assertIsInstance(result.retrieval_time, (int, float))

        # Check retrieved document format
        doc_id, score = result.retrieved_docs[0]
        self.assertIsInstance(doc_id, str)
        self.assertIsInstance(score, (int, float))
        self.assertTrue(0 <= score <= 1)

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def setUp(self):
        self.evaluator = RetrievalEvaluator()

    def test_empty_retrieval_results(self):
        # Test with empty results list
        metrics = self.evaluator.evaluate_batch([], k=5)
        self.assertIsInstance(metrics, EvaluationMetrics)
        self.assertEqual(metrics.precision, 0.0)

    def test_query_not_in_ground_truth(self):
        # Add a document but no query
        self.evaluator.add_document(Document("doc_1", "Content"))

        with self.assertRaises(ValueError):
            self.evaluator.evaluate_single_query("nonexistent_query", [("doc_1", 0.9)], k=1)

    def test_no_relevant_documents(self):
        # Query with no relevant documents
        self.evaluator.add_document(Document("doc_1", "Content"))
        self.evaluator.add_query(Query("q1", "Query", set()))  # Empty relevant set

        metrics = self.evaluator.evaluate_single_query("q1", [("doc_1", 0.9)], k=1)
        self.assertEqual(metrics['recall'], 0.0)

    def test_no_retrieved_documents(self):
        # Test with empty retrieved documents
        self.evaluator.add_document(Document("doc_1", "Content"))
        self.evaluator.add_query(Query("q1", "Query", {"doc_1"}))

        metrics = self.evaluator.evaluate_single_query("q1", [], k=5)
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)

    def test_k_larger_than_retrieved(self):
        # Test when K is larger than number of retrieved documents
        self.evaluator.add_document(Document("doc_1", "Content"))
        self.evaluator.add_query(Query("q1", "Query", {"doc_1"}))

        metrics = self.evaluator.evaluate_single_query("q1", [("doc_1", 0.9)], k=10)

        # Should still work correctly
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['hit_rate'], 1.0)

def run_all_tests():
    """Run all test suites"""
    print("🧪 Running Day 13 Retrieval Evaluation Tests")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDocument,
        TestQuery, 
        TestRetrievalResult,
        TestEvaluationMetrics,
        TestRetrievalEvaluator,
        TestSampleDataGeneration,
        TestEdgeCases
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        print("Day 13 retrieval evaluation system is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED!")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  • {test}: {traceback.split('AssertionError:')[-1].strip()}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  • {test}: {traceback.split('Error:')[-1].strip()}")

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
