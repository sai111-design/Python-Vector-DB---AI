#!/usr/bin/env python3
"""
Day 11 - Unit Tests for Prompt Engineering
"""

import unittest
from main import PromptTemplate, RAGPromptTemplate, PromptLibrary, RetrievedDocument

class TestPromptTemplate(unittest.TestCase):

    def setUp(self):
        self.template = PromptTemplate(
            name="test_template",
            template="Hello {name}, your score is {score}.",
            variables=["name", "score"]
        )

    def test_format_success(self):
        result = self.template.format(name="Alice", score=95)
        self.assertEqual(result, "Hello Alice, your score is 95.")

    def test_format_missing_variable(self):
        with self.assertRaises(ValueError):
            self.template.format(name="Alice")  # Missing score

    def test_usage_count(self):
        initial_count = self.template.usage_count
        self.template.format(name="Bob", score=88)
        self.assertEqual(self.template.usage_count, initial_count + 1)

class TestRAGPromptTemplate(unittest.TestCase):

    def setUp(self):
        self.rag_template = RAGPromptTemplate(
            name="test_rag",
            template="Context: {context}\n\nQuestion: {query}\n\nAnswer:",
            variables=["query", "context"],
            max_context_length=200
        )

        self.docs = [
            RetrievedDocument(
                content="Python is a programming language.",
                source="python_guide.pdf",
                score=0.9
            ),
            RetrievedDocument(
                content="It was created by Guido van Rossum.",
                source="history.pdf", 
                score=0.8
            )
        ]

    def test_format_with_context(self):
        result = self.rag_template.format_with_context(
            query="What is Python?",
            documents=self.docs
        )

        self.assertIn("What is Python?", result)
        self.assertIn("Python is a programming language", result)
        self.assertIn("Context:", result)

    def test_context_truncation(self):
        # Test with very small context limit
        small_template = RAGPromptTemplate(
            name="small",
            template="{context}",
            variables=["context"],
            max_context_length=50
        )

        long_docs = [
            RetrievedDocument(
                content="This is a very long document that should be truncated because it exceeds the maximum context length limit.",
                source="long_doc.pdf",
                score=0.9
            )
        ]

        result = small_template.format_with_context("test", long_docs)
        self.assertLess(len(result), 100)  # Should be truncated

    def test_empty_documents(self):
        result = self.rag_template.format_with_context(
            query="What is Python?",
            documents=[]
        )
        self.assertIn("No context found", result)

class TestPromptLibrary(unittest.TestCase):

    def setUp(self):
        self.library = PromptLibrary()

    def test_default_templates_loaded(self):
        templates = self.library.list_templates()
        self.assertIn("basic_qa", templates)
        self.assertIn("detailed_analysis", templates)

    def test_get_template(self):
        template = self.library.get_template("basic_qa")
        self.assertIsNotNone(template)
        self.assertEqual(template.name, "basic_qa")

    def test_add_custom_template(self):
        custom_template = PromptTemplate(
            name="custom",
            template="Custom: {text}",
            variables=["text"]
        )

        self.library.add_template(custom_template)
        retrieved = self.library.get_template("custom")
        self.assertEqual(retrieved.name, "custom")

    def test_nonexistent_template(self):
        result = self.library.get_template("nonexistent")
        self.assertIsNone(result)

class TestRetrievedDocument(unittest.TestCase):

    def test_document_creation(self):
        doc = RetrievedDocument(
            content="Test content",
            source="test.pdf",
            score=0.95,
            metadata={"page": 1}
        )

        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.source, "test.pdf")
        self.assertEqual(doc.score, 0.95)
        self.assertEqual(doc.metadata["page"], 1)

if __name__ == "__main__":
    # Run tests
    print("Running Day 11 Prompt Engineering Tests...")
    unittest.main(verbosity=2)
