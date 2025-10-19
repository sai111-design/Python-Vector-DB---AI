#!/usr/bin/env python3
"""
Day 11 - Prompt Engineering System
Task: Build a prompt template that takes a query + retrieved docs
"""

import json
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievedDocument:
    content: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ConversationTurn:
    role: str
    content: str

class PromptTemplate:
    def __init__(self, name: str, template: str, variables: List[str]):
        self.name = name
        self.template = template
        self.variables = variables
        self.usage_count = 0

    def format(self, **kwargs) -> str:
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        self.usage_count += 1
        return self.template.format(**kwargs)

class RAGPromptTemplate(PromptTemplate):
    def __init__(self, name: str, template: str, variables: List[str], 
                 max_context_length: int = 4000):
        super().__init__(name, template, variables)
        self.max_context_length = max_context_length

    def format_with_context(self, query: str, documents: List[RetrievedDocument]) -> str:
        context = self._build_context(documents)
        return self.format(query=query, context=context)

    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        if not documents:
            return "No context found."

        parts = []
        for i, doc in enumerate(documents, 1):
            content = f"Document {i}: {doc.content}"
            parts.append(content)
        return "\n\n".join(parts)

class PromptLibrary:
    def __init__(self):
        self.templates = {}
        self._load_defaults()

    def add_template(self, template: PromptTemplate):
        self.templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        return list(self.templates.keys())

    def _load_defaults(self):
        qa_template = RAGPromptTemplate(
            name="basic_qa",
            template="Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            variables=["query", "context"]
        )

        analysis_template = RAGPromptTemplate(
            name="detailed_analysis",
            template="Context:\n{context}\n\nQuestion: {query}\n\nProvide detailed analysis:\n1. Answer\n2. Evidence\n3. Confidence\n\nResponse:",
            variables=["query", "context"]
        )

        self.add_template(qa_template)
        self.add_template(analysis_template)

# Demo function
def demo():
    library = PromptLibrary()
    template = library.get_template("basic_qa")

    # Sample documents
    docs = [
        RetrievedDocument(
            content="The theory of relativity was introduced by Albert Einstein in 1905.",
            source="physics_textbook.pdf",
            score=0.95
        ),
        RetrievedDocument(
            content="Einstein's work revolutionized our understanding of space and time.",
            source="science_encyclopedia.pdf", 
            score=0.87
        )
    ]

    query = "Who introduced the theory of relativity and when?"
    prompt = template.format_with_context(query, docs)

    print("=" * 60)
    print("DAY 11 - PROMPT ENGINEERING DEMO")
    print("=" * 60)
    print("Query:", query)
    print("\nGenerated Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print(f"\nTemplate used: {template.name}")
    print(f"Documents processed: {len(docs)}")
    print(f"Template usage count: {template.usage_count}")

if __name__ == "__main__":
    demo()
