#!/usr/bin/env python3
"""
Day 13 - Benchmark Datasets for Retrieval Evaluation
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset"""
    name: str
    description: str
    documents: List[Dict[str, Any]]
    queries: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class BenchmarkLoader:
    """Loader for standard IR benchmark datasets"""

    def __init__(self, data_dir: Path = Path("data/benchmarks")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def create_msmarco_sample(self) -> BenchmarkDataset:
        """Create a sample dataset inspired by MS MARCO"""
        documents = [
            {
                "id": "msmarco_1",
                "content": "The capital of France is Paris. Paris is located in the north-central part of the country and is the largest city in France with over 2 million residents.",
                "metadata": {"source": "geography", "length": "short"}
            },
            {
                "id": "msmarco_2", 
                "content": "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures make it attractive for Rapid Application Development.",
                "metadata": {"source": "technology", "length": "medium"}
            },
            {
                "id": "msmarco_3",
                "content": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate changes are natural, scientific evidence shows that human activities have been the main driver since the 1800s.",
                "metadata": {"source": "environment", "length": "long"}
            },
            {
                "id": "msmarco_4",
                "content": "The human heart is a muscular organ that pumps blood throughout the body via the circulatory system. It has four chambers: two atria and two ventricles.",
                "metadata": {"source": "biology", "length": "medium"}
            },
            {
                "id": "msmarco_5",
                "content": "Machine learning algorithms can be broadly classified into supervised, unsupervised, and reinforcement learning based on the type of learning signal or feedback available.",
                "metadata": {"source": "technology", "length": "medium"}
            },
            {
                "id": "msmarco_6",
                "content": "The Renaissance was a cultural movement that spanned roughly from the 14th to the 17th century, beginning in Italy and later spreading to the rest of Europe.",
                "metadata": {"source": "history", "length": "short"}
            },
            {
                "id": "msmarco_7",
                "content": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
                "metadata": {"source": "biology", "length": "short"}
            },
            {
                "id": "msmarco_8",
                "content": "Quantum computing harnesses quantum mechanical phenomena such as superposition and entanglement to process information in ways that classical computers cannot.",
                "metadata": {"source": "technology", "length": "long"}
            }
        ]

        queries = [
            {
                "id": "q_geo_1",
                "text": "What is the capital of France?",
                "relevant_doc_ids": ["msmarco_1"],
                "metadata": {"type": "factual", "domain": "geography"}
            },
            {
                "id": "q_tech_1", 
                "text": "What is Python programming language?",
                "relevant_doc_ids": ["msmarco_2"],
                "metadata": {"type": "definition", "domain": "technology"}
            },
            {
                "id": "q_env_1",
                "text": "What causes climate change?",
                "relevant_doc_ids": ["msmarco_3"],
                "metadata": {"type": "causal", "domain": "environment"}
            },
            {
                "id": "q_bio_1",
                "text": "How many chambers does the human heart have?",
                "relevant_doc_ids": ["msmarco_4"],
                "metadata": {"type": "factual", "domain": "biology"}
            },
            {
                "id": "q_tech_2",
                "text": "What are the types of machine learning?",
                "relevant_doc_ids": ["msmarco_5"],
                "metadata": {"type": "classification", "domain": "technology"}
            },
            {
                "id": "q_hist_1",
                "text": "When did the Renaissance period occur?",
                "relevant_doc_ids": ["msmarco_6"],
                "metadata": {"type": "temporal", "domain": "history"}
            },
            {
                "id": "q_bio_2",
                "text": "How do plants make energy?",
                "relevant_doc_ids": ["msmarco_7"],
                "metadata": {"type": "process", "domain": "biology"}
            },
            {
                "id": "q_tech_3",
                "text": "What is quantum computing?",
                "relevant_doc_ids": ["msmarco_8"],
                "metadata": {"type": "definition", "domain": "technology"}
            }
        ]

        return BenchmarkDataset(
            name="MS MARCO Sample",
            description="Sample dataset inspired by MS MARCO for retrieval evaluation",
            documents=documents,
            queries=queries,
            metadata={
                "total_documents": len(documents),
                "total_queries": len(queries),
                "domains": ["geography", "technology", "environment", "biology", "history"],
                "query_types": ["factual", "definition", "causal", "classification", "temporal", "process"]
            }
        )

    def create_trec_sample(self) -> BenchmarkDataset:
        """Create a sample dataset inspired by TREC"""
        documents = [
            {
                "id": "trec_doc_1",
                "content": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans. AI research deals with the question of how to create computers that are capable of intelligent behavior.",
                "metadata": {"category": "DESCRIPTION", "complexity": "intermediate"}
            },
            {
                "id": "trec_doc_2",
                "content": "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south and is bounded by Asia and Australia in the west and the Americas in the east.",
                "metadata": {"category": "LOCATION", "complexity": "basic"}
            },
            {
                "id": "trec_doc_3", 
                "content": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language. He was born in Stratford-upon-Avon in 1564 and died in 1616.",
                "metadata": {"category": "PERSON", "complexity": "basic"}
            },
            {
                "id": "trec_doc_4",
                "content": "The speed of light in a vacuum is exactly 299,792,458 meters per second. This is a universal physical constant and forms the basis for the definition of the meter in the International System of Units.",
                "metadata": {"category": "NUMERIC", "complexity": "intermediate"}
            },
            {
                "id": "trec_doc_5",
                "content": "Photosynthesis occurs when plants use chlorophyll to convert carbon dioxide and water into glucose and oxygen using energy from sunlight. This process typically happens in the leaves of plants.",
                "metadata": {"category": "DESCRIPTION", "complexity": "advanced"}
            }
        ]

        queries = [
            {
                "id": "trec_q1",
                "text": "What is artificial intelligence?",
                "relevant_doc_ids": ["trec_doc_1"],
                "metadata": {"category": "DESCRIPTION", "answer_type": "definition"}
            },
            {
                "id": "trec_q2",
                "text": "Which is the largest ocean?",
                "relevant_doc_ids": ["trec_doc_2"],
                "metadata": {"category": "LOCATION", "answer_type": "entity"}
            },
            {
                "id": "trec_q3",
                "text": "Who wrote Hamlet?",
                "relevant_doc_ids": ["trec_doc_3"],
                "metadata": {"category": "PERSON", "answer_type": "person"}
            },
            {
                "id": "trec_q4",
                "text": "What is the speed of light?",
                "relevant_doc_ids": ["trec_doc_4"],
                "metadata": {"category": "NUMERIC", "answer_type": "number"}
            },
            {
                "id": "trec_q5",
                "text": "How does photosynthesis work?",
                "relevant_doc_ids": ["trec_doc_5"],
                "metadata": {"category": "DESCRIPTION", "answer_type": "process"}
            }
        ]

        return BenchmarkDataset(
            name="TREC Sample",
            description="Sample dataset inspired by TREC for question classification and retrieval",
            documents=documents,
            queries=queries,
            metadata={
                "total_documents": len(documents),
                "total_queries": len(queries),
                "categories": ["DESCRIPTION", "LOCATION", "PERSON", "NUMERIC"],
                "complexity_levels": ["basic", "intermediate", "advanced"]
            }
        )

    def save_dataset(self, dataset: BenchmarkDataset, filename: str):
        """Save dataset to file"""
        filepath = self.data_dir / filename

        data = {
            "name": dataset.name,
            "description": dataset.description,
            "documents": dataset.documents,
            "queries": dataset.queries,
            "metadata": dataset.metadata
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved dataset '{dataset.name}' to {filepath}")

    def load_dataset(self, filename: str) -> BenchmarkDataset:
        """Load dataset from file"""
        filepath = self.data_dir / filename

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return BenchmarkDataset(
            name=data["name"],
            description=data["description"],
            documents=data["documents"],
            queries=data["queries"],
            metadata=data["metadata"]
        )

    def create_all_benchmarks(self):
        """Create and save all benchmark datasets"""
        # MS MARCO Sample
        msmarco = self.create_msmarco_sample()
        self.save_dataset(msmarco, "msmarco_sample.json")

        # TREC Sample
        trec = self.create_trec_sample()
        self.save_dataset(trec, "trec_sample.json")

        print(f"✅ Created benchmark datasets in {self.data_dir}")
        print(f"  • MS MARCO Sample: {len(msmarco.documents)} docs, {len(msmarco.queries)} queries")
        print(f"  • TREC Sample: {len(trec.documents)} docs, {len(trec.queries)} queries")

def main():
    """Demo function"""
    loader = BenchmarkLoader()
    loader.create_all_benchmarks()

if __name__ == "__main__":
    main()
