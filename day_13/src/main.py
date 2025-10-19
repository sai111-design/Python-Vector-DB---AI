#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 13 - Evaluate Retrieval System
==================================
Task: Manually test retrieval precision/recall

This module provides comprehensive evaluation tools for retrieval systems:
- Precision, Recall, F1, MAP, NDCG metrics
- Ground truth management
- Automated evaluation pipelines
- Visualization and reporting
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document in the corpus"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

@dataclass
class Query:
    """Represents a query with ground truth"""
    id: str
    text: str
    relevant_doc_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """Represents retrieval results for a query"""
    query_id: str
    retrieved_docs: List[Tuple[str, float]]  # (doc_id, score)
    retrieval_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map_score: float = 0.0  # Mean Average Precision
    mrr: float = 0.0        # Mean Reciprocal Rank
    ndcg: float = 0.0       # Normalized Discounted Cumulative Gain
    hit_rate: float = 0.0   # Hit Rate at k

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "map": self.map_score,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate
        }

class RetrievalEvaluator:
    """Main class for evaluating retrieval systems"""

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.queries: Dict[str, Query] = {}
        self.documents: Dict[str, Document] = {}
        self.results_history: List[Dict[str, Any]] = []

    def add_document(self, document: Document):
        """Add a document to the corpus"""
        self.documents[document.id] = document
        logger.debug(f"Added document: {document.id}")

    def add_query(self, query: Query):
        """Add a query with ground truth"""
        self.queries[query.id] = query
        logger.debug(f"Added query: {query.id} with {len(query.relevant_doc_ids)} relevant docs")

    def load_ground_truth(self, filepath: Path):
        """Load ground truth from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load documents
        for doc_data in data.get("documents", []):
            document = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
            self.add_document(document)

        # Load queries
        for query_data in data.get("queries", []):
            query = Query(
                id=query_data["id"],
                text=query_data["text"],
                relevant_doc_ids=set(query_data["relevant_doc_ids"]),
                metadata=query_data.get("metadata", {})
            )
            self.add_query(query)

        logger.info(f"Loaded {len(self.documents)} documents and {len(self.queries)} queries")

    def evaluate_single_query(self, query_id: str, retrieved_docs: List[Tuple[str, float]], 
                             k: int = 10) -> Dict[str, float]:
        """Evaluate retrieval results for a single query"""
        if query_id not in self.queries:
            raise ValueError(f"Query {query_id} not found in ground truth")

        query = self.queries[query_id]
        relevant_docs = query.relevant_doc_ids

        # Get top-k results
        top_k_docs = [doc_id for doc_id, _ in retrieved_docs[:k]]

        # Calculate metrics
        metrics = {}

        # Precision@k
        relevant_retrieved = set(top_k_docs) & relevant_docs
        metrics['precision'] = len(relevant_retrieved) / len(top_k_docs) if top_k_docs else 0.0

        # Recall@k  
        metrics['recall'] = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0

        # F1@k
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0

        # Average Precision
        metrics['ap'] = self._calculate_average_precision(retrieved_docs, relevant_docs)

        # Reciprocal Rank
        metrics['rr'] = self._calculate_reciprocal_rank(retrieved_docs, relevant_docs)

        # NDCG@k
        metrics['ndcg'] = self._calculate_ndcg(retrieved_docs, relevant_docs, k)

        # Hit Rate@k
        metrics['hit_rate'] = 1.0 if relevant_retrieved else 0.0

        return metrics

    def _calculate_average_precision(self, retrieved_docs: List[Tuple[str, float]], 
                                   relevant_docs: Set[str]) -> float:
        """Calculate Average Precision"""
        if not relevant_docs:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0

        for i, (doc_id, _) in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i

        return precision_sum / len(relevant_docs) if relevant_docs else 0.0

    def _calculate_reciprocal_rank(self, retrieved_docs: List[Tuple[str, float]], 
                                 relevant_docs: Set[str]) -> float:
        """Calculate Reciprocal Rank"""
        for i, (doc_id, _) in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / i
        return 0.0

    def _calculate_ndcg(self, retrieved_docs: List[Tuple[str, float]], 
                       relevant_docs: Set[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        # DCG calculation
        dcg = 0.0
        for i, (doc_id, _) in enumerate(retrieved_docs[:k], 1):
            if doc_id in relevant_docs:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / math.log2(i + 1)

        # IDCG calculation (ideal DCG)
        idcg = 0.0
        for i in range(1, min(len(relevant_docs) + 1, k + 1)):
            idcg += 1.0 / math.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_batch(self, results: List[RetrievalResult], k: int = 10) -> EvaluationMetrics:
        """Evaluate a batch of retrieval results"""
        all_metrics = []

        for result in results:
            try:
                query_metrics = self.evaluate_single_query(
                    result.query_id, result.retrieved_docs, k
                )
                all_metrics.append(query_metrics)
            except ValueError as e:
                logger.warning(f"Skipping evaluation for {result.query_id}: {e}")

        if not all_metrics:
            return EvaluationMetrics()

        # Calculate mean metrics
        metrics = EvaluationMetrics(
            precision=np.mean([m['precision'] for m in all_metrics]),
            recall=np.mean([m['recall'] for m in all_metrics]),
            f1=np.mean([m['f1'] for m in all_metrics]),
            map_score=np.mean([m['ap'] for m in all_metrics]),
            mrr=np.mean([m['rr'] for m in all_metrics]),
            ndcg=np.mean([m['ndcg'] for m in all_metrics]),
            hit_rate=np.mean([m['hit_rate'] for m in all_metrics])
        )

        return metrics

    def evaluate_at_multiple_k(self, results: List[RetrievalResult]) -> Dict[int, EvaluationMetrics]:
        """Evaluate retrieval at multiple k values"""
        evaluation_results = {}

        for k in self.k_values:
            metrics = self.evaluate_batch(results, k)
            evaluation_results[k] = metrics
            logger.info(f"Evaluation at k={k}: P={metrics.precision:.3f}, R={metrics.recall:.3f}, F1={metrics.f1:.3f}")

        return evaluation_results

    def create_evaluation_report(self, results: List[RetrievalResult], 
                               output_dir: Path) -> Path:
        """Create comprehensive evaluation report"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate at multiple k values
        k_results = self.evaluate_at_multiple_k(results)

        # Create detailed report
        report = {
            "evaluation_summary": {
                "total_queries": len(results),
                "total_documents": len(self.documents),
                "evaluation_timestamp": datetime.now().isoformat(),
                "k_values_evaluated": self.k_values
            },
            "metrics_by_k": {
                str(k): metrics.to_dict() for k, metrics in k_results.items()
            },
            "per_query_analysis": self._create_per_query_analysis(results),
            "system_statistics": self._calculate_system_statistics(results)
        }

        # Save report
        report_path = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Create visualizations
        self._create_visualizations(k_results, output_dir)

        # Create summary table
        self._create_summary_table(k_results, output_dir)

        logger.info(f"Evaluation report saved to {report_path}")
        return report_path

    def _create_per_query_analysis(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Create detailed per-query analysis"""
        analysis = {}

        for result in results:
            if result.query_id in self.queries:
                query_analysis = {}

                for k in [5, 10]:  # Analyze at k=5 and k=10
                    metrics = self.evaluate_single_query(
                        result.query_id, result.retrieved_docs, k
                    )
                    query_analysis[f"k_{k}"] = metrics

                # Add query details
                query_analysis["query_text"] = self.queries[result.query_id].text
                query_analysis["relevant_docs_count"] = len(self.queries[result.query_id].relevant_doc_ids)
                query_analysis["retrieval_time"] = result.retrieval_time

                analysis[result.query_id] = query_analysis

        return analysis

    def _calculate_system_statistics(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Calculate system-wide statistics"""
        retrieval_times = [r.retrieval_time for r in results if r.retrieval_time > 0]

        return {
            "average_retrieval_time": np.mean(retrieval_times) if retrieval_times else 0.0,
            "median_retrieval_time": np.median(retrieval_times) if retrieval_times else 0.0,
            "total_retrieval_time": np.sum(retrieval_times),
            "queries_per_second": len(results) / np.sum(retrieval_times) if retrieval_times else 0.0
        }

    def _create_visualizations(self, k_results: Dict[int, EvaluationMetrics], 
                             output_dir: Path):
        """Create evaluation visualizations"""
        # Metrics vs K plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        k_values = list(k_results.keys())

        # Precision vs K
        precisions = [k_results[k].precision for k in k_values]
        axes[0, 0].plot(k_values, precisions, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Precision@K', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('K')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].grid(True, alpha=0.3)

        # Recall vs K
        recalls = [k_results[k].recall for k in k_values]
        axes[0, 1].plot(k_values, recalls, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Recall@K', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].grid(True, alpha=0.3)

        # F1 vs K
        f1_scores = [k_results[k].f1 for k in k_values]
        axes[1, 0].plot(k_values, f1_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('F1@K', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('K')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].grid(True, alpha=0.3)

        # Combined metrics
        axes[1, 1].plot(k_values, precisions, 'bo-', label='Precision', linewidth=2)
        axes[1, 1].plot(k_values, recalls, 'ro-', label='Recall', linewidth=2)
        axes[1, 1].plot(k_values, f1_scores, 'go-', label='F1', linewidth=2)
        axes[1, 1].set_title('Combined Metrics@K', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('K')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Advanced metrics plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # MAP vs K
        map_scores = [k_results[k].map_score for k in k_values]
        axes[0].plot(k_values, map_scores, 'mo-', linewidth=2, markersize=8)
        axes[0].set_title('Mean Average Precision@K', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('K')
        axes[0].set_ylabel('MAP')
        axes[0].grid(True, alpha=0.3)

        # NDCG vs K
        ndcg_scores = [k_results[k].ndcg for k in k_values]
        axes[1].plot(k_values, ndcg_scores, 'co-', linewidth=2, markersize=8)
        axes[1].set_title('NDCG@K', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('K')
        axes[1].set_ylabel('NDCG')
        axes[1].grid(True, alpha=0.3)

        # Hit Rate vs K
        hit_rates = [k_results[k].hit_rate for k in k_values]
        axes[2].plot(k_values, hit_rates, 'yo-', linewidth=2, markersize=8)
        axes[2].set_title('Hit Rate@K', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('K')
        axes[2].set_ylabel('Hit Rate')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'advanced_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}")

    def _create_summary_table(self, k_results: Dict[int, EvaluationMetrics], 
                            output_dir: Path):
        """Create summary table of results"""
        # Create DataFrame
        data = []
        for k, metrics in k_results.items():
            data.append({
                'K': k,
                'Precision': f"{metrics.precision:.4f}",
                'Recall': f"{metrics.recall:.4f}",
                'F1': f"{metrics.f1:.4f}",
                'MAP': f"{metrics.map_score:.4f}",
                'MRR': f"{metrics.mrr:.4f}",
                'NDCG': f"{metrics.ndcg:.4f}",
                'Hit Rate': f"{metrics.hit_rate:.4f}"
            })

        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = output_dir / 'evaluation_summary.csv'
        df.to_csv(csv_path, index=False)

        # Create formatted table visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Retrieval Evaluation Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Summary table saved to {csv_path}")

# Demo functions
def create_sample_ground_truth() -> Dict[str, Any]:
    """Create sample ground truth data for testing"""
    documents = [
        {
            "id": "doc_1",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "metadata": {"topic": "machine_learning", "difficulty": "beginner"}
        },
        {
            "id": "doc_2", 
            "content": "Deep learning is a specialized area of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
            "metadata": {"topic": "deep_learning", "difficulty": "intermediate"}
        },
        {
            "id": "doc_3",
            "content": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
            "metadata": {"topic": "nlp", "difficulty": "intermediate"}
        },
        {
            "id": "doc_4",
            "content": "Computer vision is a field of AI that trains computers to interpret and understand the visual world through digital images and videos.",
            "metadata": {"topic": "computer_vision", "difficulty": "intermediate"}
        },
        {
            "id": "doc_5",
            "content": "Supervised learning uses labeled training data to learn a mapping function from input variables to output variables.",
            "metadata": {"topic": "supervised_learning", "difficulty": "beginner"}
        },
        {
            "id": "doc_6",
            "content": "Unsupervised learning finds hidden patterns or structures in data without labeled examples or supervision.",
            "metadata": {"topic": "unsupervised_learning", "difficulty": "intermediate"}
        },
        {
            "id": "doc_7",
            "content": "Reinforcement learning is a type of machine learning where agents learn to make decisions by performing actions and receiving rewards or penalties.",
            "metadata": {"topic": "reinforcement_learning", "difficulty": "advanced"}
        }
    ]

    queries = [
        {
            "id": "query_1",
            "text": "What is machine learning?",
            "relevant_doc_ids": ["doc_1", "doc_5"],
            "metadata": {"difficulty": "beginner", "category": "definition"}
        },
        {
            "id": "query_2",
            "text": "How does deep learning work?",
            "relevant_doc_ids": ["doc_2"],
            "metadata": {"difficulty": "intermediate", "category": "explanation"}
        },
        {
            "id": "query_3",
            "text": "What are the types of machine learning?",
            "relevant_doc_ids": ["doc_5", "doc_6", "doc_7"],
            "metadata": {"difficulty": "intermediate", "category": "classification"}
        },
        {
            "id": "query_4",
            "text": "Explain natural language processing",
            "relevant_doc_ids": ["doc_3"],
            "metadata": {"difficulty": "intermediate", "category": "definition"}
        }
    ]

    return {
        "documents": documents,
        "queries": queries,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "description": "Sample ground truth for retrieval evaluation",
            "total_documents": len(documents),
            "total_queries": len(queries)
        }
    }

def simulate_retrieval_results() -> List[RetrievalResult]:
    """Simulate retrieval results for demo"""
    # Simulate imperfect retrieval system
    results = [
        RetrievalResult(
            query_id="query_1",
            retrieved_docs=[
                ("doc_1", 0.95),  # Correct
                ("doc_2", 0.78),  # Incorrect but related
                ("doc_5", 0.72),  # Correct
                ("doc_3", 0.65),  # Incorrect
                ("doc_4", 0.58)   # Incorrect
            ],
            retrieval_time=0.045
        ),
        RetrievalResult(
            query_id="query_2", 
            retrieved_docs=[
                ("doc_2", 0.92),  # Correct
                ("doc_1", 0.81),  # Related but not exact
                ("doc_6", 0.67),  # Incorrect
                ("doc_7", 0.63),  # Incorrect
                ("doc_4", 0.55)   # Incorrect
            ],
            retrieval_time=0.038
        ),
        RetrievalResult(
            query_id="query_3",
            retrieved_docs=[
                ("doc_5", 0.88),  # Correct
                ("doc_6", 0.85),  # Correct
                ("doc_1", 0.79),  # Related
                ("doc_7", 0.74),  # Correct
                ("doc_2", 0.61)   # Related
            ],
            retrieval_time=0.052
        ),
        RetrievalResult(
            query_id="query_4",
            retrieved_docs=[
                ("doc_3", 0.91),  # Correct
                ("doc_1", 0.73),  # Related
                ("doc_2", 0.68),  # Related
                ("doc_4", 0.64),  # Related (both AI)
                ("doc_5", 0.57)   # Incorrect
            ],
            retrieval_time=0.041
        )
    ]

    return results

def main():
    """Main demo function"""
    print("🔍 DAY 13 - RETRIEVAL EVALUATION DEMO")
    print("=" * 60)

    # Create evaluator
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])

    # Create and save sample ground truth
    ground_truth = create_sample_ground_truth()
    gt_path = Path("data/ground_truth_sample.json")
    gt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)

    # Load ground truth
    evaluator.load_ground_truth(gt_path)

    # Simulate retrieval results
    results = simulate_retrieval_results()

    print(f"\n📊 Evaluating {len(results)} queries against {len(evaluator.documents)} documents")

    # Evaluate at different k values
    k_results = evaluator.evaluate_at_multiple_k(results)

    # Print results
    print("\n📈 EVALUATION RESULTS")
    print("-" * 40)

    for k, metrics in k_results.items():
        print(f"\nK = {k}:")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall:    {metrics.recall:.4f}")
        print(f"  F1 Score:  {metrics.f1:.4f}")
        print(f"  MAP:       {metrics.map_score:.4f}")
        print(f"  NDCG:      {metrics.ndcg:.4f}")
        print(f"  Hit Rate:  {metrics.hit_rate:.4f}")

    # Create comprehensive report
    results_dir = Path("results")
    report_path = evaluator.create_evaluation_report(results, results_dir)

    print(f"\n📝 Comprehensive report saved to: {report_path}")
    print(f"📊 Visualizations saved to: {results_dir}")

    print("\n✅ Day 13 retrieval evaluation completed successfully!")

if __name__ == "__main__":
    main()
