from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from collections import Counter
import re
import math

logger = logging.getLogger(__name__)

class ChunkingEvaluator:
    """Evaluate the quality of text chunking"""

    def __init__(self):
        self.metrics = [
            'chunk_size_consistency',
            'semantic_coherence',
            'coverage',
            'overlap_efficiency', 
            'boundary_quality'
        ]

    def evaluate_chunks(self, chunks: List[Dict[str, Any]], 
                       original_text: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of chunking quality

        Args:
            chunks: List of chunk dictionaries
            original_text: Original text that was chunked

        Returns:
            Evaluation metrics and scores
        """
        if not chunks:
            return {"error": "No chunks provided for evaluation"}

        evaluation = {
            "total_chunks": len(chunks),
            "metrics": {}
        }

        # Basic statistics
        evaluation["basic_stats"] = self._calculate_basic_stats(chunks)

        # Chunk size consistency
        evaluation["metrics"]["size_consistency"] = self._evaluate_size_consistency(chunks)

        # Coverage analysis (if original text provided)
        if original_text:
            evaluation["metrics"]["coverage"] = self._evaluate_coverage(chunks, original_text)

        # Overlap efficiency
        evaluation["metrics"]["overlap_efficiency"] = self._evaluate_overlap_efficiency(chunks)

        # Boundary quality
        evaluation["metrics"]["boundary_quality"] = self._evaluate_boundary_quality(chunks)

        # Semantic coherence (basic version)
        evaluation["metrics"]["semantic_coherence"] = self._evaluate_semantic_coherence(chunks)

        # Overall score
        evaluation["overall_score"] = self._calculate_overall_score(evaluation["metrics"])

        return evaluation

    def _calculate_basic_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic statistics about chunks"""
        chunk_lengths = []
        token_counts = []
        sentence_counts = []

        for chunk in chunks:
            chunk_lengths.append(len(chunk["text"]))

            if "metadata" in chunk:
                metadata = chunk["metadata"]
                if "token_count" in metadata:
                    token_counts.append(metadata["token_count"])
                if "sentence_count" in metadata:
                    sentence_counts.append(metadata["sentence_count"])
                elif "word_count" in metadata:
                    # Estimate sentences from word count
                    sentence_counts.append(max(1, metadata["word_count"] // 15))

        stats = {
            "character_lengths": {
                "mean": np.mean(chunk_lengths),
                "std": np.std(chunk_lengths),
                "min": np.min(chunk_lengths),
                "max": np.max(chunk_lengths),
                "median": np.median(chunk_lengths)
            }
        }

        if token_counts:
            stats["token_counts"] = {
                "mean": np.mean(token_counts),
                "std": np.std(token_counts),
                "min": np.min(token_counts),
                "max": np.max(token_counts),
                "median": np.median(token_counts)
            }

        if sentence_counts:
            stats["sentence_counts"] = {
                "mean": np.mean(sentence_counts),
                "std": np.std(sentence_counts),
                "min": np.min(sentence_counts),
                "max": np.max(sentence_counts),
                "median": np.median(sentence_counts)
            }

        return stats

    def _evaluate_size_consistency(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate consistency of chunk sizes"""
        # Use token counts if available, otherwise character counts
        sizes = []

        for chunk in chunks:
            if "metadata" in chunk and "token_count" in chunk["metadata"]:
                sizes.append(chunk["metadata"]["token_count"])
            else:
                sizes.append(len(chunk["text"]))

        if not sizes:
            return {"error": "No size data available"}

        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        cv = std_size / mean_size if mean_size > 0 else 0  # Coefficient of variation

        # Lower coefficient of variation means more consistent sizes
        consistency_score = max(0, 1 - cv)  # Score between 0 and 1

        return {
            "mean_size": mean_size,
            "std_size": std_size,
            "coefficient_of_variation": cv,
            "consistency_score": consistency_score,
            "interpretation": self._interpret_consistency_score(consistency_score)
        }

    def _evaluate_coverage(self, chunks: List[Dict[str, Any]], 
                          original_text: str) -> Dict[str, Any]:
        """Evaluate how well chunks cover the original text"""
        total_original_chars = len(original_text)
        total_chunk_chars = sum(len(chunk["text"]) for chunk in chunks)

        # Calculate coverage ratio
        coverage_ratio = total_chunk_chars / total_original_chars if total_original_chars > 0 else 0

        # Check for content preservation (simplified)
        original_words = set(re.findall(r'\b\w+\b', original_text.lower()))
        chunk_words = set()

        for chunk in chunks:
            chunk_words.update(re.findall(r'\b\w+\b', chunk["text"].lower()))

        word_coverage = len(chunk_words.intersection(original_words)) / len(original_words) if original_words else 0

        return {
            "character_coverage_ratio": coverage_ratio,
            "word_coverage_ratio": word_coverage,
            "total_original_chars": total_original_chars,
            "total_chunk_chars": total_chunk_chars,
            "coverage_score": min(coverage_ratio, word_coverage),  # Conservative score
            "interpretation": self._interpret_coverage_score(min(coverage_ratio, word_coverage))
        }

    def _evaluate_overlap_efficiency(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the efficiency of chunk overlaps"""
        total_overlap = 0
        overlap_count = 0

        # Check consecutive chunks for overlap
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]["text"]
            next_chunk = chunks[i + 1]["text"]

            # Simple overlap detection (last words of current vs first words of next)
            current_words = current_chunk.split()
            next_words = next_chunk.split()

            # Look for overlap up to 50% of smaller chunk
            max_overlap = min(len(current_words), len(next_words)) // 2

            overlap_found = 0
            for j in range(1, max_overlap + 1):
                if current_words[-j:] == next_words[:j]:
                    overlap_found = j

            if overlap_found > 0:
                total_overlap += overlap_found
                overlap_count += 1

        avg_overlap = total_overlap / max(overlap_count, 1)
        overlap_ratio = overlap_count / max(len(chunks) - 1, 1)

        # Score based on reasonable overlap (not too much, not too little)
        efficiency_score = 1.0
        if overlap_ratio > 0.8:  # Too much overlap
            efficiency_score = 0.7
        elif overlap_ratio < 0.2 and len(chunks) > 1:  # Too little overlap for multi-chunk
            efficiency_score = 0.6

        return {
            "total_overlap_words": total_overlap,
            "overlapping_chunks": overlap_count,
            "average_overlap_words": avg_overlap,
            "overlap_ratio": overlap_ratio,
            "efficiency_score": efficiency_score,
            "interpretation": self._interpret_overlap_score(efficiency_score)
        }

    def _evaluate_boundary_quality(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the quality of chunk boundaries"""
        boundary_scores = []

        for chunk in chunks:
            text = chunk["text"].strip()
            if not text:
                boundary_scores.append(0.0)
                continue

            score = 0.0

            # Check if starts with capital letter (good sentence boundary)
            if text[0].isupper():
                score += 0.3

            # Check if ends with punctuation
            if text[-1] in '.!?':
                score += 0.3

            # Check if doesn't end mid-sentence (no comma, semicolon at end)
            if text[-1] not in ',;:':
                score += 0.2

            # Check if doesn't start with lowercase (unless it's a continuation)
            if not (text[0].islower() and len(text) > 1):
                score += 0.2

            boundary_scores.append(score)

        avg_boundary_score = np.mean(boundary_scores) if boundary_scores else 0.0

        return {
            "individual_scores": boundary_scores,
            "average_score": avg_boundary_score,
            "boundary_quality_score": avg_boundary_score,
            "interpretation": self._interpret_boundary_score(avg_boundary_score)
        }

    def _evaluate_semantic_coherence(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate semantic coherence within chunks (basic version)"""
        coherence_scores = []

        for chunk in chunks:
            text = chunk["text"]

            # Simple coherence measures
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                coherence_scores.append(1.0)  # Single sentence is coherent
                continue

            # Check for topic consistency (simplified)
            # Count repeated words across sentences
            all_words = []
            for sentence in sentences:
                words = re.findall(r'\b\w+\b', sentence.lower())
                all_words.extend(words)

            if not all_words:
                coherence_scores.append(0.5)
                continue

            # Calculate word repetition as a proxy for topic consistency
            word_counts = Counter(all_words)
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            repetition_ratio = repeated_words / len(word_counts) if word_counts else 0

            # Simple coherence score based on word repetition and sentence structure
            coherence_score = min(1.0, 0.5 + repetition_ratio * 0.5)
            coherence_scores.append(coherence_score)

        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0

        return {
            "individual_scores": coherence_scores,
            "average_coherence": avg_coherence,
            "coherence_score": avg_coherence,
            "interpretation": self._interpret_coherence_score(avg_coherence)
        }

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall chunking quality score"""
        scores = []
        weights = {
            "size_consistency": 0.2,
            "coverage": 0.3,
            "overlap_efficiency": 0.2,
            "boundary_quality": 0.2,
            "semantic_coherence": 0.1
        }

        total_weight = 0
        weighted_sum = 0

        for metric, weight in weights.items():
            if metric in metrics and isinstance(metrics[metric], dict):
                # Get the main score from each metric
                if "consistency_score" in metrics[metric]:
                    score = metrics[metric]["consistency_score"]
                elif "coverage_score" in metrics[metric]:
                    score = metrics[metric]["coverage_score"]
                elif "efficiency_score" in metrics[metric]:
                    score = metrics[metric]["efficiency_score"]
                elif "boundary_quality_score" in metrics[metric]:
                    score = metrics[metric]["boundary_quality_score"]
                elif "coherence_score" in metrics[metric]:
                    score = metrics[metric]["coherence_score"]
                else:
                    continue

                weighted_sum += score * weight
                total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            "score": overall_score,
            "grade": self._score_to_grade(overall_score),
            "interpretation": self._interpret_overall_score(overall_score),
            "weights_used": weights
        }

    def _interpret_consistency_score(self, score: float) -> str:
        """Interpret consistency score"""
        if score >= 0.8:
            return "Excellent - Very consistent chunk sizes"
        elif score >= 0.6:
            return "Good - Reasonably consistent chunk sizes"
        elif score >= 0.4:
            return "Fair - Some variation in chunk sizes"
        else:
            return "Poor - Highly variable chunk sizes"

    def _interpret_coverage_score(self, score: float) -> str:
        """Interpret coverage score"""
        if score >= 0.95:
            return "Excellent - Complete coverage of original content"
        elif score >= 0.85:
            return "Good - Most content preserved"
        elif score >= 0.7:
            return "Fair - Some content may be missing"
        else:
            return "Poor - Significant content loss"

    def _interpret_overlap_score(self, score: float) -> str:
        """Interpret overlap efficiency score"""
        if score >= 0.8:
            return "Excellent - Optimal overlap strategy"
        elif score >= 0.6:
            return "Good - Reasonable overlap"
        else:
            return "Poor - Suboptimal overlap strategy"

    def _interpret_boundary_score(self, score: float) -> str:
        """Interpret boundary quality score"""
        if score >= 0.8:
            return "Excellent - Clean sentence boundaries"
        elif score >= 0.6:
            return "Good - Mostly clean boundaries"
        elif score >= 0.4:
            return "Fair - Some awkward boundaries"
        else:
            return "Poor - Many awkward boundaries"

    def _interpret_coherence_score(self, score: float) -> str:
        """Interpret semantic coherence score"""
        if score >= 0.8:
            return "Excellent - Highly coherent chunks"
        elif score >= 0.6:
            return "Good - Generally coherent"
        elif score >= 0.4:
            return "Fair - Some coherence issues"
        else:
            return "Poor - Low coherence"

    def _interpret_overall_score(self, score: float) -> str:
        """Interpret overall chunking quality score"""
        if score >= 0.8:
            return "Excellent chunking quality - Ready for production use"
        elif score >= 0.6:
            return "Good chunking quality - Minor improvements possible"
        elif score >= 0.4:
            return "Fair chunking quality - Consider adjusting parameters"
        else:
            return "Poor chunking quality - Significant improvements needed"

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        elif score >= 0.3:
            return "D"
        else:
            return "F"

    def compare_chunking_strategies(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple chunking strategies"""
        comparison = {
            "strategies": list(strategy_results.keys()),
            "metrics_comparison": {},
            "rankings": {}
        }

        # Extract scores for each metric
        metrics = ["size_consistency", "coverage", "overlap_efficiency", "boundary_quality", "semantic_coherence"]

        for metric in metrics:
            comparison["metrics_comparison"][metric] = {}
            metric_scores = {}

            for strategy, results in strategy_results.items():
                if "evaluation" in results and "metrics" in results["evaluation"]:
                    eval_metrics = results["evaluation"]["metrics"]

                    # Extract the main score for this metric
                    if metric in eval_metrics:
                        metric_data = eval_metrics[metric]
                        if "consistency_score" in metric_data:
                            score = metric_data["consistency_score"]
                        elif "coverage_score" in metric_data:
                            score = metric_data["coverage_score"]
                        elif "efficiency_score" in metric_data:
                            score = metric_data["efficiency_score"]
                        elif "boundary_quality_score" in metric_data:
                            score = metric_data["boundary_quality_score"]
                        elif "coherence_score" in metric_data:
                            score = metric_data["coherence_score"]
                        else:
                            score = 0.0

                        metric_scores[strategy] = score
                        comparison["metrics_comparison"][metric][strategy] = score

            # Rank strategies for this metric
            if metric_scores:
                ranked = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
                comparison["rankings"][metric] = [strategy for strategy, _ in ranked]

        # Overall ranking based on overall scores
        overall_scores = {}
        for strategy, results in strategy_results.items():
            if ("evaluation" in results and 
                "overall_score" in results["evaluation"] and
                "score" in results["evaluation"]["overall_score"]):
                overall_scores[strategy] = results["evaluation"]["overall_score"]["score"]

        if overall_scores:
            overall_ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            comparison["overall_ranking"] = [strategy for strategy, _ in overall_ranked]
            comparison["overall_scores"] = overall_scores

        # Best strategy recommendation
        if "overall_ranking" in comparison and comparison["overall_ranking"]:
            best_strategy = comparison["overall_ranking"][0]
            comparison["recommendation"] = {
                "best_strategy": best_strategy,
                "reason": f"Highest overall score: {overall_scores.get(best_strategy, 0.0):.3f}"
            }

        return comparison

class PerformanceEvaluator:
    """Evaluate performance aspects of chunking and embedding"""

    def __init__(self):
        pass

    def evaluate_processing_speed(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate processing speed metrics"""
        stats = results.get("statistics", {})

        processing_time = stats.get("processing_time_seconds", 0)
        chunk_count = stats.get("chunking", {}).get("total_chunks", 1)
        total_chars = stats.get("cleaned_length", 1)

        return {
            "total_processing_time": processing_time,
            "chunks_per_second": chunk_count / processing_time if processing_time > 0 else 0,
            "characters_per_second": total_chars / processing_time if processing_time > 0 else 0,
            "avg_time_per_chunk": processing_time / chunk_count if chunk_count > 0 else 0
        }

    def evaluate_memory_efficiency(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate memory usage of chunks"""
        total_text_size = sum(len(chunk["text"]) for chunk in chunks)
        total_embedding_size = 0
        embedding_count = 0

        for chunk in chunks:
            if "embedding" in chunk:
                embedding_count += 1
                embedding_size = len(chunk["embedding"]) * 4  # Assume 4 bytes per float
                total_embedding_size += embedding_size

        return {
            "total_text_bytes": total_text_size,
            "total_embedding_bytes": total_embedding_size,
            "total_memory_bytes": total_text_size + total_embedding_size,
            "chunks_with_embeddings": embedding_count,
            "avg_chunk_memory": (total_text_size + total_embedding_size) / len(chunks) if chunks else 0
        }

def create_evaluation_report(chunks: List[Dict[str, Any]], 
                           original_text: str = None,
                           include_performance: bool = True) -> str:
    """Create a formatted evaluation report"""
    evaluator = ChunkingEvaluator()
    evaluation = evaluator.evaluate_chunks(chunks, original_text)

    report = []
    report.append("# Chunking Quality Evaluation Report")
    report.append("=" * 50)

    # Basic statistics
    report.append("\n## Basic Statistics")
    basic_stats = evaluation.get("basic_stats", {})
    report.append(f"Total chunks: {evaluation['total_chunks']}")

    if "character_lengths" in basic_stats:
        char_stats = basic_stats["character_lengths"]
        report.append(f"Average chunk length: {char_stats['mean']:.1f} characters")
        report.append(f"Length standard deviation: {char_stats['std']:.1f}")
        report.append(f"Min/Max length: {char_stats['min']:.0f}/{char_stats['max']:.0f}")

    # Metrics
    report.append("\n## Quality Metrics")
    metrics = evaluation.get("metrics", {})

    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict):
            report.append(f"\n### {metric_name.replace('_', ' ').title()}")

            # Find the main score
            main_score = None
            interpretation = ""

            if "consistency_score" in metric_data:
                main_score = metric_data["consistency_score"]
                interpretation = metric_data.get("interpretation", "")
            elif "coverage_score" in metric_data:
                main_score = metric_data["coverage_score"]
                interpretation = metric_data.get("interpretation", "")
            elif "efficiency_score" in metric_data:
                main_score = metric_data["efficiency_score"]
                interpretation = metric_data.get("interpretation", "")
            elif "boundary_quality_score" in metric_data:
                main_score = metric_data["boundary_quality_score"]
                interpretation = metric_data.get("interpretation", "")
            elif "coherence_score" in metric_data:
                main_score = metric_data["coherence_score"]
                interpretation = metric_data.get("interpretation", "")

            if main_score is not None:
                report.append(f"Score: {main_score:.3f}")
                if interpretation:
                    report.append(f"Assessment: {interpretation}")

    # Overall score
    if "overall_score" in evaluation:
        overall = evaluation["overall_score"]
        report.append("\n## Overall Assessment")
        report.append(f"Overall Score: {overall['score']:.3f}")
        report.append(f"Grade: {overall['grade']}")
        report.append(f"Summary: {overall['interpretation']}")

    return "\n".join(report)
