import tiktoken
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for text chunks"""
    chunk_id: str
    start_char: int
    end_char: int
    token_count: int
    source_document: Optional[str] = None
    chunk_type: str = "token"
    overlap_tokens: int = 0

class BaseChunker(ABC):
    """Abstract base class for text chunkers"""

    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces"""
        pass

class TokenBasedChunker(BaseChunker):
    """Token-based text chunker using tiktoken"""

    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 1000,
                 overlap_tokens: int = 100,
                 preserve_sentences: bool = True):
        """
        Initialize token-based chunker

        Args:
            model_name: Model name for tokenization
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            preserve_sentences: Whether to preserve sentence boundaries
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_sentences = preserve_sentences

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, source_document: str = None) -> List[Dict[str, Any]]:
        """
        Chunk text based on token count

        Args:
            text: Text to chunk
            source_document: Source document identifier

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []

        # Tokenize the entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= self.max_tokens:
            return [{
                "text": text,
                "metadata": ChunkMetadata(
                    chunk_id="chunk_0",
                    start_char=0,
                    end_char=len(text),
                    token_count=total_tokens,
                    source_document=source_document,
                    chunk_type="token"
                ).__dict__
            }]

        chunks = []
        start_token_idx = 0
        chunk_id = 0

        while start_token_idx < total_tokens:
            # Calculate end token index
            end_token_idx = min(start_token_idx + self.max_tokens, total_tokens)

            # Extract token slice
            chunk_tokens = tokens[start_token_idx:end_token_idx]

            # Decode tokens back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # If preserving sentences and not the last chunk, adjust boundary
            if (self.preserve_sentences and 
                end_token_idx < total_tokens and 
                len(chunk_text) > 0):
                chunk_text = self._adjust_sentence_boundary(chunk_text)
                # Re-encode to get actual token count
                chunk_tokens = self.encoding.encode(chunk_text)

            # Calculate character positions in original text
            start_char = len(self.encoding.decode(tokens[:start_token_idx]))
            end_char = start_char + len(chunk_text)

            # Create chunk metadata
            metadata = ChunkMetadata(
                chunk_id=f"chunk_{chunk_id}",
                start_char=start_char,
                end_char=end_char,
                token_count=len(chunk_tokens),
                source_document=source_document,
                chunk_type="token",
                overlap_tokens=self.overlap_tokens if chunk_id > 0 else 0
            )

            chunks.append({
                "text": chunk_text,
                "metadata": metadata.__dict__
            })

            # Move to next chunk with overlap
            if self.overlap_tokens > 0 and end_token_idx < total_tokens:
                start_token_idx = end_token_idx - self.overlap_tokens
            else:
                start_token_idx = end_token_idx

            chunk_id += 1

        return chunks

    def _adjust_sentence_boundary(self, text: str) -> str:
        """Adjust chunk boundary to preserve sentence completeness"""
        # Find the last complete sentence
        sentence_endings = ['.', '!', '?', '\n\n']

        for i in range(len(text) - 1, -1, -1):
            if text[i] in sentence_endings:
                # Check if this is a real sentence ending (not abbreviation)
                if self._is_sentence_end(text, i):
                    return text[:i + 1].strip()

        # If no sentence ending found, return as is
        return text

    def _is_sentence_end(self, text: str, pos: int) -> bool:
        """Check if position is a real sentence ending"""
        if pos >= len(text) - 1:
            return True

        # Check for abbreviations (basic check)
        if text[pos] == '.':
            # Look at surrounding context
            if pos > 0 and text[pos - 1].isupper():
                return False
            # Check for common abbreviations
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1:pos].lower()
            common_abbrevs = {'dr', 'mr', 'mrs', 'ms', 'vs', 'etc', 'inc', 'ltd'}
            if word in common_abbrevs:
                return False

        return True

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {}

        token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]
        text_lengths = [len(chunk["text"]) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_chars_per_chunk": sum(text_lengths) / len(text_lengths),
            "total_characters": sum(text_lengths)
        }

class FixedTokenChunker(TokenBasedChunker):
    """Fixed-size token chunker without sentence preservation"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", 
                 max_tokens: int = 1000, overlap_tokens: int = 100):
        super().__init__(model_name, max_tokens, overlap_tokens, preserve_sentences=False)

class AdaptiveTokenChunker(TokenBasedChunker):
    """Adaptive token chunker that adjusts based on content density"""

    def __init__(self, model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 1000,
                 min_tokens: int = 200,
                 overlap_tokens: int = 100):
        super().__init__(model_name, max_tokens, overlap_tokens, preserve_sentences=True)
        self.min_tokens = min_tokens

    def chunk_text(self, text: str, source_document: str = None) -> List[Dict[str, Any]]:
        """Adaptive chunking based on content complexity"""
        # Analyze text complexity (simplified version)
        complexity_score = self._analyze_complexity(text)

        # Adjust chunk size based on complexity
        if complexity_score > 0.8:  # High complexity
            adjusted_max_tokens = max(self.min_tokens, int(self.max_tokens * 0.7))
        elif complexity_score > 0.5:  # Medium complexity
            adjusted_max_tokens = int(self.max_tokens * 0.85)
        else:  # Low complexity
            adjusted_max_tokens = self.max_tokens

        # Temporarily adjust max_tokens
        original_max_tokens = self.max_tokens
        self.max_tokens = adjusted_max_tokens

        chunks = super().chunk_text(text, source_document)

        # Restore original max_tokens
        self.max_tokens = original_max_tokens

        return chunks

    def _analyze_complexity(self, text: str) -> float:
        """Analyze text complexity (simplified heuristic)"""
        if not text:
            return 0.0

        # Calculate various complexity metrics
        sentences = text.split('.')
        words = text.split()

        if not sentences or not words:
            return 0.0

        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)

        # Vocabulary diversity (unique words / total words)
        unique_words = len(set(word.lower() for word in words))
        vocab_diversity = unique_words / len(words) if words else 0

        # Punctuation density
        punctuation_chars = sum(1 for char in text if char in '.,!?;:')
        punctuation_density = punctuation_chars / len(text) if text else 0

        # Normalize and combine metrics (simplified)
        complexity = min(1.0, (
            min(avg_sentence_length / 20, 1.0) * 0.4 +  # Sentence length component
            vocab_diversity * 0.4 +                     # Vocabulary diversity component
            punctuation_density * 10 * 0.2              # Punctuation density component
        ))

        return complexity

def create_chunker(chunker_type: str = "token", **kwargs) -> BaseChunker:
    """Factory function to create chunkers"""
    chunker_types = {
        "token": TokenBasedChunker,
        "fixed_token": FixedTokenChunker,
        "adaptive_token": AdaptiveTokenChunker
    }

    if chunker_type not in chunker_types:
        raise ValueError(f"Unknown chunker type: {chunker_type}")

    return chunker_types[chunker_type](**kwargs)
