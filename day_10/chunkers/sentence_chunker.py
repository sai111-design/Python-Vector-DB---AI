import re
import nltk
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic sentence splitting")

@dataclass 
class SentenceChunkMetadata:
    """Metadata for sentence-based chunks"""
    chunk_id: str
    start_char: int
    end_char: int
    sentence_count: int
    word_count: int
    source_document: Optional[str] = None
    chunk_type: str = "sentence"
    sentences_overlap: int = 0

class SentenceBasedChunker:
    """Sentence-based text chunker"""

    def __init__(self, 
                 max_sentences: int = 5,
                 max_words: int = 500,
                 overlap_sentences: int = 1,
                 min_sentence_length: int = 10,
                 use_nltk: bool = True):
        """
        Initialize sentence-based chunker

        Args:
            max_sentences: Maximum sentences per chunk
            max_words: Maximum words per chunk (secondary limit)
            overlap_sentences: Number of sentences to overlap between chunks
            min_sentence_length: Minimum characters for a valid sentence
            use_nltk: Whether to use NLTK for sentence tokenization
        """
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences
        self.min_sentence_length = min_sentence_length
        self.use_nltk = use_nltk and NLTK_AVAILABLE

        # Sentence boundary patterns for fallback
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n\n)|(?<=\. )|(?<=\! )|(?<=\? )',
            re.MULTILINE
        )

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if not text.strip():
            return []

        if self.use_nltk:
            try:
                sentences = sent_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}, using fallback")
                sentences = self._fallback_sentence_split(text)
        else:
            sentences = self._fallback_sentence_split(text)

        # Filter out sentences that are too short
        valid_sentences = [
            s.strip() for s in sentences 
            if len(s.strip()) >= self.min_sentence_length
        ]

        return valid_sentences

    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitting using regex"""
        # Split on sentence boundaries
        sentences = self.sentence_pattern.split(text)

        # Clean up and merge fragments
        cleaned_sentences = []
        current_sentence = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            current_sentence += sentence

            # Check if this looks like a complete sentence
            if self._is_complete_sentence(current_sentence):
                cleaned_sentences.append(current_sentence)
                current_sentence = ""
            else:
                current_sentence += " "

        # Add any remaining text
        if current_sentence.strip():
            cleaned_sentences.append(current_sentence.strip())

        return cleaned_sentences

    def _is_complete_sentence(self, text: str) -> bool:
        """Check if text appears to be a complete sentence"""
        text = text.strip()
        if not text:
            return False

        # Check for sentence-ending punctuation
        if text[-1] in '.!?':
            return True

        # Check for paragraph breaks
        if '\n\n' in text:
            return True

        return False

    def count_words(self, text: str) -> int:
        """Count words in text"""
        if self.use_nltk:
            try:
                return len(word_tokenize(text))
            except Exception:
                pass

        # Fallback word counting
        return len(text.split())

    def chunk_text(self, text: str, source_document: str = None) -> List[Dict[str, Any]]:
        """
        Chunk text based on sentences

        Args:
            text: Text to chunk
            source_document: Source document identifier

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []

        # Split into sentences
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        # If we have fewer sentences than max, return single chunk
        if len(sentences) <= self.max_sentences:
            word_count = self.count_words(text)
            if word_count <= self.max_words:
                return [{
                    "text": text,
                    "metadata": SentenceChunkMetadata(
                        chunk_id="chunk_0",
                        start_char=0,
                        end_char=len(text),
                        sentence_count=len(sentences),
                        word_count=word_count,
                        source_document=source_document,
                        chunk_type="sentence"
                    ).__dict__
                }]

        # Create chunks
        chunks = []
        chunk_id = 0
        i = 0

        while i < len(sentences):
            chunk_sentences = []
            chunk_word_count = 0
            start_idx = i

            # Add sentences to chunk until we hit limits
            while (i < len(sentences) and 
                   len(chunk_sentences) < self.max_sentences):

                sentence = sentences[i]
                sentence_word_count = self.count_words(sentence)

                # Check if adding this sentence exceeds word limit
                if (chunk_word_count + sentence_word_count > self.max_words and 
                    chunk_sentences):
                    break

                chunk_sentences.append(sentence)
                chunk_word_count += sentence_word_count
                i += 1

            # Create chunk text
            chunk_text = ' '.join(chunk_sentences)

            # Find character positions in original text
            start_char = text.find(chunk_sentences[0])
            end_char = start_char + len(chunk_text)

            # Create metadata
            metadata = SentenceChunkMetadata(
                chunk_id=f"chunk_{chunk_id}",
                start_char=start_char,
                end_char=end_char,
                sentence_count=len(chunk_sentences),
                word_count=chunk_word_count,
                source_document=source_document,
                chunk_type="sentence",
                sentences_overlap=self.overlap_sentences if chunk_id > 0 else 0
            )

            chunks.append({
                "text": chunk_text,
                "metadata": metadata.__dict__
            })

            # Handle overlap for next chunk
            if self.overlap_sentences > 0 and i < len(sentences):
                # Move back by overlap amount
                i = max(start_idx + 1, i - self.overlap_sentences)

            chunk_id += 1

        return chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {}

        sentence_counts = [chunk["metadata"]["sentence_count"] for chunk in chunks]
        word_counts = [chunk["metadata"]["word_count"] for chunk in chunks]
        text_lengths = [len(chunk["text"]) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_sentences": sum(sentence_counts),
            "total_words": sum(word_counts),
            "avg_sentences_per_chunk": sum(sentence_counts) / len(sentence_counts),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "min_sentences": min(sentence_counts),
            "max_sentences": max(sentence_counts),
            "avg_chars_per_chunk": sum(text_lengths) / len(text_lengths),
            "total_characters": sum(text_lengths)
        }

class SemanticSentenceChunker(SentenceBasedChunker):
    """Sentence chunker with semantic similarity grouping"""

    def __init__(self, 
                 max_sentences: int = 5,
                 max_words: int = 500,
                 overlap_sentences: int = 1,
                 similarity_threshold: float = 0.7,
                 **kwargs):
        super().__init__(max_sentences, max_words, overlap_sentences, **kwargs)
        self.similarity_threshold = similarity_threshold

        # Try to import sentence transformers for semantic similarity
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.semantic_grouping = True
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to basic chunking")
            self.semantic_grouping = False

    def chunk_text(self, text: str, source_document: str = None) -> List[Dict[str, Any]]:
        """Chunk text with semantic sentence grouping"""
        if not self.semantic_grouping:
            return super().chunk_text(text, source_document)

        # Get basic sentence chunks first
        sentences = self.split_into_sentences(text)

        if len(sentences) <= self.max_sentences:
            return super().chunk_text(text, source_document)

        # Group semantically similar sentences
        sentence_groups = self._group_sentences_semantically(sentences)

        # Create chunks from groups
        chunks = []
        chunk_id = 0

        for group in sentence_groups:
            group_text = ' '.join(group)
            word_count = self.count_words(group_text)

            # Find character positions
            start_char = text.find(group[0])
            end_char = start_char + len(group_text)

            metadata = SentenceChunkMetadata(
                chunk_id=f"chunk_{chunk_id}",
                start_char=start_char,
                end_char=end_char,
                sentence_count=len(group),
                word_count=word_count,
                source_document=source_document,
                chunk_type="semantic_sentence"
            )

            chunks.append({
                "text": group_text,
                "metadata": metadata.__dict__
            })

            chunk_id += 1

        return chunks

    def _group_sentences_semantically(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences based on semantic similarity"""
        if not sentences:
            return []

        # Generate embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)

        # Group similar sentences
        groups = []
        used_indices = set()

        for i, sentence in enumerate(sentences):
            if i in used_indices:
                continue

            current_group = [sentence]
            used_indices.add(i)

            # Find similar sentences
            for j, other_sentence in enumerate(sentences):
                if j in used_indices or i == j:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])

                if (similarity >= self.similarity_threshold and 
                    len(current_group) < self.max_sentences):
                    current_group.append(other_sentence)
                    used_indices.add(j)

            groups.append(current_group)

        return groups

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class ParagraphChunker:
    """Paragraph-based text chunker"""

    def __init__(self, max_paragraphs: int = 3, 
                 overlap_paragraphs: int = 0,
                 min_paragraph_length: int = 50):
        self.max_paragraphs = max_paragraphs
        self.overlap_paragraphs = overlap_paragraphs
        self.min_paragraph_length = min_paragraph_length

    def chunk_text(self, text: str, source_document: str = None) -> List[Dict[str, Any]]:
        """Chunk text based on paragraphs"""
        if not text.strip():
            return []

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Filter out short paragraphs
        valid_paragraphs = [
            p for p in paragraphs 
            if len(p) >= self.min_paragraph_length
        ]

        if not valid_paragraphs:
            return []

        # Create chunks
        chunks = []
        chunk_id = 0
        i = 0

        while i < len(valid_paragraphs):
            chunk_paragraphs = []
            start_idx = i

            # Add paragraphs to chunk
            while (i < len(valid_paragraphs) and 
                   len(chunk_paragraphs) < self.max_paragraphs):
                chunk_paragraphs.append(valid_paragraphs[i])
                i += 1

            chunk_text = '\n\n'.join(chunk_paragraphs)

            # Find character positions
            start_char = text.find(chunk_paragraphs[0])
            end_char = start_char + len(chunk_text)

            metadata = SentenceChunkMetadata(
                chunk_id=f"chunk_{chunk_id}",
                start_char=start_char,
                end_char=end_char,
                sentence_count=len(chunk_paragraphs),  # Using as paragraph count
                word_count=len(chunk_text.split()),
                source_document=source_document,
                chunk_type="paragraph"
            )

            chunks.append({
                "text": chunk_text,
                "metadata": metadata.__dict__
            })

            # Handle overlap
            if self.overlap_paragraphs > 0 and i < len(valid_paragraphs):
                i = max(start_idx + 1, i - self.overlap_paragraphs)

            chunk_id += 1

        return chunks
