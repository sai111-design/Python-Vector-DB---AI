from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import re

from .token_chunker import TokenBasedChunker, ChunkMetadata
from .sentence_chunker import SentenceBasedChunker, SentenceChunkMetadata

logger = logging.getLogger(__name__)

@dataclass
class HybridChunkMetadata:
    """Metadata for hybrid chunks"""
    chunk_id: str
    start_char: int
    end_char: int
    token_count: int
    sentence_count: int
    word_count: int
    source_document: Optional[str] = None
    chunk_type: str = "hybrid"
    primary_strategy: str = "token"
    secondary_strategy: str = "sentence"
    content_type: str = "text"  # text, code, table, list, etc.

class ContentTypeDetector:
    """Detect content type for adaptive chunking"""

    def __init__(self):
        self.patterns = {
            'code': [
                r'```[\w]*\n[\s\S]*?```',  # Code blocks
                r'`[^`]+`',                    # Inline code
                r'def\s+\w+\([^)]*\):',     # Python functions
                r'class\s+\w+[^:]*:',        # Python classes
                r'import\s+\w+',             # Python imports
                r'from\s+\w+\s+import',     # Python from imports
                r'function\s+\w+\([^)]*\)', # JavaScript functions
                r'var\s+\w+\s*=',           # Variable declarations
            ],
            'table': [
                r'\|[^\n]*\|',              # Markdown tables
                r'\+-+\+',                   # ASCII tables
                r'\t[^\n]*\t[^\n]*\t',     # Tab-separated values
            ],
            'list': [
                r'^\s*[-*+]\s+',             # Unordered lists
                r'^\s*\d+\.\s+',            # Ordered lists
                r'^\s*[a-zA-Z]\)\s+',       # Lettered lists
            ],
            'header': [
                r'^#{1,6}\s+',                # Markdown headers
                r'^.+\n[=-]+$',               # Underlined headers
            ],
            'quote': [
                r'^>\s+',                     # Blockquotes
                r'^\s*["\'\"][^"\'\"]*["\'\"]$', # Quoted text (fixed)
            ]
        }

    def detect_content_type(self, text: str) -> str:
        """Detect the primary content type of text"""
        if not text.strip():
            return 'text'

        # Check each pattern type
        for content_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                    return content_type

        return 'text'

    def analyze_content_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure for adaptive chunking"""
        lines = text.split('\n')

        analysis = {
            'total_lines': len(lines),
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'has_code_blocks': bool(re.search(r'```', text)),
            'has_headers': bool(re.search(r'^#{1,6}\s+', text, re.MULTILINE)),
            'has_lists': bool(re.search(r'^\s*[-*+\d]+[.)\s]', text, re.MULTILINE)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'content_type': self.detect_content_type(text)
        }

        return analysis

class HybridChunker:
    """Hybrid chunker that combines multiple chunking strategies"""

    def __init__(self, 
                 primary_strategy: str = "token",
                 fallback_strategy: str = "sentence",
                 max_tokens: int = 1000,
                 max_sentences: int = 5,
                 overlap_tokens: int = 100,
                 overlap_sentences: int = 1,
                 adaptive_sizing: bool = True,
                 preserve_structure: bool = True):
        """
        Initialize hybrid chunker

        Args:
            primary_strategy: Primary chunking strategy ('token' or 'sentence')
            fallback_strategy: Fallback strategy when primary fails
            max_tokens: Maximum tokens per chunk
            max_sentences: Maximum sentences per chunk
            overlap_tokens: Token overlap between chunks
            overlap_sentences: Sentence overlap between chunks
            adaptive_sizing: Whether to adapt chunk sizes based on content
            preserve_structure: Whether to preserve document structure
        """
        self.primary_strategy = primary_strategy
        self.fallback_strategy = fallback_strategy
        self.adaptive_sizing = adaptive_sizing
        self.preserve_structure = preserve_structure

        # Initialize chunkers
        self.token_chunker = TokenBasedChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            preserve_sentences=True
        )

        self.sentence_chunker = SentenceBasedChunker(
            max_sentences=max_sentences,
            overlap_sentences=overlap_sentences
        )

        self.content_detector = ContentTypeDetector()

    def chunk_text(self, text: str, source_document: str = None) -> List[Dict[str, Any]]:
        """
        Chunk text using hybrid approach

        Args:
            text: Text to chunk
            source_document: Source document identifier

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []

        # Analyze content structure
        content_analysis = self.content_detector.analyze_content_structure(text)

        # Choose chunking strategy based on content type
        strategy = self._choose_strategy(text, content_analysis)

        # Preprocess text if needed
        processed_text = self._preprocess_text(text, content_analysis)

        # Perform chunking
        if strategy == "token":
            chunks = self._chunk_with_tokens(processed_text, source_document, content_analysis)
        elif strategy == "sentence":
            chunks = self._chunk_with_sentences(processed_text, source_document, content_analysis)
        else:
            chunks = self._chunk_with_structure(processed_text, source_document, content_analysis)

        # Post-process chunks
        processed_chunks = self._postprocess_chunks(chunks, content_analysis)

        return processed_chunks

    def _choose_strategy(self, text: str, analysis: Dict[str, Any]) -> str:
        """Choose the best chunking strategy based on content analysis"""
        content_type = analysis['content_type']

        # Strategy selection based on content type
        if content_type == 'code':
            return 'structure'  # Preserve code structure
        elif content_type in ['table', 'list']:
            return 'structure'  # Preserve structural elements
        elif analysis['has_headers'] and self.preserve_structure:
            return 'structure'  # Chunk by sections
        elif analysis['avg_line_length'] > 100:  # Long lines suggest dense text
            return 'token'
        elif analysis['paragraph_count'] < 3:  # Few paragraphs
            return 'sentence'
        else:
            return self.primary_strategy

    def _preprocess_text(self, text: str, analysis: Dict[str, Any]) -> str:
        """Preprocess text based on content analysis"""
        # Basic cleanup
        processed_text = text

        # Normalize whitespace but preserve structure
        processed_text = re.sub(r'[ \t]+', ' ', processed_text)  # Normalize spaces
        processed_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', processed_text)  # Normalize paragraph breaks

        return processed_text.strip()

    def _chunk_with_tokens(self, text: str, source_document: str, 
                          analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk using token-based approach"""
        token_chunks = self.token_chunker.chunk_text(text, source_document)

        # Convert to hybrid metadata
        hybrid_chunks = []
        for chunk in token_chunks:
            original_metadata = chunk["metadata"]

            hybrid_metadata = HybridChunkMetadata(
                chunk_id=original_metadata["chunk_id"],
                start_char=original_metadata["start_char"],
                end_char=original_metadata["end_char"],
                token_count=original_metadata["token_count"],
                sentence_count=len(self.sentence_chunker.split_into_sentences(chunk["text"])),
                word_count=len(chunk["text"].split()),
                source_document=source_document,
                chunk_type="hybrid",
                primary_strategy="token",
                secondary_strategy="sentence",
                content_type=analysis["content_type"]
            )

            hybrid_chunks.append({
                "text": chunk["text"],
                "metadata": asdict(hybrid_metadata)
            })

        return hybrid_chunks

    def _chunk_with_sentences(self, text: str, source_document: str,
                            analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk using sentence-based approach"""
        sentence_chunks = self.sentence_chunker.chunk_text(text, source_document)

        # Convert to hybrid metadata
        hybrid_chunks = []
        for chunk in sentence_chunks:
            original_metadata = chunk["metadata"]

            hybrid_metadata = HybridChunkMetadata(
                chunk_id=original_metadata["chunk_id"],
                start_char=original_metadata["start_char"],
                end_char=original_metadata["end_char"],
                token_count=self.token_chunker.count_tokens(chunk["text"]),
                sentence_count=original_metadata["sentence_count"],
                word_count=original_metadata["word_count"],
                source_document=source_document,
                chunk_type="hybrid",
                primary_strategy="sentence",
                secondary_strategy="token",
                content_type=analysis["content_type"]
            )

            hybrid_chunks.append({
                "text": chunk["text"],
                "metadata": asdict(hybrid_metadata)
            })

        return hybrid_chunks

    def _chunk_with_structure(self, text: str, source_document: str,
                            analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk based on document structure"""
        content_type = analysis["content_type"]

        if content_type == 'code':
            return self._chunk_code_blocks(text, source_document, analysis)
        elif analysis['has_headers']:
            return self._chunk_by_headers(text, source_document, analysis)
        elif content_type in ['table', 'list']:
            return self._chunk_structured_content(text, source_document, analysis)
        else:
            # Fallback to primary strategy
            if self.primary_strategy == "token":
                return self._chunk_with_tokens(text, source_document, analysis)
            else:
                return self._chunk_with_sentences(text, source_document, analysis)

    def _chunk_code_blocks(self, text: str, source_document: str,
                          analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk code while preserving code block structure"""
        # Find code blocks
        code_block_pattern = r'```[\w]*\n([\s\S]*?)```'
        code_blocks = list(re.finditer(code_block_pattern, text))

        chunks = []
        chunk_id = 0
        last_end = 0

        for match in code_blocks:
            # Add text before code block
            if match.start() > last_end:
                pre_text = text[last_end:match.start()].strip()
                if pre_text:
                    chunk = self._create_text_chunk(pre_text, chunk_id, last_end, source_document, analysis)
                    chunks.append(chunk)
                    chunk_id += 1

            # Add code block as separate chunk
            code_text = match.group(0)
            chunk = self._create_code_chunk(code_text, chunk_id, match.start(), source_document)
            chunks.append(chunk)
            chunk_id += 1
            last_end = match.end()

        # Add remaining text
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                chunk = self._create_text_chunk(remaining_text, chunk_id, last_end, source_document, analysis)
                chunks.append(chunk)

        return chunks

    def _chunk_by_headers(self, text: str, source_document: str,
                         analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text by markdown headers"""
        # Find headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')

        chunks = []
        chunk_id = 0
        current_section = []
        current_start = 0

        for i, line in enumerate(lines):
            header_match = re.match(header_pattern, line)

            if header_match and current_section:
                # Create chunk from current section
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    chunk = self._create_text_chunk(
                        section_text, chunk_id, current_start, source_document, analysis
                    )
                    chunks.append(chunk)
                    chunk_id += 1

                # Start new section
                current_section = [line]
                current_start = sum(len(l) + 1 for l in lines[:i])  # Calculate character position
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                chunk = self._create_text_chunk(
                    section_text, chunk_id, current_start, source_document, analysis
                )
                chunks.append(chunk)

        return chunks

    def _chunk_structured_content(self, text: str, source_document: str,
                                analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk tables, lists, and other structured content"""
        # For now, treat as single chunks to preserve structure
        # In a more sophisticated implementation, you might split large tables/lists

        chunk_metadata = HybridChunkMetadata(
            chunk_id="chunk_0",
            start_char=0,
            end_char=len(text),
            token_count=self.token_chunker.count_tokens(text),
            sentence_count=len(self.sentence_chunker.split_into_sentences(text)),
            word_count=len(text.split()),
            source_document=source_document,
            chunk_type="hybrid",
            primary_strategy="structure",
            secondary_strategy=self.fallback_strategy,
            content_type=analysis["content_type"]
        )

        return [{
            "text": text,
            "metadata": asdict(chunk_metadata)
        }]

    def _create_text_chunk(self, text: str, chunk_id: int, start_char: int,
                          source_document: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a text chunk with hybrid metadata"""
        metadata = HybridChunkMetadata(
            chunk_id=f"chunk_{chunk_id}",
            start_char=start_char,
            end_char=start_char + len(text),
            token_count=self.token_chunker.count_tokens(text),
            sentence_count=len(self.sentence_chunker.split_into_sentences(text)),
            word_count=len(text.split()),
            source_document=source_document,
            chunk_type="hybrid",
            primary_strategy="structure",
            secondary_strategy=self.primary_strategy,
            content_type=analysis["content_type"]
        )

        return {
            "text": text,
            "metadata": asdict(metadata)
        }

    def _create_code_chunk(self, code_text: str, chunk_id: int, start_char: int,
                          source_document: str) -> Dict[str, Any]:
        """Create a code chunk with special handling"""
        metadata = HybridChunkMetadata(
            chunk_id=f"chunk_{chunk_id}",
            start_char=start_char,
            end_char=start_char + len(code_text),
            token_count=self.token_chunker.count_tokens(code_text),
            sentence_count=0,  # Code blocks don't have sentences
            word_count=len(code_text.split()),
            source_document=source_document,
            chunk_type="hybrid",
            primary_strategy="structure",
            secondary_strategy="token",
            content_type="code"
        )

        return {
            "text": code_text,
            "metadata": asdict(metadata)
        }

    def _postprocess_chunks(self, chunks: List[Dict[str, Any]], 
                          analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Post-process chunks for quality and consistency"""
        if not chunks:
            return chunks

        processed_chunks = []

        for chunk in chunks:
            # Skip empty chunks
            if not chunk["text"].strip():
                continue

            # Ensure minimum chunk size (unless it's structured content)
            if (analysis["content_type"] == "text" and 
                len(chunk["text"]) < 50 and 
                processed_chunks):
                # Merge with previous chunk
                processed_chunks[-1]["text"] += "\n\n" + chunk["text"]
                # Update metadata
                prev_metadata = processed_chunks[-1]["metadata"]
                prev_metadata["end_char"] = chunk["metadata"]["end_char"]
                prev_metadata["token_count"] += chunk["metadata"]["token_count"]
                prev_metadata["word_count"] += chunk["metadata"]["word_count"]
                prev_metadata["sentence_count"] += chunk["metadata"]["sentence_count"]
            else:
                processed_chunks.append(chunk)

        return processed_chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get comprehensive statistics about chunks"""
        if not chunks:
            return {}

        token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]
        sentence_counts = [chunk["metadata"]["sentence_count"] for chunk in chunks]
        word_counts = [chunk["metadata"]["word_count"] for chunk in chunks]
        text_lengths = [len(chunk["text"]) for chunk in chunks]

        content_types = {}
        strategies = {}

        for chunk in chunks:
            content_type = chunk["metadata"].get("content_type", "text")
            primary_strategy = chunk["metadata"].get("primary_strategy", "unknown")

            content_types[content_type] = content_types.get(content_type, 0) + 1
            strategies[primary_strategy] = strategies.get(primary_strategy, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "total_sentences": sum(sentence_counts),
            "total_words": sum(word_counts),
            "total_characters": sum(text_lengths),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "avg_sentences_per_chunk": sum(sentence_counts) / len(sentence_counts),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "avg_chars_per_chunk": sum(text_lengths) / len(text_lengths),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "content_type_distribution": content_types,
            "strategy_distribution": strategies
        }
