"""
Sentence-based chunking strategy.

Chunks transcript by sentences with configurable overlap.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from src.audio_processing.chunking.base import BaseChunker
from src.audio_processing.chunking.base import Chunk
from src.audio_processing.chunking.base import ChunkerConfig
from src.core.logging import get_logger

logger = get_logger(__name__)


class SentenceChunker(BaseChunker):
    """
    Chunks transcript by sentences with overlap.
    
    Creates fixed-size chunks based on sentence boundaries,
    with configurable overlap between consecutive chunks.
    """
    
    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1,
    ) -> None:
        """
        Initialize the sentence chunker.
        
        Args:
            config: Chunker configuration
            sentences_per_chunk: Target number of sentences per chunk
            overlap_sentences: Number of sentences to overlap
        """
        super().__init__(config)
        self._sentences_per_chunk = sentences_per_chunk
        self._overlap_sentences = overlap_sentences
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        return "sentence"
    
    def _do_chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk transcript by sentences.
        
        Args:
            transcript_data: Transcript data
            
        Returns:
            List of sentence-based chunks
        """
        sentences = transcript_data.get("sentences", [])
        
        if sentences:
            chunks = self._chunk_from_sentences(sentences)
        else:
            # Fall back to detecting sentences from text
            text = transcript_data.get("text", "")
            chunks = self._chunk_from_text_sentences(text, transcript_data)
        
        return chunks
    
    def _chunk_from_sentences(
        self,
        sentences: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        Create chunks from sentence data.
        
        Args:
            sentences: List of sentence dictionaries
            
        Returns:
            List of Chunk objects
        """
        if not sentences:
            return []
        
        chunks = []
        step = max(1, self._sentences_per_chunk - self._overlap_sentences)
        
        for i in range(0, len(sentences), step):
            end_idx = min(i + self._sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            
            if not chunk_sentences:
                continue
            
            text = " ".join(s.get("text", "") for s in chunk_sentences)
            
            chunk = Chunk(
                text=text.strip(),
                start_time=chunk_sentences[0].get("start", 0),
                end_time=chunk_sentences[-1].get("end", 0),
                speaker=None,
                metadata={
                    "sentence_start_idx": i,
                    "sentence_end_idx": end_idx,
                    "sentence_count": len(chunk_sentences),
                },
            )
            chunks.append(chunk)
        
        logger.debug(
            "Created %d chunks from %d sentences",
            len(chunks),
            len(sentences),
        )
        
        return chunks
    
    def _chunk_from_text_sentences(
        self,
        text: str,
        transcript_data: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create chunks by detecting sentences in text.
        
        Args:
            text: Full transcript text
            transcript_data: Full data for duration info
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        sentences = self._detect_sentences(text)
        
        if not sentences:
            return []
        
        total_duration = transcript_data.get("audio_duration", 0)
        total_chars = sum(len(s) for s in sentences)
        
        chunks = []
        step = max(1, self._sentences_per_chunk - self._overlap_sentences)
        current_time = 0
        
        for i in range(0, len(sentences), step):
            end_idx = min(i + self._sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            
            if not chunk_sentences:
                continue
            
            chunk_text = " ".join(chunk_sentences)
            
            # Estimate duration proportionally
            chunk_duration = int(
                total_duration * len(chunk_text) / total_chars
            ) if total_chars else 0
            
            chunk = Chunk(
                text=chunk_text.strip(),
                start_time=current_time,
                end_time=current_time + chunk_duration,
                speaker=None,
                metadata={
                    "sentence_start_idx": i,
                    "sentence_end_idx": end_idx,
                    "sentence_count": len(chunk_sentences),
                },
            )
            chunks.append(chunk)
            
            # Advance time (accounting for overlap)
            advance_sentences = sentences[i:i + step]
            advance_duration = int(
                total_duration * sum(len(s) for s in advance_sentences) / total_chars
            ) if total_chars else 0
            current_time += advance_duration
        
        logger.debug(
            "Created %d chunks from text with %d detected sentences",
            len(chunks),
            len(sentences),
        )
        
        return chunks
    
    def _detect_sentences(self, text: str) -> List[str]:
        """
        Detect sentence boundaries in text.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        import re
        
        # Split on sentence-ending punctuation followed by space or newline
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def calculate_overlap_tokens(
        self,
        chunks: List[Chunk],
    ) -> Dict[str, Any]:
        """
        Calculate overlap statistics between chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Overlap statistics
        """
        if len(chunks) < 2:
            return {
                "total_chunks": len(chunks),
                "overlapping_pairs": 0,
                "average_overlap_chars": 0,
            }
        
        overlaps = []
        
        for i in range(len(chunks) - 1):
            curr_text = chunks[i].text
            next_text = chunks[i + 1].text
            
            # Find common suffix/prefix
            overlap_len = self._find_overlap(curr_text, next_text)
            overlaps.append(overlap_len)
        
        return {
            "total_chunks": len(chunks),
            "overlapping_pairs": sum(1 for o in overlaps if o > 0),
            "average_overlap_chars": sum(overlaps) / len(overlaps) if overlaps else 0,
            "max_overlap_chars": max(overlaps) if overlaps else 0,
        }
    
    def _find_overlap(self, text1: str, text2: str) -> int:
        """
        Find the overlap between end of text1 and start of text2.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Number of overlapping characters
        """
        max_overlap = min(len(text1), len(text2), 500)  # Limit search
        
        for length in range(max_overlap, 0, -1):
            if text1[-length:] == text2[:length]:
                return length
        
        return 0
