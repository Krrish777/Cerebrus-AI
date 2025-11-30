"""
Semantic-based chunking strategy.

Chunks transcript by semantic boundaries using paragraph or sentence structure.
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


class SemanticChunker(BaseChunker):
    """
    Chunks transcript by semantic boundaries.
    
    Uses paragraphs as primary semantic units, with fallback
    to sentence-based chunking for unstructured text.
    """
    
    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        paragraph_separator: str = "\n\n",
    ) -> None:
        """
        Initialize the semantic chunker.
        
        Args:
            config: Chunker configuration
            paragraph_separator: String that separates paragraphs
        """
        super().__init__(config)
        self._paragraph_separator = paragraph_separator
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        return "semantic"
    
    def _do_chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk transcript by semantic boundaries.
        
        Args:
            transcript_data: Transcript data
            
        Returns:
            List of semantically coherent chunks
        """
        # Try paragraphs first
        paragraphs = transcript_data.get("paragraphs", [])
        
        if paragraphs:
            chunks = self._chunk_from_paragraphs(paragraphs)
        else:
            # Fall back to text-based paragraph detection
            text = transcript_data.get("text", "")
            chunks = self._chunk_from_text_paragraphs(
                text, transcript_data
            )
        
        # Apply size constraints while preserving semantics
        return self._apply_semantic_constraints(chunks)
    
    def _chunk_from_paragraphs(
        self,
        paragraphs: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        Create chunks from paragraph data.
        
        Args:
            paragraphs: List of paragraph dictionaries
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        for idx, para in enumerate(paragraphs):
            text = para.get("text", "").strip()
            if not text:
                continue
            
            chunk = Chunk(
                text=text,
                start_time=para.get("start", 0),
                end_time=para.get("end", 0),
                speaker=None,
                metadata={
                    "paragraph_index": idx,
                    "sentences": para.get("sentences", []),
                },
            )
            chunks.append(chunk)
        
        logger.debug(
            "Created %d chunks from %d paragraphs",
            len(chunks),
            len(paragraphs),
        )
        
        return chunks
    
    def _chunk_from_text_paragraphs(
        self,
        text: str,
        transcript_data: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create chunks from text by detecting paragraph boundaries.
        
        Args:
            text: Full transcript text
            transcript_data: Full data for duration info
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        # Split by paragraph separator
        parts = text.split(self._paragraph_separator)
        parts = [p.strip() for p in parts if p.strip()]
        
        if not parts:
            return []
        
        total_duration = transcript_data.get("audio_duration", 0)
        total_chars = sum(len(p) for p in parts)
        
        chunks = []
        current_time = 0
        
        for idx, part in enumerate(parts):
            # Estimate duration proportionally
            part_duration = int(total_duration * len(part) / total_chars) if total_chars else 0
            
            chunk = Chunk(
                text=part,
                start_time=current_time,
                end_time=current_time + part_duration,
                speaker=None,
                metadata={"paragraph_index": idx},
            )
            chunks.append(chunk)
            current_time += part_duration
        
        logger.debug(
            "Created %d chunks from text paragraphs",
            len(chunks),
        )
        
        return chunks
    
    def _apply_semantic_constraints(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply size constraints while preserving semantic coherence.
        
        Args:
            chunks: Chunks to constrain
            
        Returns:
            Size-constrained chunks
        """
        result: List[Chunk] = []
        
        for chunk in chunks:
            if len(chunk) > self._config.max_chunk_size:
                # Split by sentences to preserve meaning
                sub_chunks = self._split_by_sentences(chunk)
                result.extend(sub_chunks)
            else:
                result.append(chunk)
        
        # Merge small chunks but only with adjacent ones
        return self._merge_adjacent_small(result)
    
    def _split_by_sentences(self, chunk: Chunk) -> List[Chunk]:
        """
        Split a chunk by sentence boundaries.
        
        Args:
            chunk: Large chunk to split
            
        Returns:
            List of sentence-based sub-chunks
        """
        text = chunk.text
        sentences = self._detect_sentences(text)
        
        if not sentences:
            return self._split_large_chunk(chunk)
        
        sub_chunks = []
        current_text = ""
        current_start = chunk.start_time
        duration = chunk.end_time - chunk.start_time
        
        for sentence in sentences:
            if len(current_text) + len(sentence) > self._config.max_chunk_size:
                if current_text:
                    # Calculate end time proportionally
                    progress = len(current_text) / len(text)
                    end_time = chunk.start_time + int(duration * progress)
                    
                    sub_chunks.append(Chunk(
                        text=current_text.strip(),
                        start_time=current_start,
                        end_time=end_time,
                        speaker=chunk.speaker,
                        metadata={
                            **chunk.metadata,
                            "is_sub_chunk": True,
                        },
                    ))
                    current_start = end_time
                    current_text = sentence
                else:
                    # Single sentence too long, add it anyway
                    current_text = sentence
            else:
                current_text += " " + sentence if current_text else sentence
        
        if current_text:
            sub_chunks.append(Chunk(
                text=current_text.strip(),
                start_time=current_start,
                end_time=chunk.end_time,
                speaker=chunk.speaker,
                metadata={
                    **chunk.metadata,
                    "is_sub_chunk": True,
                },
            ))
        
        return sub_chunks
    
    def _detect_sentences(self, text: str) -> List[str]:
        """
        Detect sentence boundaries in text.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Simple sentence detection
        # In production, consider using nltk or spacy
        import re
        
        # Split on sentence-ending punctuation
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_adjacent_small(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge adjacent small chunks preserving semantic boundaries.
        
        Args:
            chunks: Chunks to potentially merge
            
        Returns:
            Merged chunks
        """
        if not chunks:
            return []
        
        merged: List[Chunk] = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            combined_size = len(current) + len(next_chunk) + 1
            
            # Only merge if both are small and combined fits max
            if (
                len(current) < self._config.min_chunk_size
                and combined_size <= self._config.max_chunk_size
            ):
                current = Chunk(
                    text=current.text + " " + next_chunk.text,
                    start_time=current.start_time,
                    end_time=next_chunk.end_time,
                    speaker=current.speaker,
                    metadata=current.metadata,
                )
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        return merged
