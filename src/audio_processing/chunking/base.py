"""
Base chunking implementation.

Provides the abstract base class and common functionality for all chunkers.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """
    Represents a chunk of transcript content.
    
    Attributes:
        text: The chunk text content
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        speaker: Speaker identifier if available
        metadata: Additional metadata for the chunk
    """
    
    text: str
    start_time: int
    end_time: int
    speaker: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> int:
        """Return chunk duration in milliseconds."""
        return self.end_time - self.start_time
    
    @property
    def duration_seconds(self) -> float:
        """Return chunk duration in seconds."""
        return self.duration_ms / 1000.0
    
    def __len__(self) -> int:
        """Return the number of characters in the chunk."""
        return len(self.text)


@dataclass
class ChunkerConfig:
    """
    Configuration for chunking behavior.
    
    Attributes:
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        overlap_size: Number of characters to overlap between chunks
        preserve_sentences: Whether to avoid breaking sentences
    """
    
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    overlap_size: int = 50
    preserve_sentences: bool = True


class BaseChunker(ABC):
    """
    Base class for all chunking strategies.
    
    Provides common functionality for chunking transcripts with
    validation and error handling.
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        """
        Initialize the chunker.
        
        Args:
            config: Chunker configuration, uses defaults if not provided
        """
        self._config = config or ChunkerConfig()
        logger.debug(
            "Initialized %s with max_chunk_size=%d",
            self.__class__.__name__,
            self._config.max_chunk_size,
        )
    
    @property
    def config(self) -> ChunkerConfig:
        """Return the chunker configuration."""
        return self._config
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        pass
    
    def chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk transcript data into smaller pieces.
        
        This is a template method that validates input and delegates
        to the subclass implementation.
        
        Args:
            transcript_data: Raw transcript data from provider
            
        Returns:
            List of Chunk objects
            
        Raises:
            ValueError: If transcript data is invalid
        """
        self._validate_transcript(transcript_data)
        
        logger.info(
            "Chunking transcript with %s",
            self.__class__.__name__,
        )
        
        chunks = self._do_chunk(transcript_data)
        
        logger.info(
            "Created %d chunks from transcript",
            len(chunks),
        )
        
        return chunks
    
    @abstractmethod
    def _do_chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Perform the actual chunking.
        
        Subclasses must implement this method.
        
        Args:
            transcript_data: Validated transcript data
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def _validate_transcript(self, transcript_data: Dict[str, Any]) -> None:
        """
        Validate transcript data before chunking.
        
        Args:
            transcript_data: Transcript data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if not transcript_data:
            raise ValueError("Transcript data cannot be empty")
        
        if not isinstance(transcript_data, dict):
            raise ValueError("Transcript data must be a dictionary")
    
    def _merge_small_chunks(
        self,
        chunks: List[Chunk],
        min_size: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Merge chunks that are too small.
        
        Args:
            chunks: List of chunks to potentially merge
            min_size: Minimum chunk size, uses config default if not provided
            
        Returns:
            List of chunks with small ones merged
        """
        min_size = min_size or self._config.min_chunk_size
        
        if not chunks:
            return []
        
        merged: List[Chunk] = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            if len(current) < min_size:
                # Merge with next chunk
                current = Chunk(
                    text=current.text + " " + next_chunk.text,
                    start_time=current.start_time,
                    end_time=next_chunk.end_time,
                    speaker=current.speaker,
                    metadata={**current.metadata, **next_chunk.metadata},
                )
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        return merged
    
    def _split_large_chunk(
        self,
        chunk: Chunk,
        max_size: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Split a chunk that is too large.
        
        Args:
            chunk: Chunk to split
            max_size: Maximum chunk size, uses config default if not provided
            
        Returns:
            List of smaller chunks
        """
        max_size = max_size or self._config.max_chunk_size
        
        if len(chunk) <= max_size:
            return [chunk]
        
        text = chunk.text
        splits: List[Chunk] = []
        start_idx = 0
        
        while start_idx < len(text):
            end_idx = start_idx + max_size
            
            # Try to find sentence boundary
            if self._config.preserve_sentences and end_idx < len(text):
                # Look for sentence end near max_size
                boundary = self._find_sentence_boundary(
                    text, start_idx, end_idx
                )
                if boundary > start_idx:
                    end_idx = boundary
            
            chunk_text = text[start_idx:end_idx].strip()
            
            if chunk_text:
                # Estimate time proportionally
                progress = (start_idx + end_idx) / (2 * len(text))
                duration = chunk.end_time - chunk.start_time
                
                splits.append(Chunk(
                    text=chunk_text,
                    start_time=chunk.start_time + int(duration * start_idx / len(text)),
                    end_time=chunk.start_time + int(duration * end_idx / len(text)),
                    speaker=chunk.speaker,
                    metadata=chunk.metadata,
                ))
            
            start_idx = end_idx
        
        return splits
    
    def _find_sentence_boundary(
        self,
        text: str,
        start: int,
        end: int,
    ) -> int:
        """
        Find the nearest sentence boundary before end.
        
        Args:
            text: Full text to search
            start: Start position
            end: Target end position
            
        Returns:
            Position of sentence boundary, or end if none found
        """
        # Look backwards from end for sentence endings
        search_text = text[start:end]
        
        for marker in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            last_pos = search_text.rfind(marker)
            if last_pos > 0:
                return start + last_pos + len(marker)
        
        # Fall back to word boundary
        space_pos = search_text.rfind(" ")
        if space_pos > 0:
            return start + space_pos + 1
        
        return end
