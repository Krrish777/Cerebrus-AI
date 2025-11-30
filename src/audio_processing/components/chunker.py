"""
Chunker Haystack component.

Provides a Haystack-compatible component for chunking transcripts.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import component

from src.audio_processing.chunking.base import Chunk
from src.audio_processing.chunking.base import ChunkerConfig
from src.audio_processing.chunking.registry import ChunkerRegistry
from src.audio_processing.chunking.registry import get_global_registry
from src.core.logging import get_logger

logger = get_logger(__name__)


@component
class ChunkerComponent:
    """
    Haystack component for chunking transcripts.
    
    Splits transcript content into smaller chunks using
    various chunking strategies.
    
    Inputs:
        transcripts: List of transcript dictionaries
        
    Outputs:
        transcripts: Original transcripts
        chunks: List of chunk lists (one per transcript)
    """
    
    def __init__(
        self,
        strategy: str = "auto",
        config: Optional[ChunkerConfig] = None,
        registry: Optional[ChunkerRegistry] = None,
    ) -> None:
        """
        Initialize the chunker component.
        
        Args:
            strategy: Chunking strategy name or 'auto' for automatic selection
            config: Optional chunker configuration
            registry: Optional custom chunker registry
        """
        self._strategy = strategy
        self._config = config
        self._registry = registry or get_global_registry()
    
    @component.output_types(
        transcripts=List[Dict[str, Any]],
        chunks=List[List[Dict[str, Any]]],
    )
    def run(
        self,
        transcripts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Chunk transcripts.
        
        Args:
            transcripts: List of transcript dictionaries
            
        Returns:
            Dictionary with:
                - transcripts: Original transcripts (passed through)
                - chunks: List of chunk lists (serialized)
        """
        all_chunks = []
        
        for transcript in transcripts:
            # Skip error transcripts
            if transcript.get("status") == "error":
                all_chunks.append([])
                continue
            
            chunks = self._chunk_transcript(transcript)
            
            # Convert to serializable format
            serialized_chunks = [
                self._serialize_chunk(chunk) for chunk in chunks
            ]
            all_chunks.append(serialized_chunks)
            
            logger.info(
                "Created %d chunks from transcript %s using %s strategy",
                len(chunks),
                transcript.get("id", "unknown"),
                self._strategy,
            )
        
        return {
            "transcripts": transcripts,
            "chunks": all_chunks,
        }
    
    def _chunk_transcript(
        self,
        transcript: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Chunk a single transcript.
        
        Args:
            transcript: Transcript dictionary
            
        Returns:
            List of Chunk objects
        """
        if self._strategy == "auto":
            return self._registry.chunk_with_best(transcript, self._config)
        else:
            return self._registry.chunk_with(
                self._strategy,
                transcript,
                self._config,
            )
    
    def _serialize_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Convert a Chunk to a serializable dictionary.
        
        Args:
            chunk: Chunk object
            
        Returns:
            Serialized chunk dictionary
        """
        return {
            "text": chunk.text,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "speaker": chunk.speaker,
            "duration_ms": chunk.duration_ms,
            "duration_seconds": chunk.duration_seconds,
            "metadata": chunk.metadata,
        }
