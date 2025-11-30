"""
Speaker-based chunking strategy.

Chunks transcript by speaker turns, creating one chunk per utterance.
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


class SpeakerChunker(BaseChunker):
    """
    Chunks transcript by speaker utterances.
    
    Creates one chunk per speaker turn, optionally merging
    consecutive turns by the same speaker.
    """
    
    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        merge_consecutive: bool = True,
    ) -> None:
        """
        Initialize the speaker chunker.
        
        Args:
            config: Chunker configuration
            merge_consecutive: Whether to merge consecutive same-speaker utterances
        """
        super().__init__(config)
        self._merge_consecutive = merge_consecutive
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        return "speaker"
    
    def _do_chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk transcript by speaker utterances.
        
        Args:
            transcript_data: Transcript data with utterances
            
        Returns:
            List of chunks, one per speaker turn
        """
        utterances = transcript_data.get("utterances", [])
        
        if not utterances:
            # Fall back to full text if no utterances
            return self._chunk_from_text(transcript_data)
        
        chunks = self._chunk_from_utterances(utterances)
        
        if self._merge_consecutive:
            chunks = self._merge_same_speaker(chunks)
        
        # Handle size constraints
        chunks = self._apply_size_constraints(chunks)
        
        return chunks
    
    def _chunk_from_utterances(
        self,
        utterances: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        Create chunks from utterance data.
        
        Args:
            utterances: List of utterance dictionaries
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        for utterance in utterances:
            text = utterance.get("text", "").strip()
            if not text:
                continue
            
            chunk = Chunk(
                text=text,
                start_time=utterance.get("start", 0),
                end_time=utterance.get("end", 0),
                speaker=utterance.get("speaker", None),
                metadata={
                    "confidence": utterance.get("confidence"),
                    "words": utterance.get("words", []),
                },
            )
            chunks.append(chunk)
        
        logger.debug(
            "Created %d chunks from %d utterances",
            len(chunks),
            len(utterances),
        )
        
        return chunks
    
    def _chunk_from_text(
        self,
        transcript_data: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create a single chunk from full transcript text.
        
        Args:
            transcript_data: Transcript with text field
            
        Returns:
            List containing single chunk
        """
        text = transcript_data.get("text", "").strip()
        
        if not text:
            logger.warning("No text content found in transcript")
            return []
        
        return [Chunk(
            text=text,
            start_time=0,
            end_time=transcript_data.get("audio_duration", 0),
            speaker=None,
            metadata={},
        )]
    
    def _merge_same_speaker(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge consecutive chunks from the same speaker.
        
        Args:
            chunks: List of chunks to merge
            
        Returns:
            List with consecutive same-speaker chunks merged
        """
        if not chunks:
            return []
        
        merged: List[Chunk] = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            if (
                current.speaker == next_chunk.speaker
                and current.speaker is not None
            ):
                # Merge with next
                current = Chunk(
                    text=current.text + " " + next_chunk.text,
                    start_time=current.start_time,
                    end_time=next_chunk.end_time,
                    speaker=current.speaker,
                    metadata={
                        "word_count": (
                            len(current.metadata.get("words", []))
                            + len(next_chunk.metadata.get("words", []))
                        ),
                    },
                )
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        
        logger.debug(
            "Merged %d chunks into %d after same-speaker merge",
            len(chunks),
            len(merged),
        )
        
        return merged
    
    def _apply_size_constraints(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply size constraints to chunks.
        
        Splits large chunks and merges small ones.
        
        Args:
            chunks: Chunks to constrain
            
        Returns:
            Size-constrained chunks
        """
        result: List[Chunk] = []
        
        for chunk in chunks:
            if len(chunk) > self._config.max_chunk_size:
                result.extend(self._split_large_chunk(chunk))
            else:
                result.append(chunk)
        
        return self._merge_small_chunks(result)
    
    def get_speaker_stats(self, chunks: List[Chunk]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics per speaker.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary mapping speaker to stats
        """
        stats: Dict[str, Dict[str, Any]] = {}
        
        for chunk in chunks:
            speaker = chunk.speaker or "Unknown"
            
            if speaker not in stats:
                stats[speaker] = {
                    "chunk_count": 0,
                    "total_duration_ms": 0,
                    "total_characters": 0,
                }
            
            stats[speaker]["chunk_count"] += 1
            stats[speaker]["total_duration_ms"] += chunk.duration_ms
            stats[speaker]["total_characters"] += len(chunk)
        
        return stats
