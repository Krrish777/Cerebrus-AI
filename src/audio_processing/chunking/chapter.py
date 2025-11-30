"""
Chapter-based chunking strategy.

Chunks transcript by chapter boundaries from auto-chapters feature.
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


class ChapterChunker(BaseChunker):
    """
    Chunks transcript by chapter boundaries.
    
    Uses auto-chapters feature to create logical content divisions.
    Falls back to speaker chunking if no chapters available.
    """
    
    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        include_headline: bool = True,
        include_summary: bool = True,
    ) -> None:
        """
        Initialize the chapter chunker.
        
        Args:
            config: Chunker configuration
            include_headline: Include chapter headline in metadata
            include_summary: Include chapter summary in metadata
        """
        super().__init__(config)
        self._include_headline = include_headline
        self._include_summary = include_summary
    
    @property
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        return "chapter"
    
    def _do_chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk transcript by chapters.
        
        Args:
            transcript_data: Transcript data with chapters
            
        Returns:
            List of chunks, one per chapter
        """
        chapters = transcript_data.get("chapters", [])
        
        if not chapters:
            logger.warning(
                "No chapters found, returning single chunk from text"
            )
            return self._chunk_from_full_text(transcript_data)
        
        chunks = self._chunk_from_chapters(chapters, transcript_data)
        
        # Apply size constraints
        return self._apply_size_constraints(chunks)
    
    def _chunk_from_chapters(
        self,
        chapters: List[Dict[str, Any]],
        transcript_data: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create chunks from chapter data.
        
        Args:
            chapters: List of chapter dictionaries
            transcript_data: Full transcript for text extraction
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        full_text = transcript_data.get("text", "")
        words = transcript_data.get("words", [])
        
        for idx, chapter in enumerate(chapters):
            start_time = chapter.get("start", 0)
            end_time = chapter.get("end", 0)
            
            # Extract text for this chapter's time range
            chapter_text = self._extract_text_for_range(
                words, full_text, start_time, end_time
            )
            
            if not chapter_text:
                chapter_text = chapter.get("gist", "") or chapter.get("headline", "")
            
            metadata: Dict[str, Any] = {
                "chapter_index": idx,
            }
            
            if self._include_headline:
                metadata["headline"] = chapter.get("headline", "")
            
            if self._include_summary:
                metadata["summary"] = chapter.get("summary", "")
                metadata["gist"] = chapter.get("gist", "")
            
            chunk = Chunk(
                text=chapter_text.strip(),
                start_time=start_time,
                end_time=end_time,
                speaker=None,
                metadata=metadata,
            )
            chunks.append(chunk)
        
        logger.debug(
            "Created %d chunks from %d chapters",
            len(chunks),
            len(chapters),
        )
        
        return chunks
    
    def _extract_text_for_range(
        self,
        words: List[Dict[str, Any]],
        full_text: str,
        start_time: int,
        end_time: int,
    ) -> str:
        """
        Extract text content for a time range.
        
        Args:
            words: Word-level data with timestamps
            full_text: Full transcript text as fallback
            start_time: Range start in milliseconds
            end_time: Range end in milliseconds
            
        Returns:
            Extracted text for the range
        """
        if words:
            range_words = [
                w.get("text", "")
                for w in words
                if start_time <= w.get("start", 0) < end_time
            ]
            if range_words:
                return " ".join(range_words)
        
        # Fallback: estimate from full text
        # This is approximate when word timestamps aren't available
        return ""
    
    def _chunk_from_full_text(
        self,
        transcript_data: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create a single chunk from full transcript.
        
        Args:
            transcript_data: Transcript with text field
            
        Returns:
            List containing single chunk
        """
        text = transcript_data.get("text", "").strip()
        
        if not text:
            return []
        
        return [Chunk(
            text=text,
            start_time=0,
            end_time=transcript_data.get("audio_duration", 0),
            speaker=None,
            metadata={"chapter_index": 0, "headline": "Full Transcript"},
        )]
    
    def _apply_size_constraints(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply size constraints to chapter chunks.
        
        Args:
            chunks: Chunks to constrain
            
        Returns:
            Size-constrained chunks
        """
        result: List[Chunk] = []
        
        for chunk in chunks:
            if len(chunk) > self._config.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                # Preserve chapter metadata on sub-chunks
                for i, sub in enumerate(sub_chunks):
                    sub.metadata["sub_chunk_index"] = i
                    sub.metadata["parent_headline"] = chunk.metadata.get("headline", "")
                result.extend(sub_chunks)
            else:
                result.append(chunk)
        
        return result
    
    def get_chapter_outline(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Generate an outline from chapter chunks.
        
        Args:
            chunks: Chapter-based chunks
            
        Returns:
            List of outline entries with headlines and timestamps
        """
        outline = []
        
        for chunk in chunks:
            if "chapter_index" in chunk.metadata:
                outline.append({
                    "index": chunk.metadata.get("chapter_index"),
                    "headline": chunk.metadata.get("headline", ""),
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration_seconds": chunk.duration_seconds,
                })
        
        return outline
