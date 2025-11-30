"""
Metadata building for audio documents.

Provides utilities for creating and managing document metadata.
"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentMetadata:
    """
    Structured metadata for audio documents.
    
    Attributes:
        source_name: Name of the audio source
        transcript_id: Unique transcript identifier
        audio_duration: Duration in seconds
        confidence: Overall confidence score
        created_at: Creation timestamp
        processing_info: Processing details
        extracted_features: Available extracted features
        custom: Custom metadata fields
    """
    
    source_name: str
    transcript_id: str
    audio_duration: Optional[float] = None
    confidence: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_info: Dict[str, Any] = field(default_factory=dict)
    extracted_features: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source_name": self.source_name,
            "transcript_id": self.transcript_id,
            "audio_duration": self.audio_duration,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "processing_info": self.processing_info,
            "extracted_features": self.extracted_features,
            **self.custom,
        }


class MetadataBuilder:
    """
    Builds metadata for audio documents.
    
    Provides methods to create base metadata and enhance it
    with extracted feature data.
    """
    
    def __init__(self) -> None:
        """Initialize the metadata builder."""
        self._feature_summarizers: Dict[str, callable] = {
            "sentiment": self._summarize_sentiment,
            "entities": self._summarize_entities,
            "chapters": self._summarize_chapters,
            "topics": self._summarize_topics,
            "highlights": self._summarize_highlights,
            "content_safety": self._summarize_content_safety,
        }
    
    def create_base_metadata(
        self,
        transcript_id: str,
        source_name: str,
        audio_duration: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> DocumentMetadata:
        """
        Create base metadata for a document.
        
        Args:
            transcript_id: Unique transcript identifier
            source_name: Name of the audio source
            audio_duration: Duration in seconds
            confidence: Confidence score (0-1)
            
        Returns:
            DocumentMetadata instance
        """
        return DocumentMetadata(
            source_name=source_name,
            transcript_id=transcript_id,
            audio_duration=audio_duration,
            confidence=confidence,
            processing_info={
                "processor": "audio_processing",
                "version": "1.0.0",
            },
        )
    
    def enhance_with_extracted(
        self,
        metadata: DocumentMetadata,
        extracted_data: Dict[str, Dict[str, Any]],
    ) -> DocumentMetadata:
        """
        Enhance metadata with extracted feature data.
        
        Args:
            metadata: Base metadata to enhance
            extracted_data: Data from extractors, keyed by extractor name
            
        Returns:
            Enhanced DocumentMetadata
        """
        features = list(extracted_data.keys())
        metadata.extracted_features = features
        
        for feature_name, feature_data in extracted_data.items():
            summarizer = self._feature_summarizers.get(feature_name)
            
            if summarizer:
                summary = summarizer(feature_data)
                metadata.custom[f"{feature_name}_summary"] = summary
        
        logger.debug(
            "Enhanced metadata with %d features: %s",
            len(features),
            features,
        )
        
        return metadata
    
    def build_chunk_metadata(
        self,
        base_metadata: DocumentMetadata,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
        speaker: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build metadata for a chunk document.
        
        Args:
            base_metadata: Base document metadata
            chunk_index: Index of this chunk
            chunk_start: Start time in milliseconds
            chunk_end: End time in milliseconds
            speaker: Speaker identifier if available
            chunk_type: Type of chunking used
            
        Returns:
            Chunk metadata dictionary
        """
        chunk_meta = base_metadata.to_dict()
        
        chunk_meta.update({
            "chunk_index": chunk_index,
            "chunk_start_ms": chunk_start,
            "chunk_end_ms": chunk_end,
            "chunk_start_sec": chunk_start / 1000.0,
            "chunk_end_sec": chunk_end / 1000.0,
            "chunk_duration_sec": (chunk_end - chunk_start) / 1000.0,
        })
        
        if speaker:
            chunk_meta["speaker"] = speaker
        
        if chunk_type:
            chunk_meta["chunk_type"] = chunk_type
        
        return chunk_meta
    
    def _summarize_sentiment(
        self,
        sentiment_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize sentiment data for metadata."""
        distribution = sentiment_data.get("distribution", {})
        
        dominant = max(
            distribution.items(),
            key=lambda x: x[1],
            default=("UNKNOWN", 0),
        )
        
        return {
            "dominant_sentiment": dominant[0],
            "sentiment_distribution": distribution,
            "sentiment_count": sentiment_data.get("total_count", 0),
        }
    
    def _summarize_entities(
        self,
        entity_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize entity data for metadata."""
        by_type = entity_data.get("entities_by_type", {})
        
        return {
            "entity_types": list(by_type.keys()),
            "entity_count": entity_data.get("total_count", 0),
        }
    
    def _summarize_chapters(
        self,
        chapter_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize chapter data for metadata."""
        chapters = chapter_data.get("chapters", [])
        
        return {
            "chapter_count": len(chapters),
            "headlines": [c.get("headline", "") for c in chapters[:5]],
        }
    
    def _summarize_topics(
        self,
        topic_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize topic data for metadata."""
        topics = topic_data.get("topics", [])
        
        top_topics = sorted(
            topics,
            key=lambda t: t.get("relevance", 0),
            reverse=True,
        )[:5]
        
        return {
            "topic_count": len(topics),
            "top_topics": [t.get("label", "") for t in top_topics],
        }
    
    def _summarize_highlights(
        self,
        highlight_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize highlight data for metadata."""
        highlights = highlight_data.get("highlights", [])
        
        top = sorted(
            highlights,
            key=lambda h: h.get("rank", 0),
            reverse=True,
        )[:3]
        
        return {
            "highlight_count": len(highlights),
            "top_highlights": [h.get("text", "") for h in top],
        }
    
    def _summarize_content_safety(
        self,
        safety_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize content safety data for metadata."""
        return {
            "is_safe": safety_data.get("is_safe", True),
            "flagged_categories": safety_data.get("flagged_categories", []),
        }
