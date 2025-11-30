"""
Document builder for audio transcripts.

Converts transcript data and extracted features into Haystack Documents.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack.dataclasses import Document

from src.audio_processing.chunking.base import Chunk
from src.audio_processing.document.metadata import DocumentMetadata
from src.audio_processing.document.metadata import MetadataBuilder
from src.audio_processing.interfaces import DocumentBuilder
from src.core.logging import get_logger

logger = get_logger(__name__)


class TranscriptDocumentBuilder(DocumentBuilder):
    """
    Builds Haystack Documents from audio transcripts.
    
    Converts transcript data, chunks, and extracted features
    into properly structured Haystack Document objects.
    """
    
    def __init__(
        self,
        metadata_builder: Optional[MetadataBuilder] = None,
    ) -> None:
        """
        Initialize the document builder.
        
        Args:
            metadata_builder: Optional custom metadata builder
        """
        self._metadata_builder = metadata_builder or MetadataBuilder()
    
    def build(
        self,
        transcript_data: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
        source_name: str,
    ) -> List[Document]:
        """
        Build a single document from the full transcript.
        
        Args:
            transcript_data: Raw transcript data
            extracted_data: Data from extractors
            source_name: Name of the audio source
            
        Returns:
            List containing one Document with full transcript
        """
        metadata = self._create_full_metadata(
            transcript_data, extracted_data, source_name
        )
        
        text = transcript_data.get("text", "")
        
        doc = Document(
            content=text,
            meta=metadata.to_dict(),
        )
        
        logger.info(
            "Built document from transcript: %s (%d chars)",
            source_name,
            len(text),
        )
        
        return [doc]
    
    def build_from_chunks(
        self,
        chunks: List[Chunk],
        transcript_data: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
        source_name: str,
    ) -> List[Document]:
        """
        Build documents from chunks.
        
        Creates one document per chunk with appropriate metadata.
        
        Args:
            chunks: List of transcript chunks
            transcript_data: Full transcript data for base metadata
            extracted_data: Data from extractors
            source_name: Name of the audio source
            
        Returns:
            List of Documents, one per chunk
        """
        base_metadata = self._create_full_metadata(
            transcript_data, extracted_data, source_name
        )
        
        documents = []
        
        for idx, chunk in enumerate(chunks):
            chunk_meta = self._metadata_builder.build_chunk_metadata(
                base_metadata=base_metadata,
                chunk_index=idx,
                chunk_start=chunk.start_time,
                chunk_end=chunk.end_time,
                speaker=chunk.speaker,
                chunk_type=chunk.metadata.get("chunk_type"),
            )
            
            # Add any chunk-specific metadata
            for key, value in chunk.metadata.items():
                if key not in chunk_meta and key != "chunk_type":
                    chunk_meta[f"chunk_{key}"] = value
            
            doc = Document(
                content=chunk.text,
                meta=chunk_meta,
            )
            documents.append(doc)
        
        logger.info(
            "Built %d documents from chunks for: %s",
            len(documents),
            source_name,
        )
        
        return documents
    
    def build_metadata(
        self,
        transcript_data: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
        source_name: str,
    ) -> Dict[str, Any]:
        """
        Build metadata dictionary for documents.
        
        Args:
            transcript_data: Raw transcript data
            extracted_data: Data from extractors
            source_name: Name of the audio source
            
        Returns:
            Metadata dictionary
        """
        metadata = self._create_full_metadata(
            transcript_data, extracted_data, source_name
        )
        return metadata.to_dict()
    
    def _create_full_metadata(
        self,
        transcript_data: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
        source_name: str,
    ) -> DocumentMetadata:
        """
        Create complete metadata from transcript and extracted data.
        
        Args:
            transcript_data: Raw transcript data
            extracted_data: Data from extractors
            source_name: Name of the audio source
            
        Returns:
            Complete DocumentMetadata instance
        """
        base_metadata = self._metadata_builder.create_base_metadata(
            transcript_id=transcript_data.get("id", "unknown"),
            source_name=source_name,
            audio_duration=transcript_data.get("audio_duration"),
            confidence=transcript_data.get("confidence"),
        )
        
        if extracted_data:
            base_metadata = self._metadata_builder.enhance_with_extracted(
                base_metadata, extracted_data
            )
        
        return base_metadata
    
    def build_utterance_documents(
        self,
        transcript_data: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
        source_name: str,
    ) -> List[Document]:
        """
        Build documents from utterances (speaker turns).
        
        Creates one document per utterance with speaker information.
        
        Args:
            transcript_data: Raw transcript data with utterances
            extracted_data: Data from extractors
            source_name: Name of the audio source
            
        Returns:
            List of Documents, one per utterance
        """
        utterances = transcript_data.get("utterances", [])
        
        if not utterances:
            logger.warning(
                "No utterances found in transcript, falling back to full document"
            )
            return self.build(transcript_data, extracted_data, source_name)
        
        base_metadata = self._create_full_metadata(
            transcript_data, extracted_data, source_name
        )
        
        documents = []
        
        for idx, utterance in enumerate(utterances):
            text = utterance.get("text", "")
            if not text:
                continue
            
            utt_meta = self._metadata_builder.build_chunk_metadata(
                base_metadata=base_metadata,
                chunk_index=idx,
                chunk_start=utterance.get("start", 0),
                chunk_end=utterance.get("end", 0),
                speaker=utterance.get("speaker"),
                chunk_type="utterance",
            )
            
            utt_meta["utterance_confidence"] = utterance.get("confidence")
            
            doc = Document(
                content=text,
                meta=utt_meta,
            )
            documents.append(doc)
        
        logger.info(
            "Built %d documents from utterances for: %s",
            len(documents),
            source_name,
        )
        
        return documents
    
    def build_chapter_documents(
        self,
        transcript_data: Dict[str, Any],
        extracted_data: Dict[str, Dict[str, Any]],
        source_name: str,
    ) -> List[Document]:
        """
        Build documents from chapters.
        
        Creates one document per chapter with headline and summary.
        
        Args:
            transcript_data: Raw transcript data with chapters
            extracted_data: Data from extractors
            source_name: Name of the audio source
            
        Returns:
            List of Documents, one per chapter
        """
        chapters = transcript_data.get("chapters", [])
        
        if not chapters:
            logger.warning(
                "No chapters found in transcript, falling back to full document"
            )
            return self.build(transcript_data, extracted_data, source_name)
        
        base_metadata = self._create_full_metadata(
            transcript_data, extracted_data, source_name
        )
        
        words = transcript_data.get("words", [])
        documents = []
        
        for idx, chapter in enumerate(chapters):
            start = chapter.get("start", 0)
            end = chapter.get("end", 0)
            
            # Extract text for this chapter's time range
            chapter_text = self._extract_chapter_text(words, start, end)
            
            if not chapter_text:
                chapter_text = chapter.get("gist", "") or chapter.get("headline", "")
            
            chapter_meta = self._metadata_builder.build_chunk_metadata(
                base_metadata=base_metadata,
                chunk_index=idx,
                chunk_start=start,
                chunk_end=end,
                chunk_type="chapter",
            )
            
            chapter_meta["chapter_headline"] = chapter.get("headline", "")
            chapter_meta["chapter_gist"] = chapter.get("gist", "")
            chapter_meta["chapter_summary"] = chapter.get("summary", "")
            
            doc = Document(
                content=chapter_text,
                meta=chapter_meta,
            )
            documents.append(doc)
        
        logger.info(
            "Built %d documents from chapters for: %s",
            len(documents),
            source_name,
        )
        
        return documents
    
    def _extract_chapter_text(
        self,
        words: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> str:
        """
        Extract text for a time range from word data.
        
        Args:
            words: Word-level transcript data
            start: Start time in milliseconds
            end: End time in milliseconds
            
        Returns:
            Extracted text
        """
        if not words:
            return ""
        
        chapter_words = [
            w.get("text", "")
            for w in words
            if start <= w.get("start", 0) < end
        ]
        
        return " ".join(chapter_words)
