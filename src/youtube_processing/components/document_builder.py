"""
YouTube Document Builder Module.

This module provides a specialized document builder for YouTube content
that extends the base document builder from the audio processing module.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack.dataclasses import Document

from src.audio_processing.document import TranscriptDocumentBuilder
from src.core.logging import get_logger
from src.youtube_processing.interfaces import VideoMetadata

logger = get_logger(__name__)


class YouTubeDocumentBuilder:
    """
    Document builder specialized for YouTube content.

    This class creates Haystack Documents from YouTube video transcripts,
    enriching them with video-specific metadata.

    Example:
        builder = YouTubeDocumentBuilder()
        documents = builder.build(
            transcript_text=transcript,
            video_metadata=metadata,
            transcript_metadata=features,
        )
    """

    def __init__(
        self,
        include_utterances: bool = True,
        include_timestamps: bool = True,
        max_content_length: Optional[int] = None,
    ) -> None:
        """
        Initialize the document builder.

        Args:
            include_utterances: Whether to create separate documents for utterances.
            include_timestamps: Whether to include timestamp metadata.
            max_content_length: Optional maximum content length for chunking.
        """
        self._include_utterances = include_utterances
        self._include_timestamps = include_timestamps
        self._max_content_length = max_content_length
        self._base_builder = TranscriptDocumentBuilder()
        logger.debug("Initialized YouTubeDocumentBuilder")

    def build(
        self,
        transcript_text: str,
        video_metadata: VideoMetadata,
        transcript_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Build Haystack documents from a YouTube transcript.

        Args:
            transcript_text: The full transcript text.
            video_metadata: Video metadata from YouTube.
            transcript_metadata: Optional metadata from transcription (entities, etc.).

        Returns:
            List of Haystack Document objects.
        """
        logger.debug("Building documents for video: %s", video_metadata.video_id)

        documents: List[Document] = []

        # Build the main document
        main_meta = self._build_main_metadata(video_metadata, transcript_metadata)
        main_document = Document(content=transcript_text, meta=main_meta)
        documents.append(main_document)

        # Build utterance documents if available and enabled
        if self._include_utterances and transcript_metadata:
            utterances = transcript_metadata.get("utterances", [])
            utterance_docs = self._build_utterance_documents(
                utterances=utterances,
                video_metadata=video_metadata,
            )
            documents.extend(utterance_docs)

        logger.debug(
            "Built %d documents for video: %s",
            len(documents),
            video_metadata.video_id,
        )

        return documents

    def build_from_chunks(
        self,
        chunks: List[str],
        video_metadata: VideoMetadata,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """
        Build documents from pre-chunked transcript text.

        Args:
            chunks: List of transcript chunks.
            video_metadata: Video metadata from YouTube.
            chunk_metadata: Optional per-chunk metadata.

        Returns:
            List of Haystack Document objects.
        """
        documents: List[Document] = []
        base_meta = video_metadata.to_dict()
        base_meta["source"] = "youtube"
        base_meta["source_url"] = f"https://www.youtube.com/watch?v={video_metadata.video_id}"

        for idx, chunk in enumerate(chunks):
            chunk_meta = base_meta.copy()
            chunk_meta["chunk_index"] = idx
            chunk_meta["total_chunks"] = len(chunks)

            if chunk_metadata and idx < len(chunk_metadata):
                chunk_meta.update(chunk_metadata[idx])

            documents.append(Document(content=chunk, meta=chunk_meta))

        return documents

    def _build_main_metadata(
        self,
        video_metadata: VideoMetadata,
        transcript_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build metadata for the main document."""
        meta = video_metadata.to_dict()

        # Add source information
        meta["source"] = "youtube"
        meta["source_url"] = f"https://www.youtube.com/watch?v={video_metadata.video_id}"
        meta["document_type"] = "transcript"

        # Add transcript-derived metadata
        if transcript_metadata:
            # Add entities if available
            if "entities" in transcript_metadata:
                meta["entities"] = transcript_metadata["entities"]

            # Add sentiment if available
            if "sentiment" in transcript_metadata:
                meta["sentiment"] = transcript_metadata["sentiment"]

            # Add topics if available
            if "topics" in transcript_metadata:
                meta["topics"] = transcript_metadata["topics"]

            # Add summary if available
            if "summary" in transcript_metadata:
                meta["summary"] = transcript_metadata["summary"]

            # Add word count
            if "words" in transcript_metadata:
                meta["word_count"] = len(transcript_metadata["words"])

        return meta

    def _build_utterance_documents(
        self,
        utterances: List[Any],
        video_metadata: VideoMetadata,
    ) -> List[Document]:
        """Build documents for individual utterances."""
        documents: List[Document] = []

        for idx, utterance in enumerate(utterances):
            # Extract utterance properties
            text = getattr(utterance, "text", str(utterance))
            speaker = getattr(utterance, "speaker", None)
            start_time = getattr(utterance, "start", None)
            end_time = getattr(utterance, "end", None)

            meta = {
                "video_id": video_metadata.video_id,
                "title": video_metadata.title,
                "channel_name": video_metadata.channel_name,
                "source": "youtube",
                "source_url": f"https://www.youtube.com/watch?v={video_metadata.video_id}",
                "document_type": "utterance",
                "utterance_index": idx,
            }

            if speaker is not None:
                meta["speaker"] = speaker

            if self._include_timestamps:
                if start_time is not None:
                    meta["start_time_ms"] = start_time
                if end_time is not None:
                    meta["end_time_ms"] = end_time
                if start_time is not None and end_time is not None:
                    meta["duration_ms"] = end_time - start_time

            documents.append(Document(content=text, meta=meta))

        return documents
