"""
YouTube Audio Processor Module.

This module provides the main orchestrator for processing YouTube videos.
It coordinates downloading, transcription, metadata enhancement, and document creation.
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack.dataclasses import Document

from src.audio_processing import AudioTranscriber
from src.core.logging import get_logger
from src.youtube_processing.config import YouTubeConfig
from src.youtube_processing.exceptions import YouTubeProcessingError
from src.youtube_processing.interfaces import CacheManager
from src.youtube_processing.interfaces import DownloadResult
from src.youtube_processing.interfaces import MetadataEnhancer
from src.youtube_processing.interfaces import URLValidator
from src.youtube_processing.interfaces import VideoDownloader
from src.youtube_processing.interfaces import VideoMetadata
from src.youtube_processing.interfaces import YouTubeProcessor

logger = get_logger(__name__)


class YouTubeAudioProcessor(YouTubeProcessor):
    """
    Main orchestrator for YouTube audio processing.

    This class coordinates the complete workflow for processing YouTube videos:
    1. URL validation
    2. Audio download (with caching)
    3. Transcription (delegated to AudioTranscriber)
    4. Metadata enhancement
    5. Document creation

    All dependencies are injected, making the class highly testable and
    allowing for easy swapping of implementations.

    Example:
        processor = YouTubeAudioProcessor.create(config)
        documents = processor.process(url)

        # Or with custom dependencies:
        processor = YouTubeAudioProcessor(
            downloader=my_downloader,
            audio_transcriber=my_transcriber,
            cache_manager=my_cache,
            metadata_enhancer=my_enhancer,
            validator=my_validator,
            config=my_config,
        )
    """

    def __init__(
        self,
        downloader: VideoDownloader,
        audio_transcriber: AudioTranscriber,
        cache_manager: Optional[CacheManager],
        metadata_enhancer: MetadataEnhancer,
        validator: URLValidator,
        config: YouTubeConfig,
    ) -> None:
        """
        Initialize the YouTube audio processor.

        Args:
            downloader: Video downloader implementation.
            audio_transcriber: Audio transcriber from audio_processing module.
            cache_manager: Optional cache manager for downloaded files.
            metadata_enhancer: Metadata enhancement implementation.
            validator: URL validator implementation.
            config: YouTube processing configuration.
        """
        self._downloader = downloader
        self._audio_transcriber = audio_transcriber
        self._cache_manager = cache_manager
        self._metadata_enhancer = metadata_enhancer
        self._validator = validator
        self._config = config

        logger.info("Initialized YouTubeAudioProcessor")

    @classmethod
    def create(
        cls,
        config: YouTubeConfig,
        audio_transcriber: Optional[AudioTranscriber] = None,
    ) -> "YouTubeAudioProcessor":
        """
        Factory method to create a YouTubeAudioProcessor with default dependencies.

        Args:
            config: YouTube processing configuration.
            audio_transcriber: Optional pre-configured audio transcriber.
                If not provided, will be created from config.audio_config_path.

        Returns:
            Configured YouTubeAudioProcessor instance.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        from src.audio_processing import AudioProcessingConfig
        from src.audio_processing import AudioTranscriber as AT
        from src.youtube_processing.cache import FileCacheManager
        from src.youtube_processing.download import YtDlpDownloader
        from src.youtube_processing.download import YouTubeURLValidator
        from src.youtube_processing.metadata import DefaultMetadataEnhancer

        # Create validator
        validator = YouTubeURLValidator(config.validation)

        # Create downloader
        downloader = YtDlpDownloader(
            config=config.download,
            retry_config=config.retry,
            validator=validator,
        )

        # Create cache manager if enabled
        cache_manager = None
        if config.cache.enabled:
            cache_manager = FileCacheManager(config.cache)

        # Create metadata enhancer
        metadata_enhancer = DefaultMetadataEnhancer(config.metadata)

        # Create or use provided audio transcriber
        if audio_transcriber is None:
            if config.audio_config_path and config.audio_config_path.exists():
                audio_config = AudioProcessingConfig.from_yaml(config.audio_config_path)
                audio_transcriber = AT.from_config(audio_config)
            else:
                raise YouTubeProcessingError(
                    message="Audio transcriber not provided and audio_config_path not found",
                )

        return cls(
            downloader=downloader,
            audio_transcriber=audio_transcriber,
            cache_manager=cache_manager,
            metadata_enhancer=metadata_enhancer,
            validator=validator,
            config=config,
        )

    def process(self, url: str) -> List[Document]:
        """
        Process a YouTube video and return documents.

        This is the main entry point for processing a video. It:
        1. Validates the URL
        2. Downloads the audio (or retrieves from cache)
        3. Transcribes the audio
        4. Enhances metadata
        5. Creates Haystack Documents

        Args:
            url: YouTube video URL.

        Returns:
            List of Haystack Document objects.

        Raises:
            VideoNotFoundError: If the video does not exist.
            ValidationError: If the URL is invalid.
            DownloadError: If download fails.
            YouTubeProcessingError: For other processing errors.
        """
        logger.info("Processing YouTube video: %s", url)

        # Step 1: Validate URL (basic format check)
        video_id = self._validator.extract_video_id(url)
        if not video_id:
            from src.youtube_processing.exceptions import ValidationError
            raise ValidationError(
                message="Invalid YouTube URL format",
                video_url=url,
                field_name="url",
            )

        # Step 2: Check cache first
        download_result = self._get_or_download(url, video_id)

        # Step 3: Validate metadata constraints
        is_valid, errors = self._validator.validate(url, download_result.metadata)
        if not is_valid:
            from src.youtube_processing.exceptions import ValidationError
            raise ValidationError(
                message=f"Video validation failed: {'; '.join(errors)}",
                video_url=url,
            )

        # Step 4: Transcribe audio
        transcript_result = self._transcribe_audio(download_result.audio_path)

        # Step 5: Enhance metadata
        enhanced_metadata = self._metadata_enhancer.enhance(
            download_result.metadata,
            transcript_text=transcript_result.get("text", ""),
        )

        # Step 6: Create documents
        documents = self._create_documents(
            video_metadata=enhanced_metadata,
            transcript_result=transcript_result,
        )

        # Step 7: Cleanup if configured
        if self._config.cache.cleanup_after_processing:
            self._cleanup_temp_files(download_result.audio_path)

        logger.info(
            "Completed processing video %s: created %d documents",
            video_id,
            len(documents),
        )

        return documents

    def process_batch(self, urls: List[str]) -> Dict[str, List[Document]]:
        """
        Process multiple YouTube videos.

        Args:
            urls: List of YouTube video URLs.

        Returns:
            Dictionary mapping URLs to their document lists.
            Failed URLs map to empty lists.
        """
        results: Dict[str, List[Document]] = {}

        for url in urls:
            try:
                documents = self.process(url)
                results[url] = documents
            except YouTubeProcessingError as e:
                logger.error("Failed to process video %s: %s", url, e)
                results[url] = []
            except Exception as e:
                logger.exception("Unexpected error processing video %s: %s", url, e)
                results[url] = []

        successful = sum(1 for docs in results.values() if docs)
        logger.info(
            "Batch processing complete: %d/%d videos processed successfully",
            successful,
            len(urls),
        )

        return results

    def get_metadata(self, url: str) -> VideoMetadata:
        """
        Get metadata for a YouTube video without processing.

        Args:
            url: YouTube video URL.

        Returns:
            VideoMetadata for the video.
        """
        return self._downloader.extract_metadata(url)

    def _get_or_download(self, url: str, video_id: str) -> DownloadResult:
        """Get audio from cache or download it."""
        # Check cache first
        if self._cache_manager and self._cache_manager.has(video_id):
            cached_path = self._cache_manager.get(video_id)
            if cached_path:
                logger.debug("Using cached audio for video: %s", video_id)
                metadata = self._downloader.extract_metadata(url)
                return DownloadResult(
                    audio_path=cached_path,
                    metadata=metadata,
                    from_cache=True,
                )

        # Download audio
        download_result = self._downloader.download(url, self._config.download.temp_dir)

        # Cache the downloaded file
        if self._cache_manager:
            cached_path = self._cache_manager.put(video_id, download_result.audio_path)
            download_result = DownloadResult(
                audio_path=cached_path,
                metadata=download_result.metadata,
                file_size_bytes=download_result.file_size_bytes,
                download_duration_seconds=download_result.download_duration_seconds,
                from_cache=False,
            )

        return download_result

    def _transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe audio using the audio processing module."""
        logger.debug("Transcribing audio: %s", audio_path)

        # Use the audio transcriber to process the file
        transcript = self._audio_transcriber.transcribe_with_features(audio_path)

        # Build result dictionary from transcript
        result: Dict[str, Any] = {
            "text": transcript.text if hasattr(transcript, "text") else str(transcript),
        }

        # Extract additional features if available
        if hasattr(transcript, "words"):
            result["words"] = transcript.words
        if hasattr(transcript, "utterances"):
            result["utterances"] = transcript.utterances
        if hasattr(transcript, "entities"):
            result["entities"] = transcript.entities
        if hasattr(transcript, "sentiment_analysis_results"):
            result["sentiment"] = transcript.sentiment_analysis_results
        if hasattr(transcript, "summary"):
            result["summary"] = transcript.summary
        if hasattr(transcript, "iab_categories_result"):
            result["topics"] = transcript.iab_categories_result

        return result

    def _create_documents(
        self,
        video_metadata: VideoMetadata,
        transcript_result: Dict[str, Any],
    ) -> List[Document]:
        """Create Haystack documents from the processing results."""
        # Merge video and transcript metadata
        merged_metadata = self._metadata_enhancer.merge_with_transcript_metadata(
            video_metadata,
            {k: v for k, v in transcript_result.items() if k != "text"},
        )

        # Create the main document
        main_document = Document(
            content=transcript_result.get("text", ""),
            meta=merged_metadata,
        )

        documents = [main_document]

        # Create additional documents for utterances if available
        if "utterances" in transcript_result and transcript_result["utterances"]:
            for idx, utterance in enumerate(transcript_result["utterances"]):
                utterance_meta = {
                    "video_id": video_metadata.video_id,
                    "title": video_metadata.title,
                    "source": "youtube",
                    "utterance_index": idx,
                    "speaker": getattr(utterance, "speaker", None),
                    "start_time": getattr(utterance, "start", None),
                    "end_time": getattr(utterance, "end", None),
                }
                utterance_text = getattr(utterance, "text", str(utterance))
                documents.append(Document(content=utterance_text, meta=utterance_meta))

        return documents

    def _cleanup_temp_files(self, audio_path: Path) -> None:
        """Clean up temporary files if not caching."""
        try:
            if audio_path.exists() and not self._cache_manager:
                audio_path.unlink()
                logger.debug("Cleaned up temp file: %s", audio_path)
        except OSError as e:
            logger.warning("Failed to cleanup temp file: %s", e)
