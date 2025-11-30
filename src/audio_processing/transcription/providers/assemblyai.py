"""
AssemblyAI transcription provider implementation.

Implements the TranscriptionProvider interface for AssemblyAI service.
Uses the official AssemblyAI Python SDK for all operations.

Follows AGENTS.md design principles:
- Single Responsibility: Only handles AssemblyAI-specific transcription
- Encapsulation: SDK client is private, only exposes interface methods
- Loose Coupling: Depends on abstractions (TranscriptionConfig), not concrete classes
- Defensibility: Validates all inputs, fails fast with explicit errors
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import assemblyai as aai

from src.audio_processing.config import (
    FeatureConfig,
    ProviderConfig,
    TranscriptionConfig,
)
from src.audio_processing.exceptions import (
    ConfigurationError,
    TranscriptionError,
)
from src.audio_processing.transcription.providers.base import BaseTranscriptionProvider
from src.core.logging import get_logger

logger = get_logger(__name__)


class AssemblyAIProvider(BaseTranscriptionProvider):
    """
    AssemblyAI transcription provider.

    Provides speech-to-text transcription with advanced features:
    - Speaker diarization (speaker labels)
    - Sentiment analysis
    - Entity detection
    - Topic detection (IAB categories)
    - Auto chapters and highlights
    - Content safety detection
    - Summarization

    Example usage:
        provider_config = ProviderConfig(
            name="assemblyai",
            api_key_env="ASSEMBLYAI_API_KEY",
        )
        provider = AssemblyAIProvider(provider_config)

        transcription_config = TranscriptionConfig(
            language_code="en",
            speaker_labels=True,
        )
        provider.configure(transcription_config)

        result = provider.transcribe(Path("audio.mp3"))
    """

    def __init__(
        self,
        provider_config: ProviderConfig,
        transcription_config: Optional[TranscriptionConfig] = None,
    ) -> None:
        """
        Initialize the AssemblyAI provider.

        :param provider_config: Provider configuration with API key
        :param transcription_config: Optional transcription settings
        :raises ConfigurationError: If API key is not available
        """
        super().__init__(provider_config, transcription_config)

        # Configure the SDK with API key
        api_key = self._provider_config.get_api_key()
        if not api_key:
            raise ConfigurationError(
                "AssemblyAI API key not found. "
                f"Set environment variable: {self._provider_config.api_key_env}"
            )

        aai.settings.api_key = api_key
        self._transcriber: Optional[aai.Transcriber] = None
        self._config: Optional[aai.TranscriptionConfig] = None

        logger.info("AssemblyAI provider initialized")

    def _configure_provider(self, config: TranscriptionConfig) -> None:
        """
        Configure the AssemblyAI transcriber with settings.

        :param config: Transcription configuration
        """
        self._config = self._build_transcription_config(config)
        self._transcriber = aai.Transcriber(config=self._config)
        logger.debug("AssemblyAI transcriber configured with features")

    def _build_transcription_config(
        self, config: TranscriptionConfig
    ) -> aai.TranscriptionConfig:
        """
        Build AssemblyAI TranscriptionConfig from our config.

        :param config: Our transcription configuration
        :return: AssemblyAI TranscriptionConfig object
        """
        # Build configuration parameters
        config_params: Dict[str, Any] = {}

        # Basic settings
        if config.language_code:
            config_params["language_code"] = config.language_code

        config_params["punctuate"] = config.punctuate
        config_params["format_text"] = config.format_text
        config_params["speaker_labels"] = config.speaker_labels

        logger.debug(
            "Building AssemblyAI config with params: %s",
            {k: v for k, v in config_params.items() if not k.startswith("_")},
        )

        return aai.TranscriptionConfig(**config_params)

    def configure_features(self, features: FeatureConfig) -> None:
        """
        Configure advanced transcription features.

        This method allows enabling/disabling specific features
        after initial configuration.

        :param features: Feature configuration
        :raises ConfigurationError: If transcriber not initialized
        """
        if not self._transcription_config:
            raise ConfigurationError(
                "Provider must be configured before adding features"
            )

        # Build updated config with features
        config_params: Dict[str, Any] = {}

        # Basic settings from existing config
        config_params["language_code"] = self._transcription_config.language_code
        config_params["punctuate"] = self._transcription_config.punctuate
        config_params["format_text"] = self._transcription_config.format_text
        config_params["speaker_labels"] = self._transcription_config.speaker_labels

        # Feature flags
        config_params["sentiment_analysis"] = features.sentiment_analysis
        config_params["entity_detection"] = features.entity_detection
        config_params["iab_categories"] = features.iab_categories
        config_params["auto_chapters"] = features.auto_chapters
        config_params["auto_highlights"] = features.auto_highlights
        config_params["content_safety"] = features.content_safety

        # Summarization
        if features.summarization:
            config_params["summarization"] = True
            config_params["summary_model"] = aai.SummarizationModel.informative
            config_params["summary_type"] = aai.SummarizationType.bullets

        self._config = aai.TranscriptionConfig(**config_params)
        self._transcriber = aai.Transcriber(config=self._config)

        logger.info(
            "Features configured: sentiment=%s, entities=%s, chapters=%s",
            features.sentiment_analysis,
            features.entity_detection,
            features.auto_chapters,
        )

    def _do_transcribe(self, audio_path: Path) -> Dict[str, Any]:
        """
        Perform transcription using AssemblyAI.

        :param audio_path: Path to the audio file
        :return: Transcription result dictionary
        :raises TranscriptionError: If transcription fails
        """
        if not self._transcriber:
            # Create default transcriber if not configured
            self._transcriber = aai.Transcriber()
            logger.warning("Using default transcriber configuration")

        try:
            transcript = self._transcriber.transcribe(str(audio_path))
        except Exception as exc:
            logger.error("AssemblyAI transcription API error: %s", exc)
            raise TranscriptionError(f"AssemblyAI API error: {exc}") from exc

        if transcript.status == aai.TranscriptStatus.error:
            error_msg = transcript.error or "Unknown transcription error"
            logger.error("Transcription failed: %s", error_msg)
            raise TranscriptionError(f"Transcription failed: {error_msg}")

        return self._format_transcript_result(transcript)

    def _format_transcript_result(self, transcript: Any) -> Dict[str, Any]:
        """
        Format the AssemblyAI transcript into a standardized result.

        :param transcript: AssemblyAI Transcript object
        :return: Formatted result dictionary
        """
        result: Dict[str, Any] = {
            "id": transcript.id,
            "status": str(transcript.status),
            "text": transcript.text or "",
            "confidence": transcript.confidence or 0.0,
            "audio_duration": transcript.audio_duration,
            "words": self._format_words(transcript.words),
        }

        # Add utterances if speaker labels enabled
        if transcript.utterances:
            result["utterances"] = self._format_utterances(transcript.utterances)

        # Add chapters if available
        if transcript.chapters:
            result["chapters"] = self._format_chapters(transcript.chapters)

        # Add sentiment analysis results
        if transcript.sentiment_analysis:
            result["sentiment_analysis"] = self._format_sentiment(
                transcript.sentiment_analysis
            )

        # Add entities
        if transcript.entities:
            result["entities"] = self._format_entities(transcript.entities)

        # Add IAB categories
        if transcript.iab_categories:
            result["iab_categories"] = self._format_iab_categories(
                transcript.iab_categories
            )

        # Add content safety results
        if transcript.content_safety:
            result["content_safety"] = self._format_content_safety(
                transcript.content_safety
            )

        # Add auto highlights
        if transcript.auto_highlights:
            result["auto_highlights"] = self._format_highlights(
                transcript.auto_highlights
            )

        # Add summary
        if transcript.summary:
            result["summary"] = transcript.summary

        logger.debug(
            "Formatted transcript result with %d words, %d utterances",
            len(result.get("words", [])),
            len(result.get("utterances", [])),
        )

        return result

    def _format_words(
        self, words: Optional[List[Any]]
    ) -> List[Dict[str, Any]]:
        """Format word-level transcription data."""
        if not words:
            return []

        return [
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "confidence": word.confidence,
            }
            for word in words
        ]

    def _format_utterances(
        self, utterances: List[Any]
    ) -> List[Dict[str, Any]]:
        """Format speaker utterances."""
        return [
            {
                "speaker": utterance.speaker,
                "text": utterance.text,
                "start": utterance.start,
                "end": utterance.end,
                "confidence": utterance.confidence,
            }
            for utterance in utterances
        ]

    def _format_chapters(
        self, chapters: List[Any]
    ) -> List[Dict[str, Any]]:
        """Format auto-generated chapters."""
        return [
            {
                "headline": chapter.headline,
                "summary": chapter.summary,
                "gist": chapter.gist,
                "start": chapter.start,
                "end": chapter.end,
            }
            for chapter in chapters
        ]

    def _format_sentiment(
        self, sentiments: List[Any]
    ) -> List[Dict[str, Any]]:
        """Format sentiment analysis results."""
        return [
            {
                "text": s.text,
                "sentiment": str(s.sentiment),
                "confidence": s.confidence,
                "start": s.start,
                "end": s.end,
            }
            for s in sentiments
        ]

    def _format_entities(
        self, entities: List[Any]
    ) -> List[Dict[str, Any]]:
        """Format detected entities."""
        return [
            {
                "entity_type": str(entity.entity_type),
                "text": entity.text,
                "start": entity.start,
                "end": entity.end,
            }
            for entity in entities
        ]

    def _format_iab_categories(
        self, iab_result: Any
    ) -> Dict[str, Any]:
        """Format IAB category detection results."""
        return {
            "status": str(iab_result.status),
            "results": [
                {
                    "text": r.text,
                    "labels": [
                        {"relevance": label.relevance, "label": label.label}
                        for label in r.labels
                    ],
                }
                for r in (iab_result.results or [])
            ],
            "summary": {
                k: v for k, v in (iab_result.summary or {}).items()
            },
        }

    def _format_content_safety(
        self, safety_result: Any
    ) -> Dict[str, Any]:
        """Format content safety detection results."""
        return {
            "status": str(safety_result.status),
            "results": [
                {
                    "text": r.text,
                    "labels": [
                        {
                            "label": label.label,
                            "confidence": label.confidence,
                            "severity": label.severity,
                        }
                        for label in r.labels
                    ],
                }
                for r in (safety_result.results or [])
            ],
            "summary": {
                k: v for k, v in (safety_result.summary or {}).items()
            },
        }

    def _format_highlights(
        self, highlights_result: Any
    ) -> Dict[str, Any]:
        """Format auto-highlights results."""
        return {
            "status": str(highlights_result.status),
            "results": [
                {
                    "text": h.text,
                    "count": h.count,
                    "rank": h.rank,
                    "timestamps": [
                        {"start": ts.start, "end": ts.end}
                        for ts in h.timestamps
                    ],
                }
                for h in (highlights_result.results or [])
            ],
        }

    def _do_upload(self, audio_path: Path) -> str:
        """
        Upload an audio file to AssemblyAI.

        :param audio_path: Path to the audio file
        :return: URL of the uploaded file
        :raises TranscriptionError: If upload fails
        """
        try:
            upload_url = aai.Transcriber.upload_file(str(audio_path))
            logger.info("File uploaded to AssemblyAI: %s", audio_path.name)
            return upload_url
        except Exception as exc:
            logger.error("Failed to upload file: %s", exc)
            raise TranscriptionError(f"Upload failed: {exc}") from exc

    def _do_get_status(self, transcript_id: str) -> Dict[str, Any]:
        """
        Get the status of a transcription job.

        :param transcript_id: AssemblyAI transcript ID
        :return: Status information dictionary
        """
        try:
            transcript = aai.Transcript.get_by_id(transcript_id)
            return {
                "id": transcript.id,
                "status": str(transcript.status),
                "error": transcript.error,
            }
        except Exception as exc:
            logger.error("Failed to get transcript status: %s", exc)
            raise TranscriptionError(f"Status check failed: {exc}") from exc

    def transcribe_async(self, audio_path: Path) -> str:
        """
        Start an async transcription job.

        :param audio_path: Path to the audio file
        :return: Transcript ID for polling
        :raises TranscriptionError: If submission fails
        """
        self._validate_audio_path(audio_path)

        if not self._transcriber:
            self._transcriber = aai.Transcriber()

        try:
            transcript = self._transcriber.submit(str(audio_path))
            logger.info(
                "Async transcription submitted: %s (id=%s)",
                audio_path.name,
                transcript.id,
            )
            return transcript.id
        except Exception as exc:
            logger.error("Failed to submit async transcription: %s", exc)
            raise TranscriptionError(f"Async submission failed: {exc}") from exc

    def wait_for_completion(self, transcript_id: str) -> Dict[str, Any]:
        """
        Wait for an async transcription to complete.

        :param transcript_id: Transcript ID to wait for
        :return: Completed transcription result
        :raises TranscriptionError: If transcription fails
        """
        try:
            transcript = aai.Transcript.get_by_id(transcript_id)
            completed = transcript.wait_until_done()

            if completed.status == aai.TranscriptStatus.error:
                raise TranscriptionError(
                    f"Transcription failed: {completed.error}"
                )

            return self._format_transcript_result(completed)
        except TranscriptionError:
            raise
        except Exception as exc:
            logger.error("Error waiting for transcription: %s", exc)
            raise TranscriptionError(f"Wait failed: {exc}") from exc
