"""
Configuration adapter for audio processing.

Provides adapters to bridge between new YAML-based configuration
and the existing AssemblyAI transcription configuration.
"""

import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from src.audio_processing.config import AudioProcessingConfig
from src.audio_processing.config import get_audio_config
from src.audio_processing.exceptions import TranscriptionAPIError
from src.audio_processing.interfaces import TranscriptionConfigAdapterInterface
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LegacyAudioProcessingConfig:
    """
    Legacy configuration dataclass for backward compatibility.
    
    This class mirrors the original AudioProcessingConfig from audio_transcriber.py
    to provide a migration path while the module is being refactored.
    """
    
    # Core transcription settings
    language_code: Optional[str] = "en"
    model: str = "best"
    
    # Speaker features
    speaker_labels: bool = True
    speakers_expected: Optional[int] = None
    
    # Content analysis
    sentiment_analysis: bool = True
    entity_detection: bool = True
    iab_categories: bool = True
    content_safety: bool = True
    content_safety_confidence: int = 80
    auto_highlights: bool = True
    
    # Audio enhancement
    noise_reduction: bool = True
    automatic_punctuation: bool = True
    format_text: bool = True
    filter_profanity: bool = False
    
    # Privacy and redaction
    redact_pii: bool = False
    redact_pii_policies: List[str] = field(default_factory=lambda: [
        "credit_card_number", "email_address", "person_name", "phone_number"
    ])
    redact_pii_audio: bool = False
    
    # Advanced features
    custom_spelling: Dict[str, List[str]] = field(default_factory=dict)
    custom_vocabulary: List[str] = field(default_factory=list)
    boost_param: str = "low"
    
    # Output formats
    include_utterances: bool = True
    include_sentences: bool = True
    include_paragraphs: bool = True
    auto_chapters: bool = True
    summarization: bool = True
    summary_model: str = "informative"
    summary_type: str = "bullets"


class ConfigurationAdapter(TranscriptionConfigAdapterInterface):
    """
    Adapter to convert between new and legacy configuration formats.
    
    This adapter enables gradual migration from the legacy configuration
    format to the new YAML-based configuration without breaking changes.
    """

    def __init__(
        self,
        config: Optional[AudioProcessingConfig] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the configuration adapter.
        
        :param config: New AudioProcessingConfig instance (loads from YAML if None)
        :param api_key: API key for transcription service
        """
        self._config = config or get_audio_config()
        self._api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        
        logger.debug("ConfigurationAdapter initialized")

    @property
    def config(self) -> AudioProcessingConfig:
        """Get the underlying configuration."""
        return self._config

    def get_api_key(self) -> str:
        """
        Get the API key for the transcription service.
        
        :return: API key string
        :raises TranscriptionAPIError: If API key is not available
        """
        if not self._api_key:
            raise TranscriptionAPIError(
                "AssemblyAI",
                "API key not provided. Set ASSEMBLYAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return self._api_key

    def to_provider_config(self) -> Any:
        """
        Convert configuration to AssemblyAI TranscriptionConfig format.
        
        Note: This method returns the configuration parameters as a dictionary
        that can be used with AssemblyAI's TranscriptionConfig constructor.
        
        :return: Dictionary of configuration parameters
        """
        config = self._config
        
        return {
            "language_code": config.transcription.language_code,
            "speaker_labels": config.speaker_analysis.speaker_labels,
            "speakers_expected": config.speaker_analysis.speakers_expected,
            "sentiment_analysis": config.content_analysis.sentiment_analysis,
            "entity_detection": config.content_analysis.entity_detection,
            "iab_categories": config.content_analysis.iab_categories,
            "content_safety": config.content_analysis.content_safety,
            "content_safety_confidence": config.content_analysis.content_safety_confidence,
            "auto_highlights": config.content_analysis.auto_highlights,
            "punctuate": config.audio_enhancement.automatic_punctuation,
            "format_text": config.audio_enhancement.format_text,
            "filter_profanity": config.audio_enhancement.filter_profanity,
            "auto_chapters": config.output.auto_chapters,
            "summarization": config.output.summarization,
        }

    def to_legacy_config(self) -> LegacyAudioProcessingConfig:
        """
        Convert new configuration to legacy format for backward compatibility.
        
        :return: LegacyAudioProcessingConfig instance
        """
        config = self._config
        
        return LegacyAudioProcessingConfig(
            language_code=config.transcription.language_code,
            model=config.transcription.model,
            speaker_labels=config.speaker_analysis.speaker_labels,
            speakers_expected=config.speaker_analysis.speakers_expected,
            sentiment_analysis=config.content_analysis.sentiment_analysis,
            entity_detection=config.content_analysis.entity_detection,
            iab_categories=config.content_analysis.iab_categories,
            content_safety=config.content_analysis.content_safety,
            content_safety_confidence=config.content_analysis.content_safety_confidence,
            auto_highlights=config.content_analysis.auto_highlights,
            noise_reduction=config.audio_enhancement.noise_reduction,
            automatic_punctuation=config.audio_enhancement.automatic_punctuation,
            format_text=config.audio_enhancement.format_text,
            filter_profanity=config.audio_enhancement.filter_profanity,
            redact_pii=config.privacy.redact_pii,
            redact_pii_policies=config.privacy.redact_pii_policies,
            redact_pii_audio=config.privacy.redact_pii_audio,
            custom_spelling=config.vocabulary.custom_spelling,
            custom_vocabulary=config.vocabulary.custom_vocabulary,
            boost_param=config.vocabulary.boost_param,
            include_utterances=config.output.include_utterances,
            include_sentences=config.output.include_sentences,
            include_paragraphs=config.output.include_paragraphs,
            auto_chapters=config.output.auto_chapters,
            summarization=config.output.summarization,
            summary_model=config.output.summary_model,
            summary_type=config.output.summary_type,
        )

    @classmethod
    def from_legacy_config(
        cls,
        legacy_config: LegacyAudioProcessingConfig,
        api_key: Optional[str] = None
    ) -> "ConfigurationAdapter":
        """
        Create adapter from legacy configuration for migration purposes.
        
        :param legacy_config: LegacyAudioProcessingConfig instance
        :param api_key: Optional API key
        :return: ConfigurationAdapter instance
        """
        from src.audio_processing.config import (
            AudioEnhancementConfig,
            ContentAnalysisConfig,
            OutputConfig,
            PrivacyConfig,
            SpeakerAnalysisConfig,
            TranscriptionConfig,
            VocabularyConfig,
        )
        
        new_config = AudioProcessingConfig(
            transcription=TranscriptionConfig(
                language_code=legacy_config.language_code or "en",
                model=legacy_config.model,
            ),
            speaker_analysis=SpeakerAnalysisConfig(
                speaker_labels=legacy_config.speaker_labels,
                speakers_expected=legacy_config.speakers_expected,
            ),
            content_analysis=ContentAnalysisConfig(
                sentiment_analysis=legacy_config.sentiment_analysis,
                entity_detection=legacy_config.entity_detection,
                iab_categories=legacy_config.iab_categories,
                content_safety=legacy_config.content_safety,
                content_safety_confidence=legacy_config.content_safety_confidence,
                auto_highlights=legacy_config.auto_highlights,
            ),
            audio_enhancement=AudioEnhancementConfig(
                noise_reduction=legacy_config.noise_reduction,
                automatic_punctuation=legacy_config.automatic_punctuation,
                format_text=legacy_config.format_text,
                filter_profanity=legacy_config.filter_profanity,
            ),
            privacy=PrivacyConfig(
                redact_pii=legacy_config.redact_pii,
                redact_pii_policies=legacy_config.redact_pii_policies,
                redact_pii_audio=legacy_config.redact_pii_audio,
            ),
            vocabulary=VocabularyConfig(
                custom_spelling=legacy_config.custom_spelling,
                custom_vocabulary=legacy_config.custom_vocabulary,
                boost_param=legacy_config.boost_param,
            ),
            output=OutputConfig(
                include_utterances=legacy_config.include_utterances,
                include_sentences=legacy_config.include_sentences,
                include_paragraphs=legacy_config.include_paragraphs,
                auto_chapters=legacy_config.auto_chapters,
                summarization=legacy_config.summarization,
                summary_model=legacy_config.summary_model,
                summary_type=legacy_config.summary_type,
            ),
        )
        
        return cls(config=new_config, api_key=api_key)


def create_legacy_config_from_yaml() -> LegacyAudioProcessingConfig:
    """
    Helper function to create legacy config from YAML for backward compatibility.
    
    :return: LegacyAudioProcessingConfig populated from YAML
    """
    adapter = ConfigurationAdapter()
    return adapter.to_legacy_config()


def get_transcription_config_params() -> Dict[str, Any]:
    """
    Helper function to get transcription configuration parameters.
    
    :return: Dictionary of configuration parameters for TranscriptionConfig
    """
    adapter = ConfigurationAdapter()
    return adapter.to_provider_config()
