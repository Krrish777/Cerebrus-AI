"""
Audio Processing Configuration Management

Loads and validates configuration for audio processing from YAML files
with environment variable overrides.
"""

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from src.audio_processing.exceptions import ConfigurationLoadError
from src.audio_processing.exceptions import TranscriptionConfigurationError
from src.core.logging import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


@dataclass
class TranscriptionConfig:
    """Configuration for core transcription settings."""

    language_code: str = "en"
    model: str = "best"
    polling_interval: float = 3.0

    def __post_init__(self) -> None:
        """Validate transcription configuration."""
        valid_models = {"best", "nano", "conformer-2"}
        if self.model not in valid_models:
            raise TranscriptionConfigurationError(
                "model",
                f"must be one of {valid_models}, got '{self.model}'"
            )
        if self.polling_interval <= 0:
            raise TranscriptionConfigurationError(
                "polling_interval",
                f"must be positive, got {self.polling_interval}"
            )


@dataclass
class SpeakerAnalysisConfig:
    """Configuration for speaker analysis settings."""

    speaker_labels: bool = True
    speakers_expected: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate speaker analysis configuration."""
        if self.speakers_expected is not None and self.speakers_expected < 1:
            raise TranscriptionConfigurationError(
                "speakers_expected",
                f"must be at least 1, got {self.speakers_expected}"
            )


@dataclass
class ContentAnalysisConfig:
    """Configuration for content analysis features."""

    sentiment_analysis: bool = True
    entity_detection: bool = True
    iab_categories: bool = True
    content_safety: bool = True
    content_safety_confidence: int = 80
    auto_highlights: bool = True

    def __post_init__(self) -> None:
        """Validate content analysis configuration."""
        if not 0 <= self.content_safety_confidence <= 100:
            raise TranscriptionConfigurationError(
                "content_safety_confidence",
                f"must be between 0 and 100, got {self.content_safety_confidence}"
            )


@dataclass
class AudioEnhancementConfig:
    """Configuration for audio enhancement settings."""

    noise_reduction: bool = True
    automatic_punctuation: bool = True
    format_text: bool = True
    filter_profanity: bool = False


@dataclass
class PrivacyConfig:
    """Configuration for privacy and redaction settings."""

    redact_pii: bool = False
    redact_pii_policies: List[str] = field(default_factory=lambda: [
        "credit_card_number",
        "email_address",
        "person_name",
        "phone_number"
    ])
    redact_pii_audio: bool = False


@dataclass
class VocabularyConfig:
    """Configuration for custom vocabulary settings."""

    custom_spelling: Dict[str, List[str]] = field(default_factory=dict)
    custom_vocabulary: List[str] = field(default_factory=list)
    boost_param: str = "low"

    def __post_init__(self) -> None:
        """Validate vocabulary configuration."""
        valid_boost_params = {"low", "default", "high"}
        if self.boost_param not in valid_boost_params:
            raise TranscriptionConfigurationError(
                "boost_param",
                f"must be one of {valid_boost_params}, got '{self.boost_param}'"
            )


@dataclass
class OutputConfig:
    """Configuration for output structure settings."""

    include_utterances: bool = True
    include_sentences: bool = True
    include_paragraphs: bool = True
    auto_chapters: bool = True
    summarization: bool = True
    summary_model: str = "informative"
    summary_type: str = "bullets"

    def __post_init__(self) -> None:
        """Validate output configuration."""
        valid_summary_models = {"informative", "conversational", "catchy"}
        valid_summary_types = {"bullets", "gist", "headline", "paragraph"}
        
        if self.summary_model not in valid_summary_models:
            raise TranscriptionConfigurationError(
                "summary_model",
                f"must be one of {valid_summary_models}, got '{self.summary_model}'"
            )
        if self.summary_type not in valid_summary_types:
            raise TranscriptionConfigurationError(
                "summary_type",
                f"must be one of {valid_summary_types}, got '{self.summary_type}'"
            )


@dataclass
class SmartChunkingConfig:
    """Configuration for smart audio chunking."""

    max_chunk_length: int = 1000
    overlap: int = 100
    respect_speakers: bool = True
    respect_chapters: bool = True

    def __post_init__(self) -> None:
        """Validate smart chunking configuration."""
        if self.max_chunk_length <= 0:
            raise TranscriptionConfigurationError(
                "max_chunk_length",
                f"must be positive, got {self.max_chunk_length}"
            )
        if self.overlap < 0:
            raise TranscriptionConfigurationError(
                "overlap",
                f"must be non-negative, got {self.overlap}"
            )
        if self.overlap >= self.max_chunk_length:
            raise TranscriptionConfigurationError(
                "overlap",
                f"must be less than max_chunk_length ({self.max_chunk_length}), "
                f"got {self.overlap}"
            )


@dataclass
class YouTubeConfig:
    """Configuration for YouTube processing settings."""

    cleanup_audio: bool = True
    cache_audio: bool = True
    audio_quality: str = "best"
    max_duration: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate YouTube configuration."""
        if self.max_duration is not None and self.max_duration <= 0:
            raise TranscriptionConfigurationError(
                "max_duration",
                f"must be positive, got {self.max_duration}"
            )


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""

    enable_timing: bool = True
    enable_statistics: bool = True
    enable_progress_tracking: bool = True


@dataclass
class ValidationConfig:
    """Configuration for input/output validation."""

    validate_inputs: bool = True
    validate_outputs: bool = True
    check_file_existence: bool = True


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling behavior."""

    fail_fast: bool = True
    continue_on_individual_source_error: bool = True


@dataclass
class AudioProcessingConfig:
    """Main configuration container for audio processing."""

    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    speaker_analysis: SpeakerAnalysisConfig = field(
        default_factory=SpeakerAnalysisConfig
    )
    content_analysis: ContentAnalysisConfig = field(
        default_factory=ContentAnalysisConfig
    )
    audio_enhancement: AudioEnhancementConfig = field(
        default_factory=AudioEnhancementConfig
    )
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    vocabulary: VocabularyConfig = field(default_factory=VocabularyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    smart_chunking: SmartChunkingConfig = field(default_factory=SmartChunkingConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)


class AudioConfigLoader:
    """Loads and validates audio processing configuration from YAML files."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration loader.

        :param config_path: Path to the YAML configuration file. 
                           If None, uses default path.
        """
        if config_path is None:
            config_path = Path("src/config/audio_processing.yaml")
        
        self._config_path = config_path
        self._config: Optional[AudioProcessingConfig] = None

    @property
    def config_path(self) -> Path:
        """Get the configuration file path."""
        return self._config_path

    def load_config(self) -> AudioProcessingConfig:
        """
        Load configuration from YAML file with environment variable overrides.

        :return: Validated AudioProcessingConfig instance
        :raises ConfigurationLoadError: if configuration file cannot be loaded
        """
        if self._config is not None:
            return self._config

        logger.info("Loading audio processing configuration from %s", self._config_path)

        try:
            yaml_data = load_config(str(self._config_path))
        except FileNotFoundError:
            logger.warning(
                "Config file not found at %s, using defaults", 
                self._config_path
            )
            self._config = AudioProcessingConfig()
            return self._config
        except Exception as e:
            logger.warning(
                "Failed to load config from %s (%s), using defaults", 
                self._config_path, 
                e
            )
            self._config = AudioProcessingConfig()
            return self._config

        try:
            yaml_data = self._apply_environment_overrides(yaml_data)
            config = self._create_config_from_dict(yaml_data)
            logger.info("Audio processing configuration loaded successfully")
            self._config = config
            return config
        except TranscriptionConfigurationError:
            raise
        except Exception as e:
            logger.error("Failed to create configuration from dictionary: %s", e)
            raise ConfigurationLoadError(
                str(self._config_path),
                f"Invalid configuration structure: {e}",
                e
            )

    def _apply_environment_overrides(
        self, 
        config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration dictionary.

        :param config_dict: Configuration dictionary from YAML
        :return: Configuration dictionary with environment overrides applied
        """
        overrides: Dict[str, Any] = {}

        env_mappings = {
            'AUDIO_LANGUAGE_CODE': ('transcription', 'language_code'),
            'AUDIO_MODEL': ('transcription', 'model'),
            'AUDIO_POLLING_INTERVAL': ('transcription', 'polling_interval'),
            'AUDIO_SPEAKER_LABELS': ('speaker_analysis', 'speaker_labels'),
            'AUDIO_SENTIMENT_ANALYSIS': ('content_analysis', 'sentiment_analysis'),
            'AUDIO_ENTITY_DETECTION': ('content_analysis', 'entity_detection'),
            'AUDIO_NOISE_REDUCTION': ('audio_enhancement', 'noise_reduction'),
            'AUDIO_REDACT_PII': ('privacy', 'redact_pii'),
            'AUDIO_MAX_CHUNK_LENGTH': ('smart_chunking', 'max_chunk_length'),
            'YOUTUBE_MAX_DURATION': ('youtube', 'max_duration'),
            'YOUTUBE_CACHE_AUDIO': ('youtube', 'cache_audio'),
            'AUDIO_FAIL_FAST': ('error_handling', 'fail_fast'),
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                logger.info(
                    "Applying environment override: %s = %s", 
                    env_var, 
                    env_value
                )
                self._set_nested_value(
                    overrides, 
                    config_path, 
                    self._parse_env_value(env_value)
                )

        return self._deep_merge(config_dict, overrides)

    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.

        :param value: String value from environment
        :return: Parsed value (int, float, bool, or str)
        """
        # Check for null
        if value.lower() == 'null' or value.lower() == 'none':
            return None

        # Try boolean first
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        return value

    def _set_nested_value(
        self, 
        data: Dict[str, Any], 
        path: tuple, 
        value: Any
    ) -> None:
        """
        Set a nested value in a dictionary using a path tuple.

        :param data: Dictionary to modify
        :param path: Tuple representing the nested path
        :param value: Value to set
        """
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _deep_merge(
        self, 
        base: Dict[str, Any], 
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge override dictionary into base dictionary.

        :param base: Base configuration dictionary
        :param override: Override configuration dictionary
        :return: Merged configuration dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _create_config_from_dict(
        self, 
        config_dict: Dict[str, Any]
    ) -> AudioProcessingConfig:
        """
        Create AudioProcessingConfig instance from configuration dictionary.

        :param config_dict: Configuration dictionary
        :return: Validated AudioProcessingConfig instance
        """
        transcription_dict = config_dict.get('transcription', {})
        speaker_dict = config_dict.get('speaker_analysis', {})
        content_dict = config_dict.get('content_analysis', {})
        enhancement_dict = config_dict.get('audio_enhancement', {})
        privacy_dict = config_dict.get('privacy', {})
        vocabulary_dict = config_dict.get('vocabulary', {})
        output_dict = config_dict.get('output', {})
        chunking_dict = config_dict.get('smart_chunking', {})
        youtube_dict = config_dict.get('youtube', {})
        performance_dict = config_dict.get('performance', {})
        validation_dict = config_dict.get('validation', {})
        error_dict = config_dict.get('error_handling', {})

        return AudioProcessingConfig(
            transcription=TranscriptionConfig(**transcription_dict),
            speaker_analysis=SpeakerAnalysisConfig(**speaker_dict),
            content_analysis=ContentAnalysisConfig(**content_dict),
            audio_enhancement=AudioEnhancementConfig(**enhancement_dict),
            privacy=PrivacyConfig(**privacy_dict),
            vocabulary=VocabularyConfig(**vocabulary_dict),
            output=OutputConfig(**output_dict),
            smart_chunking=SmartChunkingConfig(**chunking_dict),
            youtube=YouTubeConfig(**youtube_dict),
            performance=PerformanceConfig(**performance_dict),
            validation=ValidationConfig(**validation_dict),
            error_handling=ErrorHandlingConfig(**error_dict),
        )


# Global configuration loader instance
_config_loader: Optional[AudioConfigLoader] = None


def get_audio_config() -> AudioProcessingConfig:
    """
    Get the global audio processing configuration instance.

    :return: Loaded and validated AudioProcessingConfig
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = AudioConfigLoader()
    return _config_loader.load_config()


def reload_audio_config() -> AudioProcessingConfig:
    """
    Reload configuration from disk (useful for testing or dynamic config updates).

    :return: Freshly loaded AudioProcessingConfig
    """
    global _config_loader
    _config_loader = AudioConfigLoader()
    return _config_loader.load_config()
