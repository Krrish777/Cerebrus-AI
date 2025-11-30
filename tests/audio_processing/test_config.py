"""
Unit tests for audio processing configuration.

Tests configuration loading, validation, and dataclass functionality
following AGENTS.md principles: deterministic, focused, comprehensive.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.audio_processing.config import (
    AudioConfigLoader,
    AudioProcessingConfig,
    TranscriptionConfig,
    SpeakerAnalysisConfig,
    ContentAnalysisConfig,
    AudioEnhancementConfig,
    PrivacyConfig,
    VocabularyConfig,
    OutputConfig,
    SmartChunkingConfig,
    YouTubeConfig,
    PerformanceConfig,
    ValidationConfig,
    ErrorHandlingConfig,
    get_audio_config,
    reload_audio_config,
)
from src.audio_processing.exceptions import (
    TranscriptionConfigurationError,
    ConfigurationLoadError,
)


class TestTranscriptionConfig:
    """Test TranscriptionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TranscriptionConfig()
        
        assert config.language_code == "en"
        assert config.model == "best"
        assert config.polling_interval == 3.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TranscriptionConfig(
            language_code="es",
            model="nano",
            polling_interval=5.0
        )
        
        assert config.language_code == "es"
        assert config.model == "nano"
        assert config.polling_interval == 5.0

    @pytest.mark.parametrize("model", ["best", "nano", "conformer-2"])
    def test_valid_models(self, model: str):
        """Test valid model values."""
        config = TranscriptionConfig(model=model)
        assert config.model == model

    def test_invalid_model_raises_error(self):
        """Test that invalid model raises TranscriptionConfigurationError."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            TranscriptionConfig(model="invalid_model")
        
        assert "model" in str(exc_info.value)
        assert "invalid_model" in str(exc_info.value)

    def test_negative_polling_interval_raises_error(self):
        """Test that negative polling interval raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            TranscriptionConfig(polling_interval=-1.0)
        
        assert "polling_interval" in str(exc_info.value)

    def test_zero_polling_interval_raises_error(self):
        """Test that zero polling interval raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            TranscriptionConfig(polling_interval=0)
        
        assert "polling_interval" in str(exc_info.value)


class TestSpeakerAnalysisConfig:
    """Test SpeakerAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SpeakerAnalysisConfig()
        
        assert config.speaker_labels is True
        assert config.speakers_expected is None

    def test_custom_speakers_expected(self):
        """Test custom speakers_expected value."""
        config = SpeakerAnalysisConfig(speakers_expected=3)
        assert config.speakers_expected == 3

    def test_invalid_speakers_expected_raises_error(self):
        """Test that invalid speakers_expected raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            SpeakerAnalysisConfig(speakers_expected=0)
        
        assert "speakers_expected" in str(exc_info.value)


class TestContentAnalysisConfig:
    """Test ContentAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ContentAnalysisConfig()
        
        assert config.sentiment_analysis is True
        assert config.entity_detection is True
        assert config.iab_categories is True
        assert config.content_safety is True
        assert config.content_safety_confidence == 80
        assert config.auto_highlights is True

    @pytest.mark.parametrize("confidence", [0, 50, 100])
    def test_valid_confidence_values(self, confidence: int):
        """Test valid content_safety_confidence values."""
        config = ContentAnalysisConfig(content_safety_confidence=confidence)
        assert config.content_safety_confidence == confidence

    def test_invalid_confidence_raises_error_negative(self):
        """Test that negative confidence raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            ContentAnalysisConfig(content_safety_confidence=-1)
        
        assert "content_safety_confidence" in str(exc_info.value)

    def test_invalid_confidence_raises_error_over_100(self):
        """Test that confidence over 100 raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            ContentAnalysisConfig(content_safety_confidence=101)
        
        assert "content_safety_confidence" in str(exc_info.value)


class TestVocabularyConfig:
    """Test VocabularyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VocabularyConfig()
        
        assert config.custom_spelling == {}
        assert config.custom_vocabulary == []
        assert config.boost_param == "low"

    @pytest.mark.parametrize("boost_param", ["low", "default", "high"])
    def test_valid_boost_params(self, boost_param: str):
        """Test valid boost_param values."""
        config = VocabularyConfig(boost_param=boost_param)
        assert config.boost_param == boost_param

    def test_invalid_boost_param_raises_error(self):
        """Test that invalid boost_param raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            VocabularyConfig(boost_param="invalid")
        
        assert "boost_param" in str(exc_info.value)


class TestOutputConfig:
    """Test OutputConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OutputConfig()
        
        assert config.include_utterances is True
        assert config.include_sentences is True
        assert config.include_paragraphs is True
        assert config.auto_chapters is True
        assert config.summarization is True
        assert config.summary_model == "informative"
        assert config.summary_type == "bullets"

    @pytest.mark.parametrize("summary_model", ["informative", "conversational", "catchy"])
    def test_valid_summary_models(self, summary_model: str):
        """Test valid summary_model values."""
        config = OutputConfig(summary_model=summary_model)
        assert config.summary_model == summary_model

    @pytest.mark.parametrize("summary_type", ["bullets", "gist", "headline", "paragraph"])
    def test_valid_summary_types(self, summary_type: str):
        """Test valid summary_type values."""
        config = OutputConfig(summary_type=summary_type)
        assert config.summary_type == summary_type

    def test_invalid_summary_model_raises_error(self):
        """Test that invalid summary_model raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            OutputConfig(summary_model="invalid")
        
        assert "summary_model" in str(exc_info.value)

    def test_invalid_summary_type_raises_error(self):
        """Test that invalid summary_type raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            OutputConfig(summary_type="invalid")
        
        assert "summary_type" in str(exc_info.value)


class TestSmartChunkingConfig:
    """Test SmartChunkingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SmartChunkingConfig()
        
        assert config.max_chunk_length == 1000
        assert config.overlap == 100
        assert config.respect_speakers is True
        assert config.respect_chapters is True

    def test_invalid_max_chunk_length_raises_error(self):
        """Test that invalid max_chunk_length raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            SmartChunkingConfig(max_chunk_length=0)
        
        assert "max_chunk_length" in str(exc_info.value)

    def test_negative_overlap_raises_error(self):
        """Test that negative overlap raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            SmartChunkingConfig(overlap=-1)
        
        assert "overlap" in str(exc_info.value)

    def test_overlap_greater_than_chunk_length_raises_error(self):
        """Test that overlap >= max_chunk_length raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            SmartChunkingConfig(max_chunk_length=100, overlap=100)
        
        assert "overlap" in str(exc_info.value)


class TestYouTubeConfig:
    """Test YouTubeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = YouTubeConfig()
        
        assert config.cleanup_audio is True
        assert config.cache_audio is True
        assert config.audio_quality == "best"
        assert config.max_duration is None

    def test_custom_max_duration(self):
        """Test custom max_duration value."""
        config = YouTubeConfig(max_duration=3600)
        assert config.max_duration == 3600

    def test_invalid_max_duration_raises_error(self):
        """Test that invalid max_duration raises error."""
        with pytest.raises(TranscriptionConfigurationError) as exc_info:
            YouTubeConfig(max_duration=0)
        
        assert "max_duration" in str(exc_info.value)


class TestAudioProcessingConfig:
    """Test AudioProcessingConfig main configuration container."""

    def test_default_initialization(self):
        """Test default initialization of main config."""
        config = AudioProcessingConfig()
        
        assert isinstance(config.transcription, TranscriptionConfig)
        assert isinstance(config.speaker_analysis, SpeakerAnalysisConfig)
        assert isinstance(config.content_analysis, ContentAnalysisConfig)
        assert isinstance(config.audio_enhancement, AudioEnhancementConfig)
        assert isinstance(config.privacy, PrivacyConfig)
        assert isinstance(config.vocabulary, VocabularyConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.smart_chunking, SmartChunkingConfig)
        assert isinstance(config.youtube, YouTubeConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.error_handling, ErrorHandlingConfig)


class TestAudioConfigLoader:
    """Test AudioConfigLoader class."""

    def test_loader_initialization(self):
        """Test loader initialization with default path."""
        loader = AudioConfigLoader()
        assert loader.config_path == Path("src/config/audio_processing.yaml")

    def test_loader_initialization_custom_path(self):
        """Test loader initialization with custom path."""
        custom_path = Path("/custom/path/config.yaml")
        loader = AudioConfigLoader(config_path=custom_path)
        assert loader.config_path == custom_path

    def test_load_config_returns_audio_processing_config(self):
        """Test that load_config returns AudioProcessingConfig."""
        loader = AudioConfigLoader()
        config = loader.load_config()
        
        assert isinstance(config, AudioProcessingConfig)

    def test_load_config_caches_result(self):
        """Test that load_config caches the result."""
        loader = AudioConfigLoader()
        config1 = loader.load_config()
        config2 = loader.load_config()
        
        assert config1 is config2

    @patch.dict(os.environ, {"AUDIO_LANGUAGE_CODE": "fr"})
    def test_environment_override_string(self):
        """Test environment variable override for string value."""
        loader = AudioConfigLoader()
        loader._config = None  # Reset cache
        config = loader.load_config()
        
        assert config.transcription.language_code == "fr"

    @patch.dict(os.environ, {"AUDIO_SPEAKER_LABELS": "false"})
    def test_environment_override_boolean(self):
        """Test environment variable override for boolean value."""
        loader = AudioConfigLoader()
        loader._config = None  # Reset cache
        config = loader.load_config()
        
        assert config.speaker_analysis.speaker_labels is False

    @patch.dict(os.environ, {"AUDIO_MAX_CHUNK_LENGTH": "2000"})
    def test_environment_override_integer(self):
        """Test environment variable override for integer value."""
        loader = AudioConfigLoader()
        loader._config = None  # Reset cache
        config = loader.load_config()
        
        assert config.smart_chunking.max_chunk_length == 2000


class TestConfigLoaderHelperMethods:
    """Test AudioConfigLoader helper methods."""

    def test_parse_env_value_boolean_true(self):
        """Test parsing boolean true from env."""
        loader = AudioConfigLoader()
        assert loader._parse_env_value("true") is True
        assert loader._parse_env_value("True") is True
        assert loader._parse_env_value("TRUE") is True

    def test_parse_env_value_boolean_false(self):
        """Test parsing boolean false from env."""
        loader = AudioConfigLoader()
        assert loader._parse_env_value("false") is False
        assert loader._parse_env_value("False") is False

    def test_parse_env_value_null(self):
        """Test parsing null from env."""
        loader = AudioConfigLoader()
        assert loader._parse_env_value("null") is None
        assert loader._parse_env_value("none") is None
        assert loader._parse_env_value("None") is None

    def test_parse_env_value_integer(self):
        """Test parsing integer from env."""
        loader = AudioConfigLoader()
        assert loader._parse_env_value("42") == 42
        assert loader._parse_env_value("-10") == -10

    def test_parse_env_value_float(self):
        """Test parsing float from env."""
        loader = AudioConfigLoader()
        assert loader._parse_env_value("3.14") == 3.14
        assert loader._parse_env_value("-2.5") == -2.5

    def test_parse_env_value_string(self):
        """Test parsing string from env."""
        loader = AudioConfigLoader()
        assert loader._parse_env_value("hello") == "hello"

    def test_deep_merge(self):
        """Test deep merge functionality."""
        loader = AudioConfigLoader()
        
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 5, "e": 6}}
        
        result = loader._deep_merge(base, override)
        
        assert result == {"a": 1, "b": {"c": 5, "d": 3, "e": 6}}

    def test_set_nested_value(self):
        """Test setting nested values."""
        loader = AudioConfigLoader()
        data: dict = {}
        
        loader._set_nested_value(data, ("a", "b", "c"), "value")
        
        assert data == {"a": {"b": {"c": "value"}}}


class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def test_get_audio_config_returns_config(self):
        """Test get_audio_config returns configuration."""
        config = get_audio_config()
        assert isinstance(config, AudioProcessingConfig)

    def test_reload_audio_config_returns_fresh_config(self):
        """Test reload_audio_config returns fresh configuration."""
        config1 = get_audio_config()
        config2 = reload_audio_config()
        
        # Both should be valid configs
        assert isinstance(config1, AudioProcessingConfig)
        assert isinstance(config2, AudioProcessingConfig)


class TestConfigWithYamlFile:
    """Test configuration loading from actual YAML file."""

    def test_load_from_existing_yaml(self):
        """Test loading configuration from existing YAML file."""
        config_path = Path("src/config/audio_processing.yaml")
        
        if config_path.exists():
            loader = AudioConfigLoader(config_path=config_path)
            config = loader.load_config()
            
            assert isinstance(config, AudioProcessingConfig)
            # Verify some expected values from the YAML
            assert config.transcription.language_code == "en"
            assert config.transcription.model == "best"

    def test_load_from_nonexistent_file_uses_defaults(self):
        """Test loading from nonexistent file falls back to defaults."""
        loader = AudioConfigLoader(config_path=Path("/nonexistent/path.yaml"))
        config = loader.load_config()
        
        # Should use defaults
        assert isinstance(config, AudioProcessingConfig)
        assert config.transcription.language_code == "en"
