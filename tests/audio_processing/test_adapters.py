"""
Unit tests for audio processing adapters.

Tests configuration adapters for proper conversion between formats
following AGENTS.md principles: deterministic, focused, comprehensive.
"""

import os
import pytest
from unittest.mock import patch

from src.audio_processing.adapters import (
    ConfigurationAdapter,
    LegacyAudioProcessingConfig,
    create_legacy_config_from_yaml,
    get_transcription_config_params,
)
from src.audio_processing.config import (
    AudioProcessingConfig,
    TranscriptionConfig,
    SpeakerAnalysisConfig,
    ContentAnalysisConfig,
)
from src.audio_processing.exceptions import TranscriptionAPIError
from src.audio_processing.interfaces import TranscriptionConfigAdapterInterface


class TestLegacyAudioProcessingConfig:
    """Test LegacyAudioProcessingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LegacyAudioProcessingConfig()
        
        assert config.language_code == "en"
        assert config.model == "best"
        assert config.speaker_labels is True
        assert config.speakers_expected is None
        assert config.sentiment_analysis is True
        assert config.entity_detection is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LegacyAudioProcessingConfig(
            language_code="es",
            model="nano",
            speaker_labels=False,
            speakers_expected=3
        )
        
        assert config.language_code == "es"
        assert config.model == "nano"
        assert config.speaker_labels is False
        assert config.speakers_expected == 3

    def test_all_fields_present(self):
        """Test that all expected fields are present."""
        config = LegacyAudioProcessingConfig()
        
        expected_fields = [
            "language_code", "model", "speaker_labels", "speakers_expected",
            "sentiment_analysis", "entity_detection", "iab_categories",
            "content_safety", "content_safety_confidence", "auto_highlights",
            "noise_reduction", "automatic_punctuation", "format_text",
            "filter_profanity", "redact_pii", "redact_pii_policies",
            "redact_pii_audio", "custom_spelling", "custom_vocabulary",
            "boost_param", "include_utterances", "include_sentences",
            "include_paragraphs", "auto_chapters", "summarization",
            "summary_model", "summary_type"
        ]
        
        for field in expected_fields:
            assert hasattr(config, field), f"Missing field: {field}"


class TestConfigurationAdapter:
    """Test ConfigurationAdapter class."""

    def test_implements_interface(self):
        """Test that adapter implements the interface."""
        adapter = ConfigurationAdapter()
        assert isinstance(adapter, TranscriptionConfigAdapterInterface)

    def test_initialization_with_default_config(self):
        """Test adapter initialization with default config."""
        adapter = ConfigurationAdapter()
        
        assert adapter.config is not None
        assert isinstance(adapter.config, AudioProcessingConfig)

    def test_initialization_with_custom_config(self):
        """Test adapter initialization with custom config."""
        custom_config = AudioProcessingConfig(
            transcription=TranscriptionConfig(language_code="fr")
        )
        
        adapter = ConfigurationAdapter(config=custom_config)
        
        assert adapter.config.transcription.language_code == "fr"

    @patch.dict(os.environ, {"ASSEMBLYAI_API_KEY": "test_api_key"})
    def test_get_api_key_from_environment(self):
        """Test getting API key from environment variable."""
        adapter = ConfigurationAdapter()
        
        assert adapter.get_api_key() == "test_api_key"

    def test_get_api_key_from_parameter(self):
        """Test getting API key from parameter."""
        adapter = ConfigurationAdapter(api_key="custom_key")
        
        assert adapter.get_api_key() == "custom_key"

    def test_get_api_key_raises_error_when_missing(self):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            # Also clear ASSEMBLYAI_API_KEY if present
            os.environ.pop("ASSEMBLYAI_API_KEY", None)
            adapter = ConfigurationAdapter(api_key=None)
            
            with pytest.raises(TranscriptionAPIError) as exc_info:
                adapter.get_api_key()
            
            assert "API key" in str(exc_info.value)

    def test_to_provider_config_returns_dict(self):
        """Test that to_provider_config returns a dictionary."""
        adapter = ConfigurationAdapter()
        provider_config = adapter.to_provider_config()
        
        assert isinstance(provider_config, dict)

    def test_to_provider_config_contains_expected_keys(self):
        """Test that provider config contains expected keys."""
        adapter = ConfigurationAdapter()
        provider_config = adapter.to_provider_config()
        
        expected_keys = [
            "language_code", "speaker_labels", "sentiment_analysis",
            "entity_detection", "iab_categories", "content_safety",
            "auto_highlights", "punctuate", "format_text", "filter_profanity",
            "auto_chapters", "summarization"
        ]
        
        for key in expected_keys:
            assert key in provider_config, f"Missing key: {key}"

    def test_to_legacy_config_returns_legacy_config(self):
        """Test that to_legacy_config returns LegacyAudioProcessingConfig."""
        adapter = ConfigurationAdapter()
        legacy_config = adapter.to_legacy_config()
        
        assert isinstance(legacy_config, LegacyAudioProcessingConfig)

    def test_to_legacy_config_preserves_values(self):
        """Test that to_legacy_config preserves configuration values."""
        custom_config = AudioProcessingConfig(
            transcription=TranscriptionConfig(language_code="de", model="nano"),
            speaker_analysis=SpeakerAnalysisConfig(speaker_labels=False),
        )
        
        adapter = ConfigurationAdapter(config=custom_config)
        legacy_config = adapter.to_legacy_config()
        
        assert legacy_config.language_code == "de"
        assert legacy_config.model == "nano"
        assert legacy_config.speaker_labels is False


class TestConfigurationAdapterFromLegacy:
    """Test ConfigurationAdapter.from_legacy_config method."""

    def test_from_legacy_config_creates_adapter(self):
        """Test creating adapter from legacy config."""
        legacy = LegacyAudioProcessingConfig(
            language_code="es",
            model="conformer-2",
            speaker_labels=True
        )
        
        adapter = ConfigurationAdapter.from_legacy_config(legacy)
        
        assert isinstance(adapter, ConfigurationAdapter)
        assert adapter.config.transcription.language_code == "es"
        assert adapter.config.transcription.model == "conformer-2"

    def test_from_legacy_config_with_api_key(self):
        """Test creating adapter from legacy config with API key."""
        legacy = LegacyAudioProcessingConfig()
        
        adapter = ConfigurationAdapter.from_legacy_config(legacy, api_key="my_key")
        
        assert adapter.get_api_key() == "my_key"

    def test_roundtrip_conversion(self):
        """Test that conversion is reversible."""
        original_config = AudioProcessingConfig(
            transcription=TranscriptionConfig(language_code="it"),
            content_analysis=ContentAnalysisConfig(sentiment_analysis=False),
        )
        
        adapter1 = ConfigurationAdapter(config=original_config)
        legacy = adapter1.to_legacy_config()
        
        adapter2 = ConfigurationAdapter.from_legacy_config(legacy)
        
        assert adapter2.config.transcription.language_code == "it"
        assert adapter2.config.content_analysis.sentiment_analysis is False


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_create_legacy_config_from_yaml(self):
        """Test create_legacy_config_from_yaml function."""
        legacy_config = create_legacy_config_from_yaml()
        
        assert isinstance(legacy_config, LegacyAudioProcessingConfig)

    def test_get_transcription_config_params(self):
        """Test get_transcription_config_params function."""
        params = get_transcription_config_params()
        
        assert isinstance(params, dict)
        assert "language_code" in params
        assert "speaker_labels" in params


class TestConfigurationValues:
    """Test specific configuration value mappings."""

    def test_content_analysis_mapping(self):
        """Test content analysis configuration mapping."""
        config = AudioProcessingConfig(
            content_analysis=ContentAnalysisConfig(
                sentiment_analysis=True,
                entity_detection=False,
                content_safety_confidence=90
            )
        )
        
        adapter = ConfigurationAdapter(config=config)
        provider_config = adapter.to_provider_config()
        
        assert provider_config["sentiment_analysis"] is True
        assert provider_config["entity_detection"] is False
        assert provider_config["content_safety_confidence"] == 90

    def test_audio_enhancement_mapping(self):
        """Test audio enhancement configuration mapping."""
        from src.audio_processing.config import AudioEnhancementConfig
        
        config = AudioProcessingConfig(
            audio_enhancement=AudioEnhancementConfig(
                automatic_punctuation=False,
                format_text=True,
                filter_profanity=True
            )
        )
        
        adapter = ConfigurationAdapter(config=config)
        provider_config = adapter.to_provider_config()
        
        assert provider_config["punctuate"] is False
        assert provider_config["format_text"] is True
        assert provider_config["filter_profanity"] is True

    def test_output_mapping(self):
        """Test output configuration mapping."""
        from src.audio_processing.config import OutputConfig
        
        config = AudioProcessingConfig(
            output=OutputConfig(
                auto_chapters=False,
                summarization=True
            )
        )
        
        adapter = ConfigurationAdapter(config=config)
        provider_config = adapter.to_provider_config()
        
        assert provider_config["auto_chapters"] is False
        assert provider_config["summarization"] is True
