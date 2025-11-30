"""Configuration management for audio processing module."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from src.audio_processing.exceptions import ConfigurationError
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionConfig:
    """Core transcription settings."""
    
    language_code: str = "en"
    model: str = "best"
    polling_interval: float = 3.0
    punctuate: bool = True
    format_text: bool = True
    speaker_labels: bool = True


@dataclass
class FeatureConfig:
    """Feature enablement configuration."""
    
    speaker_labels: bool = True
    speakers_expected: Optional[int] = None
    sentiment_analysis: bool = True
    entity_detection: bool = True
    iab_categories: bool = True
    content_safety: bool = True
    content_safety_confidence: int = 80
    auto_highlights: bool = True
    auto_chapters: bool = True
    summarization: bool = True


@dataclass
class EnhancementConfig:
    """Audio enhancement settings."""
    
    noise_reduction: bool = True
    automatic_punctuation: bool = True
    format_text: bool = True
    filter_profanity: bool = False


@dataclass
class PrivacyConfig:
    """Privacy and redaction settings."""
    
    redact_pii: bool = False
    redact_pii_policies: List[str] = field(default_factory=lambda: [
        "credit_card_number",
        "email_address",
        "person_name",
        "phone_number"
    ])
    redact_pii_audio: bool = False


@dataclass
class OutputConfig:
    """Output format settings."""
    
    include_utterances: bool = True
    include_sentences: bool = True
    include_paragraphs: bool = True
    auto_chapters: bool = True


@dataclass
class SummarizationConfig:
    """Summarization settings."""
    
    enabled: bool = True
    summary_model: str = "informative"
    summary_type: str = "bullets"


@dataclass
class ChunkingConfig:
    """Chunking strategy configuration."""
    
    max_chunk_length: int = 1000
    overlap: int = 100
    respect_speakers: bool = True
    respect_chapters: bool = True
    respect_sentences: bool = True


@dataclass
class VocabularyConfig:
    """Custom vocabulary settings."""
    
    custom_spelling: Dict[str, List[str]] = field(default_factory=dict)
    custom_vocabulary: List[str] = field(default_factory=list)
    boost_param: str = "low"


@dataclass
class ProviderConfig:
    """Configuration for a transcription provider."""
    
    name: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5
    supports: Dict[str, bool] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)
    supported_languages: List[str] = field(default_factory=list)
    
    def get_api_key(self) -> Optional[str]:
        """
        Get API key from environment variable.
        
        :return: API key or None if not set
        """
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


@dataclass
class AudioProcessingConfig:
    """Main configuration container for audio processing."""
    
    transcription: TranscriptionConfig
    features: FeatureConfig
    enhancement: EnhancementConfig
    privacy: PrivacyConfig
    output: OutputConfig
    summarization: SummarizationConfig
    chunking: ChunkingConfig
    vocabulary: VocabularyConfig
    providers: List[ProviderConfig] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "AudioProcessingConfig":
        """
        Load configuration from YAML file.
        
        :param config_path: Path to configuration YAML file
        :return: AudioProcessingConfig instance
        :raises ConfigurationError: If config file is invalid or missing
        """
        if not config_path.exists():
            logger.error("Configuration file not found: %s", config_path)
            raise ConfigurationError(
                f"Configuration file not found: {config_path}"
            )
        
        try:
            logger.debug("Loading audio configuration from %s", config_path)
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if not data:
                raise ConfigurationError(
                    f"Configuration file is empty: {config_path}"
                )
            
            # Build configuration objects
            transcription = TranscriptionConfig(**data.get("transcription", {}))
            features = FeatureConfig(**data.get("features", {}))
            enhancement = EnhancementConfig(**data.get("enhancement", {}))
            privacy = PrivacyConfig(**data.get("privacy", {}))
            output = OutputConfig(**data.get("output", {}))
            summarization = SummarizationConfig(**data.get("summarization", {}))
            chunking = ChunkingConfig(**data.get("chunking", {}))
            vocabulary = VocabularyConfig(**data.get("vocabulary", {}))
            
            logger.info("Audio configuration loaded successfully")
            
            return cls(
                transcription=transcription,
                features=features,
                enhancement=enhancement,
                privacy=privacy,
                output=output,
                summarization=summarization,
                chunking=chunking,
                vocabulary=vocabulary
            )
            
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML configuration: %s", e)
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}"
            )
        except TypeError as e:
            logger.error("Invalid configuration structure: %s", e)
            raise ConfigurationError(
                f"Invalid configuration structure: {e}"
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AudioProcessingConfig":
        """
        Create configuration from dictionary.
        
        :param config_dict: Configuration dictionary
        :return: AudioProcessingConfig instance
        """
        transcription = TranscriptionConfig(**config_dict.get("transcription", {}))
        features = FeatureConfig(**config_dict.get("features", {}))
        enhancement = EnhancementConfig(**config_dict.get("enhancement", {}))
        privacy = PrivacyConfig(**config_dict.get("privacy", {}))
        output = OutputConfig(**config_dict.get("output", {}))
        summarization = SummarizationConfig(**config_dict.get("summarization", {}))
        chunking = ChunkingConfig(**config_dict.get("chunking", {}))
        vocabulary = VocabularyConfig(**config_dict.get("vocabulary", {}))
        
        return cls(
            transcription=transcription,
            features=features,
            enhancement=enhancement,
            privacy=privacy,
            output=output,
            summarization=summarization,
            chunking=chunking,
            vocabulary=vocabulary
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        :return: Configuration as dictionary
        """
        return {
            "transcription": self.transcription.__dict__,
            "features": self.features.__dict__,
            "enhancement": self.enhancement.__dict__,
            "privacy": self.privacy.__dict__,
            "output": self.output.__dict__,
            "summarization": self.summarization.__dict__,
            "chunking": self.chunking.__dict__,
            "vocabulary": self.vocabulary.__dict__
        }


class ProviderConfigLoader:
    """Utility for loading provider configurations."""
    
    @staticmethod
    def load_providers(config_path: Path) -> Dict[str, ProviderConfig]:
        """
        Load provider configurations from YAML file.
        
        :param config_path: Path to providers.yml file
        :return: Dictionary of provider name to ProviderConfig
        :raises ConfigurationError: If config file is invalid
        """
        if not config_path.exists():
            logger.error("Provider configuration file not found: %s", config_path)
            raise ConfigurationError(
                f"Provider configuration file not found: {config_path}"
            )
        
        try:
            logger.debug("Loading provider configurations from %s", config_path)
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if not data or "providers" not in data:
                raise ConfigurationError(
                    "Invalid provider configuration: missing 'providers' key"
                )
            
            providers = {}
            for provider_name, provider_data in data["providers"].items():
                providers[provider_name] = ProviderConfig(**provider_data)
            
            logger.info(
                "Loaded %d provider configurations",
                len(providers)
            )
            
            return providers
            
        except yaml.YAMLError as e:
            logger.error("Failed to parse provider configuration YAML: %s", e)
            raise ConfigurationError(
                f"Invalid YAML in provider configuration: {e}"
            )
        except TypeError as e:
            logger.error("Invalid provider configuration structure: %s", e)
            raise ConfigurationError(
                f"Invalid provider configuration structure: {e}"
            )
    
    @staticmethod
    def get_api_key(provider_config: ProviderConfig) -> str:
        """
        Get API key for provider from environment.
        
        :param provider_config: Provider configuration
        :return: API key
        :raises ConfigurationError: If API key not found
        """
        if not provider_config.api_key_env:
            raise ConfigurationError(
                f"No API key environment variable configured for {provider_config.name}"
            )
        
        api_key = os.getenv(provider_config.api_key_env)
        if not api_key:
            logger.error(
                "API key not found in environment: %s",
                provider_config.api_key_env
            )
            raise ConfigurationError(
                f"API key not found: set {provider_config.api_key_env} environment variable"
            )
        
        logger.debug("API key loaded from environment: %s", provider_config.api_key_env)
        return api_key
