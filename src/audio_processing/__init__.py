# Audio processing module

from src.audio_processing.adapters import ConfigurationAdapter
from src.audio_processing.adapters import create_legacy_config_from_yaml
from src.audio_processing.adapters import get_transcription_config_params
from src.audio_processing.adapters import LegacyAudioProcessingConfig
from src.audio_processing.config import AudioConfigLoader
from src.audio_processing.config import AudioProcessingConfig
from src.audio_processing.config import ContentAnalysisConfig
from src.audio_processing.config import get_audio_config
from src.audio_processing.config import reload_audio_config
from src.audio_processing.config import SmartChunkingConfig
from src.audio_processing.config import SpeakerAnalysisConfig
from src.audio_processing.config import TranscriptionConfig
from src.audio_processing.config import YouTubeConfig
from src.audio_processing.exceptions import AudioDownloadError
from src.audio_processing.exceptions import AudioProcessingException
from src.audio_processing.exceptions import AudioSourceError
from src.audio_processing.exceptions import ChunkingError
from src.audio_processing.exceptions import ConfigurationLoadError
from src.audio_processing.exceptions import TranscriptionAPIError
from src.audio_processing.exceptions import TranscriptionConfigurationError
from src.audio_processing.exceptions import TranscriptionError
from src.audio_processing.exceptions import TranscriptionServiceUnavailableError
from src.audio_processing.exceptions import YouTubeVideoError
from src.audio_processing.interfaces import AudioChunkerInterface
from src.audio_processing.interfaces import AudioSourceValidatorInterface
from src.audio_processing.interfaces import MetadataExtractorInterface
from src.audio_processing.interfaces import TranscriptionConfigAdapterInterface
from src.audio_processing.interfaces import TranscriptionServiceInterface
from src.audio_processing.interfaces import YouTubeProcessorInterface

__all__ = [
    # Configuration classes
    "AudioConfigLoader",
    "AudioProcessingConfig",
    "ContentAnalysisConfig",
    "SmartChunkingConfig",
    "SpeakerAnalysisConfig",
    "TranscriptionConfig",
    "YouTubeConfig",
    "get_audio_config",
    "reload_audio_config",
    # Adapter classes
    "ConfigurationAdapter",
    "LegacyAudioProcessingConfig",
    "create_legacy_config_from_yaml",
    "get_transcription_config_params",
    # Interface classes
    "AudioChunkerInterface",
    "AudioSourceValidatorInterface",
    "MetadataExtractorInterface",
    "TranscriptionConfigAdapterInterface",
    "TranscriptionServiceInterface",
    "YouTubeProcessorInterface",
    # Exception classes
    "AudioDownloadError",
    "AudioProcessingException",
    "AudioSourceError",
    "ChunkingError",
    "ConfigurationLoadError",
    "TranscriptionAPIError",
    "TranscriptionConfigurationError",
    "TranscriptionError",
    "TranscriptionServiceUnavailableError",
    "YouTubeVideoError",
]