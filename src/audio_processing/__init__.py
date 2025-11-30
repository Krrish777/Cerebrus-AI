"""
Audio Processing Module.

Provides a comprehensive pipeline for transcribing audio files,
extracting features, chunking content, and creating structured
documents for downstream processing.

Example:
    from src.audio_processing import AudioPipelineBuilder, AudioPipelineRunner
    
    pipeline = (
        AudioPipelineBuilder()
        .with_provider("assemblyai")
        .with_chunking("speaker")
        .build()
    )
    
    runner = AudioPipelineRunner(pipeline=pipeline)
    documents = runner.process_audio("audio.mp3")
"""

# Pipeline - Main entry point
from src.audio_processing.pipeline import AudioPipelineBuilder
from src.audio_processing.pipeline import AudioPipelineRunner
from src.audio_processing.pipeline import PipelineConfig

# Transcription
from src.audio_processing.transcription.factory import TranscriptionFactory
from src.audio_processing.transcription.orchestrator import AudioTranscriber

# Extractors
from src.audio_processing.extractors.registry import ExtractorRegistry
from src.audio_processing.extractors.registry import get_registry as get_extractor_registry

# Chunkers
from src.audio_processing.chunking.registry import ChunkerRegistry
from src.audio_processing.chunking.registry import get_global_registry as get_chunker_registry
from src.audio_processing.chunking.base import Chunk

# Document Building
from src.audio_processing.document.builder import TranscriptDocumentBuilder
from src.audio_processing.document.metadata import DocumentMetadata

# Configuration
from src.audio_processing.config import AudioProcessingConfig
from src.audio_processing.config import FeatureConfig
from src.audio_processing.config import TranscriptionConfig
from src.audio_processing.config import ProviderConfig

# Exceptions
from src.audio_processing.exceptions import AudioProcessingError
from src.audio_processing.exceptions import TranscriptionError
from src.audio_processing.exceptions import ConfigurationError

__all__ = [
    # Pipeline
    "AudioPipelineBuilder",
    "AudioPipelineRunner",
    "PipelineConfig",
    # Transcription
    "TranscriptionFactory",
    "AudioTranscriber",
    # Extractors
    "ExtractorRegistry",
    "get_extractor_registry",
    # Chunkers
    "ChunkerRegistry",
    "get_chunker_registry",
    "Chunk",
    # Documents
    "TranscriptDocumentBuilder",
    "DocumentMetadata",
    # Configuration
    "AudioProcessingConfig",
    "FeatureConfig",
    "TranscriptionConfig",
    "ProviderConfig",
    # Exceptions
    "AudioProcessingError",
    "TranscriptionError",
    "ConfigurationError",
]