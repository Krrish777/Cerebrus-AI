"""
Haystack pipeline components for audio processing.

This module provides Haystack-compatible components that can be
used in Haystack pipelines for audio transcription and processing.
"""

from src.audio_processing.components.transcriber import AudioTranscriberComponent
from src.audio_processing.components.extractor import DataExtractorComponent
from src.audio_processing.components.chunker import ChunkerComponent
from src.audio_processing.components.document_converter import DocumentConverterComponent

__all__ = [
    "AudioTranscriberComponent",
    "DataExtractorComponent",
    "ChunkerComponent",
    "DocumentConverterComponent",
]
