"""
Document building module for audio processing.

This module provides components for building Haystack Documents
from transcript data and extracted features.
"""

from src.audio_processing.document.builder import TranscriptDocumentBuilder
from src.audio_processing.document.metadata import MetadataBuilder

__all__ = [
    "TranscriptDocumentBuilder",
    "MetadataBuilder",
]
