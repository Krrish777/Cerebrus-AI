"""
YouTube Processing Components Package.

This package contains high-level processing components that orchestrate
the YouTube audio processing workflow.
"""

from src.youtube_processing.components.document_builder import YouTubeDocumentBuilder
from src.youtube_processing.components.processor import YouTubeAudioProcessor
from src.youtube_processing.components.transcriber import YouTubeTranscriber

__all__ = [
    "YouTubeAudioProcessor",
    "YouTubeDocumentBuilder",
    "YouTubeTranscriber",
]
