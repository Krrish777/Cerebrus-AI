"""
Chunking strategies for audio transcripts.

This module provides various strategies for chunking audio transcripts:
- SpeakerChunker: Chunks by speaker turns
- ChapterChunker: Chunks by transcript chapters  
- SemanticChunker: Chunks by semantic boundaries
- SentenceChunker: Chunks by sentences with overlap
"""

from src.audio_processing.chunking.base import BaseChunker
from src.audio_processing.chunking.speaker import SpeakerChunker
from src.audio_processing.chunking.chapter import ChapterChunker
from src.audio_processing.chunking.semantic import SemanticChunker
from src.audio_processing.chunking.sentence import SentenceChunker
from src.audio_processing.chunking.registry import ChunkerRegistry
from src.audio_processing.chunking.registry import get_global_registry

__all__ = [
    "BaseChunker",
    "SpeakerChunker",
    "ChapterChunker",
    "SemanticChunker",
    "SentenceChunker",
    "ChunkerRegistry",
    "get_global_registry",
]
