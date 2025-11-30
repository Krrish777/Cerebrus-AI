"""
Registry for chunking strategies.

Provides factory-style access to chunkers with configuration.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from src.audio_processing.chunking.base import BaseChunker
from src.audio_processing.chunking.base import Chunk
from src.audio_processing.chunking.base import ChunkerConfig
from src.core.logging import get_logger

logger = get_logger(__name__)

# Global registry instance
_registry: Optional["ChunkerRegistry"] = None


def get_global_registry() -> "ChunkerRegistry":
    """
    Get the global chunker registry instance.
    
    Returns:
        Singleton ChunkerRegistry instance
    """
    global _registry
    
    if _registry is None:
        _registry = ChunkerRegistry()
        _registry.register_defaults()
    
    return _registry


class ChunkerRegistry:
    """
    Registry for chunking strategies.
    
    Manages available chunkers and provides factory methods
    for creating configured instances.
    """
    
    def __init__(self) -> None:
        """Initialize the registry."""
        self._chunkers: Dict[str, Type[BaseChunker]] = {}
        self._default_configs: Dict[str, ChunkerConfig] = {}
    
    def register(
        self,
        name: str,
        chunker_class: Type[BaseChunker],
        default_config: Optional[ChunkerConfig] = None,
    ) -> None:
        """
        Register a chunker class.
        
        Args:
            name: Name to register under
            chunker_class: Chunker class to register
            default_config: Default configuration for this chunker
        """
        self._chunkers[name.lower()] = chunker_class
        
        if default_config:
            self._default_configs[name.lower()] = default_config
        
        logger.debug("Registered chunker: %s", name)
    
    def register_defaults(self) -> None:
        """Register all built-in chunkers."""
        from src.audio_processing.chunking.speaker import SpeakerChunker
        from src.audio_processing.chunking.chapter import ChapterChunker
        from src.audio_processing.chunking.semantic import SemanticChunker
        from src.audio_processing.chunking.sentence import SentenceChunker
        
        self.register("speaker", SpeakerChunker)
        self.register("chapter", ChapterChunker)
        self.register("semantic", SemanticChunker)
        self.register("sentence", SentenceChunker)
    
    def get(
        self,
        name: str,
        config: Optional[ChunkerConfig] = None,
        **kwargs: Any,
    ) -> BaseChunker:
        """
        Get a chunker instance by name.
        
        Args:
            name: Chunker name
            config: Optional configuration override
            **kwargs: Additional arguments for chunker constructor
            
        Returns:
            Configured chunker instance
            
        Raises:
            ValueError: If chunker name is not registered
        """
        name_lower = name.lower()
        
        if name_lower not in self._chunkers:
            raise ValueError(
                f"Unknown chunker: {name}. "
                f"Available: {list(self._chunkers.keys())}"
            )
        
        chunker_class = self._chunkers[name_lower]
        
        # Use provided config, or default, or create new
        if config is None:
            config = self._default_configs.get(name_lower)
        
        return chunker_class(config=config, **kwargs)
    
    def available(self) -> List[str]:
        """
        Get list of available chunker names.
        
        Returns:
            List of registered chunker names
        """
        return list(self._chunkers.keys())
    
    def chunk_with(
        self,
        name: str,
        transcript_data: Dict[str, Any],
        config: Optional[ChunkerConfig] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        """
        Chunk data using a named chunker.
        
        Convenience method that creates a chunker and chunks in one call.
        
        Args:
            name: Chunker name to use
            transcript_data: Data to chunk
            config: Optional configuration
            **kwargs: Additional chunker arguments
            
        Returns:
            List of chunks
        """
        chunker = self.get(name, config, **kwargs)
        return chunker.chunk(transcript_data)
    
    def chunk_with_best(
        self,
        transcript_data: Dict[str, Any],
        config: Optional[ChunkerConfig] = None,
    ) -> List[Chunk]:
        """
        Chunk data using the best available strategy.
        
        Selects chunker based on available data:
        1. Chapter if chapters present
        2. Speaker if utterances present
        3. Semantic if paragraphs present
        4. Sentence as fallback
        
        Args:
            transcript_data: Data to chunk
            config: Optional configuration
            
        Returns:
            List of chunks
        """
        if transcript_data.get("chapters"):
            strategy = "chapter"
        elif transcript_data.get("utterances"):
            strategy = "speaker"
        elif transcript_data.get("paragraphs"):
            strategy = "semantic"
        else:
            strategy = "sentence"
        
        logger.info("Auto-selected chunking strategy: %s", strategy)
        
        return self.chunk_with(strategy, transcript_data, config)
