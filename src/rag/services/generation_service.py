"""
Generation service.
Orchestrates response generation with LLMs.
"""

from typing import Any, Dict, List

from haystack.dataclasses import ChatMessage

from src.core.logging import get_logger
from src.rag.providers.base import GeneratorProvider

logger = get_logger(__name__)


class GenerationService:
    """Service for LLM response generation."""
    
    def __init__(self, generator: GeneratorProvider):
        """
        Initialize generation service.
        
        Args:
            generator: Generator provider instance
        """
        self.generator = generator
        logger.info("Initialized GenerationService")
    
    def generate(
        self,
        messages: List[ChatMessage],
        **generation_kwargs: Any
    ) -> str:
        """
        Generate response from messages.
        
        Args:
            messages: List of chat messages
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not messages:
            logger.warning("No messages provided for generation")
            return "No messages provided."
        
        try:
            logger.debug(f"Generating response from {len(messages)} messages")
            
            result = self.generator.run(
                messages=messages,
                **generation_kwargs
            )
            
            # Extract response text
            replies = result.get("replies", [])
            
            if not replies:
                logger.warning("No replies generated")
                return "No response generated."
            
            # Get text from first reply
            response_text = self._extract_text_from_reply(replies[0])
            
            logger.info(f"Generated response of length {len(response_text)}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def _extract_text_from_reply(self, reply: Any) -> str:
        """
        Extract text from reply object.
        
        Args:
            reply: Reply object (ChatMessage or other)
            
        Returns:
            Reply text
        """
        # Handle ChatMessage
        if hasattr(reply, 'text'):
            return reply.text or ""
        
        # Handle string
        if isinstance(reply, str):
            return reply
        
        # Handle dict
        if isinstance(reply, dict):
            return reply.get('text', '') or reply.get('content', '')
        
        # Fallback
        logger.warning(f"Unknown reply format: {type(reply)}")
        return str(reply)
    
    def warm_up(self) -> None:
        """Warm up generator."""
        try:
            self.generator.warm_up()
            logger.info("GenerationService warmed up successfully")
        except Exception as e:
            logger.warning(f"Failed to warm up generator: {e}")
