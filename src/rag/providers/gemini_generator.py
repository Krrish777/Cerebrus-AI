"""
Google Gemini generator provider.
"""

import os
from typing import Any, Dict, List, Optional

from haystack.utils import Secret
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

from src.core.logging import get_logger

logger = get_logger(__name__)


class GeminiGeneratorProvider:
    """Google Gemini generator provider implementation."""
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        api_key_env: str = "GEMINI_API_KEY",
        fallback_models: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95
    ):
        """
        Initialize Gemini generator.
        
        Args:
            model: Gemini model name
            api_key: API key (if not provided, loaded from environment)
            api_key_env: Environment variable name for API key
            fallback_models: List of fallback models to try
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Get API key
        effective_api_key = api_key or os.getenv(api_key_env) or os.getenv("GOOGLE_API_KEY")
        if not effective_api_key:
            raise ValueError(
                f"Gemini API key required. Set {api_key_env} environment variable "
                f"or pass api_key parameter"
            )
        
        # Define fallback models
        default_fallbacks = [
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-pro"
        ]
        self.fallback_models = fallback_models or default_fallbacks
        
        # Try to initialize with primary model, then fallbacks
        self._generator = self._initialize_generator(
            effective_api_key,
            [model] + [m for m in self.fallback_models if m != model]
        )
        
        logger.info(
            f"Initialized GeminiGeneratorProvider with "
            f"model={self.model}, temperature={temperature}"
        )
    
    def _initialize_generator(
        self,
        api_key: str,
        models_to_try: List[str]
    ) -> GoogleGenAIChatGenerator:
        """
        Try to initialize generator with models in order.
        
        Args:
            api_key: API key
            models_to_try: List of models to try in order
            
        Returns:
            Initialized generator
            
        Raises:
            ValueError: If all models fail
        """
        last_error = None
        
        for model in models_to_try:
            try:
                generator = GoogleGenAIChatGenerator(
                    model=model,
                    api_key=Secret.from_token(api_key)
                )
                logger.info(f"Successfully initialized with model: {model}")
                self.model = model  # Update to successful model
                return generator
            except Exception as e:
                logger.warning(f"Failed to initialize with {model}: {e}")
                last_error = e
                continue
        
        raise ValueError(
            f"Failed to initialize Gemini generator with any model. "
            f"Last error: {last_error}"
        )
    
    def run(
        self,
        messages: List[Any],
        **generation_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate response from messages.
        
        Args:
            messages: List of chat messages
            **generation_kwargs: Override generation parameters
            
        Returns:
            Dictionary containing generated response
        """
        try:
            # Prepare generation kwargs
            gen_kwargs = {
                "temperature": generation_kwargs.get("temperature", self.temperature),
                "max_tokens": generation_kwargs.get("max_tokens", self.max_tokens),
                "top_p": generation_kwargs.get("top_p", self.top_p),
            }
            
            # Generate response
            result = self._generator.run(messages=messages, **gen_kwargs)
            
            logger.debug(
                f"Generated response with {len(result.get('replies', []))} replies"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during Gemini generation: {e}")
            raise
    
    def warm_up(self) -> None:
        """Warm up generator (test connection)."""
        try:
            # Gemini generators don't require explicit warm-up
            logger.debug("GeminiGeneratorProvider does not require warm-up")
        except Exception as e:
            logger.warning(f"Failed to warm up Gemini generator: {e}")
