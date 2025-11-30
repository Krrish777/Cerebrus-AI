"""
Audio transcriber Haystack component.

Provides a Haystack-compatible component for audio transcription.
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import component

from src.audio_processing.config import FeatureConfig
from src.audio_processing.config import ProviderConfig
from src.audio_processing.config import TranscriptionConfig
from src.audio_processing.transcription.factory import TranscriptionFactory
from src.audio_processing.transcription.providers.base import BaseTranscriptionProvider
from src.core.logging import get_logger

logger = get_logger(__name__)


@component
class AudioTranscriberComponent:
    """
    Haystack component for audio transcription.
    
    Transcribes audio files using configured transcription providers.
    Can be used as part of a Haystack pipeline for audio processing.
    
    Inputs:
        audio_paths: List of paths to audio files
        
    Outputs:
        transcripts: List of transcript dictionaries
    """
    
    def __init__(
        self,
        provider_name: str = "assemblyai",
        provider_config: Optional[ProviderConfig] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        """
        Initialize the audio transcriber component.
        
        Args:
            provider_name: Name of the transcription provider
            provider_config: Provider configuration with API key
            transcription_config: Transcription settings
            feature_config: Feature extraction settings
        """
        self._provider_name = provider_name
        self._provider_config = provider_config
        self._transcription_config = transcription_config
        self._feature_config = feature_config
        self._provider: Optional[BaseTranscriptionProvider] = None
    
    def warm_up(self) -> None:
        """
        Initialize the transcription provider.
        
        Called by Haystack before the pipeline runs.
        """
        if self._provider is None:
            self._provider = self._create_provider()
            logger.info(
                "Initialized transcription provider: %s",
                self._provider_name,
            )
    
    def _create_provider(self) -> BaseTranscriptionProvider:
        """Create and configure the transcription provider."""
        factory = TranscriptionFactory()
        
        provider = factory.create(
            provider_name=self._provider_name,
            provider_config=self._provider_config,
        )
        
        if self._transcription_config:
            provider.configure(self._transcription_config)
        
        return provider
    
    @component.output_types(transcripts=List[Dict[str, Any]])
    def run(
        self,
        audio_paths: List[Path],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Transcribe audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            Dictionary with 'transcripts' key containing transcript data
        """
        if self._provider is None:
            self.warm_up()
        
        transcripts = []
        
        for audio_path in audio_paths:
            logger.info("Transcribing: %s", audio_path)
            
            try:
                transcript = self._provider.transcribe(audio_path)
                transcripts.append(transcript)
                
                logger.info(
                    "Transcribed %s: %d chars, confidence %.2f",
                    audio_path.name,
                    len(transcript.get("text", "")),
                    transcript.get("confidence", 0),
                )
            except Exception as e:
                logger.error(
                    "Failed to transcribe %s: %s",
                    audio_path,
                    str(e),
                )
                # Include error info in result
                transcripts.append({
                    "id": None,
                    "text": "",
                    "status": "error",
                    "error": str(e),
                    "source": str(audio_path),
                })
        
        return {"transcripts": transcripts}
