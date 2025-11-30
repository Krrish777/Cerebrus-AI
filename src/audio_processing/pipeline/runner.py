"""
Pipeline runner for audio processing.

Provides high-level interface for running audio processing pipelines.
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from haystack import Pipeline
from haystack.dataclasses import Document

from src.audio_processing.pipeline.builder import AudioPipelineBuilder
from src.audio_processing.pipeline.builder import PipelineConfig
from src.core.logging import get_logger

logger = get_logger(__name__)


class AudioPipelineRunner:
    """
    High-level runner for audio processing pipelines.
    
    Provides a simple interface for processing audio files
    through the pipeline.
    
    Example:
        runner = AudioPipelineRunner(api_key="...")
        documents = runner.process_audio("audio.mp3")
    """
    
    def __init__(
        self,
        pipeline: Optional[Pipeline] = None,
        config: Optional[PipelineConfig] = None,
        api_key: Optional[str] = None,
        provider_name: str = "assemblyai",
    ) -> None:
        """
        Initialize the pipeline runner.
        
        Args:
            pipeline: Optional pre-built pipeline
            config: Optional pipeline configuration
            api_key: API key for transcription provider
            provider_name: Name of the transcription provider
        """
        if pipeline:
            self._pipeline = pipeline
        else:
            self._pipeline = self._build_pipeline(
                config=config,
                api_key=api_key,
                provider_name=provider_name,
            )
    
    def _build_pipeline(
        self,
        config: Optional[PipelineConfig],
        api_key: Optional[str],
        provider_name: str,
    ) -> Pipeline:
        """Build the pipeline from configuration."""
        builder = AudioPipelineBuilder()
        
        if config:
            builder._config = config
        else:
            builder = builder.with_provider(provider_name, api_key=api_key)
        
        return builder.build()
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        source_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Process a single audio file.
        
        Args:
            audio_path: Path to the audio file
            source_name: Optional custom source name
            
        Returns:
            List of processed documents
        """
        audio_path = Path(audio_path)
        
        if source_name is None:
            source_name = audio_path.name
        
        logger.info("Processing audio file: %s", audio_path)
        
        result = self._pipeline.run({
            "transcriber": {
                "audio_paths": [audio_path],
            },
            "converter": {
                "source_names": [source_name],
            },
        })
        
        documents = result.get("converter", {}).get("documents", [])
        
        logger.info(
            "Processed %s: %d documents created",
            audio_path.name,
            len(documents),
        )
        
        return documents
    
    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        source_names: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Process multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            source_names: Optional list of source names
            
        Returns:
            List of all processed documents
        """
        paths = [Path(p) for p in audio_paths]
        
        if source_names is None:
            source_names = [p.name for p in paths]
        
        logger.info("Processing %d audio files", len(paths))
        
        result = self._pipeline.run({
            "transcriber": {
                "audio_paths": paths,
            },
            "converter": {
                "source_names": source_names,
            },
        })
        
        documents = result.get("converter", {}).get("documents", [])
        
        logger.info(
            "Batch processed %d files: %d documents created",
            len(paths),
            len(documents),
        )
        
        return documents
    
    def process_transcript(
        self,
        transcript_data: Dict[str, Any],
        source_name: str = "transcript",
    ) -> List[Document]:
        """
        Process a pre-transcribed transcript.
        
        Useful when transcription is done separately.
        
        Args:
            transcript_data: Raw transcript dictionary
            source_name: Source name for the transcript
            
        Returns:
            List of processed documents
        """
        logger.info("Processing pre-transcribed transcript: %s", source_name)
        
        # For processing without transcriber, we need a different pipeline
        pipeline = (
            AudioPipelineBuilder()
            .without_transcriber()
            .with_chunking(self._get_chunking_strategy())
            .build()
        )
        
        result = pipeline.run({
            "extractor": {
                "transcripts": [transcript_data],
            },
            "converter": {
                "source_names": [source_name],
            },
        })
        
        documents = result.get("converter", {}).get("documents", [])
        
        logger.info(
            "Processed transcript: %d documents created",
            len(documents),
        )
        
        return documents
    
    def _get_chunking_strategy(self) -> str:
        """Get the chunking strategy from the current pipeline."""
        # Try to get from chunker component if present
        if hasattr(self._pipeline, "_graph"):
            for node in self._pipeline._graph.nodes:
                component = self._pipeline._graph.nodes[node].get("instance")
                if hasattr(component, "_strategy"):
                    return component._strategy
        return "auto"
    
    @property
    def pipeline(self) -> Pipeline:
        """Return the underlying pipeline."""
        return self._pipeline
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline information
        """
        components = []
        
        if hasattr(self._pipeline, "_graph"):
            for node in self._pipeline._graph.nodes:
                component = self._pipeline._graph.nodes[node].get("instance")
                components.append({
                    "name": node,
                    "type": type(component).__name__,
                })
        
        return {
            "component_count": len(components),
            "components": components,
        }
