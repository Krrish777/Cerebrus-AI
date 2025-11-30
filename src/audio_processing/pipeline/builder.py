"""
Pipeline builder for audio processing.

Provides a fluent builder interface for constructing audio processing pipelines.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from haystack import Pipeline

from src.audio_processing.chunking.base import ChunkerConfig
from src.audio_processing.components.chunker import ChunkerComponent
from src.audio_processing.components.document_converter import DocumentConverterComponent
from src.audio_processing.components.extractor import DataExtractorComponent
from src.audio_processing.components.transcriber import AudioTranscriberComponent
from src.audio_processing.config import FeatureConfig
from src.audio_processing.config import ProviderConfig
from src.audio_processing.config import TranscriptionConfig
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for the audio processing pipeline.
    
    Attributes:
        provider_name: Transcription provider name
        provider_config: Provider configuration
        transcription_config: Transcription settings
        feature_config: Feature extraction settings
        chunking_strategy: Chunking strategy name
        chunker_config: Chunker settings
        extractors: List of extractors to use
        use_chunks: Whether to create documents from chunks
    """
    
    provider_name: str = "assemblyai"
    provider_config: Optional[ProviderConfig] = None
    transcription_config: Optional[TranscriptionConfig] = None
    feature_config: Optional[FeatureConfig] = None
    chunking_strategy: str = "auto"
    chunker_config: Optional[ChunkerConfig] = None
    extractors: Optional[List[str]] = None
    use_chunks: bool = True


class AudioPipelineBuilder:
    """
    Builder for creating audio processing pipelines.
    
    Provides a fluent interface for configuring and building
    Haystack pipelines for audio processing.
    
    Example:
        pipeline = (
            AudioPipelineBuilder()
            .with_provider("assemblyai", api_key="...")
            .with_chunking("chapter")
            .with_extractors(["sentiment", "entities"])
            .build()
        )
    """
    
    def __init__(self) -> None:
        """Initialize the pipeline builder."""
        self._config = PipelineConfig()
        self._include_transcriber = True
        self._include_extractor = True
        self._include_chunker = True
        self._include_converter = True
    
    def with_provider(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> "AudioPipelineBuilder":
        """
        Configure the transcription provider.
        
        Args:
            provider_name: Name of the provider
            api_key: Optional API key (can also be set via config)
            provider_config: Optional full provider configuration
            
        Returns:
            Self for chaining
        """
        self._config.provider_name = provider_name
        
        if provider_config:
            self._config.provider_config = provider_config
        elif api_key:
            self._config.provider_config = ProviderConfig(
                provider_name=provider_name,
                api_key=api_key,
            )
        
        return self
    
    def with_transcription_config(
        self,
        config: TranscriptionConfig,
    ) -> "AudioPipelineBuilder":
        """
        Set transcription configuration.
        
        Args:
            config: Transcription configuration
            
        Returns:
            Self for chaining
        """
        self._config.transcription_config = config
        return self
    
    def with_features(
        self,
        config: FeatureConfig,
    ) -> "AudioPipelineBuilder":
        """
        Configure feature extraction.
        
        Args:
            config: Feature configuration
            
        Returns:
            Self for chaining
        """
        self._config.feature_config = config
        return self
    
    def with_chunking(
        self,
        strategy: str = "auto",
        config: Optional[ChunkerConfig] = None,
    ) -> "AudioPipelineBuilder":
        """
        Configure chunking strategy.
        
        Args:
            strategy: Chunking strategy name
            config: Optional chunker configuration
            
        Returns:
            Self for chaining
        """
        self._config.chunking_strategy = strategy
        self._config.chunker_config = config
        return self
    
    def with_extractors(
        self,
        extractors: List[str],
    ) -> "AudioPipelineBuilder":
        """
        Configure which extractors to use.
        
        Args:
            extractors: List of extractor names
            
        Returns:
            Self for chaining
        """
        self._config.extractors = extractors
        return self
    
    def with_document_options(
        self,
        use_chunks: bool = True,
    ) -> "AudioPipelineBuilder":
        """
        Configure document creation options.
        
        Args:
            use_chunks: Whether to create documents from chunks
            
        Returns:
            Self for chaining
        """
        self._config.use_chunks = use_chunks
        return self
    
    def without_transcriber(self) -> "AudioPipelineBuilder":
        """
        Exclude the transcriber component.
        
        Useful when transcripts are provided externally.
        
        Returns:
            Self for chaining
        """
        self._include_transcriber = False
        return self
    
    def without_extractor(self) -> "AudioPipelineBuilder":
        """
        Exclude the extractor component.
        
        Returns:
            Self for chaining
        """
        self._include_extractor = False
        return self
    
    def without_chunker(self) -> "AudioPipelineBuilder":
        """
        Exclude the chunker component.
        
        Returns:
            Self for chaining
        """
        self._include_chunker = False
        return self
    
    def without_converter(self) -> "AudioPipelineBuilder":
        """
        Exclude the document converter component.
        
        Returns:
            Self for chaining
        """
        self._include_converter = False
        return self
    
    def build(self) -> Pipeline:
        """
        Build the configured pipeline.
        
        Returns:
            Configured Haystack Pipeline
        """
        pipeline = Pipeline()
        
        components = self._create_components()
        connections = self._get_connections()
        
        # Add components
        for name, component in components.items():
            pipeline.add_component(name, component)
            logger.debug("Added component: %s", name)
        
        # Connect components
        for source, target in connections:
            if source.split(".")[0] in components and target.split(".")[0] in components:
                pipeline.connect(source, target)
                logger.debug("Connected: %s -> %s", source, target)
        
        logger.info(
            "Built audio pipeline with %d components",
            len(components),
        )
        
        return pipeline
    
    def _create_components(self) -> Dict[str, Any]:
        """Create the pipeline components."""
        components = {}
        
        if self._include_transcriber:
            components["transcriber"] = AudioTranscriberComponent(
                provider_name=self._config.provider_name,
                provider_config=self._config.provider_config,
                transcription_config=self._config.transcription_config,
                feature_config=self._config.feature_config,
            )
        
        if self._include_extractor:
            components["extractor"] = DataExtractorComponent(
                extractors=self._config.extractors,
            )
        
        if self._include_chunker:
            components["chunker"] = ChunkerComponent(
                strategy=self._config.chunking_strategy,
                config=self._config.chunker_config,
            )
        
        if self._include_converter:
            components["converter"] = DocumentConverterComponent(
                use_chunks=self._config.use_chunks and self._include_chunker,
            )
        
        return components
    
    def _get_connections(self) -> List[tuple]:
        """Get the component connections."""
        connections = []
        
        # Transcriber -> Extractor
        if self._include_transcriber and self._include_extractor:
            connections.append(
                ("transcriber.transcripts", "extractor.transcripts")
            )
        
        # Extractor -> Chunker
        if self._include_extractor and self._include_chunker:
            connections.append(
                ("extractor.transcripts", "chunker.transcripts")
            )
        elif self._include_transcriber and self._include_chunker:
            connections.append(
                ("transcriber.transcripts", "chunker.transcripts")
            )
        
        # -> Converter
        if self._include_converter:
            if self._include_chunker:
                connections.append(
                    ("chunker.transcripts", "converter.transcripts")
                )
                connections.append(
                    ("chunker.chunks", "converter.chunks")
                )
            elif self._include_extractor:
                connections.append(
                    ("extractor.transcripts", "converter.transcripts")
                )
            elif self._include_transcriber:
                connections.append(
                    ("transcriber.transcripts", "converter.transcripts")
                )
            
            if self._include_extractor:
                connections.append(
                    ("extractor.extracted_data", "converter.extracted_data")
                )
        
        return connections
    
    @classmethod
    def create_default(
        cls,
        provider_name: str = "assemblyai",
        api_key: Optional[str] = None,
    ) -> Pipeline:
        """
        Create a default audio processing pipeline.
        
        Args:
            provider_name: Transcription provider name
            api_key: Optional API key
            
        Returns:
            Default configured pipeline
        """
        return (
            cls()
            .with_provider(provider_name, api_key=api_key)
            .with_chunking("auto")
            .build()
        )
    
    @classmethod
    def create_extraction_only(
        cls,
        extractors: Optional[List[str]] = None,
    ) -> Pipeline:
        """
        Create a pipeline for extraction only (no transcription).
        
        Args:
            extractors: List of extractors to use
            
        Returns:
            Extraction-only pipeline
        """
        return (
            cls()
            .without_transcriber()
            .with_extractors(extractors or [])
            .build()
        )
