"""
Tests for audio processing pipeline module.

Tests the pipeline builder and runner functionality.
"""

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.audio_processing.config import ProviderConfig
from src.audio_processing.pipeline.builder import AudioPipelineBuilder
from src.audio_processing.pipeline.builder import PipelineConfig
from src.audio_processing.pipeline.runner import AudioPipelineRunner


@pytest.fixture
def sample_transcript_data() -> Dict[str, Any]:
    """Fixture providing sample transcript data."""
    fixture_path = Path("data/fixtures/mock_transcript.json")
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    return {
        "text": "This is a sample transcript.",
        "utterances": [
            {
                "speaker": "A",
                "text": "This is a sample transcript.",
                "start": 0,
                "end": 3000,
            }
        ],
        "words": [
            {"text": "This", "start": 0, "end": 500},
            {"text": "is", "start": 500, "end": 800},
            {"text": "a", "start": 800, "end": 900},
            {"text": "sample", "start": 900, "end": 1500},
            {"text": "transcript", "start": 1500, "end": 2500},
        ],
    }


@pytest.fixture
def sample_analysis_data() -> Dict[str, Any]:
    """Fixture providing sample analysis data."""
    fixture_path = Path("data/fixtures/mock_analysis_data.json")
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    return {
        "sentiment_analysis_results": [
            {"sentiment": "POSITIVE", "confidence": 0.85}
        ],
        "entities": [
            {"text": "Test", "entity_type": "ORGANIZATION"}
        ],
    }


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_default_config_values(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.provider_name == "assemblyai"
        assert config.provider_config is None
        assert config.chunking_strategy == "auto"
        assert config.extractors is None
        assert config.use_chunks is True
    
    def test_config_with_custom_values(self) -> None:
        """Test configuration with custom values."""
        provider_cfg = ProviderConfig(name="custom", api_key_env="TEST_KEY")
        config = PipelineConfig(
            provider_name="custom",
            provider_config=provider_cfg,
            chunking_strategy="speaker",
            extractors=["sentiment", "entities"],
        )
        
        assert config.provider_name == "custom"
        assert config.provider_config is provider_cfg
        assert config.chunking_strategy == "speaker"
        assert config.extractors == ["sentiment", "entities"]
    
    def test_config_with_chunker_config(self) -> None:
        """Test configuration with chunker config."""
        config = PipelineConfig(
            chunking_strategy="chapter",
            use_chunks=False,
        )
        
        assert config.chunking_strategy == "chapter"
        assert config.use_chunks is False


class TestAudioPipelineBuilder:
    """Tests for AudioPipelineBuilder fluent interface."""
    
    def test_builder_initialization(self) -> None:
        """Test builder initializes with default config."""
        builder = AudioPipelineBuilder()
        
        assert builder._config is not None
        assert isinstance(builder._config, PipelineConfig)
    
    def test_with_provider_config(self) -> None:
        """Test configuring provider with full config."""
        provider_cfg = ProviderConfig(name="custom", api_key_env="TEST_KEY")
        builder = AudioPipelineBuilder()
        result = builder.with_provider("custom", provider_config=provider_cfg)
        
        assert result is builder  # Fluent interface
        assert builder._config.provider_name == "custom"
        assert builder._config.provider_config is provider_cfg
    
    def test_with_provider_name_only(self) -> None:
        """Test configuring provider with name only."""
        builder = AudioPipelineBuilder()
        result = builder.with_provider("assemblyai")
        
        assert result is builder
        assert builder._config.provider_name == "assemblyai"
    
    def test_with_chunking(self) -> None:
        """Test configuring chunking strategy."""
        builder = AudioPipelineBuilder()
        result = builder.with_chunking("speaker")
        
        assert result is builder
        assert builder._config.chunking_strategy == "speaker"
    
    def test_with_extractors(self) -> None:
        """Test configuring extractors."""
        builder = AudioPipelineBuilder()
        extractors = ["sentiment", "entities", "topics"]
        result = builder.with_extractors(extractors)
        
        assert result is builder
        assert builder._config.extractors == extractors
    
    def test_without_transcriber(self) -> None:
        """Test disabling transcription."""
        builder = AudioPipelineBuilder()
        result = builder.without_transcriber()
        
        assert result is builder
        assert builder._include_transcriber is False
    
    def test_without_extractor(self) -> None:
        """Test disabling extractor."""
        builder = AudioPipelineBuilder()
        result = builder.without_extractor()
        
        assert result is builder
        assert builder._include_extractor is False
    
    def test_without_chunker(self) -> None:
        """Test disabling chunker."""
        builder = AudioPipelineBuilder()
        result = builder.without_chunker()
        
        assert result is builder
        assert builder._include_chunker is False
    
    def test_method_chaining(self) -> None:
        """Test fluent method chaining."""
        builder = (
            AudioPipelineBuilder()
            .with_provider("custom")
            .with_chunking("chapter")
            .with_extractors(["sentiment"])
        )
        
        assert builder._config.provider_name == "custom"
        assert builder._config.chunking_strategy == "chapter"
        assert builder._config.extractors == ["sentiment"]
    
    @patch("src.audio_processing.pipeline.builder.Pipeline")
    @patch("src.audio_processing.pipeline.builder.AudioTranscriberComponent")
    @patch("src.audio_processing.pipeline.builder.DataExtractorComponent")
    @patch("src.audio_processing.pipeline.builder.ChunkerComponent")
    @patch("src.audio_processing.pipeline.builder.DocumentConverterComponent")
    def test_build_creates_pipeline(
        self,
        mock_converter: MagicMock,
        mock_chunker: MagicMock,
        mock_extractor: MagicMock,
        mock_transcriber: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """Test building a pipeline creates components."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        
        builder = AudioPipelineBuilder()
        result = builder.build()
        
        # Should create a pipeline
        mock_pipeline_cls.assert_called_once()
        assert result is mock_pipeline
    
    @patch("src.audio_processing.pipeline.builder.Pipeline")
    @patch("src.audio_processing.pipeline.builder.DocumentConverterComponent")
    def test_build_minimal_pipeline(
        self,
        mock_converter: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """Test building minimal pipeline without transcriber."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        
        builder = (
            AudioPipelineBuilder()
            .without_transcriber()
            .without_extractor()
            .without_chunker()
        )
        result = builder.build()
        
        # Should still create a pipeline
        mock_pipeline_cls.assert_called_once()
        assert result is mock_pipeline


class TestAudioPipelineRunner:
    """Tests for AudioPipelineRunner."""
    
    def test_runner_with_provided_pipeline(self) -> None:
        """Test runner accepts pre-built pipeline."""
        mock_pipeline = MagicMock()
        
        runner = AudioPipelineRunner(pipeline=mock_pipeline)
        
        assert runner._pipeline is mock_pipeline
    
    @patch("src.audio_processing.pipeline.runner.AudioPipelineBuilder")
    def test_runner_builds_pipeline_from_config(
        self,
        mock_builder_cls: MagicMock,
    ) -> None:
        """Test runner builds pipeline from configuration."""
        mock_builder = MagicMock()
        mock_pipeline = MagicMock()
        mock_builder.with_provider.return_value = mock_builder
        mock_builder.build.return_value = mock_pipeline
        mock_builder_cls.return_value = mock_builder
        
        runner = AudioPipelineRunner(
            provider_name="custom",
        )
        
        assert runner._pipeline is mock_pipeline
    
    def test_process_audio(self) -> None:
        """Test processing a single audio file."""
        mock_pipeline = MagicMock()
        mock_documents = [MagicMock()]
        mock_pipeline.run.return_value = {
            "converter": {"documents": mock_documents}
        }
        
        runner = AudioPipelineRunner(pipeline=mock_pipeline)
        result = runner.process_audio(Path("test.mp3"))
        
        assert result == mock_documents
        mock_pipeline.run.assert_called_once()
    
    def test_process_audio_with_source_name(self) -> None:
        """Test processing audio with custom source name."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"converter": {"documents": []}}
        
        runner = AudioPipelineRunner(pipeline=mock_pipeline)
        runner.process_audio(
            Path("test.mp3"),
            source_name="custom_source",
        )
        
        call_args = mock_pipeline.run.call_args[0][0]
        assert call_args["converter"]["source_names"] == ["custom_source"]
    
    def test_process_batch(self) -> None:
        """Test batch processing multiple audio files."""
        mock_pipeline = MagicMock()
        mock_documents = [MagicMock(), MagicMock()]
        mock_pipeline.run.return_value = {
            "converter": {"documents": mock_documents}
        }
        
        runner = AudioPipelineRunner(pipeline=mock_pipeline)
        paths: List[Any] = [Path("test1.mp3"), Path("test2.mp3")]
        result = runner.process_batch(paths)
        
        assert result == mock_documents
        mock_pipeline.run.assert_called_once()
    
    def test_process_batch_with_source_names(self) -> None:
        """Test batch processing with custom source names."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"converter": {"documents": []}}
        
        runner = AudioPipelineRunner(pipeline=mock_pipeline)
        paths: List[Any] = [Path("a.mp3"), Path("b.mp3")]
        names = ["source_a", "source_b"]
        
        runner.process_batch(paths, source_names=names)
        
        call_args = mock_pipeline.run.call_args[0][0]
        assert call_args["converter"]["source_names"] == names
    
    def test_pipeline_property(self) -> None:
        """Test pipeline property returns underlying pipeline."""
        mock_pipeline = MagicMock()
        
        runner = AudioPipelineRunner(pipeline=mock_pipeline)
        
        assert runner.pipeline is mock_pipeline
    
    def test_get_pipeline_info(self) -> None:
        """Test getting pipeline information."""
        mock_pipeline = MagicMock()
        mock_pipeline._graph.nodes = {}
        
        runner = AudioPipelineRunner(pipeline=mock_pipeline)
        info = runner.get_pipeline_info()
        
        assert "component_count" in info
        assert "components" in info
        assert isinstance(info["components"], list)


class TestPipelineIntegration:
    """Integration tests for pipeline builder and runner."""
    
    @patch("src.audio_processing.pipeline.builder.Pipeline")
    @patch("src.audio_processing.pipeline.builder.AudioTranscriberComponent")
    @patch("src.audio_processing.pipeline.builder.DataExtractorComponent")
    @patch("src.audio_processing.pipeline.builder.ChunkerComponent")
    @patch("src.audio_processing.pipeline.builder.DocumentConverterComponent")
    def test_builder_and_runner_integration(
        self,
        mock_converter: MagicMock,
        mock_chunker: MagicMock,
        mock_extractor: MagicMock,
        mock_transcriber: MagicMock,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        """Test builder and runner work together."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "converter": {"documents": []}
        }
        mock_pipeline_cls.return_value = mock_pipeline
        
        # Build pipeline
        pipeline = (
            AudioPipelineBuilder()
            .with_provider("assemblyai")
            .with_chunking("speaker")
            .build()
        )
        
        # Use runner
        runner = AudioPipelineRunner(pipeline=pipeline)
        result = runner.process_audio(Path("test.mp3"))
        
        assert isinstance(result, list)
    
    def test_fluent_configuration(self) -> None:
        """Test full fluent configuration pattern."""
        builder = (
            AudioPipelineBuilder()
            .with_provider("assemblyai")
            .with_chunking("semantic")
            .with_extractors(["sentiment", "entities", "topics"])
        )
        
        config = builder._config
        
        assert config.provider_name == "assemblyai"
        assert config.chunking_strategy == "semantic"
        assert config.extractors is not None
        assert len(config.extractors) == 3
        assert "sentiment" in config.extractors
        assert "entities" in config.extractors
        assert "topics" in config.extractors
