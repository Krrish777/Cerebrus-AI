# Audio Processing Module

The audio processing module provides a comprehensive pipeline for transcribing audio files, extracting features, chunking content, and creating structured documents for downstream processing.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Audio Processing Pipeline                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │  Transcriber │ → │  Extractors  │ → │   Chunkers   │ → │  Documents   │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│                                                                             │
│  - AssemblyAI       - Sentiment        - Speaker          - Haystack       │
│  - Custom           - Entities         - Chapter          - Metadata       │
│    providers        - Chapters         - Semantic         - Chunks         │
│                     - Topics           - Sentence                          │
│                     - Safety                                               │
│                     - Highlights                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

Ensure you have the required dependencies:

```bash
uv add assemblyai haystack-ai
```

## Quick Start

### Basic Usage with Pipeline Builder

```python
from src.audio_processing.pipeline import AudioPipelineBuilder, AudioPipelineRunner

# Build a pipeline with the fluent interface
pipeline = (
    AudioPipelineBuilder()
    .with_provider("assemblyai")
    .with_chunking("speaker")
    .with_extractors(["sentiment", "entities", "topics"])
    .build()
)

# Run the pipeline
runner = AudioPipelineRunner(pipeline=pipeline)
documents = runner.process_audio("path/to/audio.mp3")

# Access results
for doc in documents:
    print(doc.content)
    print(doc.meta)
```

### Using Individual Components

```python
from pathlib import Path
from src.audio_processing.transcription.factory import TranscriptionFactory
from src.audio_processing.extractors.registry import get_registry as get_extractor_registry
from src.audio_processing.chunking.registry import get_global_registry as get_chunker_registry
from src.audio_processing.document.builder import TranscriptDocumentBuilder

# 1. Transcribe audio
provider = TranscriptionFactory.create("assemblyai", api_key="your-key")
provider.configure()
transcript = provider.transcribe(Path("audio.mp3"))

# 2. Extract features
extractor_registry = get_extractor_registry()
extracted = extractor_registry.extract_all(transcript)

# 3. Chunk the transcript
chunker_registry = get_chunker_registry()
chunks = chunker_registry.chunk_with_best(transcript)

# 4. Build documents
builder = TranscriptDocumentBuilder()
documents = builder.build_from_chunks(
    chunks=chunks,
    transcript_data=transcript,
    extracted_data=extracted,
    source_name="audio.mp3",
)
```

## Module Components

### 1. Transcription Providers (`transcription/`)

Handles audio transcription with support for multiple providers.

#### Available Providers
- **AssemblyAI**: Full-featured transcription with speaker diarization, sentiment analysis, and more

#### Example
```python
from src.audio_processing.transcription.factory import TranscriptionFactory
from src.audio_processing.config import FeatureConfig

# Create provider
provider = TranscriptionFactory.create("assemblyai", api_key="your-key")

# Configure with features
feature_config = FeatureConfig(
    speaker_labels=True,
    sentiment_analysis=True,
    entity_detection=True,
    auto_chapters=True,
)
provider.configure(feature_config=feature_config)

# Transcribe
result = provider.transcribe(Path("audio.mp3"))
```

### 2. Extractors (`extractors/`)

Extract structured data from transcripts.

#### Available Extractors
| Extractor | Data Extracted |
|-----------|---------------|
| `sentiment` | Sentiment analysis (positive/negative/neutral) |
| `entity` | Named entities (people, organizations, locations) |
| `chapter` | Auto-detected chapters with headlines |
| `topic` | IAB content categories |
| `content_safety` | Flagged content categories |
| `highlights` | Key phrases and highlights |

#### Example
```python
from src.audio_processing.extractors.registry import get_registry

registry = get_registry()

# Extract all available data
all_extracted = registry.extract_all(transcript_data)

# Extract specific features
sentiment = registry.get("sentiment").extract(transcript_data)
entities = registry.get("entity").extract(transcript_data)
```

### 3. Chunkers (`chunking/`)

Split transcripts into manageable chunks.

#### Available Strategies
| Strategy | Description |
|----------|-------------|
| `speaker` | Chunk by speaker turns |
| `chapter` | Chunk by detected chapters |
| `semantic` | Chunk by paragraphs/topics |
| `sentence` | Chunk by sentences with overlap |
| `auto` | Auto-select best strategy |

#### Example
```python
from src.audio_processing.chunking.registry import get_global_registry

registry = get_global_registry()

# Auto-select best strategy
chunks = registry.chunk_with_best(transcript_data)

# Use specific strategy
speaker_chunks = registry.chunk_with("speaker", transcript_data)
chapter_chunks = registry.chunk_with("chapter", transcript_data)
```

### 4. Document Builder (`document/`)

Convert transcripts and chunks into Haystack Documents.

#### Example
```python
from src.audio_processing.document.builder import TranscriptDocumentBuilder

builder = TranscriptDocumentBuilder()

# Build from full transcript
documents = builder.build(
    transcript_data=transcript,
    extracted_data=extracted,
    source_name="audio.mp3",
)

# Build from chunks
documents = builder.build_from_chunks(
    chunks=chunks,
    transcript_data=transcript,
    extracted_data=extracted,
    source_name="audio.mp3",
)

# Build from utterances (speaker turns)
documents = builder.build_utterance_documents(
    transcript_data=transcript,
    extracted_data=extracted,
    source_name="audio.mp3",
)
```

### 5. Haystack Components (`components/`)

Pre-built Haystack 2.0 components for pipeline integration.

| Component | Input | Output |
|-----------|-------|--------|
| `AudioTranscriberComponent` | Audio paths | Transcripts |
| `DataExtractorComponent` | Transcripts | Extracted data |
| `ChunkerComponent` | Transcripts | Chunks |
| `DocumentConverterComponent` | Transcripts, chunks, extracted | Documents |

### 6. Pipeline Builder (`pipeline/`)

Fluent builder for constructing complete pipelines.

```python
from src.audio_processing.pipeline import AudioPipelineBuilder

# Full pipeline
pipeline = (
    AudioPipelineBuilder()
    .with_provider("assemblyai", api_key="...")
    .with_chunking("chapter")
    .with_extractors(["sentiment", "entities"])
    .build()
)

# Minimal pipeline (no extraction)
pipeline = (
    AudioPipelineBuilder()
    .with_provider("assemblyai")
    .without_extractor()
    .build()
)

# Extraction-only pipeline (no transcription)
pipeline = AudioPipelineBuilder.create_extraction_only(
    extractors=["sentiment", "topics"]
)
```

## Configuration

### Audio Configuration (`config/audio_config.yml`)

```yaml
transcription:
  language_code: en
  model: best
  punctuate: true
  format_text: true
  speaker_labels: true

features:
  speaker_labels: true
  sentiment_analysis: true
  entity_detection: true
  auto_chapters: true
  auto_highlights: true

chunking:
  max_chunk_length: 1000
  overlap: 100
  respect_speakers: true
```

### Provider Configuration (`config/providers.yml`)

```yaml
providers:
  assemblyai:
    name: assemblyai
    api_key_env: ASSEMBLYAI_API_KEY
    timeout: 300
    max_retries: 3
```

## Design Principles

The module follows these design principles:

1. **Single Responsibility**: Each component has one purpose
2. **Loose Coupling**: Components are connected via interfaces
3. **Extensibility**: New providers, extractors, chunkers can be added
4. **Registry Pattern**: Components registered and retrieved by name
5. **Factory Pattern**: Providers created via factory
6. **Builder Pattern**: Fluent interface for pipeline construction

## Testing

Run all audio processing tests:

```bash
uv run pytest tests/audio_processing/ -v
```

Run specific test modules:

```bash
# Transcription tests
uv run pytest tests/audio_processing/test_transcription_providers.py -v

# Extractor tests
uv run pytest tests/audio_processing/test_extractors.py -v

# Chunking tests
uv run pytest tests/audio_processing/test_chunking.py -v

# Pipeline tests
uv run pytest tests/audio_processing/test_pipeline.py -v

# Integration tests
uv run pytest tests/audio_processing/test_integration.py -v
```

## Module Structure

```
src/audio_processing/
├── __init__.py
├── config.py                 # Configuration dataclasses
├── exceptions.py             # Custom exceptions
├── interfaces.py             # Abstract base classes
├── transcription/
│   ├── providers/
│   │   ├── base.py          # Base provider
│   │   └── assemblyai.py    # AssemblyAI implementation
│   ├── factory.py           # Provider factory
│   └── orchestrator.py      # Workflow orchestration
├── extractors/
│   ├── base.py              # Base extractor
│   ├── sentiment.py         # Sentiment extractor
│   ├── entity.py            # Entity extractor
│   ├── chapter.py           # Chapter extractor
│   ├── topic.py             # Topic extractor
│   ├── content_safety.py    # Safety extractor
│   ├── highlights.py        # Highlights extractor
│   └── registry.py          # Extractor registry
├── chunking/
│   ├── base.py              # Base chunker
│   ├── speaker.py           # Speaker-based chunker
│   ├── chapter.py           # Chapter-based chunker
│   ├── semantic.py          # Semantic chunker
│   ├── sentence.py          # Sentence chunker
│   └── registry.py          # Chunker registry
├── document/
│   ├── metadata.py          # Metadata handling
│   └── builder.py           # Document builder
├── components/
│   ├── transcriber.py       # Haystack transcriber component
│   ├── extractor.py         # Haystack extractor component
│   ├── chunker.py           # Haystack chunker component
│   └── document_converter.py # Haystack converter component
└── pipeline/
    ├── __init__.py
    ├── builder.py           # Pipeline builder
    └── runner.py            # Pipeline runner
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ASSEMBLYAI_API_KEY` | API key for AssemblyAI provider |

## Error Handling

The module uses custom exceptions:

```python
from src.audio_processing.exceptions import (
    AudioProcessingError,
    TranscriptionError,
    ConfigurationError,
)

try:
    result = provider.transcribe(audio_path)
except TranscriptionError as e:
    logger.error("Transcription failed: %s", e)
except ConfigurationError as e:
    logger.error("Configuration invalid: %s", e)
```

## Contributing

When adding new components:

1. Create the component following existing patterns
2. Add comprehensive tests in `tests/audio_processing/`
3. Register in appropriate registry
4. Update documentation

### Adding a New Extractor

```python
from src.audio_processing.extractors.base import BaseExtractor

class CustomExtractor(BaseExtractor):
    def __init__(self):
        super().__init__(name="custom", data_key="custom_data")
    
    def is_available(self, data: Dict[str, Any]) -> bool:
        return "custom_data" in data
    
    def _do_extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw = data.get("custom_data", [])
        return {"items": raw, "count": len(raw)}
```

### Adding a New Chunker

```python
from src.audio_processing.chunking.base import BaseChunker, Chunk

class CustomChunker(BaseChunker):
    @property
    def strategy_name(self) -> str:
        return "custom"
    
    def chunk(self, transcript_data: Dict[str, Any]) -> List[Chunk]:
        # Implement chunking logic
        pass
```
