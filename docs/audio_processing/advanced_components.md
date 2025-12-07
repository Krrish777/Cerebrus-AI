# Audio Processing - Configuration & Advanced Components

## Table of Contents
1. [Configuration Management](#configuration-management)
2. [Chunking Strategies](#chunking-strategies)
3. [Feature Extractors](#feature-extractors)
4. [Pipeline Components](#pipeline-components)
5. [Complete Examples](#complete-examples)

---

## Configuration Management

### Overview

The `config.py` module provides comprehensive configuration management through 10 specialized dataclasses organized hierarchically.

### Configuration Architecture

```python
AudioProcessingConfig
├── TranscriptionConfig      # Core transcription settings
├── FeatureConfig            # Feature enablement
├── EnhancementConfig        # Audio enhancement
├── PrivacyConfig            # PII redaction
├── OutputConfig             # Output formats
├── SummarizationConfig      # Summarization settings
├── ChunkingConfig           # Chunking strategy
├── VocabularyConfig         # Custom vocabulary
└── ProviderConfig[]         # Multiple provider configs
```

### Core Configuration Classes

#### 1. TranscriptionConfig

```python
@dataclass
class TranscriptionConfig:
    """Core transcription settings."""
    language_code: str = "en"
    model: str = "best"
    polling_interval: float = 3.0
    punctuate: bool = True
    format_text: bool = True
    speaker_labels: bool = True
```

#### 2. FeatureConfig

```python
@dataclass
class FeatureConfig:
    """Feature enablement configuration."""
    speaker_labels: bool = True
    speakers_expected: Optional[int] = None
    sentiment_analysis: bool = True
    entity_detection: bool = True
    iab_categories: bool = True
    content_safety: bool = True
    content_safety_confidence: int = 80
    auto_highlights: bool = True
    auto_chapters: bool = True
    summarization: bool = True
```

#### 3. ChunkingConfig

```python
@dataclass
class ChunkingConfig:
    """Chunking strategy configuration."""
    max_chunk_length: int = 1000
    overlap: int = 100
    respect_speakers: bool = True
    respect_chapters: bool = True
    respect_sentences: bool = True
```

### Loading Configuration

#### From YAML

```python
from pathlib import Path
from src.audio_processing.config import AudioProcessingConfig

# Load from YAML file
config = AudioProcessingConfig.from_yaml(Path("audio_config.yaml"))

# YAML structure:
"""
transcription:
  language_code: "en"
  model: "best"
  polling_interval: 3.0

features:
  speaker_labels: true
  sentiment_analysis: true
  entity_detection: true
  iab_categories: true

enhancement:
  noise_reduction: true
  automatic_punctuation: true

privacy:
  redact_pii: false
  redact_pii_policies:
    - credit_card_number
    - email_address

chunking:
  max_chunk_length: 1000
  overlap: 100
  respect_speakers: true
"""
```

#### From Dictionary

```python
config_dict = {
    "transcription": {"model": "best", "language_code": "en"},
    "features": {"speaker_labels": True, "sentiment_analysis": True},
    "enhancement": {"noise_reduction": True},
    "chunking": {"max_chunk_length": 800}
}

config = AudioProcessingConfig.from_dict(config_dict)
```

#### Provider Configuration

```python
@dataclass
class ProviderConfig:
    """Configuration for a transcription provider."""
    name: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5
    supports: Dict[str, bool] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)

# Usage
assemblyai_config = ProviderConfig(
    name="assemblyai",
    api_key_env="ASSEMBLYAI_API_KEY",
    timeout=300,
    max_retries=3,
    supports={
        "speaker_diarization": True,
        "sentiment_analysis": True,
        "entity_detection": True
    },
    models=["best", "nano", "conformer-2"]
)
```

---

## Chunking Strategies

### Overview

The chunking system provides 6 specialized strategies for intelligent text splitting with context preservation.

### Base Architecture

```python
from src.audio_processing.chunking.base import BaseChunker, Chunk, ChunkerConfig

@dataclass
class Chunk:
    """Represents a chunk of transcript content."""
    text: str
    start_time: int              # milliseconds
    end_time: int                # milliseconds
    speaker: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> int:
        return self.end_time - self.start_time
    
    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0
```

### Chunking Strategies

#### 1. Speaker-Based Chunking

**File**: `chunking/speaker.py`

**Purpose**: Split by speaker changes (best for conversations)

```python
from src.audio_processing.chunking.speaker import SpeakerChunker

chunker = SpeakerChunker(
    config=ChunkerConfig(
        max_chunk_size=1000,
        min_chunk_size=100,
        overlap_size=50
    )
)

# Chunk by speaker boundaries
chunks = chunker.chunk(transcript_with_utterances)

for chunk in chunks:
    print(f"Speaker {chunk.speaker}: {chunk.text}")
    print(f"Duration: {chunk.duration_seconds}s")
```

**Use Cases**:
- Meeting transcriptions
- Podcast episodes
- Interviews
- Multi-speaker discussions

#### 2. Chapter-Based Chunking

**File**: `chunking/chapter.py`

**Purpose**: Split by auto-generated chapters

```python
from src.audio_processing.chunking.chapter import ChapterChunker

chunker = ChapterChunker()

# Chunk by chapters (requires auto_chapters=True in config)
chunks = chunker.chunk(transcript_with_chapters)

for chunk in chunks:
    print(f"Chapter: {chunk.metadata.get('chapter_headline')}")
    print(f"Text: {chunk.text}")
```

**Use Cases**:
- Long-form content
- Structured presentations
- Educational videos
- Tutorial series

#### 3. Semantic Chunking

**File**: `chunking/semantic.py`

**Purpose**: Split by semantic similarity

```python
from src.audio_processing.chunking.semantic import SemanticChunker

chunker = SemanticChunker(
    model="sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold=0.7
)

# Chunk by semantic boundaries
chunks = chunker.chunk(transcript)

# Chunks are semantically coherent
for chunk in chunks:
    print(f"Topic cluster: {chunk.text[:100]}...")
```

**Use Cases**:
- Unstructured content
- Topic-based segmentation
- Content without clear breaks
- General-purpose chunking

#### 4. Sentence-Based Chunking

**File**: `chunking/sentence.py`

**Purpose**: Split by sentences with max length

```python
from src.audio_processing.chunking.sentence import SentenceChunker

chunker = SentenceChunker(
    config=ChunkerConfig(
        max_chunk_size=800,
        overlap_size=100,
        preserve_sentences=True
    )
)

# Chunk by sentences (never breaks mid-sentence)
chunks = chunker.chunk(transcript)
```

**Use Cases**:
- Simple baseline chunking
- When other strategies unavailable
- Preserving sentence integrity
- General text processing

### Registry Pattern

```python
from src.audio_processing.chunking.registry import ChunkerRegistry

# Register custom chunker
@ChunkerRegistry.register("custom")
class CustomChunker(BaseChunker):
    def chunk(self, transcript):
        # Custom chunking logic
        pass

# Get chunker by name
chunker = ChunkerRegistry.get("speaker")(config=ChunkerConfig())
chunks = chunker.chunk(transcript)
```

### Complete Chunking Example

```python
from haystack import Pipeline
from src.audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
from src.audio_processing.chunking.speaker import SpeakerChunker
from src.audio_processing.chunking.registry import ChunkerRegistry

# 1. Transcribe with speaker labels
transcriber = AssemblyAITranscriber(
    config=AudioProcessingConfig(speaker_labels=True)
)
result = transcriber.run(sources=["meeting.mp3"])

# 2. Get transcript with speaker data
transcript_doc = result["documents"][0]
speaker_utterances = transcript_doc.meta.get("speaker_utterances", [])

# 3. Chunk by speakers
chunker = SpeakerChunker()
chunks = chunker.chunk_utterances(speaker_utterances)

# 4. Process chunks
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"  Speaker: {chunk.speaker}")
    print(f"  Time: {chunk.start_time/1000:.1f}s - {chunk.end_time/1000:.1f}s")
    print(f"  Text: {chunk.text[:100]}...")
    
    # Convert to Document for RAG
    doc = Document(
        content=chunk.text,
        meta={
            "speaker": chunk.speaker,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "chunk_index": i
        }
    )
```

---

## Feature Extractors

### Overview

The extractors system provides 8 specialized extractors for feature extraction from transcripts.

### Base Extractor

```python
from src.audio_processing.extractors.base import BaseExtractor

class BaseExtractor(ABC):
    """Abstract base class for all extractors."""
    
    @abstractmethod
    def extract(self, transcript: Any) -> List[Dict[str, Any]]:
        """Extract features from transcript."""
        pass
```

### Available Extractors

#### 1. Sentiment Extractor

**File**: `extractors/sentiment.py`

```python
from src.audio_processing.extractors.sentiment import SentimentExtractor

extractor = SentimentExtractor()
sentiment_data = extractor.extract(transcript)

# Output structure:
[
    {
        "text": "This product is amazing!",
        "sentiment": "POSITIVE",
        "confidence": 0.98,
        "start_time": 1000,
        "end_time": 3000
    },
    {
        "text": "However, the price is high.",
        "sentiment": "NEGATIVE",
        "confidence": 0.85,
        "start_time": 3100,
        "end_time": 5500
    }
]
```

#### 2. Entity Extractor

**File**: `extractors/entity.py`

```python
from src.audio_processing.extractors.entity import EntityExtractor

extractor = EntityExtractor()
entities = extractor.extract(transcript)

# Extracts: person_name, location, organization, date, etc.
[
    {"text": "New York", "entity_type": "location", "start_time": 5000},
    {"text": "Microsoft", "entity_type": "organization", "start_time": 10000}
]
```

#### 3. Topic Extractor

**File**: `extractors/topic.py`

```python
from src.audio_processing.extractors.topic import TopicExtractor

extractor = TopicExtractor()
topics = extractor.extract(transcript)

# IAB topic classification
[
    {"topic": "Technology>Artificial Intelligence", "relevance": 0.92},
    {"topic": "Business>Cloud Computing", "relevance": 0.85}
]
```

#### 4. Content Safety Extractor

**File**: `extractors/content_safety.py`

```python
from src.audio_processing.extractors.content_safety import ContentSafetyExtractor

extractor = ContentSafetyExtractor(confidence_threshold=80)
safety_labels = extractor.extract(transcript)

# Flags: violence, hate_speech, profanity, sexual_content, etc.
[
    {
        "label": "profanity",
        "confidence": 0.95,
        "text": "...",
        "timestamp": 15000,
        "severity": "high"
    }
]
```

#### 5. Highlights Extractor

**File**: `extractors/highlights.py`

```python
from src.audio_processing.extractors.highlights import HighlightsExtractor

extractor = HighlightsExtractor()
highlights = extractor.extract(transcript)

# Key moments
[
    {
        "text": "The main finding of our research is...",
        "start_time": 30000,
        "end_time": 45000,
        "rank": 0.95  # Importance score
    }
]
```

#### 6. Chapter Extractor

**File**: `extractors/chapter.py`

```python
from src.audio_processing.extractors.chapter import ChapterExtractor

extractor = ChapterExtractor()
chapters = extractor.extract(transcript)

# Auto-generated chapters
[
    {
        "headline": "Introduction",
        "summary": "Overview of the topic",
        "gist": "Introduction and agenda",
        "start_time": 0,
        "end_time": 60000
    }
]
```

### Extractor Registry

```python
from src.audio_processing.extractors.registry import ExtractorRegistry

# Register custom extractor
@ExtractorRegistry.register("custom")
class CustomExtractor(BaseExtractor):
    def extract(self, transcript):
        # Custom extraction logic
        return []

# Get extractor by name
extractor = ExtractorRegistry.get("sentiment")()
features = extractor.extract(transcript)
```

### Complete Extraction Example

```python
from src.audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
from src.audio_processing.extractors.registry import ExtractorRegistry

# 1. Transcribe with all features
transcriber = AssemblyAITranscriber(
    config=AudioProcessingConfig(
        sentiment_analysis=True,
        entity_detection=True,
        iab_categories=True,
        content_safety=True,
        auto_highlights=True,
        auto_chapters=True
    )
)

result = transcriber.run(sources=["content.mp3"])
doc = result["documents"][0]

# 2. Extract all features (already in metadata)
sentiment_data = doc.meta.get("sentiment_analysis", [])
entities = doc.meta.get("entities", [])
topics = doc.meta.get("topics", [])
safety_labels = doc.meta.get("content_safety", [])
highlights = doc.meta.get("highlights", [])
chapters = doc.meta.get("chapters", [])

# 3. Analyze extracted features
print(f"Sentiment Distribution:")
sentiments = {}
for item in sentiment_data:
    sent = item["sentiment"]
    sentiments[sent] = sentiments.get(sent, 0) + 1
print(sentiments)  # {'POSITIVE': 45, 'NEUTRAL': 30, 'NEGATIVE': 12}

print(f"\nEntity Types:")
entity_types = {}
for entity in entities:
    etype = entity["entity_type"]
    entity_types[etype] = entity_types.get(etype, 0) + 1
print(entity_types)  # {'person_name': 15, 'location': 8, 'organization': 6}

print(f"\nTop Topics:")
top_topics = sorted(topics, key=lambda x: x["relevance"], reverse=True)[:5]
for topic in top_topics:
    print(f"  {topic['topic']}: {topic['relevance']:.2%}")
```

---

## Pipeline Components

### Haystack Components

The `components/` directory provides Haystack-compatible components for pipeline integration.

#### 1. TranscriberComponent

**File**: `components/transcriber.py`

```python
from haystack import component, Document
from src.audio_processing.components.transcriber import TranscriberComponent

@component
class TranscriberComponent:
    """Haystack component for audio transcription."""
    
    @component.output_types(documents=List[Document])
    def run(self, audio_paths: List[str]) -> Dict[str, List[Document]]:
        # Transcription logic
        pass

# Usage in pipeline
from haystack import Pipeline

pipeline = Pipeline()
pipeline.add_component("transcriber", TranscriberComponent(api_key="..."))
```

#### 2. ChunkerComponent

**File**: `components/chunker.py`

```python
from src.audio_processing.components.chunker import ChunkerComponent

@component
class ChunkerComponent:
    """Haystack component for intelligent chunking."""
    
    def __init__(self, strategy: str = "speaker"):
        self.strategy = strategy
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        # Chunking logic
        pass
```

#### 3. ExtractorComponent

**File**: `components/extractor.py`

```python
from src.audio_processing.components.extractor import ExtractorComponent

@component
class ExtractorComponent:
    """Haystack component for feature extraction."""
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        # Extraction logic
        pass
```

#### 4. DocumentConverterComponent

**File**: `components/document_converter.py`

```python
from src.audio_processing.components.document_converter import DocumentConverterComponent

@component
class DocumentConverterComponent:
    """Convert audio transcripts to structured documents."""
    
    @component.output_types(documents=List[Document])
    def run(self, transcripts: List[Dict]) -> Dict[str, List[Document]]:
        # Conversion logic
        pass
```

---

## Complete Examples

### Example 1: Full RAG Pipeline with Audio

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from src.audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
from src.audio_processing.chunking.speaker import SpeakerChunker
from src.vector_database.qdrant_db import QdrantDocumentWriter

# Build complete pipeline
pipeline = Pipeline()

# 1. Transcribe
transcriber = AssemblyAITranscriber(
    config=AudioProcessingConfig(
        speaker_labels=True,
        sentiment_analysis=True,
        entity_detection=True,
        iab_categories=True,
        auto_highlights=True
    )
)
pipeline.add_component("transcriber", transcriber)

# 2. Chunk by speakers
from src.audio_processing.audio_transcriber import SmartAudioProcessor
processor = SmartAudioProcessor(
    chunk_strategy="speaker",
    max_chunk_length=800,
    preserve_context=True
)
pipeline.add_component("processor", processor)

# 3. Embed
embedder = SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")
pipeline.add_component("embedder", embedder)

# 4. Store
writer = QdrantDocumentWriter(collection_name="audio_transcripts")
pipeline.add_component("writer", writer)

# Connect
pipeline.connect("transcriber.documents", "processor.documents")
pipeline.connect("processor.documents", "embedder.documents")
pipeline.connect("embedder.documents", "writer.documents")

# Run
audio_files = ["meeting1.mp3", "meeting2.mp3", "meeting3.mp3"]
result = pipeline.run({
    "transcriber": {"sources": audio_files}
})

print(f"Processed {len(audio_files)} audio files")
print(f"Stored {len(result['writer']['documents_written'])} chunks")
```

### Example 2: YouTube Channel Archive with RAG

```python
import yt_dlp
from src.audio_processing.yt_audio_transcriber import YouTubeAudioTranscriber
from src.vector_database.qdrant_db import QdrantDocumentWriter

def get_channel_videos(channel_url: str, max_videos: int = 50):
    """Get recent videos from YouTube channel."""
    ydl_opts = {'quiet': True, 'extract_flat': True, 'playlistend': max_videos}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        return [f"https://youtube.com/watch?v={e['id']}" for e in info['entries'] if e]

# Get channel videos
channel_url = "https://www.youtube.com/@ChannelName/videos"
video_urls = get_channel_videos(channel_url, max_videos=100)

# Transcribe all
transcriber = YouTubeAudioTranscriber(
    cache_audio=True,
    max_duration=3600  # 1 hour limit
)

all_documents = []
for i, url in enumerate(video_urls, 1):
    print(f"Processing {i}/{len(video_urls)}: {url}")
    try:
        result = transcriber.run(sources=[url])
        all_documents.extend(result["documents"])
    except Exception as e:
        print(f"Failed: {e}")

# Store in vector database
writer = QdrantDocumentWriter(collection_name="youtube_channel")
writer.run(documents=all_documents)

print(f"\n✅ Complete! Stored {len(all_documents)} documents")
```

### Example 3: Multi-Feature Analysis Dashboard

```python
from src.audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
import pandas as pd
import json

# Transcribe with all features
transcriber = AssemblyAITranscriber(
    config=AudioProcessingConfig(
        speaker_labels=True,
        sentiment_analysis=True,
        entity_detection=True,
        iab_categories=True,
        content_safety=True,
        auto_highlights=True,
        auto_chapters=True
    )
)

result = transcriber.run(sources=["content.mp3"])
doc = result["documents"][0]

# Extract all analysis
analysis = {
    "video_title": doc.meta.get("source"),
    "duration_seconds": doc.meta.get("audio_duration_seconds"),
    "confidence": doc.meta.get("confidence"),
    
    # Sentiment analysis
    "sentiment": {
        "positive": len([s for s in doc.meta.get("sentiment_analysis", []) if s["sentiment"] == "POSITIVE"]),
        "neutral": len([s for s in doc.meta.get("sentiment_analysis", []) if s["sentiment"] == "NEUTRAL"]),
        "negative": len([s for s in doc.meta.get("sentiment_analysis", []) if s["sentiment"] == "NEGATIVE"]),
    },
    
    # Entity analysis
    "entities": {},
    
    # Topic analysis
    "top_topics": [],
    
    # Safety
    "content_flags": len(doc.meta.get("content_safety", [])),
    
    # Structure
    "num_chapters": len(doc.meta.get("chapters", [])),
    "num_highlights": len(doc.meta.get("highlights", []))
}

# Count entity types
for entity in doc.meta.get("entities", []):
    etype = entity["entity_type"]
    analysis["entities"][etype] = analysis["entities"].get(etype, 0) + 1

# Get top topics
topics = doc.meta.get("topics", [])
top_topics = sorted(topics, key=lambda x: x["relevance"], reverse=True)[:5]
analysis["top_topics"] = [{"topic": t["topic"], "relevance": t["relevance"]} for t in top_topics]

# Save analysis
with open("content_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

print("Analysis complete!")
print(json.dumps(analysis, indent=2))
```

### Example 4: Batch Processing with Progress Tracking

```python
from pathlib import Path
from tqdm import tqdm
import json
from src.audio_processing.audio_transcriber import AssemblyAITranscriber

# Find all audio files
audio_dir = Path("audio_archive")
audio_files = list(audio_dir.glob("**/*.mp3"))

# Setup transcriber
transcriber = AssemblyAITranscriber()

# Progress tracking
checkpoint_file = "transcription_progress.json"
completed = set()
if Path(checkpoint_file).exists():
    with open(checkpoint_file) as f:
        completed = set(json.load(f))

# Process with progress bar
for audio_file in tqdm(audio_files, desc="Transcribing"):
    if str(audio_file) in completed:
        continue
    
    try:
        # Transcribe
        result = transcriber.run(sources=[audio_file])
        
        # Save transcript
        output_file = audio_file.with_suffix('.txt')
        output_file.write_text(result["documents"][0].content)
        
        # Update checkpoint
        completed.add(str(audio_file))
        with open(checkpoint_file, 'w') as f:
            json.dump(list(completed), f)
            
    except Exception as e:
        print(f"\nFailed {audio_file}: {e}")

print(f"\n✅ Complete! Transcribed {len(completed)}/{len(audio_files)} files")
```

---

## See Also

- [overview.md](overview.md) - Audio processing module overview
- [audio_transcriber.md](audio_transcriber.md) - Core AssemblyAI transcription
- [yt_audio_transcriber.md](yt_audio_transcriber.md) - YouTube transcription
- [../document_processing/overview.md](../document_processing/overview.md) - Document processing integration
- [../embeddings/overview.md](../embeddings/overview.md) - Embeddings generation
- [../vector_database/overview.md](../vector_database/overview.md) - Vector storage
- [../rag/overview.md](../rag/overview.md) - RAG system integration
