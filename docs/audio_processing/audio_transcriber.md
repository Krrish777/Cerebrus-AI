# audio_transcriber.py - AssemblyAI Transcription Component

## Table of Contents
1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [AudioProcessingConfig](#audioprocessingconfig)
4. [AssemblyAITranscriber Component](#assemblyaitranscriber-component)
5. [SmartAudioProcessor](#smartaudioprocessor)
6. [Quick Start Examples](#quick-start-examples)
7. [Advanced Features](#advanced-features)
8. [Configuration Guide](#configuration-guide)
9. [Integration Patterns](#integration-patterns)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**Purpose**: Provides comprehensive AssemblyAI transcription capabilities as a Haystack component with advanced features including speaker diarization, sentiment analysis, entity detection, and intelligent content processing.

**File**: `src/audio_processing/audio_transcriber.py` (976 lines)

**Key Features**:
- 🎯 **Haystack @component decorator** for pipeline integration
- 🎤 **Speaker diarization** with automatic speaker detection
- 💭 **Sentiment analysis** on transcript segments
- 🏷️ **Entity detection** (people, locations, organizations)
- 📊 **Topic classification** (IAB categories)
- 🛡️ **Content safety labeling** for moderation
- ✨ **Auto-highlights** extraction
- 📖 **Auto-chapters** generation
- 📝 **AI summarization** (bullets, gist, headline, paragraph)
- 🔒 **PII redaction** for privacy
- 🧠 **Smart chunking** with context awareness

**Dependencies**:
```python
import assemblyai as aai  # Primary transcription SDK
from haystack import component, Document
from src.core.logging import CustomLogger
```

---

## Core Classes

### Class Hierarchy

```
AudioProcessingConfig (dataclass)
  └─ Configuration for all AssemblyAI features

AssemblyAITranscriber (@component)
  ├─ Main transcription component
  └─ Integrates with Haystack pipelines

SmartAudioProcessor (@component)
  └─ Intelligent chunking with context
```

---

## AudioProcessingConfig

**Purpose**: Comprehensive configuration dataclass for all AssemblyAI transcription features.

### Definition

```python
@dataclass
class AudioProcessingConfig:
    """Configuration for AssemblyAI audio processing features."""
    
    # Core transcription settings (3 attributes)
    language_code: Optional[str] = "en"
    model: str = "best"  # 'best', 'nano', 'conformer-2'
    
    # Speaker features (2 attributes)
    speaker_labels: bool = True
    speakers_expected: Optional[int] = None
    
    # Content analysis (7 attributes)
    sentiment_analysis: bool = True
    entity_detection: bool = True
    iab_categories: bool = True  # Topic detection
    content_safety: bool = True
    content_safety_confidence: int = 80
    auto_highlights: bool = True
    
    # Audio enhancement (4 attributes)
    noise_reduction: bool = True
    automatic_punctuation: bool = True
    format_text: bool = True
    filter_profanity: bool = False
    
    # Privacy and redaction (3 attributes)
    redact_pii: bool = False
    redact_pii_policies: List[str] = field(default_factory=lambda: [
        "credit_card_number", "email_address", "person_name", "phone_number"
    ])
    redact_pii_audio: bool = False
    
    # Advanced features (4 attributes)
    custom_spelling: Dict[str, List[str]] = field(default_factory=dict)
    custom_vocabulary: List[str] = field(default_factory=list)
    boost_param: str = "low"  # 'low', 'default', 'high'
    
    # Output formats (7 attributes)
    include_utterances: bool = True
    include_sentences: bool = True
    include_paragraphs: bool = True
    auto_chapters: bool = True
    summarization: bool = True
    summary_model: str = "informative"  # 'informative', 'conversational', 'catchy'
    summary_type: str = "bullets"  # 'bullets', 'gist', 'headline', 'paragraph'
```

**Total Attributes**: 30+ configuration options

### Configuration Attributes Explained

#### Core Transcription Settings
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `language_code` | `Optional[str]` | `"en"` | ISO language code (e.g., "en", "es", "fr") |
| `model` | `str` | `"best"` | Model quality: "best", "nano" (fast), "conformer-2" (legacy) |

#### Speaker Features
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `speaker_labels` | `bool` | `True` | Enable speaker diarization |
| `speakers_expected` | `Optional[int]` | `None` | Hint for expected number of speakers |

#### Content Analysis
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentiment_analysis` | `bool` | `True` | Sentiment per sentence (positive/neutral/negative) |
| `entity_detection` | `bool` | `True` | Extract entities (person, location, organization) |
| `iab_categories` | `bool` | `True` | Topic classification (700+ IAB categories) |
| `content_safety` | `bool` | `True` | Label sensitive content (violence, hate, etc.) |
| `content_safety_confidence` | `int` | `80` | Confidence threshold (0-100) |
| `auto_highlights` | `bool` | `True` | Extract key points automatically |

#### Audio Enhancement
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_reduction` | `bool` | `True` | Reduce background noise |
| `automatic_punctuation` | `bool` | `True` | Add punctuation automatically |
| `format_text` | `bool` | `True` | Format numbers, dates, times |
| `filter_profanity` | `bool` | `False` | Replace profanity with asterisks |

#### Privacy and Redaction
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `redact_pii` | `bool` | `False` | Enable PII redaction |
| `redact_pii_policies` | `List[str]` | `["credit_card_number", ...]` | PII types to redact |
| `redact_pii_audio` | `bool` | `False` | Redact PII from audio file |

#### Output Formats
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_utterances` | `bool` | `True` | Include speaker utterances |
| `include_sentences` | `bool` | `True` | Include sentence-level data |
| `include_paragraphs` | `bool` | `True` | Include paragraph-level data |
| `auto_chapters` | `bool` | `True` | Generate chapters automatically |
| `summarization` | `bool` | `True` | Generate AI summary |
| `summary_model` | `str` | `"informative"` | Summary style |
| `summary_type` | `str` | `"bullets"` | Summary format |

---

## AssemblyAITranscriber Component

**Purpose**: Main Haystack component for AssemblyAI transcription with full feature support.

### Class Definition

```python
@component
class AssemblyAITranscriber:
    """
    A comprehensive Haystack component for AssemblyAI speech-to-text transcription.
    
    This component provides full access to AssemblyAI's advanced features including
    speaker diarization, content analysis, sentiment analysis, and more.
    """
```

### Constructor

```python
def __init__(
    self,
    api_key: Optional[str] = None,
    config: Optional[AudioProcessingConfig] = None,
    polling_interval: float = 3.0
):
    """
    Initialize the AssemblyAI Transcriber component.
    
    :param api_key: AssemblyAI API key. If None, uses ASSEMBLYAI_API_KEY env var
    :param config: Audio processing configuration
    :param polling_interval: Polling interval for checking transcription status
    """
```

**Parameters**:
- `api_key`: AssemblyAI API key (optional, can use env var `ASSEMBLYAI_API_KEY`)
- `config`: `AudioProcessingConfig` instance (optional, uses defaults if None)
- `polling_interval`: Float seconds between status checks (default: 3.0)

**Raises**:
- `ImportError`: If `assemblyai` package not installed
- `ValueError`: If no API key provided (via parameter or env var)

### Core Methods

#### 1. `run()` - Main Transcription Method

```python
@component.output_types(documents=List[Document])
def run(
    self, 
    sources: List[Union[str, Path, bytes]]
) -> Dict[str, List[Document]]:
    """
    Transcribe audio files or URLs using AssemblyAI.
    
    :param sources: List of audio file paths, URLs, or bytes
    :return: Dictionary with 'documents' key containing transcribed documents
    """
```

**Parameters**:
- `sources`: List of audio sources supporting:
  - Local file paths (`Path` or `str`)
  - Remote URLs (`str` starting with http/https)
  - Raw bytes (`bytes` data)

**Returns**: Dictionary with structure:
```python
{
    "documents": [
        Document(
            content="# Transcription: file.mp3\n## Full Transcript\n...",
            meta={
                "source": "file.mp3",
                "transcript_id": "abc123...",
                "audio_duration_seconds": 120.5,
                "confidence": 0.95,
                "sentiment_analysis": [...],
                "entities": [...],
                "topics": [...],
                # ... more structured data
            }
        ),
        # ... more documents
    ]
}
```

#### 2. `_create_transcription_config()` - Config Builder

```python
def _create_transcription_config(self):
    """Create AssemblyAI TranscriptionConfig from our config."""
```

**Purpose**: Converts `AudioProcessingConfig` to AssemblyAI SDK's `TranscriptionConfig`.

**Handles**:
- API version compatibility
- Feature availability checks
- Fallback to property setting if constructor fails
- Logging of all settings

#### 3. `_add_structured_documents()` - Document Generator

```python
def _add_structured_documents(
    self,
    documents: List[Document],
    transcript: Any,
    source_name: str
) -> None:
    """
    Add structured documents for each feature analysis result.
    
    Creates separate documents for:
    - Speaker utterances
    - Sentiment analysis results
    - Entity detections
    - Topic classifications
    - Content safety labels
    - Auto highlights
    - Auto chapters
    """
```

**Purpose**: Creates additional Haystack `Document` objects for each feature, enabling independent retrieval and processing.

#### 4. Feature Extraction Methods

All extract structured data from AssemblyAI responses:

```python
def _extract_sentiment_data(self, sentiment_results) -> List[Dict]:
    """Extract sentiment analysis into structured format."""

def _extract_entity_data(self, entities) -> List[Dict]:
    """Extract entity detection into structured format."""

def _extract_topic_data(self, topics) -> List[Dict]:
    """Extract IAB topic classification into structured format."""

def _extract_content_safety_data(self, safety_labels) -> List[Dict]:
    """Extract content safety labels into structured format."""

def _extract_highlights_data(self, highlights) -> List[Dict]:
    """Extract auto-highlights into structured format."""
```

---

## SmartAudioProcessor

**Purpose**: Intelligent audio chunking component that creates context-aware chunks using speaker boundaries, chapters, or semantic segmentation.

### Class Definition

```python
@component
class SmartAudioProcessor:
    """
    Smart audio processing component that creates intelligent chunks
    from transcribed audio based on speakers, chapters, or semantic boundaries.
    """
```

### Constructor

```python
def __init__(
    self,
    chunk_strategy: str = "speaker",  # 'speaker', 'chapter', 'semantic', 'sentence'
    max_chunk_length: int = 1000,
    overlap: int = 100,
    preserve_context: bool = True
):
    """
    Initialize the Smart Audio Processor.
    
    :param chunk_strategy: Chunking strategy to use
    :param max_chunk_length: Maximum characters per chunk
    :param overlap: Characters to overlap between chunks
    :param preserve_context: Whether to preserve speaker/chapter context
    """
```

**Parameters**:
- `chunk_strategy`: One of:
  - `"speaker"`: Split by speaker changes (best for conversations)
  - `"chapter"`: Split by auto-generated chapters (best for structured content)
  - `"semantic"`: Split by semantic boundaries (best for varied content)
  - `"sentence"`: Split by sentences (simple baseline)
- `max_chunk_length`: Maximum characters per chunk (default: 1000)
- `overlap`: Overlap characters for context continuity (default: 100)
- `preserve_context`: Add speaker/chapter metadata to chunks (default: True)

### Core Methods

#### 1. `run()` - Main Processing Method

```python
@component.output_types(documents=List[Document])
def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
    """
    Process transcribed documents into smart chunks.
    
    :param documents: Documents from AssemblyAITranscriber
    :return: Dictionary with 'documents' key containing chunked documents
    """
```

**Input**: Documents from `AssemblyAITranscriber` containing metadata with:
- `utterances` (for speaker-based chunking)
- `chapters` (for chapter-based chunking)
- `sentences` (for semantic/sentence chunking)

**Output**: New documents representing intelligent chunks with preserved metadata.

#### 2. Chunking Strategy Methods

```python
def _chunk_by_speakers(
    self, 
    transcript: Any, 
    base_metadata: Dict
) -> List[Document]:
    """Create chunks based on speaker boundaries."""

def _chunk_by_chapters(
    self, 
    transcript: Any, 
    base_metadata: Dict
) -> List[Document]:
    """Create chunks based on auto-generated chapters."""

def _chunk_by_semantic_boundaries(
    self, 
    transcript: Any, 
    base_metadata: Dict
) -> List[Document]:
    """Create chunks based on semantic similarity."""
```

---

## Quick Start Examples

### Example 1: Basic Transcription

```python
from pathlib import Path
from haystack import Pipeline
from src.audio_processing.audio_transcriber import (
    AssemblyAITranscriber, AudioProcessingConfig
)

# Create basic configuration
config = AudioProcessingConfig(
    speaker_labels=True,
    sentiment_analysis=True,
    entity_detection=True
)

# Initialize transcriber
transcriber = AssemblyAITranscriber(
    api_key="your_api_key",  # Or set ASSEMBLYAI_API_KEY env var
    config=config
)

# Transcribe a file
result = transcriber.run(sources=[Path("podcast.mp3")])
documents = result["documents"]

# Access results
for doc in documents:
    print(f"Transcription: {doc.content[:200]}")
    print(f"Metadata: {doc.meta}")
```

### Example 2: Advanced Features with All Options

```python
from src.audio_processing.audio_transcriber import (
    AssemblyAITranscriber, AudioProcessingConfig
)

# Enable all advanced features
config = AudioProcessingConfig(
    # Core
    language_code="en",
    model="best",
    
    # Speaker analysis
    speaker_labels=True,
    speakers_expected=2,
    
    # Content analysis
    sentiment_analysis=True,
    entity_detection=True,
    iab_categories=True,
    content_safety=True,
    content_safety_confidence=75,
    auto_highlights=True,
    
    # Enhancement
    noise_reduction=True,
    automatic_punctuation=True,
    format_text=True,
    filter_profanity=False,
    
    # Privacy
    redact_pii=True,
    redact_pii_policies=[
        "credit_card_number",
        "email_address",
        "phone_number",
        "person_name"
    ],
    redact_pii_audio=True,
    
    # Custom vocabulary
    custom_vocabulary=["Kubernetes", "Docker", "AssemblyAI"],
    boost_param="high",
    
    # Output
    include_utterances=True,
    include_sentences=True,
    auto_chapters=True,
    summarization=True,
    summary_model="informative",
    summary_type="bullets"
)

transcriber = AssemblyAITranscriber(
    config=config,
    polling_interval=2.0
)

# Transcribe with all features
result = transcriber.run(sources=["interview.mp3"])
doc = result["documents"][0]

# Access structured features
print("Sentiment:", doc.meta.get("sentiment_analysis"))
print("Entities:", doc.meta.get("entities"))
print("Topics:", doc.meta.get("topics"))
print("Safety:", doc.meta.get("content_safety"))
print("Highlights:", doc.meta.get("highlights"))
```

### Example 3: Smart Chunking Pipeline

```python
from haystack import Pipeline
from src.audio_processing.audio_transcriber import (
    AssemblyAITranscriber,
    SmartAudioProcessor,
    AudioProcessingConfig
)

# Create pipeline with smart chunking
pipeline = Pipeline()

# Add transcriber
transcriber = AssemblyAITranscriber(
    config=AudioProcessingConfig(
        speaker_labels=True,
        auto_chapters=True,
        sentiment_analysis=True
    )
)
pipeline.add_component("transcriber", transcriber)

# Add smart processor
processor = SmartAudioProcessor(
    chunk_strategy="speaker",  # Chunk by speaker changes
    max_chunk_length=800,
    overlap=100,
    preserve_context=True
)
pipeline.add_component("processor", processor)

# Connect components
pipeline.connect("transcriber.documents", "processor.documents")

# Run pipeline
result = pipeline.run({
    "transcriber": {"sources": ["meeting.mp3"]}
})

# Get smart chunks
chunks = result["processor"]["documents"]
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Speaker: {chunk.meta.get('speaker', 'N/A')}")
    print(f"  Text: {chunk.content[:100]}...")
    print(f"  Time: {chunk.meta.get('start_time')}ms - {chunk.meta.get('end_time')}ms")
```

### Example 4: Multiple Source Types

```python
from pathlib import Path
from src.audio_processing.audio_transcriber import AssemblyAITranscriber

transcriber = AssemblyAITranscriber()

# Mix of source types
sources = [
    Path("local_file.mp3"),                          # Local file
    "https://example.com/remote_audio.wav",           # Remote URL
    Path("another_file.wav"),                         # Another local file
]

result = transcriber.run(sources=sources)

# Each source becomes a document
for doc in result["documents"]:
    source_name = doc.meta["source"]
    duration = doc.meta["audio_duration_seconds"]
    print(f"{source_name}: {duration}s")
```

### Example 5: Content Safety Filtering

```python
from src.audio_processing.audio_transcriber import (
    AssemblyAITranscriber, AudioProcessingConfig
)

# Configure content safety
config = AudioProcessingConfig(
    content_safety=True,
    content_safety_confidence=85,  # Higher threshold = stricter
    filter_profanity=True
)

transcriber = AssemblyAITranscriber(config=config)
result = transcriber.run(sources=["user_content.mp3"])

doc = result["documents"][0]

# Check for flagged content
safety_labels = doc.meta.get("content_safety", [])
for label in safety_labels:
    if label["confidence"] > 0.85:
        print(f"⚠️  Flagged: {label['label']} (confidence: {label['confidence']})")
        print(f"   Text: {label['text']}")
        print(f"   Time: {label['timestamp']}ms")
```

### Example 6: Entity-Based Analysis

```python
from src.audio_processing.audio_transcriber import (
    AssemblyAITranscriber, AudioProcessingConfig
)

config = AudioProcessingConfig(
    entity_detection=True,
    iab_categories=True
)

transcriber = AssemblyAITranscriber(config=config)
result = transcriber.run(sources=["news_segment.mp3"])

doc = result["documents"][0]

# Analyze entities
entities = doc.meta.get("entities", [])
entity_types = {}
for entity in entities:
    entity_type = entity["entity_type"]
    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

print("Entity Distribution:")
for entity_type, count in entity_types.items():
    print(f"  {entity_type}: {count}")

# Analyze topics
topics = doc.meta.get("topics", [])
top_topics = sorted(topics, key=lambda x: x["relevance"], reverse=True)[:5]
print("\nTop 5 Topics:")
for topic in top_topics:
    print(f"  {topic['topic']}: {topic['relevance']:.2%}")
```

### Example 7: Custom Vocabulary for Technical Content

```python
from src.audio_processing.audio_transcriber import (
    AssemblyAITranscriber, AudioProcessingConfig
)

# Define technical terms
tech_vocab = [
    "Kubernetes", "Docker", "PostgreSQL", "Redis",
    "AssemblyAI", "FastAPI", "Pydantic", "SQLAlchemy"
]

custom_spelling = {
    "kates": ["Kubernetes"],
    "docker": ["Docker"],
    "post gray sequel": ["PostgreSQL"]
}

config = AudioProcessingConfig(
    custom_vocabulary=tech_vocab,
    custom_spelling=custom_spelling,
    boost_param="high",  # High boost for better recognition
    format_text=True
)

transcriber = AssemblyAITranscriber(config=config)
result = transcriber.run(sources=["tech_talk.mp3"])

# Vocabulary boost ensures technical terms are recognized correctly
doc = result["documents"][0]
print(doc.content)  # Will have correct technical terms
```

---

## Advanced Features

### 1. Speaker Diarization

**What it does**: Identifies who spoke when and labels each utterance.

**Configuration**:
```python
config = AudioProcessingConfig(
    speaker_labels=True,
    speakers_expected=3  # Optional hint
)
```

**Output Metadata**:
```python
{
    "speaker_utterances": [
        {
            "speaker": "A",
            "text": "Hello, how are you?",
            "start_time": 0,
            "end_time": 2000,
            "confidence": 0.95
        },
        {
            "speaker": "B",
            "text": "I'm doing great, thanks!",
            "start_time": 2100,
            "end_time": 4500,
            "confidence": 0.92
        }
    ]
}
```

**Use Cases**:
- Meeting transcription
- Interview analysis
- Podcast processing
- Multi-speaker conversation analysis

### 2. Sentiment Analysis

**What it does**: Labels each sentence with sentiment (positive/neutral/negative).

**Configuration**:
```python
config = AudioProcessingConfig(
    sentiment_analysis=True
)
```

**Output Metadata**:
```python
{
    "sentiment_analysis": [
        {
            "text": "This product is amazing!",
            "sentiment": "POSITIVE",
            "confidence": 0.98,
            "start_time": 1000,
            "end_time": 3000
        },
        {
            "text": "However, the price is too high.",
            "sentiment": "NEGATIVE",
            "confidence": 0.85,
            "start_time": 3100,
            "end_time": 5500
        }
    ]
}
```

**Use Cases**:
- Customer feedback analysis
- Call center quality monitoring
- Social media content analysis
- Brand perception tracking

### 3. Entity Detection

**What it does**: Extracts named entities (people, locations, organizations, etc.).

**Configuration**:
```python
config = AudioProcessingConfig(
    entity_detection=True
)
```

**Detected Entity Types**:
- `person_name`: People mentioned
- `location`: Cities, countries, addresses
- `organization`: Companies, institutions
- `date`: Temporal references
- `phone_number`: Contact numbers
- `email_address`: Email addresses
- `occupation`: Job titles
- `medical_condition`: Health-related terms
- `medical_process`: Medical procedures
- `drug`: Medication names

**Output Metadata**:
```python
{
    "entities": [
        {
            "text": "New York",
            "entity_type": "location",
            "start_time": 5000,
            "end_time": 6500
        },
        {
            "text": "Microsoft",
            "entity_type": "organization",
            "start_time": 10000,
            "end_time": 11200
        }
    ]
}
```

### 4. Topic Classification (IAB Categories)

**What it does**: Classifies content into 700+ standardized IAB topics.

**Configuration**:
```python
config = AudioProcessingConfig(
    iab_categories=True
)
```

**Output Metadata**:
```python
{
    "topics": [
        {
            "topic": "Technology>Artificial Intelligence",
            "relevance": 0.92
        },
        {
            "topic": "Business>Cloud Computing",
            "relevance": 0.85
        }
    ]
}
```

**Use Cases**:
- Content categorization
- Ad targeting
- Content recommendations
- SEO optimization

### 5. Content Safety Labeling

**What it does**: Flags sensitive content (violence, hate speech, profanity, etc.).

**Configuration**:
```python
config = AudioProcessingConfig(
    content_safety=True,
    content_safety_confidence=80  # 0-100 threshold
)
```

**Detected Labels**:
- `violence`: Violence or harm
- `hate_speech`: Hateful content
- `profanity`: Explicit language
- `sexual_content`: Adult content
- `terrorism`: Extremist content
- `alcohol`: Alcohol references
- `drugs`: Drug references
- `accidents`: Accident descriptions

**Output Metadata**:
```python
{
    "content_safety": [
        {
            "label": "profanity",
            "confidence": 0.95,
            "text": "...",
            "timestamp": 15000,
            "severity": "high"
        }
    ]
}
```

### 6. Auto-Highlights

**What it does**: Automatically identifies key points and important moments.

**Configuration**:
```python
config = AudioProcessingConfig(
    auto_highlights=True
)
```

**Output Metadata**:
```python
{
    "highlights": [
        {
            "text": "The main finding of our research is...",
            "start_time": 30000,
            "end_time": 45000,
            "rank": 0.95  # Importance score
        }
    ]
}
```

**Use Cases**:
- Meeting notes generation
- Video highlight reels
- Podcast clip identification
- Content summarization

### 7. Auto-Chapters

**What it does**: Automatically segments content into chapters with summaries.

**Configuration**:
```python
config = AudioProcessingConfig(
    auto_chapters=True
)
```

**Output Metadata**:
```python
{
    "chapters": [
        {
            "headline": "Introduction",
            "summary": "Overview of the topic and agenda",
            "gist": "Introduction and agenda overview",
            "start_time": 0,
            "end_time": 60000
        },
        {
            "headline": "Main Discussion",
            "summary": "Detailed analysis of the key findings",
            "gist": "Analysis of key findings",
            "start_time": 60000,
            "end_time": 180000
        }
    ]
}
```

**Use Cases**:
- Video navigation
- Podcast chapters
- Long-form content indexing
- Timestamp generation

### 8. AI Summarization

**What it does**: Generates AI-powered summaries in various formats.

**Configuration**:
```python
config = AudioProcessingConfig(
    summarization=True,
    summary_model="informative",  # or "conversational", "catchy"
    summary_type="bullets"         # or "gist", "headline", "paragraph"
)
```

**Summary Models**:
- `informative`: Objective, fact-based
- `conversational`: Casual, dialogue-like
- `catchy`: Engaging, attention-grabbing

**Summary Types**:
- `bullets`: Bullet-point list
- `gist`: Brief one-sentence summary
- `headline`: Short title-like summary
- `paragraph`: Full paragraph summary

**Output**: Added to document content as `## Summary` section.

### 9. PII Redaction

**What it does**: Automatically detects and redacts personally identifiable information.

**Configuration**:
```python
config = AudioProcessingConfig(
    redact_pii=True,
    redact_pii_policies=[
        "credit_card_number",
        "email_address",
        "person_name",
        "phone_number",
        "date_of_birth",
        "social_security_number",
        "drivers_license"
    ],
    redact_pii_audio=True  # Also redact from audio
)
```

**Output**: PII replaced with `[REDACTED]` in transcript and audio.

---

## Configuration Guide

### Presets for Common Use Cases

#### Preset 1: Basic Transcription (Fast, Low Cost)

```python
basic_config = AudioProcessingConfig(
    model="nano",  # Faster model
    speaker_labels=False,
    sentiment_analysis=False,
    entity_detection=False,
    iab_categories=False,
    content_safety=False,
    auto_highlights=False,
    auto_chapters=False,
    summarization=False
)
```

**Use when**: Simple transcription needed, cost-sensitive, speed critical.

#### Preset 2: Meeting Transcription

```python
meeting_config = AudioProcessingConfig(
    model="best",
    speaker_labels=True,
    speakers_expected=None,  # Auto-detect
    sentiment_analysis=True,
    auto_highlights=True,
    auto_chapters=True,
    summarization=True,
    summary_model="informative",
    summary_type="bullets"
)
```

**Use when**: Business meetings, conferences, group discussions.

#### Preset 3: Content Moderation

```python
moderation_config = AudioProcessingConfig(
    content_safety=True,
    content_safety_confidence=75,  # Lower threshold = more sensitive
    filter_profanity=True,
    redact_pii=True,
    redact_pii_policies=[
        "credit_card_number",
        "email_address",
        "phone_number",
        "person_name"
    ],
    sentiment_analysis=True
)
```

**Use when**: User-generated content, social media, call centers.

#### Preset 4: Technical Content

```python
technical_config = AudioProcessingConfig(
    model="best",
    format_text=True,
    custom_vocabulary=[
        "Kubernetes", "Docker", "PostgreSQL", "FastAPI",
        "AssemblyAI", "Python", "TypeScript"
    ],
    boost_param="high",
    entity_detection=True,
    iab_categories=True,
    auto_chapters=True
)
```

**Use when**: Technical talks, developer podcasts, training videos.

#### Preset 5: Podcast Production

```python
podcast_config = AudioProcessingConfig(
    model="best",
    speaker_labels=True,
    auto_chapters=True,
    auto_highlights=True,
    summarization=True,
    summary_model="catchy",
    summary_type="headline",
    sentiment_analysis=True,
    entity_detection=True,
    iab_categories=True
)
```

**Use when**: Podcast episodes, interviews, storytelling content.

### Environment Variables

```bash
# Required
export ASSEMBLYAI_API_KEY="your_api_key_here"

# Optional (defaults shown)
export ASSEMBLYAI_POLLING_INTERVAL="3.0"
export ASSEMBLYAI_MODEL="best"
export ASSEMBLYAI_LANGUAGE="en"
```

### YAML Configuration

```yaml
# audio_config.yaml
transcription:
  language_code: "en"
  model: "best"
  polling_interval: 3.0

features:
  speaker_labels: true
  sentiment_analysis: true
  entity_detection: true
  iab_categories: true
  content_safety: true
  auto_highlights: true
  auto_chapters: true
  summarization: true

enhancement:
  noise_reduction: true
  automatic_punctuation: true
  format_text: true
  filter_profanity: false

privacy:
  redact_pii: false
  redact_pii_policies:
    - credit_card_number
    - email_address
    - phone_number
    - person_name

vocabulary:
  custom_vocabulary:
    - Kubernetes
    - Docker
    - PostgreSQL
  boost_param: "high"

summarization:
  summary_model: "informative"
  summary_type: "bullets"
```

Load from YAML:
```python
from pathlib import Path
from src.audio_processing.config import AudioProcessingConfig

config = AudioProcessingConfig.from_yaml(Path("audio_config.yaml"))
transcriber = AssemblyAITranscriber(config=config)
```

---

## Integration Patterns

### Pattern 1: RAG Pipeline Integration

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from src.audio_processing.audio_transcriber import (
    AssemblyAITranscriber, SmartAudioProcessor
)
from src.vector_database.qdrant_db import QdrantDocumentWriter

# Build RAG ingestion pipeline
pipeline = Pipeline()

# 1. Transcribe audio
transcriber = AssemblyAITranscriber(
    config=AudioProcessingConfig(speaker_labels=True, auto_chapters=True)
)
pipeline.add_component("transcriber", transcriber)

# 2. Smart chunking
processor = SmartAudioProcessor(chunk_strategy="chapter")
pipeline.add_component("processor", processor)

# 3. Generate embeddings
embedder = SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")
pipeline.add_component("embedder", embedder)

# 4. Store in vector database
writer = QdrantDocumentWriter(collection_name="audio_transcripts")
pipeline.add_component("writer", writer)

# Connect
pipeline.connect("transcriber.documents", "processor.documents")
pipeline.connect("processor.documents", "embedder.documents")
pipeline.connect("embedder.documents", "writer.documents")

# Run
result = pipeline.run({
    "transcriber": {"sources": ["podcast_episode.mp3"]}
})
```

### Pattern 2: Batch Processing

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.audio_processing.audio_transcriber import AssemblyAITranscriber

transcriber = AssemblyAITranscriber()

# Find all audio files
audio_files = list(Path("audio_archive").glob("**/*.mp3"))

# Process in batches
batch_size = 10
for i in range(0, len(audio_files), batch_size):
    batch = audio_files[i:i+batch_size]
    result = transcriber.run(sources=batch)
    
    # Save results
    for doc in result["documents"]:
        output_file = Path("transcripts") / f"{doc.meta['source']}.txt"
        output_file.write_text(doc.content)
```

### Pattern 3: Real-Time Monitoring

```python
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.audio_processing.audio_transcriber import AssemblyAITranscriber

class AudioTranscriptionHandler(FileSystemEventHandler):
    def __init__(self):
        self.transcriber = AssemblyAITranscriber()
    
    def on_created(self, event):
        if event.src_path.endswith(('.mp3', '.wav', '.m4a')):
            print(f"New audio file: {event.src_path}")
            result = self.transcriber.run(sources=[event.src_path])
            
            # Save transcript
            output_path = Path(event.src_path).with_suffix('.txt')
            output_path.write_text(result["documents"][0].content)
            print(f"Transcript saved: {output_path}")

# Monitor directory
handler = AudioTranscriptionHandler()
observer = Observer()
observer.schedule(handler, path="audio_inbox", recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

---

## Troubleshooting

### Issue 1: Import Error - assemblyai not found

**Symptom**:
```python
ImportError: assemblyai package is required
```

**Solution**:
```bash
pip install assemblyai
```

### Issue 2: API Key Not Found

**Symptom**:
```python
ValueError: AssemblyAI API key required
```

**Solution**:
```bash
# Set environment variable
export ASSEMBLYAI_API_KEY="your_api_key"

# Or pass directly
transcriber = AssemblyAITranscriber(api_key="your_api_key")
```

### Issue 3: Feature Not Available

**Symptom**: Warnings like "Property sentiment_analysis not available"

**Cause**: assemblyai package version too old or feature unavailable.

**Solution**:
```bash
pip install --upgrade assemblyai
```

### Issue 4: Slow Transcription

**Symptoms**: Transcription takes longer than expected.

**Solutions**:

1. **Use faster model**:
```python
config = AudioProcessingConfig(model="nano")  # Faster than "best"
```

2. **Disable unnecessary features**:
```python
config = AudioProcessingConfig(
    sentiment_analysis=False,
    entity_detection=False,
    iab_categories=False,
    content_safety=False,
    auto_highlights=False
)
```

3. **Adjust polling interval**:
```python
transcriber = AssemblyAITranscriber(polling_interval=1.0)  # Check more frequently
```

### Issue 5: Poor Speaker Diarization

**Symptoms**: Speakers misidentified or merged.

**Solutions**:

1. **Provide speaker hint**:
```python
config = AudioProcessingConfig(
    speaker_labels=True,
    speakers_expected=2  # If you know exact count
)
```

2. **Ensure audio quality**:
- Clear audio with minimal background noise
- Distinct speaker voices
- Avoid overlapping speech

3. **Enable noise reduction**:
```python
config = AudioProcessingConfig(noise_reduction=True)
```

### Issue 6: High API Costs

**Symptoms**: Unexpected AssemblyAI billing.

**Solutions**:

1. **Use nano model for non-critical content**:
```python
config = AudioProcessingConfig(model="nano")  # Cheaper
```

2. **Disable advanced features selectively**:
```python
config = AudioProcessingConfig(
    summarization=False,  # These cost extra
    auto_chapters=False,
    auto_highlights=False
)
```

3. **Pre-filter audio**:
- Only transcribe necessary files
- Skip silent audio
- Remove duplicate content

### Issue 7: Memory Issues with Large Files

**Symptoms**: Out of memory errors with large audio files.

**Solutions**:

1. **Process in smaller batches**:
```python
# Instead of processing 100 files at once
for audio_file in audio_files:
    result = transcriber.run(sources=[audio_file])
    # Process immediately, don't accumulate
```

2. **Use streaming approach**:
```python
# Process files one at a time
for source in sources:
    result = transcriber.run(sources=[source])
    yield result["documents"][0]  # Stream results
```

---

## Performance Considerations

### Model Comparison

| Model | Speed | Accuracy | Cost | Best For |
|-------|-------|----------|------|----------|
| `nano` | ⚡⚡⚡ Fast | ★★☆ Good | $ Low | Quick drafts, high volume |
| `best` | ⚡⚡ Medium | ★★★ Excellent | $$$ High | Production, critical accuracy |
| `conformer-2` | ⚡ Slow | ★★★ Excellent | $$ Medium | Legacy compatibility |

### Feature Impact

| Feature | Processing Time | Cost Impact | Use Case |
|---------|----------------|-------------|----------|
| Speaker labels | +20% | Medium | Multi-speaker content |
| Sentiment analysis | +15% | Medium | Feedback analysis |
| Entity detection | +10% | Low | Content tagging |
| IAB categories | +15% | Medium | Classification |
| Content safety | +10% | Low | Moderation |
| Auto-highlights | +25% | High | Key point extraction |
| Auto-chapters | +30% | High | Long-form content |
| Summarization | +20% | High | Content overview |
| PII redaction | +15% | Medium | Privacy compliance |

### Optimization Tips

1. **Feature Selection**: Only enable features you'll use
2. **Model Choice**: Use `nano` for drafts, `best` for final
3. **Batch Processing**: Group similar files together
4. **Caching**: Cache results, don't re-transcribe
5. **Parallel Processing**: Process independent files in parallel
6. **Polling Interval**: Balance between responsiveness and API calls

---

## See Also

- [overview.md](overview.md) - Audio processing module overview
- [yt_audio_transcriber.md](yt_audio_transcriber.md) - YouTube-specific transcription
- [config.md](config.md) - Configuration management
- [chunking.md](chunking.md) - Chunking strategies
- [extractors.md](extractors.md) - Feature extraction
- [AssemblyAI Documentation](https://www.assemblyai.com/docs) - Official API docs
