import os
import sys
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
from urllib.parse import urlparse
import io

from haystack import component, Document
from haystack.core.serialization import default_from_dict, default_to_dict
from core.logging import CustomLogger

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False
    aai = None

# Configure logging with type safety
import logging
from typing import Union, Any

try:
    from core.logging import CustomLogger
    _logger_instance = CustomLogger()
    _temp_logger = _logger_instance.get_logger(__name__)
    if _temp_logger is not None:
        logger: logging.Logger = _temp_logger
    else:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Type checking helpers
if ASSEMBLYAI_AVAILABLE and aai is not None:
    TranscriptionConfigType = aai.TranscriptionConfig
else:
    TranscriptionConfigType = None

@dataclass
class AudioProcessingConfig:
    """Configuration for AssemblyAI audio processing features."""
    
    # Core transcription settings
    language_code: Optional[str] = "en"
    model: str = "best"  # 'best', 'nano', 'conformer-2'
    
    # Speaker features
    speaker_labels: bool = True
    speakers_expected: Optional[int] = None
    
    # Content analysis
    sentiment_analysis: bool = True
    entity_detection: bool = True
    iab_categories: bool = True  # Topic detection
    content_safety: bool = True
    content_safety_confidence: int = 80
    auto_highlights: bool = True
    
    # Audio enhancement
    noise_reduction: bool = True
    automatic_punctuation: bool = True
    format_text: bool = True
    filter_profanity: bool = False
    
    # Privacy and redaction
    redact_pii: bool = False
    redact_pii_policies: List[str] = field(default_factory=lambda: [
        "credit_card_number", "email_address", "person_name", "phone_number"
    ])
    redact_pii_audio: bool = False
    
    # Advanced features
    custom_spelling: Dict[str, List[str]] = field(default_factory=dict)
    custom_vocabulary: List[str] = field(default_factory=list)
    boost_param: str = "low"  # 'low', 'default', 'high'
    
    # Output formats
    include_utterances: bool = True
    include_sentences: bool = True
    include_paragraphs: bool = True
    auto_chapters: bool = True
    summarization: bool = True
    summary_model: str = "informative"  # 'informative', 'conversational', 'catchy'
    summary_type: str = "bullets"  # 'bullets', 'gist', 'headline', 'paragraph'

@component
class AssemblyAITranscriber:
    """
    A comprehensive Haystack component for AssemblyAI speech-to-text transcription.
    
    This component provides full access to AssemblyAI's advanced features including
    speaker diarization, content analysis, sentiment analysis, and more.
    """

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
        
        logger.info("Initializing AssemblyAI Transcriber component")
        
        if not ASSEMBLYAI_AVAILABLE or aai is None:
            logger.error("AssemblyAI package is not available")
            raise ImportError(
                "assemblyai package is required. Install with: pip install assemblyai"
            )
        
        logger.debug("AssemblyAI package is available")
        
        # Get API key
        api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            logger.error("No API key provided for AssemblyAI")
            raise ValueError(
                "AssemblyAI API key required. Set ASSEMBLYAI_API_KEY env var or pass api_key parameter."
            )
        
        logger.debug("API key obtained successfully")
        
        # Set API key and polling interval
        aai.settings.api_key = api_key
        aai.settings.polling_interval = polling_interval
        logger.debug(f"Set polling interval to {polling_interval} seconds")
        
        # Initialize configuration
        self.config = config or AudioProcessingConfig()
        self.api_key = api_key
        self.polling_interval = polling_interval
        
        logger.info(f"Configuration initialized - Speaker labels: {self.config.speaker_labels}, Sentiment: {self.config.sentiment_analysis}, Entities: {self.config.entity_detection}")
        
        # Initialize transcriber
        self.transcriber = aai.Transcriber()
        
        logger.info("AssemblyAI Transcriber initialized successfully")

    def _create_transcription_config(self):
        """Create AssemblyAI TranscriptionConfig from our config."""
        
        if not ASSEMBLYAI_AVAILABLE or aai is None:
            raise RuntimeError("AssemblyAI not available")
            
        logger.debug("Creating AssemblyAI TranscriptionConfig from AudioProcessingConfig")
        
        # Use the proper constructor with parameters instead of setting properties
        try:
            config = aai.TranscriptionConfig(
                speaker_labels=self.config.speaker_labels,
                sentiment_analysis=self.config.sentiment_analysis,
                entity_detection=self.config.entity_detection,
                iab_categories=self.config.iab_categories,
                content_safety=self.config.content_safety,
                content_safety_confidence=self.config.content_safety_confidence,
                auto_highlights=self.config.auto_highlights,
                language_code=self.config.language_code,
                punctuate=self.config.automatic_punctuation,
                format_text=self.config.format_text,
                filter_profanity=self.config.filter_profanity,
                auto_chapters=self.config.auto_chapters,
                summarization=self.config.summarization
            )
            logger.debug("TranscriptionConfig created with constructor parameters")
        except Exception as constructor_error:
            logger.warning(f"Constructor approach failed: {constructor_error}")
            # Fallback: create base config and try to set properties individually
            config = aai.TranscriptionConfig()
            logger.debug("Base TranscriptionConfig created, trying individual property setting")
            
            # Try setting each property with error handling
            config_items = [
                ('language_code', self.config.language_code),
                ('speaker_labels', self.config.speaker_labels),
                ('sentiment_analysis', self.config.sentiment_analysis),
                ('entity_detection', self.config.entity_detection),
                ('iab_categories', self.config.iab_categories),
                ('content_safety', self.config.content_safety),
                ('content_safety_confidence', self.config.content_safety_confidence),
                ('auto_highlights', self.config.auto_highlights),
                ('punctuate', self.config.automatic_punctuation),
                ('format_text', self.config.format_text),
                ('filter_profanity', self.config.filter_profanity),
                ('auto_chapters', self.config.auto_chapters),
                ('summarization', self.config.summarization)
            ]
            
            for prop_name, prop_value in config_items:
                try:
                    if hasattr(config, prop_name):
                        setattr(config, prop_name, prop_value)
                        logger.debug(f"Set {prop_name} to {prop_value}")
                    else:
                        logger.warning(f"Property {prop_name} not available in this AssemblyAI version")
                except Exception as prop_error:
                    logger.warning(f"Failed to set {prop_name}: {prop_error}")
        
        # Handle speakers_expected separately if provided
        if self.config.speakers_expected:
            try:
                if hasattr(config, 'speakers_expected'):
                    setattr(config, 'speakers_expected', self.config.speakers_expected)
                    logger.debug(f"Set speakers_expected to {self.config.speakers_expected}")
            except Exception as e:
                logger.warning(f"Failed to set speakers_expected: {e}")
        
        logger.debug(f"Content analysis features configured - Sentiment: {self.config.sentiment_analysis}, Entities: {self.config.entity_detection}, Topics: {self.config.iab_categories}, Safety: {self.config.content_safety}, Highlights: {self.config.auto_highlights}")
        
        # Privacy settings
        if self.config.redact_pii:
            try:
                if hasattr(config, 'redact_pii'):
                    setattr(config, 'redact_pii', True)
                    logger.debug("PII redaction enabled")
                    
                    if hasattr(aai, 'PIIRedactionPolicy') and hasattr(config, 'redact_pii_policies'):
                        pii_policies = [
                            getattr(aai.PIIRedactionPolicy, policy, policy)
                            for policy in self.config.redact_pii_policies
                            if hasattr(aai.PIIRedactionPolicy, policy)
                        ]
                        setattr(config, 'redact_pii_policies', pii_policies)
                        logger.debug(f"PII policies set: {pii_policies}")
                        
                    if hasattr(config, 'redact_pii_audio'):
                        setattr(config, 'redact_pii_audio', self.config.redact_pii_audio)
                        logger.debug(f"PII audio redaction: {self.config.redact_pii_audio}")
            except Exception as e:
                logger.warning(f"PII redaction features not available: {e}")
        
        # Custom vocabulary
        try:
            if self.config.custom_spelling and hasattr(config, 'set_custom_spelling'):
                config.set_custom_spelling(self.config.custom_spelling)
                logger.debug(f"Custom spelling set: {list(self.config.custom_spelling.keys())}")
            
            if self.config.custom_vocabulary and hasattr(config, 'word_boost'):
                setattr(config, 'word_boost', self.config.custom_vocabulary)
                logger.debug(f"Custom vocabulary set: {len(self.config.custom_vocabulary)} words")
                
                if hasattr(aai, 'BoostParam') and hasattr(config, 'boost_param'):
                    boost_param = getattr(aai.BoostParam, self.config.boost_param, self.config.boost_param)
                    setattr(config, 'boost_param', boost_param)
                    logger.debug(f"Boost param set: {self.config.boost_param}")
        except Exception as e:
            logger.warning(f"Custom vocabulary features not available: {e}")
        
        # Summarization settings
        if self.config.summarization:
            try:
                if hasattr(aai, 'SummarizationModel') and hasattr(config, 'summary_model'):
                    summary_model = getattr(aai.SummarizationModel, self.config.summary_model, self.config.summary_model)
                    setattr(config, 'summary_model', summary_model)
                    logger.debug(f"Summary model set: {self.config.summary_model}")
                    
                if hasattr(aai, 'SummarizationType') and hasattr(config, 'summary_type'):
                    summary_type = getattr(aai.SummarizationType, self.config.summary_type, self.config.summary_type)
                    setattr(config, 'summary_type', summary_type)
                    logger.debug(f"Summary type set: {self.config.summary_type}")
            except Exception as e:
                logger.warning(f"Summarization configuration failed: {e}")
        
        return config

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
        
        logger.info(f"Starting transcription for {len(sources)} sources")
        logger.debug(f"Source types: {[type(source).__name__ for source in sources]}")
        
        documents = []
        
        for i, source in enumerate(sources, 1):
            documents_before_source = len(documents)
            logger.info(f"Processing source {i}/{len(sources)}: {str(source)[:100]}...")
            try:
                # Handle different source types
                if isinstance(source, bytes):
                    # Upload bytes to AssemblyAI using BytesIO
                    import io
                    upload_url = self.transcriber.upload_file(io.BytesIO(source))
                    source_url = upload_url
                    source_name = "uploaded_audio"
                elif isinstance(source, (str, Path)):
                    source_str = str(source)
                    if self._is_url(source_str):
                        source_url = source_str
                        source_name = Path(urlparse(source_str).path).name or "web_audio"
                    else:
                        # Local file - upload to AssemblyAI
                        with open(source, 'rb') as f:
                            upload_url = self.transcriber.upload_file(f)
                        source_url = upload_url
                        source_name = Path(source).name
                else:
                    raise ValueError(f"Unsupported source type: {type(source)}")
                
                # Create transcription config
                transcript_config = self._create_transcription_config()
                
                # Transcribe
                logger.info(f"Starting transcription for: {source_name}")
                transcript = self.transcriber.transcribe(source_url, transcript_config)
                
                if hasattr(transcript, 'status') and hasattr(aai, 'TranscriptStatus'):
                    if transcript.status == aai.TranscriptStatus.error:  # type: ignore
                        logger.error(f"Transcription failed for {source_name}: {transcript.error}")
                        continue
                elif hasattr(transcript, 'error') and transcript.error:
                    logger.error(f"Transcription failed for {source_name}: {transcript.error}")
                    continue
                
                # Extract comprehensive content
                content_parts = [f"# Transcription: {source_name}\n"]
                
                # Main transcript
                if transcript.text:
                    content_parts.append(f"## Full Transcript\n{transcript.text}\n")
                    logger.info(f"Full transcription completed for {source_name}:")
                    logger.info(f"Text: {transcript.text}")
                    logger.info(f"Text length: {len(transcript.text)} characters")
                
                # Log audio metadata
                duration = getattr(transcript, 'audio_duration_seconds', None)
                confidence = getattr(transcript, 'confidence', None)
                logger.info(f"Audio metadata - Duration: {duration}s, Confidence: {confidence}")
                
                # Speaker-labeled transcript
                if self.config.speaker_labels and hasattr(transcript, 'utterances') and transcript.utterances:
                    content_parts.append("## Speaker Transcript\n")
                    logger.info(f"Speaker analysis found {len(transcript.utterances)} utterances:")
                    speaker_count = {}
                    for i, utterance in enumerate(transcript.utterances):
                        content_parts.append(f"**Speaker {utterance.speaker}** ({utterance.start}ms - {utterance.end}ms): {utterance.text}\n")
                        speaker_count[utterance.speaker] = speaker_count.get(utterance.speaker, 0) + 1
                        logger.info(f"  Utterance {i+1}: Speaker {utterance.speaker} ({utterance.start}-{utterance.end}ms): {utterance.text}")
                    logger.info(f"Speaker distribution: {dict(speaker_count)}")
                
                # Auto chapters
                if self.config.auto_chapters and hasattr(transcript, 'chapters') and transcript.chapters:
                    content_parts.append("## Chapters\n")
                    logger.info(f"Auto chapters detected: {len(transcript.chapters)} chapters")
                    for i, chapter in enumerate(transcript.chapters):
                        content_parts.append(f"### Chapter {i+1}: {chapter.headline}\n")
                        content_parts.append(f"**Time**: {chapter.start}ms - {chapter.end}ms\n")
                        content_parts.append(f"**Summary**: {chapter.summary}\n")
                        content_parts.append(f"**Gist**: {chapter.gist}\n\n")
                        logger.info(f"  Chapter {i+1}: '{chapter.headline}' ({chapter.start}-{chapter.end}ms)")
                        logger.info(f"    Summary: {chapter.summary}")
                        logger.info(f"    Gist: {chapter.gist}")
                
                # Summary
                if self.config.summarization and hasattr(transcript, 'summary') and transcript.summary:
                    content_parts.append(f"## Summary\n{transcript.summary}\n")
                    logger.info(f"AI Summary generated: {transcript.summary}")
                
                # Create comprehensive metadata
                metadata = {
                    "source": source_name,
                    "transcript_id": transcript.id,
                    "audio_duration_seconds": getattr(transcript, 'audio_duration_seconds', None),
                    "language_code": self.config.language_code,
                    "confidence": getattr(transcript, 'confidence', None),
                    "audio_url": source_url if isinstance(source, str) and self._is_url(str(source)) else None
                }
                
                # Add analysis results to metadata
                if self.config.sentiment_analysis and hasattr(transcript, 'sentiment_analysis'):
                    sentiment_data = self._extract_sentiment_data(transcript.sentiment_analysis)
                    metadata['sentiment_analysis'] = sentiment_data
                    logger.info(f"Sentiment analysis completed: {len(sentiment_data)} segments")
                    for i, sentiment in enumerate(sentiment_data[:5]):  # Log first 5
                        logger.info(f"  Sentiment {i+1}: '{sentiment['text'][:50]}...' → {sentiment['sentiment']} (confidence: {sentiment['confidence']})")
                
                if self.config.entity_detection and hasattr(transcript, 'entities'):
                    entity_data = self._extract_entity_data(transcript.entities)
                    metadata['entities'] = entity_data
                    logger.info(f"Entity detection completed: {len(entity_data)} entities found")
                    entity_types = {}
                    for entity in entity_data:
                        entity_type = entity['entity_type']
                        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                        logger.info(f"  Entity: '{entity['text']}' → {entity['entity_type']} ({entity['start_time']}-{entity['end_time']}ms)")
                    if entity_types:
                        logger.info(f"Entity types found: {dict(entity_types)}")
                
                if self.config.iab_categories and hasattr(transcript, 'iab_categories'):
                    topic_data = self._extract_topic_data(transcript.iab_categories)
                    metadata['topics'] = topic_data
                    logger.info(f"Topic detection completed: {len(topic_data.get('results', []))} topic segments")
                    if topic_data.get('summary'):
                        logger.info(f"Topic summary: {topic_data['summary']}")
                
                if self.config.content_safety and hasattr(transcript, 'content_safety'):
                    safety_data = self._extract_content_safety_data(transcript.content_safety)
                    metadata['content_safety'] = safety_data
                    logger.info(f"Content safety analysis: {len(safety_data.get('results', []))} segments analyzed")
                    if safety_data.get('summary'):
                        logger.info(f"Safety summary: {safety_data['summary']}")
                
                if self.config.auto_highlights and hasattr(transcript, 'auto_highlights'):
                    highlights_data = self._extract_highlights_data(transcript.auto_highlights)
                    metadata['highlights'] = highlights_data
                    logger.info(f"Auto highlights extracted: {len(highlights_data)} key highlights")
                    for i, highlight in enumerate(highlights_data[:3]):  # Log top 3
                        logger.info(f"  Highlight {i+1}: '{highlight['text']}' (rank: {highlight['rank']}, count: {highlight['count']})")
                
                # Create main document
                main_document = Document(
                    content="\n".join(content_parts),
                    meta=metadata
                )
                documents.append(main_document)
                
                logger.info(f"Main document created - Content length: {len(main_document.content or '')} characters")
                logger.info(f"Document metadata keys: {list(metadata.keys())}")
                
                # Create additional structured documents if requested
                structured_docs_count = len(documents)
                self._add_structured_documents(transcript, documents, metadata)
                new_structured_count = len(documents) - structured_docs_count
                if new_structured_count > 0:
                    logger.info(f"Created {new_structured_count} additional structured documents (sentences/paragraphs)")
                
                total_docs_for_source = len(documents) - len(documents_before_source) if 'documents_before_source' in locals() else len(documents)
                logger.info(f"Successfully transcribed {source_name} - Generated {total_docs_for_source} documents total")
                
            except Exception as e:
                logger.error(f"Error processing source {source}: {str(e)}")
                continue
        
        logger.info(f"Transcription completed for all {len(sources)} sources")
        logger.info(f"Total documents generated: {len(documents)}")
        
        # Log summary of document types
        doc_types = {}
        for doc in documents:
            content_type = doc.meta.get('content_type', 'main')
            doc_types[content_type] = doc_types.get(content_type, 0) + 1
        logger.info(f"Document type distribution: {dict(doc_types)}")
        
        return {"documents": documents}
    
    def _add_structured_documents(self, transcript, documents: List[Document], base_metadata: Dict):
        """Add structured documents (sentences, paragraphs) if requested."""
        
        if self.config.include_sentences and hasattr(transcript, 'get_sentences'):
            try:
                sentences = transcript.get_sentences()
                logger.info(f"Extracting {len(sentences)} individual sentences")
                for i, sentence in enumerate(sentences):
                    sentence_doc = Document(
                        content=sentence.text,
                        meta={
                            **base_metadata,
                            "content_type": "sentence",
                            "sentence_index": i,
                            "start_time": sentence.start,
                            "end_time": sentence.end
                        }
                    )
                    documents.append(sentence_doc)
                    if i < 3:  # Log first 3 sentences
                        logger.info(f"  Sentence {i+1} ({sentence.start}-{sentence.end}ms): {sentence.text[:50]}...")
                logger.info(f"Successfully created {len(sentences)} sentence documents")
            except Exception as e:
                logger.warning(f"Failed to extract sentences: {e}")
        
        if self.config.include_paragraphs and hasattr(transcript, 'get_paragraphs'):
            try:
                paragraphs = transcript.get_paragraphs()
                logger.info(f"Extracting {len(paragraphs)} individual paragraphs")
                for i, paragraph in enumerate(paragraphs):
                    paragraph_doc = Document(
                        content=paragraph.text,
                        meta={
                            **base_metadata,
                            "content_type": "paragraph", 
                            "paragraph_index": i,
                            "start_time": paragraph.start,
                            "end_time": paragraph.end
                        }
                    )
                    documents.append(paragraph_doc)
                    if i < 2:  # Log first 2 paragraphs
                        logger.info(f"  Paragraph {i+1} ({paragraph.start}-{paragraph.end}ms): {paragraph.text[:80]}...")
                logger.info(f"Successfully created {len(paragraphs)} paragraph documents")
            except Exception as e:
                logger.warning(f"Failed to extract paragraphs: {e}")
    
    def _is_url(self, string: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _extract_sentiment_data(self, sentiment_results) -> List[Dict]:
        """Extract sentiment analysis data."""
        if not sentiment_results:
            return []
        
        return [
            {
                "text": result.text,
                "sentiment": getattr(result.sentiment, 'value', result.sentiment) if hasattr(result, 'sentiment') else str(result.sentiment),
                "confidence": result.confidence,
                "start_time": result.start,
                "end_time": result.end,
                "speaker": getattr(result, 'speaker', None)
            }
            for result in sentiment_results
        ]
    
    def _extract_entity_data(self, entities) -> List[Dict]:
        """Extract entity detection data."""
        if not entities:
            return []
        
        return [
            {
                "text": entity.text,
                "entity_type": getattr(entity.entity_type, 'value', entity.entity_type) if hasattr(entity, 'entity_type') else str(entity.entity_type),
                "start_time": entity.start,
                "end_time": entity.end
            }
            for entity in entities
        ]
    
    def _extract_topic_data(self, iab_categories) -> Dict:
        """Extract topic detection data."""
        if not iab_categories:
            return {}
        
        data = {
            "summary": dict(iab_categories.summary) if hasattr(iab_categories, 'summary') else {},
            "results": []
        }
        
        if hasattr(iab_categories, 'results'):
            data["results"] = [
                {
                    "text": result.text,
                    "labels": [
                        {"label": label.label, "relevance": label.relevance}
                        for label in result.labels
                    ],
                    "start_time": result.timestamp.start,
                    "end_time": result.timestamp.end
                }
                for result in iab_categories.results
            ]
        
        return data
    
    def _extract_content_safety_data(self, content_safety) -> Dict:
        """Extract content safety data."""
        if not content_safety:
            return {}
        
        data = {
            "summary": dict(content_safety.summary) if hasattr(content_safety, 'summary') else {},
            "results": []
        }
        
        if hasattr(content_safety, 'results'):
            data["results"] = [
                {
                    "text": result.text,
                    "labels": [
                        {
                            "label": label.label,
                            "confidence": label.confidence,
                            "severity": getattr(label, 'severity', None)
                        }
                        for label in result.labels
                    ],
                    "start_time": result.timestamp.start,
                    "end_time": result.timestamp.end
                }
                for result in content_safety.results
            ]
        
        return data
    
    def _extract_highlights_data(self, auto_highlights) -> List[Dict]:
        """Extract auto highlights data."""
        if not auto_highlights or not hasattr(auto_highlights, 'results'):
            return []
        
        return [
            {
                "text": result.text,
                "rank": result.rank,
                "count": result.count,
                "timestamps": [
                    {"start_time": ts.start, "end_time": ts.end}
                    for ts in result.timestamps
                ]
            }
            for result in auto_highlights.results
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            api_key="***",  # Don't serialize the actual API key
            config=self.config.__dict__,
            polling_interval=self.polling_interval
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssemblyAITranscriber":
        """Deserialize the component from a dictionary."""
        init_params = data.get("init_parameters", {})
        config_data = init_params.get("config", {})
        config = AudioProcessingConfig(**config_data) if config_data else AudioProcessingConfig()
        
        return cls(
            api_key=init_params.get("api_key", "test_key"),
            config=config,
            polling_interval=init_params.get("polling_interval", 3.0)
        )


@component
class SmartAudioProcessor:
    """
    Advanced audio processor that mimics mew.py's smart document chunking
    but for audio content with speaker awareness and content boundaries.
    """
    
    def __init__(
        self,
        assemblyai_transcriber: AssemblyAITranscriber,
        max_chunk_length: int = 1000,
        overlap: int = 100,
        respect_speakers: bool = True,
        respect_chapters: bool = True
    ):
        logger.info("Initializing SmartAudioProcessor")
        self.transcriber = assemblyai_transcriber
        self.max_chunk_length = max_chunk_length
        self.overlap = overlap
        self.respect_speakers = respect_speakers
        self.respect_chapters = respect_chapters
        
        logger.info(f"SmartAudioProcessor configured - Max chunk length: {max_chunk_length}, Overlap: {overlap}, Respect speakers: {respect_speakers}, Respect chapters: {respect_chapters}")
    
    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, bytes]]) -> Dict[str, List[Document]]:
        """Process audio with smart chunking like mew.py does for documents."""
        
        logger.info(f"Starting smart audio processing for {len(sources)} sources")
        
        # First get the transcription with all features
        logger.debug("Getting transcription from AssemblyAI transcriber")
        transcription_result = self.transcriber.run(sources)
        raw_documents = transcription_result["documents"]
        
        logger.info(f"Received {len(raw_documents)} raw documents from transcriber")
        
        smart_chunks = []
        
        for doc in raw_documents:
            if doc.meta.get("content_type") in ["sentence", "paragraph"]:
                # These are already structured chunks, skip
                continue
            
            # Process main transcript document with smart chunking
            if "content_type" not in doc.meta:
                chunks = self._create_smart_audio_chunks(doc)
                smart_chunks.extend(chunks)
        
        return {"documents": smart_chunks}
    
    def _create_smart_audio_chunks(self, document: Document) -> List[Document]:
        """Create smart chunks from audio transcript similar to mew.py's approach."""
        
        logger.debug(f"Creating smart chunks for document: {document.meta.get('source', 'unknown')}")
        
        chunks = []
        content = document.content or ""
        metadata = document.meta.copy()
        
        logger.debug(f"Document content length: {len(content)} characters")
        
        # Check if we have speaker or chapter information
        speaker_data = metadata.get("sentiment_analysis", [])
        chapter_info = content.find("## Chapters") != -1
        
        logger.debug(f"Analysis - Speaker data available: {len(speaker_data) > 0}, Chapter info: {chapter_info}")
        
        if self.respect_speakers and speaker_data:
            # Speaker-aware chunking
            logger.info("Using speaker-aware chunking strategy")
            chunks = self._chunk_by_speakers(content, metadata, speaker_data)
        elif self.respect_chapters and chapter_info:
            # Chapter-aware chunking
            logger.info("Using chapter-aware chunking strategy")
            chunks = self._chunk_by_chapters(content, metadata)
        else:
            # Semantic boundary-aware chunking
            logger.info("Using semantic boundary-aware chunking strategy")
            chunks = self._chunk_by_semantic_boundaries(content, metadata)
        
        logger.info(f"Created {len(chunks)} smart chunks using selected strategy")
        
        return chunks
    
    def _chunk_by_speakers(self, content: str, metadata: Dict, speaker_data: List) -> List[Document]:
        """Chunk content based on speaker changes."""
        logger.debug(f"Starting speaker-based chunking with {len(speaker_data)} speaker data points")
        
        chunks = []
        lines = content.split('\n')
        
        logger.debug(f"Processing {len(lines)} lines for speaker chunking")
        
        current_chunk = []
        current_speaker = None
        chunk_id = 0
        
        for line in lines:
            if line.startswith("**Speaker "):
                # New speaker detected
                if current_chunk and current_speaker:
                    # Save previous chunk
                    chunk_content = '\n'.join(current_chunk)
                    if len(chunk_content.strip()) > 0:
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "chunk_id": chunk_id,
                            "chunk_type": "speaker_segment",
                            "speaker": current_speaker,
                            "chunk_length": len(chunk_content),
                            "processing_strategy": "speaker_aware"
                        })
                        
                        chunks.append(Document(content=chunk_content, meta=chunk_metadata))
                        chunk_id += 1
                
                # Start new chunk
                current_chunk = [line]
                # Extract speaker info
                if "Speaker " in line:
                    try:
                        current_speaker = line.split("Speaker ")[1].split("**")[0]
                    except:
                        current_speaker = "Unknown"
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if len(chunk_content.strip()) > 0:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_type": "speaker_segment",
                    "speaker": current_speaker or "Unknown",
                    "chunk_length": len(chunk_content),
                    "processing_strategy": "speaker_aware"
                })
                
                chunks.append(Document(content=chunk_content, meta=chunk_metadata))
        
        return chunks
    
    def _chunk_by_chapters(self, content: str, metadata: Dict) -> List[Document]:
        """Chunk content based on auto-generated chapters."""
        logger.debug("Starting chapter-based chunking")
        
        chunks = []
        
        # Find chapter sections
        sections = content.split("### Chapter")
        logger.debug(f"Found {len(sections)} chapter sections")
        
        for i, section in enumerate(sections):
            if i == 0:  # Skip the part before first chapter
                continue
            
            if len(section.strip()) > 0:
                chunk_content = f"### Chapter{section}"
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i - 1,
                    "chunk_type": "chapter",
                    "chapter_number": i,
                    "chunk_length": len(chunk_content),
                    "processing_strategy": "chapter_aware"
                })
                
                chunks.append(Document(content=chunk_content, meta=chunk_metadata))
        
        return chunks
    
    def _chunk_by_semantic_boundaries(self, content: str, metadata: Dict) -> List[Document]:
        """Chunk content based on semantic boundaries like sentences and paragraphs."""
        logger.debug("Starting semantic boundary-based chunking")
        
        chunks = []
        
        # Split by double newlines (paragraph breaks) and other semantic indicators
        sections = content.split('\n\n')
        logger.debug(f"Found {len(sections)} semantic sections")
        
        current_chunk = ""
        chunk_id = 0
        
        for section in sections:
            if len(current_chunk) + len(section) > self.max_chunk_length and current_chunk:
                # Save current chunk
                if current_chunk.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": chunk_id,
                        "chunk_type": "semantic_boundary",
                        "chunk_length": len(current_chunk),
                        "processing_strategy": "semantic_aware"
                    })
                    
                    chunks.append(Document(content=current_chunk.strip(), meta=chunk_metadata))
                    chunk_id += 1
                
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_type": "semantic_boundary", 
                "chunk_length": len(current_chunk),
                "processing_strategy": "semantic_aware"
            })
            
            chunks.append(Document(content=current_chunk.strip(), meta=chunk_metadata))
        
        return chunks

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            assemblyai_transcriber=self.transcriber.to_dict(),
            max_chunk_length=self.max_chunk_length,
            overlap=self.overlap,
            respect_speakers=self.respect_speakers,
            respect_chapters=self.respect_chapters
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmartAudioProcessor":
        """Deserialize the component from a dictionary."""
        init_params = data.get("init_parameters", {})
        transcriber_data = init_params.get("assemblyai_transcriber", {})
        transcriber = AssemblyAITranscriber.from_dict({"init_parameters": transcriber_data})
        
        return cls(
            assemblyai_transcriber=transcriber,
            max_chunk_length=init_params.get("max_chunk_length", 1000),
            overlap=init_params.get("overlap", 100),
            respect_speakers=init_params.get("respect_speakers", True),
            respect_chapters=init_params.get("respect_chapters", True)
        )


# Example usage and integration functions
def create_audio_pipeline(
    api_key: Optional[str] = None,
    config: Optional[AudioProcessingConfig] = None
):
    """
    Create a complete audio processing pipeline with AssemblyAI.
    
    :param api_key: AssemblyAI API key
    :param config: Audio processing configuration
    :return: Configured Haystack pipeline
    """
    logger.info("Creating comprehensive audio processing pipeline")
    
    from haystack import Pipeline
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.components.writers import DocumentWriter
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    
    logger.debug("Imported required Haystack components")
    
    # Create components
    transcriber = AssemblyAITranscriber(api_key=api_key, config=config)
    smart_processor = SmartAudioProcessor(assemblyai_transcriber=transcriber)
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    document_store = InMemoryDocumentStore()
    writer = DocumentWriter(document_store=document_store)
    
    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_component("smart_processor", smart_processor)
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("writer", writer)
    
    # Connect components
    pipeline.connect("smart_processor", "embedder")
    pipeline.connect("embedder", "writer")
    
    return pipeline


def create_advanced_audio_config() -> AudioProcessingConfig:
    """Create an advanced configuration with all AssemblyAI features enabled."""
    logger.info("Creating advanced audio processing configuration with all features enabled")
    
    return AudioProcessingConfig(
        # Speaker analysis
        speaker_labels=True,
        speakers_expected=None,  # Auto-detect
        
        # Content analysis - all features enabled
        sentiment_analysis=True,
        entity_detection=True,
        iab_categories=True,
        content_safety=True,
        content_safety_confidence=75,
        auto_highlights=True,
        
        # Audio enhancement
        noise_reduction=True,
        automatic_punctuation=True,
        format_text=True,
        
        # Privacy features
        redact_pii=False,  # Can be enabled as needed
        redact_pii_policies=["person_name", "phone_number", "email_address"],
        
        # Custom vocabulary
        custom_spelling={
            "AssemblyAI": ["assembly ai", "assembly AI"],
            "Haystack": ["hay stack"],
            "API": ["api", "A.P.I."]
        },
        custom_vocabulary=["transcription", "speech-to-text", "AI", "machine learning"],
        boost_param="high",
        
        # Output structure
        include_utterances=True,
        include_sentences=True,
        include_paragraphs=True,
        auto_chapters=True,
        summarization=True,
        summary_model="informative",
        summary_type="bullets"
    )