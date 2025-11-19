"""
YouTube Audio Transcriber Module

This module provides comprehensive YouTube audio transcription capabilities
by integrating with the existing AssemblyAI transcriber features and adding
YouTube-specific functionality for downloading and processing video audio.
"""

import logging
import os
import tempfile
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse
from datetime import datetime

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    yt_dlp = None

from haystack import component, Document
from .audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
from src.core.logging import CustomLogger

# Configure logging with type safety
try:
    from src.core.logging import CustomLogger
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


@dataclass
class YouTubeVideoInfo:
    """Contains metadata about a YouTube video."""
    video_id: str
    title: str
    description: str
    uploader: str
    upload_date: str
    duration: Optional[float]
    view_count: Optional[int]
    like_count: Optional[int]
    channel: str
    channel_id: str
    tags: List[str]
    categories: List[str]
    thumbnail: Optional[str]
    webpage_url: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata storage."""
        return {
            'video_id': self.video_id,
            'title': self.title,
            'description': self.description,
            'uploader': self.uploader,
            'upload_date': self.upload_date,
            'duration_seconds': self.duration,
            'view_count': self.view_count,
            'like_count': self.like_count,
            'channel': self.channel,
            'channel_id': self.channel_id,
            'tags': self.tags,
            'categories': self.categories,
            'thumbnail': self.thumbnail,
            'webpage_url': self.webpage_url
        }


@component
class YouTubeAudioTranscriber:
    """
    Comprehensive YouTube Audio Transcriber that combines YouTube audio extraction
    with advanced AssemblyAI transcription features.
    
    Features:
    - YouTube URL validation and video ID extraction
    - High-quality audio downloading with yt-dlp
    - Full integration with AssemblyAI's advanced features
    - Smart caching to avoid re-downloading
    - Rich metadata extraction from YouTube
    - Comprehensive error handling and logging
    """
    
    def __init__(
        self,
        assemblyai_api_key: Optional[str] = None,
        audio_config: Optional[AudioProcessingConfig] = None,
        temp_dir: Optional[str] = None,
        cleanup_audio: bool = True,
        cache_audio: bool = True,
        audio_quality: str = "best",
        max_duration: Optional[int] = None  # Maximum duration in seconds
    ):
        """
        Initialize the YouTube Audio Transcriber.
        
        :param assemblyai_api_key: AssemblyAI API key
        :param audio_config: Audio processing configuration for transcription
        :param temp_dir: Directory for temporary audio files
        :param cleanup_audio: Whether to delete audio files after transcription
        :param cache_audio: Whether to cache downloaded audio files
        :param audio_quality: Audio quality preference ('best', 'worst', or specific format)
        :param max_duration: Maximum video duration to process (in seconds)
        """
        
        logger.info("=" * 80)
        logger.info("🎬 INITIALIZING YOUTUBE AUDIO TRANSCRIBER")
        logger.info("=" * 80)
        
        if not YT_DLP_AVAILABLE or yt_dlp is None:
            logger.error("❌ yt-dlp is not available. Please install it: pip install yt-dlp")
            raise ImportError("yt-dlp is required for YouTube audio transcription")
        
        logger.info("✅ yt-dlp is available")
        
        # Initialize AssemblyAI transcriber with advanced config
        if audio_config is None:
            audio_config = self._create_youtube_optimized_config()
            
        logger.info("🔧 Initializing AssemblyAI transcriber with YouTube-optimized configuration")
        self.transcriber = AssemblyAITranscriber(
            api_key=assemblyai_api_key,
            config=audio_config
        )
        
        # Setup temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "youtube_audio_transcriber"
        
        self.temp_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Temporary directory: {self.temp_dir}")
        
        # Configuration
        self.cleanup_audio = cleanup_audio
        self.cache_audio = cache_audio
        self.audio_quality = audio_quality
        self.max_duration = max_duration
        
        logger.info("⚙️ Configuration:")
        logger.info(f"   • Audio Quality: {audio_quality}")
        logger.info(f"   • Cache Audio: {cache_audio}")
        logger.info(f"   • Cleanup After: {cleanup_audio}")
        logger.info(f"   • Max Duration: {max_duration}s" if max_duration else "   • Max Duration: Unlimited")
        
        logger.info("🚀 YouTube Audio Transcriber initialized successfully")
        logger.info("-" * 80)
    
    def _create_youtube_optimized_config(self) -> AudioProcessingConfig:
        """Create an optimized configuration for YouTube audio transcription."""
        logger.info("🎯 Creating YouTube-optimized transcription configuration")
        
        return AudioProcessingConfig(
            # Core settings optimized for YouTube content
            language_code="en",
            model="best",
            
            # Speaker analysis (important for interviews, podcasts, discussions)
            speaker_labels=True,
            speakers_expected=None,  # Auto-detect
            
            # Content analysis (valuable for YouTube content categorization)
            sentiment_analysis=True,
            entity_detection=True,
            iab_categories=True,  # Topic detection for content categorization
            content_safety=True,
            content_safety_confidence=75,
            auto_highlights=True,  # Extract key moments
            
            # Audio enhancement (YouTube audio can be noisy)
            noise_reduction=True,
            automatic_punctuation=True,
            format_text=True,
            filter_profanity=False,  # Preserve original content
            
            # Structure extraction (helpful for long-form content)
            include_utterances=True,
            include_sentences=True,
            include_paragraphs=True,
            # Note: auto_chapters and summarization cannot be enabled together
            auto_chapters=False,  # Explicitly disabled to allow summarization
            
            # Summarization (valuable for long YouTube videos) 
            # Note: AssemblyAI only allows ONE of: auto_chapters OR summarization
            summarization=True,
            
            # Custom vocabulary for common YouTube terms
            custom_vocabulary=[
                "YouTube", "subscribe", "notification", "like", "comment",
                "channel", "playlist", "timestamp", "description", "pinned",
                "livestream", "premiere", "tutorial", "vlog", "podcast"
            ],
            boost_param="high"
        )
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        logger.debug(f"🔍 Extracting video ID from URL: {url}")
        
        # Regular expressions for different YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                logger.debug(f"✅ Extracted video ID: {video_id}")
                return video_id
        
        logger.warning(f"⚠️ Could not extract video ID from URL: {url}")
        return None
    
    def validate_youtube_url(self, url: str) -> bool:
        """Validate if the URL is a valid YouTube URL."""
        logger.debug(f"✅ Validating YouTube URL: {url}")
        
        youtube_domains = ['youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com']
        
        try:
            parsed = urlparse(url)
            is_valid = parsed.netloc in youtube_domains and self.extract_video_id(url) is not None
            logger.debug(f"URL validation result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.warning(f"⚠️ URL validation failed: {e}")
            return False
    
    def get_video_info(self, url: str) -> Optional[YouTubeVideoInfo]:
        """Extract comprehensive video metadata using yt-dlp."""
        logger.info(f"📊 Extracting video metadata for: {url}")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
                info = ydl.extract_info(url, download=False)
                
                video_info = YouTubeVideoInfo(
                    video_id=info.get('id', ''),
                    title=info.get('title', 'Unknown Title'), # type: ignore
                    description=info.get('description', ''),
                    uploader=info.get('uploader', 'Unknown'), # type: ignore
                    upload_date=info.get('upload_date', ''),
                    duration=info.get('duration'),
                    view_count=info.get('view_count'),
                    like_count=info.get('like_count'),
                    channel=info.get('channel', info.get('uploader', 'Unknown')),
                    channel_id=info.get('channel_id', ''),
                    tags=info.get('tags', []), # type: ignore
                    categories=info.get('categories', []),
                    thumbnail=info.get('thumbnail'),
                    webpage_url=info.get('webpage_url', url)
                )
                
                logger.info("📈 Video metadata extracted:")
                logger.info(f"   • Title: {video_info.title}")
                logger.info(f"   • Channel: {video_info.channel}")
                logger.info(f"   • Duration: {video_info.duration}s" if video_info.duration else "   • Duration: Unknown")
                logger.info(f"   • Views: {video_info.view_count:,}" if video_info.view_count else "   • Views: Unknown")
                
                return video_info
                
        except Exception as e:
            logger.error(f"❌ Failed to extract video metadata: {e}")
            return None
    
    def download_audio(self, url: str, video_info: Optional[YouTubeVideoInfo] = None) -> str:
        """Download audio from YouTube video with caching and quality control."""
        
        if video_info is None:
            video_info = self.get_video_info(url)
            if not video_info:
                raise ValueError("Could not extract video information")
        
        logger.info(f"🎵 Downloading audio for: {video_info.title}")
        logger.info(f"   • Video ID: {video_info.video_id}")
        logger.info(f"   • Channel: {video_info.channel}")
        
        # Check duration limit
        if self.max_duration and video_info.duration and video_info.duration > self.max_duration:
            raise ValueError(f"Video duration ({video_info.duration}s) exceeds maximum allowed ({self.max_duration}s)")
        
        # Generate cache filename
        audio_filename = f"{video_info.video_id}_{self.audio_quality}.m4a"
        audio_path = self.temp_dir / audio_filename
        
        # Check if cached audio exists
        if self.cache_audio and audio_path.exists():
            logger.info(f"✅ Using cached audio: {audio_path}")
            return str(audio_path)
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': self._get_audio_format(),
            'outtmpl': str(self.temp_dir / f"{video_info.video_id}_%(quality)s.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        logger.info(f"⬇️ Downloading audio with quality: {self.audio_quality}")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
                ydl.download([url])
            
            # Find the downloaded file
            for potential_file in self.temp_dir.glob(f"{video_info.video_id}_*.m4a"):
                if potential_file.exists():
                    # Rename to standard format if needed
                    if potential_file != audio_path:
                        potential_file.rename(audio_path)
                    
                    file_size = audio_path.stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"✅ Audio downloaded successfully: {audio_path}")
                    logger.info(f"   • File size: {file_size:.1f} MB")
                    return str(audio_path)
            
            raise FileNotFoundError(f"Downloaded audio file not found in {self.temp_dir}")
            
        except Exception as e:
            logger.error(f"❌ Audio download failed: {e}")
            raise
    
    def _get_audio_format(self) -> str:
        """Get yt-dlp format string based on quality preference."""
        if self.audio_quality == "best":
            return "bestaudio[ext=m4a]/bestaudio/best"
        elif self.audio_quality == "worst":
            return "worstaudio[ext=m4a]/worstaudio/worst"
        else:
            return f"{self.audio_quality}/bestaudio/best"
    
    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]) -> Dict[str, List[Document]]:
        """
        Transcribe YouTube videos with comprehensive analysis.
        
        :param sources: List of YouTube URLs
        :return: Dictionary with 'documents' key containing transcribed documents
        """
        
        logger.info("=" * 100)
        logger.info("🎬 STARTING YOUTUBE AUDIO TRANSCRIPTION PIPELINE")
        logger.info("=" * 100)
        
        logger.info("📋 Input Analysis:")
        logger.info(f"   • Number of YouTube URLs: {len(sources)}")
        
        all_documents = []
        
        for i, url in enumerate(sources, 1):
            logger.info(f"\n🎯 Processing YouTube video {i}/{len(sources)}")
            logger.info(f"   • URL: {url}")
            
            try:
                # Validate YouTube URL
                if not self.validate_youtube_url(url):
                    logger.error(f"❌ Invalid YouTube URL: {url}")
                    continue
                
                # Get video metadata
                video_info = self.get_video_info(url)
                if not video_info:
                    logger.error(f"❌ Could not extract video metadata for: {url}")
                    continue
                
                # Download audio
                audio_path = self.download_audio(url, video_info)
                
                # Transcribe using AssemblyAI
                logger.info("🎙️ Starting transcription with AssemblyAI...")
                transcription_result = self.transcriber.run(sources=[audio_path])
                
                # Process and enhance documents with YouTube metadata
                youtube_documents = self._enhance_documents_with_youtube_metadata(
                    transcription_result["documents"],
                    video_info,
                    url
                )
                
                all_documents.extend(youtube_documents)
                
                logger.info(f"✅ YouTube video {i} processed successfully")
                logger.info(f"   • Generated {len(youtube_documents)} documents")
                
                # Cleanup audio file if requested
                if self.cleanup_audio and not self.cache_audio:
                    try:
                        os.unlink(audio_path)
                        logger.debug(f"🗑️ Cleaned up audio file: {audio_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not cleanup audio file: {e}")
                        
            except Exception as e:
                logger.error(f"❌ Error processing YouTube video {i}: {e}")
                continue
        
        # Final summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 YOUTUBE TRANSCRIPTION SUMMARY")
        logger.info("=" * 100)
        logger.info(f"✅ Successfully processed {len([d for d in all_documents if 'main_transcript' in d.meta.get('content_type', '')])}/{len(sources)} YouTube videos")
        logger.info(f"📄 Total documents generated: {len(all_documents)}")
        
        # Document type breakdown
        doc_types = {}
        for doc in all_documents:
            content_type = doc.meta.get('content_type', 'unknown')
            doc_types[content_type] = doc_types.get(content_type, 0) + 1
        
        if doc_types:
            logger.info("📋 Document type distribution:")
            for doc_type, count in doc_types.items():
                logger.info(f"   • {doc_type}: {count}")
        
        logger.info("=" * 100)
        
        return {"documents": all_documents}
    
    def _enhance_documents_with_youtube_metadata(
        self,
        documents: List[Document],
        video_info: YouTubeVideoInfo,
        original_url: str
    ) -> List[Document]:
        """Enhance transcription documents with YouTube-specific metadata."""
        
        logger.info(f"🔧 Enhancing {len(documents)} documents with YouTube metadata")
        
        enhanced_documents = []
        
        for i, doc in enumerate(documents):
            # Create enhanced metadata
            enhanced_meta = doc.meta.copy()
            
            # Add YouTube-specific metadata
            enhanced_meta.update({
                # Source information
                'source_type': 'youtube_video',
                'youtube_url': original_url,
                'original_source': original_url,
                
                # Video metadata
                'video_info': video_info.to_dict(),
                
                # Content identification
                'video_id': video_info.video_id,
                'video_title': video_info.title,
                'channel_name': video_info.channel,
                'channel_id': video_info.channel_id,
                'upload_date': video_info.upload_date,
                
                # Processing metadata
                'processed_timestamp': datetime.now().isoformat(),
                'transcriber_version': '2.0',
                'processing_method': 'youtube_audio_transcription',
                
                # Citation information
                'citation': {
                    'title': video_info.title,
                    'channel': video_info.channel,
                    'url': original_url,
                    'video_id': video_info.video_id,
                    'upload_date': video_info.upload_date,
                    'access_date': datetime.now().isoformat(),
                    'duration': video_info.duration
                }
            })
            
            # Determine content type for better organization
            if 'content_type' not in enhanced_meta:
                if i == 0 and len(documents) > 1:
                    enhanced_meta['content_type'] = 'main_transcript'
                elif 'sentence' in str(doc.content).lower():
                    enhanced_meta['content_type'] = 'sentence'
                elif 'paragraph' in str(doc.content).lower():
                    enhanced_meta['content_type'] = 'paragraph'
                else:
                    enhanced_meta['content_type'] = 'transcript_chunk'
            
            # Create enhanced document
            enhanced_doc = Document(
                content=doc.content,
                meta=enhanced_meta
            )
            
            enhanced_documents.append(enhanced_doc)
        
        logger.info("✅ Enhanced all documents with YouTube metadata")
        
        return enhanced_documents
    
    def cleanup_temp_files(self, keep_cache: bool = True) -> None:
        """Clean up temporary files."""
        logger.info("🗑️ Cleaning up temporary files...")
        
        try:
            if self.temp_dir.exists():
                for file_path in self.temp_dir.glob("*.m4a"):
                    if keep_cache and self.cache_audio:
                        continue  # Keep cached files
                    file_path.unlink()
                    logger.debug(f"Deleted: {file_path}")
                
                # Remove directory if empty
                if not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
                    logger.info("✅ Temporary directory removed")
                else:
                    logger.info(f"✅ Temporary files cleaned (cache preserved: {keep_cache and self.cache_audio})")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up temporary files: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        return {
            'type': 'YouTubeAudioTranscriber',
            'init_parameters': {
                'assemblyai_api_key': '***',  # Hide API key
                'audio_config': self.transcriber.config.__dict__,
                'temp_dir': str(self.temp_dir),
                'cleanup_audio': self.cleanup_audio,
                'cache_audio': self.cache_audio,
                'audio_quality': self.audio_quality,
                'max_duration': self.max_duration
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "YouTubeAudioTranscriber":
        """Deserialize the component from a dictionary."""
        init_params = data.get('init_parameters', {})
        # Remove sensitive data
        init_params.pop('assemblyai_api_key', None)
        return cls(**init_params)


def create_youtube_transcription_pipeline(
    assemblyai_api_key: Optional[str] = None,
    enable_advanced_features: bool = True,
    max_video_duration: Optional[int] = 3600  # 1 hour default
) -> 'YouTubeAudioTranscriber':
    """
    Create a pre-configured YouTube transcription pipeline.
    
    :param assemblyai_api_key: AssemblyAI API key
    :param enable_advanced_features: Whether to enable all advanced features
    :param max_video_duration: Maximum video duration in seconds
    :return: Configured YouTubeAudioTranscriber
    """
    
    logger.info("🏗️ Creating YouTube transcription pipeline")
    
    if enable_advanced_features:
        config = AudioProcessingConfig(
            # Enable all advanced features for comprehensive analysis
            speaker_labels=True,
            sentiment_analysis=True,
            entity_detection=True,
            iab_categories=True,
            content_safety=True,
            auto_highlights=True,
            auto_chapters=True,
            summarization=True,
            include_sentences=True,
            include_paragraphs=True,
            # YouTube-optimized settings
            noise_reduction=True,
            automatic_punctuation=True,
            format_text=True,
            boost_param="high"
        )
    else:
        config = AudioProcessingConfig(
            # Basic configuration for faster processing
            speaker_labels=True,
            automatic_punctuation=True,
            format_text=True
        )
    
    transcriber = YouTubeAudioTranscriber(
        assemblyai_api_key=assemblyai_api_key,
        audio_config=config,
        cleanup_audio=True,
        cache_audio=True,
        audio_quality="best",
        max_duration=max_video_duration
    )
    
    logger.info("✅ YouTube transcription pipeline created successfully")
    return transcriber


# Example usage functions for testing and demonstration
if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("❌ Please set ASSEMBLYAI_API_KEY environment variable")
        exit(1)
    
    # Create transcriber
    transcriber = create_youtube_transcription_pipeline(
        assemblyai_api_key=api_key,
        enable_advanced_features=True,
        max_video_duration=1800  # 30 minutes
    )
    
    # Test URLs (replace with actual YouTube URLs)
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example URL
    ]
    
    try:
        # Run transcription
        result = transcriber.run(sources=test_urls)
        documents = result["documents"]
        
        print("\n🎉 Transcription completed!")
        print(f"📊 Generated {len(documents)} documents")
        
        # Display sample results
        for i, doc in enumerate(documents[:3], 1):
            print(f"\n📄 Document {i}:")
            print(f"   Type: {doc.meta.get('content_type', 'unknown')}")
            print(f"   Video: {doc.meta.get('video_title', 'Unknown')}")
            print(f"   Content: {str(doc.content)[:200]}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        transcriber.cleanup_temp_files(keep_cache=True)
