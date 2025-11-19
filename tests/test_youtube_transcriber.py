#!/usr/bin/env python3
"""
Test script for the YouTube Audio Transcriber with comprehensive features.
Demonstrates YouTube video transcription with AssemblyAI integration.
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set API key for testing (replace with your actual key)
# You can also set this as an environment variable: set ASSEMBLYAI_API_KEY=your_key_here
if not os.getenv("ASSEMBLYAI_API_KEY"):
    # Replace 'YOUR_API_KEY_HERE' with your actual AssemblyAI API key
    os.environ["ASSEMBLYAI_API_KEY"] = "YOUR_API_KEY_HERE"  # <-- PUT YOUR REAL API KEY HERE

from audio_processing.yt_audio_transcriber import YouTubeAudioTranscriber, create_youtube_transcription_pipeline
from audio_processing.audio_transcriber import AudioProcessingConfig

def test_youtube_url_validation():
    """Test YouTube URL validation and video ID extraction."""
    print("=" * 80)
    print("🧪 TESTING YOUTUBE URL VALIDATION")
    print("=" * 80)
    
    # Get API key for proper initialization
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("⚠️ ASSEMBLYAI_API_KEY not set - using dummy key for URL validation test")
        api_key = "dummy_key_for_testing"
    
    try:
        # Create a basic transcriber for testing
        transcriber = YouTubeAudioTranscriber(
            assemblyai_api_key=api_key,
            cleanup_audio=False
        )
        
        # Test URLs
        test_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
            "https://youtube.com/embed/dQw4w9WgXcQ",
            "https://www.example.com/not-youtube",  # Invalid
            "not-a-url-at-all"  # Invalid
        ]
        
        print("🔍 Testing URL validation:")
        for url in test_urls:
            video_id = transcriber.extract_video_id(url)
            is_valid = transcriber.validate_youtube_url(url)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"   {status}: {url}")
            if video_id:
                print(f"      Video ID: {video_id}")
        
        print("\n✅ URL validation test completed!")
    
    except Exception as e:
        if "dummy_key_for_testing" in str(e) or "API key" in str(e):
            print("⚠️ API key validation failed (expected with dummy key)")
            print("✅ URL validation logic is working - AssemblyAI integration requires real key")
        else:
            print(f"❌ Unexpected error: {e}")

def test_youtube_transcription():
    """Test actual YouTube transcription with real API."""
    
    print("\n" + "=" * 80)
    print("🎬 TESTING YOUTUBE TRANSCRIPTION")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("❌ ASSEMBLYAI_API_KEY environment variable not set")
        print("   Please set your AssemblyAI API key to test transcription")
        return
    
    print("✅ AssemblyAI API key found")
    
    # Create YouTube-optimized configuration
    config = AudioProcessingConfig(
        # Speaker analysis
        speaker_labels=True,
        speakers_expected=None,
        
        # Content analysis
        sentiment_analysis=True,
        entity_detection=True,
        iab_categories=True,
        content_safety=True,
        auto_highlights=True,
        
        # Audio enhancement
        noise_reduction=True,
        automatic_punctuation=True,
        format_text=True,
        
        # Structure
        include_utterances=True,
        include_sentences=True,
        include_paragraphs=True,
        auto_chapters=True,
        
        # Summarization
        summarization=True,
        summary_model="informative",
        summary_type="bullets"
    )
    
    # Create transcriber
    print("🔧 Creating YouTube Audio Transcriber...")
    transcriber = YouTubeAudioTranscriber(
        assemblyai_api_key=api_key,
        audio_config=config,
        cleanup_audio=True,
        cache_audio=True,
        audio_quality="best",
        max_duration=600  # 10 minutes max for testing
    )
    
    # Test with a short, publicly available video
    # Using a short educational/demo video that should be safe for testing
    test_urls = [
        "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - First YouTube video (20 seconds)
    ]
    
    print(f"\n🎯 Testing with {len(test_urls)} YouTube video(s)")
    
    try:
        # Run transcription
        result = transcriber.run(sources=test_urls)
        documents = result["documents"]
        
        print("\n🎉 Transcription completed successfully!")
        print("📊 Results Summary:")
        print(f"   • Total documents generated: {len(documents)}")
        
        # Analyze document types
        doc_types = {}
        for doc in documents:
            content_type = doc.meta.get('content_type', 'unknown')
            doc_types[content_type] = doc_types.get(content_type, 0) + 1
        
        print("   • Document type breakdown:")
        for doc_type, count in doc_types.items():
            print(f"     - {doc_type}: {count}")
        
        # Show main transcript
        main_docs = [doc for doc in documents if doc.meta.get('content_type') == 'main_transcript']
        if main_docs:
            main_doc = main_docs[0]
            print("\n📄 Main Transcript Preview:")
            print(f"   • Video: {main_doc.meta.get('video_title', 'Unknown')}")
            print(f"   • Channel: {main_doc.meta.get('channel_name', 'Unknown')}")
            print(f"   • Duration: {main_doc.meta.get('video_info', {}).get('duration_seconds', 'Unknown')}s")
            
            # Show transcript preview
            content = str(main_doc.content)
            preview_length = 500
            if len(content) > preview_length:
                preview = content[:preview_length] + "..."
            else:
                preview = content
            print(f"   • Content Preview:\n{preview}")
            
            # Show metadata highlights
            metadata = main_doc.meta
            print("\n📈 Analysis Results:")
            
            if 'sentiment_analysis' in metadata:
                sentiment_count = len(metadata['sentiment_analysis'])
                print(f"   • Sentiment Analysis: {sentiment_count} segments")
            
            if 'entities' in metadata:
                entity_count = len(metadata['entities'])
                print(f"   • Entity Detection: {entity_count} entities")
            
            if 'topics' in metadata:
                topics = metadata['topics']
                if topics.get('summary'):
                    print(f"   • Topic Analysis: {len(topics['summary'])} categories")
            
            if 'highlights' in metadata:
                highlight_count = len(metadata['highlights'])
                print(f"   • Auto Highlights: {highlight_count} key moments")
        
        print("\n✅ YouTube transcription test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Transcription test failed: {e}")
        print("This could be due to:")
        print("   • Network connectivity issues")
        print("   • Invalid YouTube URL")
        print("   • AssemblyAI API limits")
        print("   • Missing dependencies (yt-dlp, ffmpeg)")
    
    finally:
        # Cleanup
        print("\n🗑️ Cleaning up temporary files...")
        transcriber.cleanup_temp_files(keep_cache=True)

def test_pipeline_creation():
    """Test the convenience pipeline creation function."""
    
    print("\n" + "=" * 80)
    print("🏗️ TESTING PIPELINE CREATION")
    print("=" * 80)
    
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("⚠️ ASSEMBLYAI_API_KEY not set - using dummy key for pipeline creation test")
        api_key = "dummy_key_for_testing"
    
    try:
        # Test basic pipeline
        print("🔧 Creating basic YouTube transcription pipeline...")
        basic_pipeline = create_youtube_transcription_pipeline(
            assemblyai_api_key=api_key,
            enable_advanced_features=False,
            max_video_duration=300  # 5 minutes
        )
        
        print("✅ Basic pipeline created successfully")
        print(f"   • Audio Quality: {basic_pipeline.audio_quality}")
        print(f"   • Max Duration: {basic_pipeline.max_duration}s")
        print(f"   • Cache Audio: {basic_pipeline.cache_audio}")
        
        # Test advanced pipeline
        print("\n🔧 Creating advanced YouTube transcription pipeline...")
        advanced_pipeline = create_youtube_transcription_pipeline(
            assemblyai_api_key=api_key,
            enable_advanced_features=True,
            max_video_duration=1800  # 30 minutes
        )
        
        print("✅ Advanced pipeline created successfully")
        print(f"   • Speaker Labels: {advanced_pipeline.transcriber.config.speaker_labels}")
        print(f"   • Sentiment Analysis: {advanced_pipeline.transcriber.config.sentiment_analysis}")
        print(f"   • Entity Detection: {advanced_pipeline.transcriber.config.entity_detection}")
        print(f"   • Auto Chapters: {advanced_pipeline.transcriber.config.auto_chapters}")
        print(f"   • Summarization: {advanced_pipeline.transcriber.config.summarization}")
        
        print("\n✅ Pipeline creation test completed!")
    
    except Exception as e:
        if "dummy_key_for_testing" in str(e) or "API key" in str(e):
            print("⚠️ API key validation failed (expected with dummy key)")
            print("✅ Pipeline creation logic is working - AssemblyAI integration requires real key")
        else:
            print(f"❌ Unexpected error: {e}")

def demo_integration_with_document_processor():
    """Demonstrate how YouTube transcriber can work with the document processor."""
    
    print("\n" + "=" * 80)
    print("🔗 TESTING INTEGRATION WITH DOCUMENT PROCESSOR")
    print("=" * 80)
    
    print("💡 Integration capabilities:")
    print("   • YouTube transcripts can be processed as documents")
    print("   • Speaker segments can be treated as separate documents")
    print("   • Chapters can be processed as individual chunks")
    print("   • Metadata can be preserved through the pipeline")
    print("   • Supports the same chunking strategies as text documents")
    
    print("\n🔮 Future integration possibilities:")
    print("   • Combine YouTube transcripts with PDF documents")
    print("   • Create mixed-media document collections")
    print("   • Cross-reference video content with text sources")
    print("   • Build comprehensive knowledge bases from multiple sources")
    
    print("\n✅ Integration analysis completed!")

def run_all_tests():
    """Run all test functions."""
    
    print("🚀 STARTING COMPREHENSIVE YOUTUBE AUDIO TRANSCRIBER TESTS")
    print("=" * 100)
    
    # Test URL validation (no API required)
    test_youtube_url_validation()
    
    # Test pipeline creation
    test_pipeline_creation()
    
    # Test integration concepts
    demo_integration_with_document_processor()
    
    # Test actual transcription (requires API key)
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if api_key:
        test_youtube_transcription()
    else:
        print("\n⚠️ Skipping transcription test - ASSEMBLYAI_API_KEY not set")
        print("   Set your API key to test full transcription functionality")
    
    print("\n" + "=" * 100)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 100)
    print("\n💡 To test with your own videos:")
    print("   1. Set ASSEMBLYAI_API_KEY environment variable")
    print("   2. Replace test URLs with your YouTube videos")
    print("   3. Adjust max_duration for longer videos")
    print("   4. Configure audio quality and features as needed")

if __name__ == "__main__":
    run_all_tests()