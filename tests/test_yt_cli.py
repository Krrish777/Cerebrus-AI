#!/usr/bin/env python3
"""
Simple YouTube Transcriber CLI Test Tool

This tool allows you to quickly test YouTube audio transcription with your API key.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded environment variables from .env file")
    else:
        print("ℹ️ No .env file found")

def setup_api_key():
    """Setup AssemblyAI API key from environment."""
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    if not api_key:
        print("❌ ASSEMBLYAI_API_KEY environment variable not found!")
        print("Please set your API key using:")
        print("   PowerShell: $env:ASSEMBLYAI_API_KEY=\"your_api_key_here\"")
        print("   CMD: set ASSEMBLYAI_API_KEY=your_api_key_here")
        print("   Or add it to your system environment variables")
        print("\nGet a free API key at: https://www.assemblyai.com/")
        sys.exit(1)
    
    print("✅ API key found in environment")
    return api_key

def test_basic_functionality():
    """Test basic YouTube URL validation."""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        from audio_processing.yt_audio_transcriber import YouTubeAudioTranscriber
        
        # Create transcriber
        transcriber = YouTubeAudioTranscriber(
            cleanup_audio=True,
            cache_audio=True,
            audio_quality="best",
            max_duration=300  # 5 minutes max for testing
        )
        
        print("✅ YouTube transcriber created successfully")
        
        # Test URL validation
        test_urls = [
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # "Me at the zoo" - First YouTube video
            "https://youtu.be/jNQXAC9IVRw",
            "https://www.example.com/invalid",  # Invalid URL
        ]
        
        print("\n🔍 Testing URL validation:")
        for url in test_urls:
            video_id = transcriber.extract_video_id(url)
            is_valid = transcriber.validate_youtube_url(url)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"   {status}: {url[:50]}...")
            if video_id:
                print(f"      Video ID: {video_id}")
        
        return transcriber
    
    except Exception as e:
        print(f"❌ Error creating transcriber: {e}")
        return None

def test_youtube_transcription(transcriber):
    """Test actual YouTube transcription."""
    print("\n🎬 Testing YouTube Transcription")
    print("=" * 50)
    
    # Use the first YouTube video (very short - 19 seconds)
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    
    print(f"📹 Testing with: {test_url}")
    print("   (This is 'Me at the zoo' - the first YouTube video, only 19 seconds)")
    
    try:
        # Get video info first
        video_info = transcriber.get_video_info(test_url)
        if video_info:
            print("\n📊 Video Information:")
            print(f"   • Title: {video_info.title}")
            print(f"   • Channel: {video_info.channel}")
            print(f"   • Duration: {video_info.duration}s")
            print(f"   • Views: {video_info.view_count:,}" if video_info.view_count else "   • Views: Unknown")
        
        # Ask user confirmation
        response = input("\n🤔 Proceed with transcription? This will use your AssemblyAI credits. (y/n): ").strip().lower()
        
        if response not in ['y', 'yes']:
            print("⏭️ Skipping transcription test")
            return
        
        print("\n🎙️ Starting transcription...")
        print("⏳ This may take a minute...")
        
        # Run transcription
        result = transcriber.run(sources=[test_url])
        documents = result["documents"]
        
        print("\n🎉 Transcription completed!")
        print("📊 Results:")
        print(f"   • Total documents: {len(documents)}")
        
        # Show main transcript
        main_docs = [doc for doc in documents if doc.meta.get('content_type') == 'main_transcript']
        if main_docs:
            main_doc = main_docs[0]
            print("\n📄 Main Transcript:")
            print(f"   • Video: {main_doc.meta.get('video_title', 'Unknown')}")
            
            content = str(main_doc.content)
            if len(content) > 300:
                preview = content[:300] + "..."
            else:
                preview = content
            print(f"   • Content:\n{preview}")
            
            # Show analysis results
            metadata = main_doc.meta
            print("\n📈 Analysis Results:")
            
            if 'sentiment_analysis' in metadata:
                print(f"   • Sentiment Analysis: {len(metadata['sentiment_analysis'])} segments")
            
            if 'entities' in metadata:
                print(f"   • Entity Detection: {len(metadata['entities'])} entities")
            
            if 'topics' in metadata:
                topics = metadata['topics']
                if topics.get('summary'):
                    print("   • Topic Analysis: Available")
            
            if 'highlights' in metadata:
                print(f"   • Auto Highlights: {len(metadata['highlights'])} key moments")
        
        print("\n✅ YouTube transcription test completed successfully!")
    
    except Exception as e:
        print(f"\n❌ Transcription failed: {e}")
        print("This could be due to:")
        print("   • Network connectivity issues")
        print("   • AssemblyAI API limits or invalid key")
        print("   • Missing dependencies (yt-dlp, ffmpeg)")

def test_custom_url(transcriber):
    """Test with user-provided YouTube URL."""
    print("\n🎯 Test Custom YouTube URL")
    print("=" * 50)
    
    print("Enter a YouTube URL to test (or press Enter to skip):")
    custom_url = input("URL: ").strip()
    
    if not custom_url:
        print("⏭️ Skipping custom URL test")
        return
    
    # Validate URL
    if not transcriber.validate_youtube_url(custom_url):
        print("❌ Invalid YouTube URL")
        return
    
    # Get video info
    video_info = transcriber.get_video_info(custom_url)
    if not video_info:
        print("❌ Could not get video information")
        return
    
    print("\n📊 Video Information:")
    print(f"   • Title: {video_info.title}")
    print(f"   • Channel: {video_info.channel}")
    print(f"   • Duration: {video_info.duration}s" if video_info.duration else "   • Duration: Unknown")
    
    # Check duration
    if video_info.duration and video_info.duration > 300:
        print(f"⚠️ Video is {video_info.duration}s long - this may take a while and use more credits")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("⏭️ Skipping long video")
            return
    
    try:
        print("\n🎙️ Starting transcription...")
        result = transcriber.run(sources=[custom_url])
        documents = result["documents"]
        
        print("\n🎉 Custom URL transcription completed!")
        print(f"📊 Generated {len(documents)} documents")
        
    except Exception as e:
        print(f"\n❌ Custom URL transcription failed: {e}")

def main():
    """Main CLI function."""
    print("🎬 YouTube Audio Transcriber - CLI Test Tool")
    print("=" * 60)
    
    # Load .env file first
    load_env_file()
    
    # Setup API key
    api_key = setup_api_key()
    
    # Test basic functionality
    transcriber = test_basic_functionality()
    if not transcriber:
        print("❌ Basic functionality test failed. Exiting.")
        sys.exit(1)
    
    # Menu loop
    while True:
        print("\n🎯 What would you like to test?")
        print("1. Test with short demo video (recommended)")
        print("2. Test with custom YouTube URL")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            test_youtube_transcription(transcriber)
        elif choice == "2":
            test_custom_url(transcriber)
        elif choice == "3":
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    print("\n🎉 Testing completed! Thank you for trying the YouTube Audio Transcriber!")
    
    # Cleanup
    print("\n🗑️ Cleaning up temporary files...")
    transcriber.cleanup_temp_files(keep_cache=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)