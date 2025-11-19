#!/usr/bin/env python3
"""
Simple direct test of YouTube audio transcription without Haystack dependencies.
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded environment variables from .env file")

# Load the environment
load_env_file()

# Check API key
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    print("❌ No ASSEMBLYAI_API_KEY found in environment!")
    sys.exit(1)

print("✅ API key found")
print("🎬 Testing Direct YouTube Audio Download and Transcription")
print("=" * 60)

try:
    import yt_dlp
    print("✅ yt-dlp available")
except ImportError:
    print("❌ yt-dlp not available. Install with: pip install yt-dlp")
    sys.exit(1)

try:
    import assemblyai as aai
    print("✅ AssemblyAI SDK available")
except ImportError:
    print("❌ AssemblyAI SDK not available. Install with: pip install assemblyai")
    sys.exit(1)

# Set up AssemblyAI
aai.settings.api_key = api_key

def test_youtube_transcription():
    """Test direct YouTube transcription without Haystack."""
    print("\n🎬 Testing Direct YouTube Transcription")
    print("=" * 50)
    
    # Use the first YouTube video (19 seconds)
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    print(f"📹 Testing with: {test_url}")
    
    # Download audio using yt-dlp
    print("🎵 Downloading audio...")
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': 'temp_%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info
            info = ydl.extract_info(test_url, download=False)
            print(f"   • Title: {info.get('title', 'Unknown')}")
            print(f"   • Duration: {info.get('duration', 'Unknown')}s")
            
            # Download audio
            ydl.download([test_url])
            
            # Find the downloaded file
            video_id = info.get('id', 'unknown')
            audio_file = f"temp_{video_id}.m4a"
            
            if os.path.exists(audio_file):
                print(f"✅ Audio downloaded: {audio_file}")
                
                # Transcribe with AssemblyAI
                print("🎙️ Starting transcription...")
                
                config = aai.TranscriptionConfig(
                    speaker_labels=True,
                    sentiment_analysis=True,
                    entity_detection=True,
                    summarization=True,  # Only summarization, no auto_chapters
                    auto_highlights=True
                )
                
                transcriber = aai.Transcriber(config=config)
                transcript = transcriber.transcribe(audio_file)
                
                if transcript.status == aai.TranscriptStatus.error:
                    print(f"❌ Transcription failed: {transcript.error}")
                else:
                    print("✅ Transcription completed!")
                    print("\n📄 Transcript:")
                    print(f"   Text: {transcript.text}")
                    
                    if hasattr(transcript, 'summary') and transcript.summary:
                        print(f"\n📋 Summary: {transcript.summary}")
                    
                    if hasattr(transcript, 'sentiment_analysis_results') and transcript.sentiment_analysis_results:
                        print(f"\n😊 Sentiment: {len(transcript.sentiment_analysis_results)} segments analyzed")
                    
                    if hasattr(transcript, 'entities') and transcript.entities:
                        print(f"\n🏷️ Entities: {len(transcript.entities)} entities detected")
                        
                    if hasattr(transcript, 'auto_highlights_result') and transcript.auto_highlights_result:
                        print(f"\n⭐ Highlights: {len(transcript.auto_highlights_result.results)} key moments")
                
                # Cleanup
                os.remove(audio_file)
                print(f"\n🗑️ Cleaned up temporary file: {audio_file}")
                
            else:
                print(f"❌ Audio file not found: {audio_file}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        test_youtube_transcription()
        print("\n🎉 Direct transcription test completed!")
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")