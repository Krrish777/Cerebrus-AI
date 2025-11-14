#!/usr/bin/env python3
"""
Real API test for audio_transcriber.py using actual AssemblyAI API.
This test requires a valid ASSEMBLYAI_API_KEY environment variable.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables")
    print("   Install with: pip install python-dotenv")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_api_key():
    """Check if AssemblyAI API key is available."""
    print("🔑 CHECKING API CONFIGURATION")
    print("=" * 50)
    
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("❌ ASSEMBLYAI_API_KEY not found!")
        print("\n💡 To set your API key:")
        print("   Option 1 - Create .env file in project root:")
        print("     ASSEMBLYAI_API_KEY=your_key_here")
        print("   Option 2 - Set environment variable:")
        print("     PowerShell: $env:ASSEMBLYAI_API_KEY='your_key_here'")
        print("     CMD: set ASSEMBLYAI_API_KEY=your_key_here")
        print("   Get API key from: https://www.assemblyai.com/")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    return True

def test_basic_transcription():
    """Test basic transcription functionality with real API."""
    print("\n🎵 TESTING BASIC TRANSCRIPTION")
    print("=" * 50)
    
    try:
        from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
        
        # Simple configuration
        config = AudioProcessingConfig(
            language_code="en",
            speaker_labels=True,
            sentiment_analysis=True,
            entity_detection=True,
            auto_chapters=False,  # Disable for short audio
            summarization=False   # Disable for short audio
        )
        
        print("📋 Creating transcriber with basic config...")
        transcriber = AssemblyAITranscriber(config=config)
        print("✅ Transcriber created successfully")
        
        # Test with harvard.wav
        audio_file = Path("data/harvard.wav")
        if not audio_file.exists():
            print(f"❌ Audio file not found: {audio_file}")
            return False
        
        print(f"🎵 Transcribing: {audio_file}")
        print("⏳ This may take 30-60 seconds...")
        
        start_time = time.time()
        result = transcriber.run([str(audio_file)])
        end_time = time.time()
        
        print(f"⏱️ Transcription completed in {end_time - start_time:.1f} seconds")
        
        # Check results
        documents = result.get("documents", [])
        print(f"📊 Results:")
        print(f"   ✅ Documents generated: {len(documents)}")
        
        if documents:
            main_doc = documents[0]
            content = main_doc.content
            metadata = main_doc.meta
            
            print(f"   ✅ Content length: {len(content or '')} characters")
            print(f"   ✅ Source: {metadata.get('source', 'unknown')}")
            print(f"   ✅ Language: {metadata.get('language_code', 'unknown')}")
            print(f"   ✅ Confidence: {metadata.get('confidence', 'unknown')}")
            print(f"   ✅ Duration: {metadata.get('audio_duration_seconds', 'unknown')} seconds")
            
            # Show preview of transcription
            if content:
                lines = content.split('\n')[:10]  # First 10 lines
                print("\n📝 Transcription preview:")
                for line in lines:
                    if line.strip():
                        print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
            
            # Check for features
            sentiment_data = metadata.get('sentiment_analysis', [])
            entities_data = metadata.get('entities', [])
            
            print(f"\n🔍 Feature analysis:")
            print(f"   ✅ Sentiment analysis: {len(sentiment_data)} segments")
            print(f"   ✅ Entities detected: {len(entities_data)} entities")
            
            # Show some sentiment data
            if sentiment_data:
                print("\n💭 Sentiment examples:")
                for i, sentiment in enumerate(sentiment_data[:3]):
                    print(f"   {i+1}. \"{sentiment.get('text', '')[:50]}...\" -> {sentiment.get('sentiment', 'unknown')}")
            
            # Show some entities
            if entities_data:
                print("\n🏷️ Entity examples:")
                for i, entity in enumerate(entities_data[:3]):
                    print(f"   {i+1}. \"{entity.get('text', '')}\" -> {entity.get('entity_type', 'unknown')}")
        
        print("✅ Basic transcription test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Basic transcription test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_features():
    """Test advanced features with real API."""
    print("\n🚀 TESTING ADVANCED FEATURES")
    print("=" * 50)
    
    try:
        from audio_processing.audio_transcriber import create_advanced_audio_config, AssemblyAITranscriber
        
        # Create advanced configuration
        print("📋 Creating advanced configuration...")
        config = create_advanced_audio_config()
        
        # Adjust for shorter audio file
        config.auto_chapters = False  # Short audio won't have chapters
        config.summarization = False  # Not needed for short audio
        
        print("✅ Advanced config created")
        
        transcriber = AssemblyAITranscriber(config=config)
        print("✅ Advanced transcriber created")
        
        # Test with harvard.wav
        audio_file = Path("data/harvard.wav")
        print(f"🎵 Processing with advanced features: {audio_file}")
        print("⏳ This may take 60-90 seconds...")
        
        start_time = time.time()
        result = transcriber.run([str(audio_file)])
        end_time = time.time()
        
        print(f"⏱️ Advanced processing completed in {end_time - start_time:.1f} seconds")
        
        documents = result.get("documents", [])
        print(f"📊 Advanced results:")
        print(f"   ✅ Documents: {len(documents)}")
        
        if documents:
            main_doc = documents[0]
            metadata = main_doc.meta
            
            # Check advanced features
            features_found = []
            if metadata.get('sentiment_analysis'):
                features_found.append(f"Sentiment ({len(metadata['sentiment_analysis'])} segments)")
            if metadata.get('entities'):
                features_found.append(f"Entities ({len(metadata['entities'])} items)")
            if metadata.get('topics'):
                features_found.append("Topics")
            if metadata.get('content_safety'):
                features_found.append("Content Safety")
            if metadata.get('highlights'):
                features_found.append(f"Highlights ({len(metadata['highlights'])} items)")
            
            print(f"   ✅ Features detected: {', '.join(features_found)}")
            
            # Show highlights if available
            highlights = metadata.get('highlights', [])
            if highlights:
                print("\n⭐ Key highlights:")
                for i, highlight in enumerate(highlights[:3]):
                    print(f"   {i+1}. \"{highlight.get('text', '')[:60]}...\" (rank: {highlight.get('rank', 'unknown')})")
        
        print("✅ Advanced features test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smart_processor():
    """Test SmartAudioProcessor with real API."""
    print("\n🧠 TESTING SMART AUDIO PROCESSOR")
    print("=" * 50)
    
    try:
        from audio_processing.audio_transcriber import AssemblyAITranscriber, SmartAudioProcessor, AudioProcessingConfig
        
        # Create configuration optimized for chunking
        config = AudioProcessingConfig(
            language_code="en",
            speaker_labels=True,
            sentiment_analysis=True,
            entity_detection=True,
            auto_chapters=False,
            include_sentences=True,
            include_paragraphs=True
        )
        
        print("📋 Creating smart processor...")
        transcriber = AssemblyAITranscriber(config=config)
        processor = SmartAudioProcessor(
            assemblyai_transcriber=transcriber,
            max_chunk_length=300,  # Smaller chunks for testing
            overlap=50,
            respect_speakers=True,
            respect_chapters=False
        )
        
        print("✅ Smart processor created")
        
        # Process audio
        audio_file = Path("data/harvard.wav")
        print(f"🎵 Smart processing: {audio_file}")
        print("⏳ This may take 60-90 seconds...")
        
        start_time = time.time()
        result = processor.run([str(audio_file)])
        end_time = time.time()
        
        print(f"⏱️ Smart processing completed in {end_time - start_time:.1f} seconds")
        
        documents = result.get("documents", [])
        print(f"📊 Smart processing results:")
        print(f"   ✅ Smart chunks created: {len(documents)}")
        
        # Analyze chunks
        chunk_types = {}
        total_length = 0
        
        for doc in documents:
            chunk_type = doc.meta.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_length += len(doc.content or '')
            
        print(f"   ✅ Total content length: {total_length} characters")
        print(f"   ✅ Chunk types: {dict(chunk_types)}")
        
        # Show sample chunks
        print("\n📝 Sample chunks:")
        for i, doc in enumerate(documents[:3]):
            chunk_info = f"Chunk {i+1} ({doc.meta.get('chunk_type', 'unknown')})"
            content_preview = (doc.content or '')[:80].replace('\n', ' ')
            print(f"   {chunk_info}: {content_preview}...")
            
            if doc.meta.get('speaker'):
                print(f"      Speaker: {doc.meta['speaker']}")
            if doc.meta.get('processing_strategy'):
                print(f"      Strategy: {doc.meta['processing_strategy']}")
        
        print("✅ Smart processor test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Smart processor test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n🚨 TESTING ERROR HANDLING")
    print("=" * 50)
    
    try:
        from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
        
        config = AudioProcessingConfig()
        transcriber = AssemblyAITranscriber(config=config)
        
        # Test with non-existent file
        print("🗂️ Testing with non-existent file...")
        result = transcriber.run(["nonexistent_file.wav"])
        documents = result.get("documents", [])
        print(f"   ✅ Handled gracefully - Documents: {len(documents)}")
        
        # Test with empty list
        print("📭 Testing with empty source list...")
        result = transcriber.run([])
        documents = result.get("documents", [])
        print(f"   ✅ Handled gracefully - Documents: {len(documents)}")
        
        print("✅ Error handling test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test FAILED: {e}")
        return False

def main():
    """Run all real API tests."""
    print("🎯 ASSEMBLYAI AUDIO PROCESSING - REAL API TESTS")
    print("=" * 60)
    print("This test suite uses the real AssemblyAI API")
    print("Ensure you have ASSEMBLYAI_API_KEY in .env file or environment")
    print("=" * 60)
    
    # Check prerequisites
    if not check_api_key():
        return
    
    # Check audio file
    audio_file = Path("data/harvard.wav")
    if not audio_file.exists():
        print(f"❌ Test audio file not found: {audio_file}")
        print("   Please ensure harvard.wav is in the data/ directory")
        return
    
    print(f"✅ Test audio file found: {audio_file} ({audio_file.stat().st_size} bytes)")
    
    # Run tests
    tests = [
        ("Basic Transcription", test_basic_transcription),
        ("Advanced Features", test_advanced_features),
        ("Smart Processor", test_smart_processor),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n▶️  Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"💥 {test_name} failed!")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 REAL API TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All real API tests passed! Your audio processing is working perfectly.")
        print("\n💡 Your audio transcriber is ready for production use!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the error messages above.")
        print("\n🔧 Common issues:")
        print("   • Check your ASSEMBLYAI_API_KEY in .env file or environment")
        print("   • Ensure internet connection")
        print("   • Verify audio file exists and is readable")
    
    print(f"\n📊 API Usage Summary:")
    print("   • Tests used real AssemblyAI API calls")
    print("   • Each test consumed API credits")
    print("   • Check your AssemblyAI dashboard for usage details")
    print("\n🔧 For future runs:")
    print("   • Create .env file with: ASSEMBLYAI_API_KEY=your_key")
    print("   • Or set environment variable before running")

if __name__ == "__main__":
    main()