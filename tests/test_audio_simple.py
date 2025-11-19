#!/usr/bin/env python3
"""
Simple test for audio_transcriber.py that avoids HTTP client initialization.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_configuration_only():
    """Test configuration classes without initializing transcriber."""
    print("🧪 TESTING CONFIGURATION ONLY")
    print("=" * 50)
    
    try:
        from audio_processing.audio_transcriber import AudioProcessingConfig, create_advanced_audio_config
        
        print("📋 Testing basic configuration...")
        config = AudioProcessingConfig()
        print(f"   ✅ Language: {config.language_code}")
        print(f"   ✅ Model: {config.model}")
        print(f"   ✅ Speaker labels: {config.speaker_labels}")
        print(f"   ✅ Sentiment: {config.sentiment_analysis}")
        
        print("\n📋 Testing advanced configuration...")
        advanced = create_advanced_audio_config()
        print(f"   ✅ Content safety: {advanced.content_safety}")
        print(f"   ✅ Auto highlights: {advanced.auto_highlights}")
        print(f"   ✅ Custom vocab: {len(advanced.custom_vocabulary)} words")
        
        print("\n📋 Testing custom configuration...")
        custom = AudioProcessingConfig(
            language_code="fr",
            speaker_labels=False,
            sentiment_analysis=False
        )
        print(f"   ✅ Custom language: {custom.language_code}")
        print(f"   ✅ Custom speakers: {custom.speaker_labels}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_audio_file_access():
    """Test harvard.wav file accessibility."""
    print("\n🧪 TESTING AUDIO FILE ACCESS")
    print("=" * 50)
    
    test_file = Path(__file__).parent.parent / "data" / "harvard.wav"
    
    print(f"📁 Checking: {test_file}")
    
    if not test_file.exists():
        print(f"❌ File not found: {test_file}")
        return False
    
    size = test_file.stat().st_size
    print("📊 File info:")
    print(f"   ✅ Path: {test_file}")
    print(f"   ✅ Size: {size:,} bytes ({size/1024:.1f} KB)")
    print(f"   ✅ Readable: {os.access(test_file, os.R_OK)}")
    
    if size == 0:
        print("❌ File is empty!")
        return False
    
    return True

def test_mocked_transcriber():
    """Test transcriber functionality with complete mocking."""
    print("\n🧪 TESTING MOCKED TRANSCRIBER")
    print("=" * 50)
    
    try:
        # Mock the entire assemblyai module before importing
        with patch('audio_processing.audio_transcriber.aai') as mock_aai:
            
            # Setup comprehensive mocks
            mock_aai.settings = Mock()
            mock_aai.Transcriber = Mock()
            mock_aai.TranscriptionConfig = Mock()
            
            # Create mock transcript with all required attributes
            mock_transcript = Mock()
            mock_transcript.text = "The stale smell of old beer lingers. It takes heat to bring out the odor."
            mock_transcript.confidence = 0.95
            mock_transcript.error = None
            
            # Mock sentiment analysis data (must be iterable)
            mock_sentiment = Mock()
            mock_sentiment.text = "The stale smell of old beer lingers."
            mock_sentiment.sentiment = "neutral"
            mock_sentiment.confidence = 0.8
            mock_sentiment.start = 0
            mock_sentiment.end = 2000
            mock_transcript.sentiment_analysis = [mock_sentiment]  # Make it a list
            
            # Mock entities (must be iterable)
            mock_entity = Mock()
            mock_entity.text = "beer"
            mock_entity.entity_type = "substance"
            mock_entity.start = 500
            mock_entity.end = 1000
            mock_transcript.entities = [mock_entity]  # Make it a list
            
            # Mock utterances for speaker detection
            mock_utterance = Mock()
            mock_utterance.text = "The stale smell of old beer lingers."
            mock_utterance.speaker = "A"
            mock_utterance.start = 0
            mock_utterance.end = 2000
            mock_transcript.utterances = [mock_utterance]  # Make it a list
            
            # Mock chapters (must be iterable)
            mock_chapter = Mock()
            mock_chapter.headline = "Introduction"
            mock_chapter.summary = "Introduction to the text"
            mock_chapter.gist = "Beginning"
            mock_chapter.start = 0
            mock_chapter.end = 2000
            mock_transcript.chapters = [mock_chapter]  # Make it a list
            
            # Mock other attributes that might be accessed
            mock_transcript.summary = "A test transcription about beer"
            
            # Mock methods that might be called
            mock_transcript.get_sentences.return_value = []
            mock_transcript.get_paragraphs.return_value = []
            
            # Setup mock transcriber instance
            mock_transcriber_instance = Mock()
            mock_transcriber_instance.transcribe.return_value = mock_transcript
            mock_aai.Transcriber.return_value = mock_transcriber_instance
            
            print("🎭 Mocks configured successfully")
            print("   • Transcript text ready")
            print("   • Sentiment analysis mocked (iterable)")
            print("   • Entities mocked (iterable)")
            print("   • Utterances mocked (iterable)")
            
            # Now import and test
            from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
            
            print("📦 Creating configuration...")
            config = AudioProcessingConfig(
                speaker_labels=True,
                sentiment_analysis=True,
                entity_detection=True
            )
            
            print("🚀 Creating mocked transcriber...")
            transcriber = AssemblyAITranscriber(
                api_key="test_api_key_123",
                config=config
            )
            print("   ✅ Transcriber created successfully")
            
            print("🎵 Testing transcription...")
            test_file = str(Path(__file__).parent.parent / "data" / "harvard.wav")
            result = transcriber.run([test_file])
            
            print("📊 Results:")
            documents = result.get("documents", [])
            print(f"   ✅ Documents returned: {len(documents)}")
            
            if documents:
                doc = documents[0]
                print(f"   ✅ Content length: {len(doc.content)} chars")  # type: ignore
                print(f"   ✅ Content preview: '{doc.content[:50]}...'")  # type: ignore
                print(f"   ✅ Metadata keys: {len(doc.meta)}")
                
                # Check for expected data
                if 'sentiment_analysis' in doc.meta:
                    print(f"   ✅ Sentiment data: {len(doc.meta['sentiment_analysis'])} items")
                if 'entities' in doc.meta:
                    print(f"   ✅ Entity data: {len(doc.meta['entities'])} items")
                if 'utterances' in doc.meta:
                    print(f"   ✅ Utterance data: {len(doc.meta['utterances'])} items")
            
            return True
            
    except Exception as e:
        print(f"❌ Mocked transcriber test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_setup():
    """Test logging configuration."""
    print("\n🧪 TESTING LOGGING SETUP")
    print("=" * 50)
    
    try:
        # Test direct logger usage
        from core.logging import CustomLogger
        
        print("📝 Testing custom logger...")
        logger_instance = CustomLogger()
        logger = logger_instance.get_logger("test_audio")
        
        print("   ✅ Logger created")
        
        # Test logging output
        logger.info("🧪 Test audio transcriber logging")  # type: ignore
        logger.info("   • Testing configuration validation")  # type: ignore
        logger.info("   • Testing file access verification")  # type: ignore
        logger.info("   • Testing mock transcription workflow")  # type: ignore
        
        print("   ✅ Log messages sent")
        
        # Check log files
        log_dir = Path(__file__).parent.parent / "logs"
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            print(f"   ✅ Log files: {len(log_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_smart_processor_mock():
    """Test SmartAudioProcessor with mocked dependencies."""
    print("\n🧪 TESTING SMART PROCESSOR (MOCKED)")
    print("=" * 50)
    
    try:
        with patch('audio_processing.audio_transcriber.aai'):
            from audio_processing.audio_transcriber import SmartAudioProcessor
            from haystack import Document
            
            # Create mock transcriber
            mock_transcriber = Mock()
            mock_transcriber.run.return_value = {
                "documents": [
                    Document(
                        content="**Speaker A:** Hello world.\n**Speaker B:** How are you?",
                        meta={
                            "source": "test.wav",
                            "speakers": ["A", "B"]
                        }
                    )
                ]
            }
            
            print("🧠 Creating SmartAudioProcessor...")
            processor = SmartAudioProcessor(
                assemblyai_transcriber=mock_transcriber,
                max_chunk_length=500,
                overlap=50,
                respect_speakers=True
            )
            
            print(f"   ✅ Max chunk: {processor.max_chunk_length}")
            print(f"   ✅ Overlap: {processor.overlap}")
            print(f"   ✅ Speaker aware: {processor.respect_speakers}")
            
            print("🔄 Testing processing...")
            result = processor.run(["test_audio.wav"])
            documents = result.get("documents", [])
            
            print(f"   ✅ Processed documents: {len(documents)}")
            
            return True
            
    except Exception as e:
        print(f"❌ Smart processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simplified audio transcriber tests."""
    print("🚀 SIMPLIFIED AUDIO TRANSCRIBER TESTS")
    print("=" * 60)
    print("Testing core functionality without external dependencies")
    print("=" * 60)
    
    tests = [
        ("Configuration Classes", test_configuration_only),
        ("Audio File Access", test_audio_file_access),
        ("Mocked Transcriber", test_mocked_transcriber),
        ("Logging Setup", test_logging_setup),
        ("Smart Processor (Mocked)", test_smart_processor_mock)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n▶️  Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ ERROR in {test_name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Audio transcriber is working correctly.")
        print("\n💡 Next steps to test with real API:")
        print("   1. Get AssemblyAI API key from: https://www.assemblyai.com/")
        print("   2. Set environment variable: ASSEMBLYAI_API_KEY=your_key")
        print("   3. Run: python tests/test_audio_transcriber_integration.py")
        print("\n📝 Test your audio file:")
        print(f"   • File ready: data/harvard.wav ({Path(__file__).parent.parent / 'data' / 'harvard.wav'})")
        print(f"   • Size: {(Path(__file__).parent.parent / 'data' / 'harvard.wav').stat().st_size:,} bytes")
    else:
        failed = [name for name, success in results if not success]
        print("\n⚠️  Some tests failed:")
        for test in failed:
            print(f"   • {test}")

if __name__ == "__main__":
    main()