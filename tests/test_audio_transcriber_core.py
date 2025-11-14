#!/usr/bin/env python3
"""
Core functionality test for audio_transcriber.py
Tests internal components without requiring external APIs.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_audio_transcriber_imports():
    """Test that all audio transcriber components can be imported."""
    print("🧪 TESTING IMPORTS")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import (
            AudioProcessingConfig,
            AssemblyAITranscriber,
            SmartAudioProcessor,
            create_audio_pipeline,
            create_advanced_audio_config
        )
        print("✅ All components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_audio_config_creation():
    """Test audio configuration creation and validation."""
    print("\n🧪 TESTING CONFIGURATION")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import AudioProcessingConfig, create_advanced_audio_config
        
        # Test basic config
        print("📋 Testing basic configuration...")
        basic_config = AudioProcessingConfig()
        
        print(f"   • Language: {basic_config.language_code}")
        print(f"   • Model: {basic_config.model}")
        print(f"   • Speaker labels: {basic_config.speaker_labels}")
        print(f"   • Sentiment analysis: {basic_config.sentiment_analysis}")
        print(f"   • Entity detection: {basic_config.entity_detection}")
        
        # Test advanced config
        print("\n📋 Testing advanced configuration...")
        advanced_config = create_advanced_audio_config()
        
        print(f"   • Content safety: {advanced_config.content_safety}")
        print(f"   • Auto highlights: {advanced_config.auto_highlights}")
        print(f"   • Summarization: {advanced_config.summarization}")
        print(f"   • Custom vocabulary: {len(advanced_config.custom_vocabulary)} words")
        print(f"   • PII redaction: {advanced_config.redact_pii}")
        
        # Test custom config
        print("\n📋 Testing custom configuration...")
        custom_config = AudioProcessingConfig(
            language_code="es",
            speaker_labels=False,
            sentiment_analysis=False,
            custom_vocabulary=["transcription", "AI", "machine learning"]
        )
        
        print(f"   • Custom language: {custom_config.language_code}")
        print(f"   • Custom vocabulary: {custom_config.custom_vocabulary}")
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_harvard_wav_file():
    """Test access to the harvard.wav test file."""
    print("\n🧪 TESTING AUDIO FILE")
    print("=" * 40)
    
    test_file = Path(__file__).parent.parent / "data" / "harvard.wav"
    
    print(f"📁 Checking file: {test_file}")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    file_size = test_file.stat().st_size
    print(f"📊 File details:")
    print(f"   • Path: {test_file}")
    print(f"   • Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"   • Exists: ✅")
    print(f"   • Readable: {'✅' if os.access(test_file, os.R_OK) else '❌'}")
    
    if file_size == 0:
        print("❌ Audio file is empty")
        return False
    
    print("✅ Audio file tests passed")
    return True

def test_serialization():
    """Test component serialization/deserialization."""
    print("\n🧪 TESTING SERIALIZATION")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
        
        # Create config
        config = AudioProcessingConfig(
            language_code="en",
            speaker_labels=True,
            sentiment_analysis=True
        )
        
        print("📦 Testing component serialization...")
        
        # Create transcriber (without real API key)
        transcriber = AssemblyAITranscriber(
            api_key="test_key_for_serialization",
            config=config
        )
        
        # Test serialization
        print("   • Serializing to dict...")
        serialized = transcriber.to_dict()
        print(f"   • Serialized keys: {list(serialized.keys())}")
        
        # Test deserialization
        print("   • Deserializing from dict...")
        deserialized = AssemblyAITranscriber.from_dict(serialized)
        print(f"   • Deserialized type: {type(deserialized).__name__}")
        
        # Compare configs
        original_speaker_labels = transcriber.config.speaker_labels
        deserialized_speaker_labels = deserialized.config.speaker_labels
        
        print(f"   • Original speaker labels: {original_speaker_labels}")
        print(f"   • Deserialized speaker labels: {deserialized_speaker_labels}")
        
        assert original_speaker_labels == deserialized_speaker_labels
        
        print("✅ Serialization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smart_audio_processor_creation():
    """Test SmartAudioProcessor creation and configuration."""
    print("\n🧪 TESTING SMART AUDIO PROCESSOR")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import (
            SmartAudioProcessor, 
            AssemblyAITranscriber, 
            AudioProcessingConfig
        )
        
        # Create transcriber for the processor
        config = AudioProcessingConfig()
        transcriber = AssemblyAITranscriber(
            api_key="test_key_for_processor",
            config=config
        )
        
        print("🧠 Creating SmartAudioProcessor...")
        processor = SmartAudioProcessor(
            assemblyai_transcriber=transcriber,
            max_chunk_length=1000,
            overlap=100,
            respect_speakers=True,
            respect_chapters=True
        )
        
        print(f"   • Max chunk length: {processor.max_chunk_length}")
        print(f"   • Overlap: {processor.overlap}")
        print(f"   • Respect speakers: {processor.respect_speakers}")
        print(f"   • Respect chapters: {processor.respect_chapters}")
        
        # Test serialization
        print("\n📦 Testing processor serialization...")
        serialized = processor.to_dict()
        print(f"   • Serialized keys: {list(serialized.keys())}")
        
        deserialized = SmartAudioProcessor.from_dict(serialized)
        print(f"   • Deserialized successfully")
        
        print("✅ Smart processor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Smart processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_integration():
    """Test logging integration with the audio transcriber."""
    print("\n🧪 TESTING LOGGING INTEGRATION")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
        
        print("📝 Testing logging during initialization...")
        
        config = AudioProcessingConfig(speaker_labels=True, sentiment_analysis=True)
        
        # This should generate logs
        transcriber = AssemblyAITranscriber(
            api_key="test_logging_key",
            config=config
        )
        
        print("   • Transcriber initialized (check logs for details)")
        
        # Check if log files were created
        log_dir = Path(__file__).parent.parent / "logs"
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            print(f"   • Log files found: {len(log_files)}")
            
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                print(f"   • Latest log: {latest_log.name}")
                print(f"   • Log size: {latest_log.stat().st_size} bytes")
        
        print("✅ Logging integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def main():
    """Run all core audio transcriber tests."""
    print("🚀 AUDIO TRANSCRIBER CORE TESTS")
    print("=" * 60)
    print("Testing audio_transcriber.py core functionality")
    print("These tests verify components work without external APIs")
    print("=" * 60)
    
    tests = [
        ("Component Imports", test_audio_transcriber_imports),
        ("Configuration Creation", test_audio_config_creation),
        ("Audio File Access", test_harvard_wav_file),
        ("Component Serialization", test_serialization),
        ("Smart Processor Creation", test_smart_audio_processor_creation),
        ("Logging Integration", test_logging_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n▶️  Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success, None))
            if success:
                print(f"✅ PASSED: {test_name}")
            else:
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ ERROR: {test_name} - {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"🎯 TEST SUMMARY")
    print(f"=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print(f"\n🎉 All core tests passed! The audio transcriber is ready to use.")
        print(f"💡 Next steps:")
        print(f"   • Set ASSEMBLYAI_API_KEY for real transcription")
        print(f"   • Run: python tests/test_audio_transcriber_integration.py")
        print(f"   • Or use: pytest tests/ -v")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
        
        # Show failed tests
        failed_tests = [name for name, success, error in results if not success]
        if failed_tests:
            print(f"\n❌ Failed tests:")
            for test in failed_tests:
                print(f"   • {test}")

if __name__ == "__main__":
    main()