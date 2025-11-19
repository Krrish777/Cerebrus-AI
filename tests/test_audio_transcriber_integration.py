#!/usr/bin/env python3
"""
Integration test for the audio transcriber with real audio file.
Tests the complete audio processing pipeline using harvard.wav.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestAudioTranscriberIntegration:
    """Integration tests for audio transcriber with real files."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.test_audio_file = Path(__file__).parent.parent / "data" / "harvard.wav"
        self.test_api_key = "test_api_key_12345"
        
        # Ensure test file exists
        assert self.test_audio_file.exists(), f"Test audio file not found: {self.test_audio_file}"
        
    def test_audio_file_exists(self):
        """Test that the harvard.wav file exists and is accessible."""
        print("\n🧪 TESTING AUDIO FILE ACCESSIBILITY")
        print("=" * 60)
        
        print(f"📁 Checking file: {self.test_audio_file}")
        assert self.test_audio_file.exists(), "Harvard.wav file should exist"
        
        file_size = self.test_audio_file.stat().st_size
        print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        assert file_size > 0, "Audio file should not be empty"
        
        print("✅ Audio file is accessible and has content")
        
    def test_audio_processing_config_creation(self):
        """Test creating audio processing configuration."""
        print("\n🧪 TESTING AUDIO PROCESSING CONFIG")
        print("=" * 60)
        
        try:
            from audio_processing.audio_transcriber import AudioProcessingConfig, create_advanced_audio_config
            
            print("📋 Testing basic configuration...")
            basic_config = AudioProcessingConfig()
            print(f"   ✅ Basic config created - Speaker labels: {basic_config.speaker_labels}")
            print(f"   ✅ Sentiment analysis: {basic_config.sentiment_analysis}")
            print(f"   ✅ Language: {basic_config.language_code}")
            
            print("📋 Testing advanced configuration...")
            advanced_config = create_advanced_audio_config()
            print("   ✅ Advanced config created")
            print(f"   ✅ Entity detection: {advanced_config.entity_detection}")
            print(f"   ✅ Content safety: {advanced_config.content_safety}")
            print(f"   ✅ Auto highlights: {advanced_config.auto_highlights}")
            print(f"   ✅ Summarization: {advanced_config.summarization}")
            
            assert basic_config.speaker_labels == True
            assert advanced_config.sentiment_analysis == True
            
        except ImportError as e:
            pytest.skip(f"Audio transcriber dependencies not available: {e}")
    
    @pytest.mark.skipif(not os.getenv("ASSEMBLYAI_API_KEY"), reason="No AssemblyAI API key provided")
    def test_real_assemblyai_transcription(self):
        """Test real AssemblyAI transcription (requires API key)."""
        print("\n🧪 TESTING REAL ASSEMBLYAI TRANSCRIPTION")
        print("=" * 60)
        
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            pytest.skip("ASSEMBLYAI_API_KEY environment variable not set")
            
        try:
            from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
            
            print(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else 'short_key'}")
            
            # Create simple config for faster processing
            config = AudioProcessingConfig(
                language_code="en",
                speaker_labels=True,
                sentiment_analysis=False,  # Disable for speed
                entity_detection=False,
                auto_highlights=False,
                summarization=False
            )
            
            print("📋 Configuration:")
            print(f"   • Language: {config.language_code}")
            print(f"   • Speaker labels: {config.speaker_labels}")
            print("   • Enhanced features: disabled for speed")
            
            # Initialize transcriber
            print("\n🚀 Initializing transcriber...")
            transcriber = AssemblyAITranscriber(api_key=api_key, config=config)
            print("   ✅ Transcriber initialized")
            
            # Transcribe the harvard.wav file
            print(f"\n🎵 Transcribing: {self.test_audio_file.name}")
            print(f"   📁 File path: {self.test_audio_file}")
            
            result = transcriber.run([str(self.test_audio_file)])
            
            print("\n📊 TRANSCRIPTION RESULTS:")
            documents = result.get("documents", [])
            print(f"   📄 Documents returned: {len(documents)}")
            
            if documents:
                main_doc = documents[0]
                content = main_doc.content
                metadata = main_doc.meta
                
                print(f"   📝 Content length: {len(content):,} characters") # pyright: ignore[reportArgumentType]
                print(f"   🗂️  Metadata fields: {len(metadata)}")
                print(f"   📋 Content preview: '{content[:100]}...'" if len(content) > 100 else f"   📋 Full content: '{content}'") # type: ignore # type: ignore
                
                # Check for expected content (Harvard sentences typically contain "The...")
                assert len(content) > 0, "Transcription should not be empty" # type: ignore
                assert isinstance(content, str), "Content should be a string"
                
                print("   ✅ Transcription successful!")
                
                # Show metadata details
                print("\n🗂️  METADATA ANALYSIS:")
                for key, value in list(metadata.items())[:10]:  # Show first 10
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:50] + "..."
                    print(f"   • {key}: {value}")
                
            else:
                pytest.fail("No documents returned from transcription")
                
        except ImportError as e:
            pytest.skip(f"Audio transcriber dependencies not available: {e}")
        except Exception as e:
            pytest.fail(f"Real transcription failed: {e}")
    
    def test_mocked_assemblyai_transcription(self):
        """Test transcriber with mocked AssemblyAI responses."""
        print("\n🧪 TESTING MOCKED ASSEMBLYAI TRANSCRIPTION")
        print("=" * 60)
        
        try:
            from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
            
            # Mock transcript object
            mock_transcript = Mock()
            mock_transcript.text = "The stale smell of old beer lingers. It takes heat to bring out the odor."
            mock_transcript.confidence = 0.95
            mock_transcript.words = []
            mock_transcript.utterances = []
            mock_transcript.error = None
            
            # Add speaker labels
            mock_utterance = Mock()
            mock_utterance.text = "The stale smell of old beer lingers."
            mock_utterance.speaker = "A"
            mock_utterance.start = 0
            mock_utterance.end = 2000
            mock_transcript.utterances = [mock_utterance]
            
            print("🎭 Created mock transcript:")
            print(f"   📝 Text: {mock_transcript.text}")
            print(f"   🎯 Confidence: {mock_transcript.confidence}")
            print(f"   👥 Speakers: {len(mock_transcript.utterances)}")
            
            with patch('audio_processing.audio_transcriber.aai') as mock_aai:
                # Setup mock transcriber
                mock_transcriber_instance = Mock()
                mock_transcriber_instance.transcribe.return_value = mock_transcript
                mock_aai.Transcriber.return_value = mock_transcriber_instance
                mock_aai.TranscriptionConfig.return_value = Mock()
                mock_aai.settings = Mock()
                
                print("\n🔧 Setting up mocked transcriber...")
                
                # Create config
                config = AudioProcessingConfig(
                    speaker_labels=True,
                    sentiment_analysis=True
                )
                
                # Initialize transcriber (mocked)
                transcriber = AssemblyAITranscriber(api_key=self.test_api_key, config=config)
                print("   ✅ Mocked transcriber initialized")
                
                # Run transcription
                print("\n🎵 Running mocked transcription...")
                result = transcriber.run([str(self.test_audio_file)])
                
                print("\n📊 MOCKED TRANSCRIPTION RESULTS:")
                documents = result.get("documents", [])
                print(f"   📄 Documents returned: {len(documents)}")
                
                assert len(documents) > 0, "Should return at least one document"
                
                main_doc = documents[0]
                print(f"   📝 Content: '{main_doc.content}'")
                print(f"   🗂️  Metadata keys: {list(main_doc.meta.keys())[:5]}...")
                
                # Verify content
                assert "beer" in main_doc.content.lower(), "Content should contain expected words" # type: ignore
                assert main_doc.meta.get("source") == str(self.test_audio_file)
                
                print("   ✅ Mocked transcription successful!")
                
        except ImportError as e:
            pytest.skip(f"Audio transcriber dependencies not available: {e}")
    
    def test_smart_audio_processor(self):
        """Test the SmartAudioProcessor with mocked data."""
        print("\n🧪 TESTING SMART AUDIO PROCESSOR")
        print("=" * 60)
        
        try:
            from audio_processing.audio_transcriber import SmartAudioProcessor, AssemblyAITranscriber, AudioProcessingConfig
            from haystack import Document
            
            print("🧠 Testing smart audio processing...")
            
            # Create mock transcriber
            mock_transcriber = Mock(spec=AssemblyAITranscriber)
            
            # Create mock documents that the transcriber would return
            mock_documents = [
                Document(
                    content="**Speaker A:** The stale smell of old beer lingers.\n**Speaker B:** It takes heat to bring out the odor.",
                    meta={
                        "source": str(self.test_audio_file),
                        "content_type": "transcript",
                        "speakers": ["A", "B"],
                        "duration": 5.2
                    }
                )
            ]
            
            mock_transcriber.run.return_value = {"documents": mock_documents}
            
            print(f"   📄 Mock documents prepared: {len(mock_documents)}")
            
            # Create smart processor
            processor = SmartAudioProcessor(
                assemblyai_transcriber=mock_transcriber,
                max_chunk_length=500,
                overlap=50,
                respect_speakers=True
            )
            
            print("   🧠 Smart processor created")
            print("   ⚙️  Chunk length: 500, Overlap: 50, Speaker-aware: True")
            
            # Process the audio
            print("\n🔄 Processing audio...")
            result = processor.run([str(self.test_audio_file)])
            
            print("\n📊 SMART PROCESSING RESULTS:")
            documents = result.get("documents", [])
            print(f"   📄 Output documents: {len(documents)}")
            
            assert len(documents) > 0, "Should return processed documents"
            
            # Analyze the results
            for i, doc in enumerate(documents[:3]):  # Show first 3
                print(f"   📋 Document {i+1}:")
                print(f"      Content: '{doc.content[:80]}...'" if len(doc.content) > 80 else f"      Content: '{doc.content}'") # type: ignore
                print(f"      Strategy: {doc.meta.get('processing_strategy', 'unknown')}")
                print(f"      Chunk type: {doc.meta.get('chunk_type', 'unknown')}")
                
            print("   ✅ Smart processing successful!")
            
        except ImportError as e:
            pytest.skip(f"Audio transcriber dependencies not available: {e}")
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        print("\n🧪 TESTING ERROR HANDLING")
        print("=" * 60)
        
        try:
            from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
            
            print("🚫 Testing invalid API key...")
            
            config = AudioProcessingConfig()
            
            # Test with invalid API key
            try:
                transcriber = AssemblyAITranscriber(api_key="invalid_key", config=config)
                print("   ✅ Transcriber created (validation happens on use)")
                
                # This should fail when actually trying to transcribe
                with pytest.raises(Exception):
                    transcriber.run([str(self.test_audio_file)])
                    
            except ValueError as e:
                print(f"   ✅ Caught expected error: {e}")
            
            print("🚫 Testing missing file...")
            
            # Test with non-existent file
            fake_file = self.test_audio_file.parent / "nonexistent.wav"
            transcriber = AssemblyAITranscriber(api_key=self.test_api_key, config=config)
            
            try:
                result = transcriber.run([str(fake_file)])
                # Depending on implementation, this might not fail immediately
                print("   ⚠️  No immediate error (file validation may happen later)")
            except Exception as e:
                print(f"   ✅ Caught expected file error: {type(e).__name__}")
            
        except ImportError as e:
            pytest.skip(f"Audio transcriber dependencies not available: {e}")

def main():
    """Run the audio transcriber tests manually."""
    print("🚀 AUDIO TRANSCRIBER INTEGRATION TESTS")
    print("=" * 80)
    print("Testing the audio_transcriber.py with harvard.wav")
    print("These tests verify:")
    print("  • Configuration creation and validation")
    print("  • File accessibility and processing")  
    print("  • Mocked AssemblyAI integration")
    print("  • Smart audio processing pipeline")
    print("  • Error handling scenarios")
    print("\n💡 To run real API tests, set ASSEMBLYAI_API_KEY environment variable")
    print("=" * 80)
    
    # Create test instance
    test_instance = TestAudioTranscriberIntegration()
    test_instance.setup_method()
    
    # Run each test
    tests = [
        ("File Accessibility", test_instance.test_audio_file_exists),
        ("Config Creation", test_instance.test_audio_processing_config_creation),
        ("Mocked Transcription", test_instance.test_mocked_assemblyai_transcription),
        ("Smart Processing", test_instance.test_smart_audio_processor),
        ("Error Handling", test_instance.test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n▶️  Running: {test_name}")
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🎯 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check the output above")
    
    print("\n💡 To run with pytest: pytest tests/test_audio_transcriber_integration.py -v")

if __name__ == "__main__":
    main()