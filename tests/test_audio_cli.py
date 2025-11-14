#!/usr/bin/env python3
"""
CLI tool for testing audio transcription features directly.
Usage: python test_audio_cli.py [options]
"""

import os
import sys
import argparse
from pathlib import Path
import json
from typing import Dict, Any, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_setup():
    """Check if everything is set up correctly."""
    print("🔍 CHECKING SETUP")
    print("=" * 30)
    
    # Check API key
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("❌ ASSEMBLYAI_API_KEY not found!")
        print("💡 Create .env file with: ASSEMBLYAI_API_KEY=your_key_here")
        return False
    
    print(f"✅ API key: {api_key[:10]}...{api_key[-4:]}")
    
    # Check default audio file
    default_audio = Path("data/harvard.wav")
    if default_audio.exists():
        print(f"✅ Default audio file: {default_audio} ({default_audio.stat().st_size} bytes)")
    else:
        print(f"⚠️  Default audio file not found: {default_audio}")
        print("   You can specify a different file with --file option")
    
    return True

def test_basic_transcription(audio_file: str, show_full: bool = False):
    """Test basic transcription."""
    print("\n🎵 BASIC TRANSCRIPTION TEST")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
        
        config = AudioProcessingConfig(
            language_code="en",
            speaker_labels=True,
            sentiment_analysis=True,
            entity_detection=True,
            auto_chapters=False,
            summarization=False
        )
        
        print(f"📋 Creating transcriber for: {audio_file}")
        transcriber = AssemblyAITranscriber(config=config)
        
        print("⏳ Transcribing (this may take 30-60 seconds)...")
        result = transcriber.run([audio_file])
        
        documents = result.get("documents", [])
        print(f"✅ Generated {len(documents)} documents")
        
        if documents:
            main_doc = documents[0]
            content = main_doc.content or ""
            metadata = main_doc.meta
            
            print(f"\n📊 Results:")
            print(f"   • Content length: {len(content)} characters")
            print(f"   • Source: {metadata.get('source')}")
            print(f"   • Duration: {metadata.get('audio_duration_seconds')} seconds")
            print(f"   • Language: {metadata.get('language_code')}")
            print(f"   • Confidence: {metadata.get('confidence')}")
            
            # Show features
            sentiment_data = metadata.get('sentiment_analysis', [])
            entities_data = metadata.get('entities', [])
            
            print(f"\n🔍 Features:")
            print(f"   • Sentiment segments: {len(sentiment_data)}")
            print(f"   • Entities: {len(entities_data)}")
            
            # Preview
            if show_full:
                print(f"\n📝 Full transcription:")
                print(content)
            else:
                lines = content.split('\n')[:5]
                print(f"\n📝 Preview (first 5 lines):")
                for line in lines:
                    if line.strip():
                        preview = line[:80] + "..." if len(line) > 80 else line
                        print(f"   {preview}")
                print("   (use --full to see complete transcription)")
            
            # Show sentiment examples
            if sentiment_data and len(sentiment_data) > 0:
                print(f"\n💭 Sentiment examples:")
                for i, sentiment in enumerate(sentiment_data[:3]):
                    text = sentiment.get('text', '')[:40] + "..."
                    sentiment_val = sentiment.get('sentiment', 'unknown')
                    print(f"   {i+1}. \"{text}\" → {sentiment_val}")
            
            # Show entities
            if entities_data and len(entities_data) > 0:
                print(f"\n🏷️  Entities:")
                for i, entity in enumerate(entities_data[:5]):
                    print(f"   {i+1}. \"{entity.get('text')}\" → {entity.get('entity_type')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_features(audio_file: str):
    """Test advanced features."""
    print("\n🚀 ADVANCED FEATURES TEST")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import create_advanced_audio_config, AssemblyAITranscriber
        
        config = create_advanced_audio_config()
        config.auto_chapters = False  # Disable for short audio
        config.summarization = False
        
        print(f"📋 Creating advanced transcriber for: {audio_file}")
        transcriber = AssemblyAITranscriber(config=config)
        
        print("⏳ Processing with advanced features (this may take 60-90 seconds)...")
        result = transcriber.run([audio_file])
        
        documents = result.get("documents", [])
        print(f"✅ Generated {len(documents)} documents")
        
        if documents:
            main_doc = documents[0]
            metadata = main_doc.meta
            
            features_found = []
            if metadata.get('sentiment_analysis'):
                features_found.append(f"Sentiment ({len(metadata['sentiment_analysis'])})")
            if metadata.get('entities'):
                features_found.append(f"Entities ({len(metadata['entities'])})")
            if metadata.get('topics'):
                features_found.append("Topics")
            if metadata.get('content_safety'):
                features_found.append("Content Safety")
            if metadata.get('highlights'):
                features_found.append(f"Highlights ({len(metadata['highlights'])})")
            
            print(f"🎯 Advanced features detected: {', '.join(features_found)}")
            
            # Show highlights
            highlights = metadata.get('highlights', [])
            if highlights:
                print(f"\n⭐ Key highlights:")
                for i, highlight in enumerate(highlights[:3]):
                    text = highlight.get('text', '')[:50]
                    rank = highlight.get('rank', 'unknown')
                    print(f"   {i+1}. \"{text}...\" (rank: {rank})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smart_chunking(audio_file: str):
    """Test smart audio processing."""
    print("\n🧠 SMART CHUNKING TEST")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import AssemblyAITranscriber, SmartAudioProcessor, AudioProcessingConfig
        
        config = AudioProcessingConfig(
            language_code="en",
            speaker_labels=True,
            sentiment_analysis=True,
            entity_detection=True,
            include_sentences=True,
            include_paragraphs=True
        )
        
        print(f"📋 Creating smart processor for: {audio_file}")
        transcriber = AssemblyAITranscriber(config=config)
        processor = SmartAudioProcessor(
            assemblyai_transcriber=transcriber,
            max_chunk_length=300,
            overlap=50,
            respect_speakers=True
        )
        
        print("⏳ Smart processing (this may take 60-90 seconds)...")
        result = processor.run([audio_file])
        
        documents = result.get("documents", [])
        print(f"✅ Generated {len(documents)} smart chunks")
        
        chunk_types = {}
        total_length = 0
        
        for doc in documents:
            chunk_type = doc.meta.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_length += len(doc.content or '')
        
        print(f"📊 Chunk analysis:")
        print(f"   • Total content: {total_length} characters")
        print(f"   • Chunk types: {dict(chunk_types)}")
        
        print(f"\n📝 Sample chunks:")
        for i, doc in enumerate(documents[:3]):
            chunk_type = doc.meta.get('chunk_type', 'unknown')
            content_preview = (doc.content or '')[:60].replace('\n', ' ')
            print(f"   {i+1}. [{chunk_type}] {content_preview}...")
            if doc.meta.get('speaker'):
                print(f"      Speaker: {doc.meta['speaker']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_results(audio_file: str, output_file: str):
    """Save transcription results to file."""
    print(f"\n💾 SAVING RESULTS TO: {output_file}")
    print("=" * 40)
    
    try:
        from audio_processing.audio_transcriber import AssemblyAITranscriber, AudioProcessingConfig
        
        config = AudioProcessingConfig(
            speaker_labels=True,
            sentiment_analysis=True,
            entity_detection=True,
            auto_highlights=True
        )
        
        transcriber = AssemblyAITranscriber(config=config)
        result = transcriber.run([audio_file])
        
        documents = result.get("documents", [])
        
        # Prepare output data
        output_data = {
            "source_file": audio_file,
            "documents_count": len(documents),
            "documents": []
        }
        
        for i, doc in enumerate(documents):
            doc_data = {
                "index": i,
                "content": doc.content,
                "metadata": doc.meta
            }
            output_data["documents"].append(doc_data)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_file}")
        print(f"   • Documents: {len(documents)}")
        print(f"   • File size: {Path(output_file).stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for testing audio transcription features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_audio_cli.py                           # Basic test with default file
  python test_audio_cli.py --file myaudio.wav       # Test with custom file
  python test_audio_cli.py --advanced               # Test advanced features
  python test_audio_cli.py --smart                  # Test smart chunking
  python test_audio_cli.py --full                   # Show full transcription
  python test_audio_cli.py --save results.json     # Save results to file
  python test_audio_cli.py --all                    # Run all tests
        """
    )
    
    parser.add_argument(
        "--file", "-f",
        default="data/harvard.wav",
        help="Audio file to transcribe (default: data/harvard.wav)"
    )
    
    parser.add_argument(
        "--basic", "-b",
        action="store_true",
        help="Run basic transcription test (default)"
    )
    
    parser.add_argument(
        "--advanced", "-a",
        action="store_true",
        help="Test advanced features (sentiment, entities, highlights)"
    )
    
    parser.add_argument(
        "--smart", "-s",
        action="store_true",
        help="Test smart audio chunking"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full transcription (not just preview)"
    )
    
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    print("🎯 ASSEMBLYAI AUDIO TRANSCRIBER CLI")
    print("=" * 50)
    
    # Check setup
    if not check_setup():
        return
    
    # Check audio file
    audio_file = Path(args.file)
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print(f"🎵 Using audio file: {audio_file}")
    
    # Determine which tests to run
    run_basic = args.basic or args.all or not any([args.advanced, args.smart, args.save])
    run_advanced = args.advanced or args.all
    run_smart = args.smart or args.all
    run_save = args.save is not None
    
    success_count = 0
    total_tests = sum([run_basic, run_advanced, run_smart, run_save])
    
    # Run tests
    if run_basic:
        if test_basic_transcription(str(audio_file), args.full):
            success_count += 1
    
    if run_advanced:
        if test_advanced_features(str(audio_file)):
            success_count += 1
    
    if run_smart:
        if test_smart_chunking(str(audio_file)):
            success_count += 1
    
    if run_save:
        if save_results(str(audio_file), args.save):
            success_count += 1
    
    # Summary
    print(f"\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! Audio transcription is working perfectly.")
    else:
        print(f"\n⚠️  Some tests failed. Check the error messages above.")
    
    print(f"\n💡 Tips:")
    print("  • Use --help for more options")
    print("  • Use --full to see complete transcriptions")
    print("  • Use --save results.json to save output")
    print("  • Set ASSEMBLYAI_API_KEY in .env file")

if __name__ == "__main__":
    main()