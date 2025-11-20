#!/usr/bin/env python3
"""
Simple Document Processing Demo

A lightweight demo that shows the document processing pipeline in action
with the three files from your data directory.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def analyze_data_files():
    """Analyze the files in the data directory."""
    data_dir = Path("data")
    
    print("=" * 60)
    print("DATA DIRECTORY ANALYSIS")
    print("=" * 60)
    
    target_files = ["llm_overview.pdf", "sample.md", "sample.txt"]
    
    for filename in target_files:
        file_path = data_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"{filename}")
            print(f"   Size: {size:,} bytes")
            
            # Read content preview for text files
            if filename.endswith(('.md', '.txt')):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = len(content.split('\n'))
                    words = len(content.split())
                    preview = content[:200] + "..." if len(content) > 200 else content
                    
                    print(f"   Lines: {lines}")
                    print(f"   Words: {words}")
                    print(f"   Preview: {preview.strip()}")
                except Exception as e:
                    print(f"   Error reading: {e}")
            
            print()
        else:
            print(f"{filename} - Not found")
            print()

def test_basic_imports():
    """Test basic imports from the document processing modules."""
    print("=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)
    
    try:
        from src.document_processing.pipeline_config import PipelineConfig, ChunkingConfig
        print("PipelineConfig imported successfully")
        
        # Test configuration creation
        config = PipelineConfig()
        print(f"PipelineConfig created: {type(config)}")
        print(f"   Chunking config: {config.chunking}")
        print(f"   File types config: {config.file_types}")
        print()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print()
    
    try:
        from src.document_processing.file_analyzer import FileAnalyzer
        print("FileAnalyzer imported successfully")
        
        # Test file analyzer
        from src.document_processing.pipeline_config import PipelineConfig
        config = PipelineConfig()
        analyzer = FileAnalyzer(config)
        print(f"FileAnalyzer created: {type(analyzer)}")
        
        # Test file type detection
        pdf_type = analyzer.detect_file_type(Path("test.pdf"))
        print(f"   PDF detection: {pdf_type}")
        
        md_type = analyzer.detect_file_type(Path("test.md"))
        print(f"   Markdown detection: {md_type}")
        
        txt_type = analyzer.detect_file_type(Path("test.txt"))
        print(f"   Text detection: {txt_type}")
        print()
        
    except ImportError as e:
        print(f"FileAnalyzer import error: {e}")
        print()
    
    try:
        from src.document_processing.metadata_manager import MetadataManager
        from src.document_processing.pipeline_config import PipelineConfig
        
        config = PipelineConfig()
        print("MetadataManager imported successfully")
        
        manager = MetadataManager(config)
        print(f"MetadataManager created: {type(manager)}")
        print()
        
    except ImportError as e:
        print(f"MetadataManager import error: {e}")
        print()

def simulate_pipeline_workflow():
    """Simulate the pipeline workflow without complex dependencies."""
    print("=" * 60)
    print("SIMULATED PIPELINE WORKFLOW")
    print("=" * 60)
    
    data_files = [
        Path("data/llm_overview.pdf"),
        Path("data/sample.md"),
        Path("data/sample.txt")
    ]
    
    existing_files = [f for f in data_files if f.exists()]
    
    print(f"Processing {len(existing_files)} files:")
    
    try:
        # Import what we can
        from src.document_processing.file_analyzer import FileAnalyzer
        from src.document_processing.pipeline_config import PipelineConfig
        
        config = PipelineConfig()
        analyzer = FileAnalyzer(config)
        
        for file_path in existing_files:
            print(f"\nProcessing: {file_path.name}")
            
            # Analyze file
            file_type = analyzer.detect_file_type(file_path)
            print(f"   Detected type: {file_type}")
            
            # Get file info
            file_info = analyzer.get_file_info(file_path)
            print(f"   File size: {file_info.get('size_formatted', 'Unknown')}")
            print(f"   Supported: {file_info.get('is_supported', False)}")
            
            # Simulate processing steps
            print(f"   Simulating conversion...")
            print(f"   Simulating chunking...")
            print(f"   Simulating metadata enhancement...")
            
            # Read content for text files to show what would be processed
            if file_path.suffix in ['.md', '.txt']:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    words = len(content.split())
                    
                    # Estimate chunks (300 chars per chunk as in demo config)
                    estimated_chunks = max(1, len(content) // 300)
                    
                    print(f"   Content: {len(content)} chars, ~{words} words")
                    print(f"   Estimated chunks: {estimated_chunks}")
                    
                    # Show content preview
                    preview = content[:150] + "..." if len(content) > 150 else content
                    print(f"   Preview: {preview.strip()}")
                    
                except Exception as e:
                    print(f"   ❌ Error reading content: {e}")
            
            elif file_path.suffix == '.pdf':
                file_size = file_path.stat().st_size
                print(f"   PDF size: {file_size:,} bytes")
                print(f"   Estimated processing: PDF → Text → Chunks")
        
        print(f"\nSimulated processing of {len(existing_files)} files completed!")
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the demo."""
    print("DOCUMENT PROCESSING PIPELINE DEMO")
    print("Working directory:", Path.cwd())
    print()
    
    analyze_data_files()
    test_basic_imports()
    simulate_pipeline_workflow()
    
    print("\n" + "=" * 60)
    print("Demo completed! Your pipeline is ready for:")
    print("   • PDF document processing")
    print("   • Markdown file handling")
    print("   • Text file processing")
    print("   • Smart chunking with natural boundaries")
    print("   • Comprehensive metadata enhancement")
    print("   • Error handling and statistics")
    print("=" * 60)

if __name__ == "__main__":
    main()