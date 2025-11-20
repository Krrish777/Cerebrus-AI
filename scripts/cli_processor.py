#!/usr/bin/env python3
"""
Document Processor CLI Tool

A command-line interface for testing the document processing pipeline.
Follows AGENTS.md standards with no emojis and proper logging.

Usage:
    python cli_processor.py --help
    python cli_processor.py analyze data/sample.txt
    python cli_processor.py process data/sample.md --chunk-size 200
    python cli_processor.py batch data/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.document_processing.pipeline_config import (
        PipelineConfig, ChunkingConfig, ProcessingConfig
    )
    from src.document_processing.file_analyzer import FileAnalyzer
    from src.document_processing.pipeline_orchestrator import DocumentPipelineOrchestrator
    from src.core.logging import get_logger
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure virtual environment is activated and dependencies are installed")
    sys.exit(1)

logger = get_logger(__name__)


def create_config(chunk_size: int = 300, chunk_overlap: int = 50) -> PipelineConfig:
    """Create pipeline configuration with specified parameters."""
    return PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            boundary_preferences=["paragraph", "sentence"]
        ),
        processing=ProcessingConfig(
            enable_pdf_processing=True,
            enable_text_processing=True,
            enable_markdown_processing=True
        )
    )


def analyze_file(file_path: Path, config: PipelineConfig) -> dict:
    """Analyze a single file and return results."""
    logger.info("Analyzing file: %s", file_path)
    
    analyzer = FileAnalyzer(config)
    
    if not file_path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    try:
        # File type detection
        detected_type = analyzer.detect_file_type(file_path)
        
        # File information
        file_info = analyzer.get_file_info(file_path)
        
        # Content analysis for text files
        content_info = {}
        if file_path.suffix in ['.md', '.txt']:
            content = file_path.read_text(encoding='utf-8')
            content_info = {
                "content_length": len(content),
                "word_count": len(content.split()),
                "line_count": len(content.splitlines()),
                "estimated_chunks": max(1, len(content) // config.chunking.chunk_size)
            }
        
        result = {
            "file": str(file_path),
            "detected_type": detected_type,
            "is_supported": file_info.get("is_supported", False),
            "size_bytes": file_info.get("size_bytes", 0),
            "size_formatted": file_info.get("size_formatted", "Unknown"),
            **content_info
        }
        
        logger.info("Analysis completed for %s: type=%s, supported=%s", 
                   file_path.name, detected_type, result["is_supported"])
        
        return result
        
    except Exception as e:
        error_msg = f"Analysis failed: {e}"
        logger.error("Failed to analyze %s: %s", file_path, str(e))
        return {"error": error_msg, "file": str(file_path)}


def process_file(file_path: Path, config: PipelineConfig) -> dict:
    """Process a single file through the complete pipeline."""
    logger.info("Processing file: %s", file_path)
    
    try:
        orchestrator = DocumentPipelineOrchestrator(config)
        result = orchestrator.process_documents([str(file_path)])
        
        logger.info("Processing completed for %s: %d documents generated", 
                   file_path.name, len(result.get("documents", [])))
        
        return result
        
    except Exception as e:
        error_msg = f"Processing failed: {e}"
        logger.error("Failed to process %s: %s", file_path, str(e))
        return {"error": error_msg, "file": str(file_path)}


def batch_process(directory: Path, config: PipelineConfig) -> dict:
    """Process all supported files in a directory."""
    logger.info("Batch processing directory: %s", directory)
    
    if not directory.is_dir():
        error_msg = f"Directory not found: {directory}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    # Find supported files
    supported_extensions = ['.pdf', '.txt', '.md', '.markdown']
    files = []
    
    for ext in supported_extensions:
        files.extend(directory.glob(f"*{ext}"))
    
    if not files:
        logger.warning("No supported files found in %s", directory)
        return {"warning": "No supported files found", "directory": str(directory)}
    
    logger.info("Found %d files for processing", len(files))
    
    try:
        orchestrator = DocumentPipelineOrchestrator(config)
        result = orchestrator.process_documents([str(f) for f in files])
        
        logger.info("Batch processing completed: %d files processed", 
                   result.get("stats", {}).get("files_processed", 0))
        
        return result
        
    except Exception as e:
        error_msg = f"Batch processing failed: {e}"
        logger.error("Failed to process directory %s: %s", directory, str(e))
        return {"error": error_msg, "directory": str(directory)}


def print_analysis_result(result: dict) -> None:
    """Print analysis results in a clean format."""
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    
    print(f"File: {result['file']}")
    print(f"Type: {result['detected_type']}")
    print(f"Supported: {result['is_supported']}")
    print(f"Size: {result['size_formatted']}")
    
    if "content_length" in result:
        print(f"Content: {result['content_length']} characters")
        print(f"Words: {result['word_count']}")
        print(f"Lines: {result['line_count']}")
        print(f"Estimated Chunks: {result['estimated_chunks']}")


def print_processing_result(result: dict) -> None:
    """Print processing results in a clean format."""
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    
    documents = result.get("documents", [])
    errors = result.get("errors", [])
    stats = result.get("stats", {})
    
    print(f"Documents Generated: {len(documents)}")
    print(f"Errors: {len(errors)}")
    
    if stats:
        print(f"Files Processed: {stats.get('files_processed', 0)}")
        print(f"Processing Time: {stats.get('total_time', 0):.2f} seconds")
        print(f"Success Rate: {stats.get('success_rate', 0):.1f}%")
    
    if errors:
        print("Errors encountered:")
        for error in errors:
            print(f"  - {error}")
    
    # Show first few document previews
    for i, doc in enumerate(documents[:3]):
        content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
        print(f"Document {i+1}: {content_preview}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Document Processing Pipeline CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_processor.py analyze data/sample.txt
  python cli_processor.py process data/sample.md --chunk-size 200
  python cli_processor.py batch data/ --chunk-overlap 30
  python cli_processor.py analyze data/llm_overview.pdf --json
        """
    )
    
    parser.add_argument(
        "command",
        choices=["analyze", "process", "batch"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "path",
        type=Path,
        help="File or directory path"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Chunk size for processing (default: 300)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap for processing (default: 50)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = create_config(args.chunk_size, args.chunk_overlap)
    
    # Execute command
    result = None
    
    if args.command == "analyze":
        result = analyze_file(args.path, config)
        if not args.json:
            print_analysis_result(result)
    
    elif args.command == "process":
        result = process_file(args.path, config)
        if not args.json:
            print_processing_result(result)
    
    elif args.command == "batch":
        result = batch_process(args.path, config)
        if not args.json:
            print_processing_result(result)
    
    # Output JSON if requested
    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()