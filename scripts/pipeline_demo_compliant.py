#!/usr/bin/env python3
"""
Document Processing Pipeline Demo - AGENTS.md Compliant

This script demonstrates the document processing pipeline by processing 
files from the data directory. All output follows AGENTS.md logging standards
with no emojis or decorative characters.

Usage: python pipeline_demo_compliant.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import traceback

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src.document_processing.pipeline_config import (
        PipelineConfig, ChunkingConfig, ProcessingConfig, FileTypeConfig,
        MetadataConfig, ErrorHandlingConfig, PerformanceConfig
    )
    from src.document_processing.file_analyzer import FileAnalyzer
    from src.core.logging import get_logger
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure virtual environment is activated and dependencies are installed")
    sys.exit(1)

# Configure logging - following AGENTS.md standards
logger = get_logger(__name__)


def create_demo_config() -> PipelineConfig:
    """Create demonstration configuration for the pipeline."""
    logger.info("Creating pipeline configuration with demo settings")
    
    return PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=300,
            chunk_overlap=50,
            min_chunk_size_ratio=0.2,
            boundary_preferences=["paragraph", "sentence"],
            enable_statistics=True
        ),
        processing=ProcessingConfig(
            enable_pdf_processing=True,
            enable_text_processing=True,
            enable_markdown_processing=True,
            enable_markdown_fallback=True
        ),
        file_types=FileTypeConfig(),
        metadata=MetadataConfig(),
        error_handling=ErrorHandlingConfig(
            fail_fast=False,
            continue_on_individual_file_error=True
        ),
        performance=PerformanceConfig(
            enable_timing=True,
            enable_statistics=True
        )
    )


def analyze_data_files(data_dir: Path) -> Dict[str, Any]:
    """
    Analyze files in the data directory and return analysis results.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary containing file analysis results
    """
    logger.info("Analyzing files in data directory: %s", data_dir)
    
    target_files = ["llm_overview.pdf", "sample.md", "sample.txt"]
    results = {
        "total_files": len(target_files),
        "existing_files": [],
        "missing_files": [],
        "file_details": {}
    }
    
    for filename in target_files:
        file_path = data_dir / filename
        
        if file_path.exists():
            file_size = file_path.stat().st_size
            results["existing_files"].append(str(file_path))
            results["file_details"][filename] = {
                "size": file_size,
                "path": str(file_path),
                "exists": True
            }
            
            logger.debug("Found file %s with size %d bytes", filename, file_size)
            
            # Analyze text files for additional details
            if filename.endswith(('.md', '.txt')):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    word_count = len(content.split())
                    line_count = len(content.split('\n'))
                    
                    results["file_details"][filename].update({
                        "content_length": len(content),
                        "word_count": word_count,
                        "line_count": line_count
                    })
                    
                    logger.debug("Text file %s has %d words and %d lines", 
                               filename, word_count, line_count)
                    
                except Exception as e:
                    logger.warning("Could not analyze text content of %s: %s", 
                                 filename, str(e))
        else:
            results["missing_files"].append(filename)
            results["file_details"][filename] = {"exists": False}
            logger.warning("File not found: %s", filename)
    
    logger.info("File analysis completed. Found %d of %d target files", 
               len(results["existing_files"]), results["total_files"])
    
    return results


def test_file_analyzer(config: PipelineConfig, file_paths: List[Path]) -> Dict[str, Any]:
    """
    Test the FileAnalyzer component with the provided files.
    
    Args:
        config: Pipeline configuration
        file_paths: List of file paths to analyze
        
    Returns:
        Dictionary containing analyzer test results
    """
    logger.info("Testing FileAnalyzer component with %d files", len(file_paths))
    
    try:
        analyzer = FileAnalyzer(config)
        results = {
            "analyzer_created": True,
            "file_analyses": {},
            "supported_files": [],
            "unsupported_files": []
        }
        
        for file_path in file_paths:
            logger.debug("Analyzing file: %s", file_path.name)
            
            try:
                # Detect file type
                file_type = analyzer.detect_file_type(file_path)
                
                # Get file information
                file_info = analyzer.get_file_info(file_path)
                
                analysis = {
                    "detected_type": file_type,
                    "is_supported": file_info.get("is_supported", False),
                    "size_bytes": file_info.get("size_bytes", 0),
                    "size_formatted": file_info.get("size_formatted", "Unknown")
                }
                
                results["file_analyses"][str(file_path)] = analysis
                
                if analysis["is_supported"]:
                    results["supported_files"].append(str(file_path))
                    logger.info("File %s detected as %s (supported)", 
                              file_path.name, file_type)
                else:
                    results["unsupported_files"].append(str(file_path))
                    logger.warning("File %s detected as %s (not supported)", 
                                 file_path.name, file_type)
                
            except Exception as e:
                logger.error("Failed to analyze file %s: %s", file_path.name, str(e))
                results["file_analyses"][str(file_path)] = {
                    "error": str(e),
                    "is_supported": False
                }
        
        logger.info("FileAnalyzer testing completed. %d supported, %d unsupported files",
                   len(results["supported_files"]), len(results["unsupported_files"]))
        
        return results
        
    except Exception as e:
        logger.error("Failed to create or test FileAnalyzer: %s", str(e))
        return {
            "analyzer_created": False,
            "error": str(e),
            "file_analyses": {},
            "supported_files": [],
            "unsupported_files": []
        }


def simulate_document_processing(file_paths: List[Path]) -> Dict[str, Any]:
    """
    Simulate document processing workflow for demonstration.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        Dictionary containing simulation results
    """
    logger.info("Simulating document processing for %d files", len(file_paths))
    
    simulation_results = {
        "total_files": len(file_paths),
        "processed_files": [],
        "estimated_chunks": 0,
        "total_content_chars": 0,
        "processing_steps": []
    }
    
    for file_path in file_paths:
        logger.debug("Simulating processing for file: %s", file_path.name)
        
        file_result = {
            "filename": file_path.name,
            "path": str(file_path),
            "steps_completed": []
        }
        
        try:
            # Step 1: File Analysis
            file_result["steps_completed"].append("file_analysis")
            logger.debug("Step 1: File analysis completed for %s", file_path.name)
            
            # Step 2: Document Conversion (simulated)
            file_result["steps_completed"].append("document_conversion")
            logger.debug("Step 2: Document conversion simulated for %s", file_path.name)
            
            # Step 3: Content Analysis
            if file_path.suffix in ['.md', '.txt']:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    content_length = len(content)
                    word_count = len(content.split())
                    
                    # Estimate chunks based on demo config (300 chars per chunk)
                    estimated_chunks = max(1, content_length // 300)
                    
                    file_result.update({
                        "content_length": content_length,
                        "word_count": word_count,
                        "estimated_chunks": estimated_chunks
                    })
                    
                    simulation_results["total_content_chars"] += content_length
                    simulation_results["estimated_chunks"] += estimated_chunks
                    
                    logger.debug("Content analysis for %s: %d chars, %d words, %d estimated chunks",
                               file_path.name, content_length, word_count, estimated_chunks)
                    
                except Exception as e:
                    logger.warning("Could not read content from %s: %s", file_path.name, str(e))
                    file_result["estimated_chunks"] = 1
                    simulation_results["estimated_chunks"] += 1
            
            elif file_path.suffix == '.pdf':
                # Simulate PDF processing
                file_size = file_path.stat().st_size
                estimated_chunks = max(1, file_size // 10000)  # Rough estimation
                
                file_result.update({
                    "file_size": file_size,
                    "estimated_chunks": estimated_chunks
                })
                
                simulation_results["estimated_chunks"] += estimated_chunks
                
                logger.debug("PDF analysis for %s: %d bytes, %d estimated chunks",
                           file_path.name, file_size, estimated_chunks)
            
            # Step 4: Chunking (simulated)
            file_result["steps_completed"].append("chunking")
            logger.debug("Step 3: Chunking simulated for %s", file_path.name)
            
            # Step 5: Metadata Enhancement (simulated)
            file_result["steps_completed"].append("metadata_enhancement")
            logger.debug("Step 4: Metadata enhancement simulated for %s", file_path.name)
            
            simulation_results["processed_files"].append(file_result)
            
        except Exception as e:
            logger.error("Simulation failed for file %s: %s", file_path.name, str(e))
            file_result["error"] = str(e)
            simulation_results["processed_files"].append(file_result)
    
    logger.info("Processing simulation completed. %d files processed, %d total estimated chunks",
               len(simulation_results["processed_files"]), simulation_results["estimated_chunks"])
    
    return simulation_results


def print_results(file_analysis: Dict[str, Any], 
                  analyzer_results: Dict[str, Any], 
                  simulation_results: Dict[str, Any]) -> None:
    """
    Print results in a clean, AGENTS.md compliant format.
    
    Args:
        file_analysis: Results from file analysis
        analyzer_results: Results from FileAnalyzer testing
        simulation_results: Results from processing simulation
    """
    print("=" * 80)
    print("DOCUMENT PROCESSING PIPELINE DEMONSTRATION RESULTS")
    print("=" * 80)
    print()
    
    # File Analysis Results
    print("FILE ANALYSIS RESULTS:")
    print(f"  Total target files: {file_analysis['total_files']}")
    print(f"  Files found: {len(file_analysis['existing_files'])}")
    print(f"  Files missing: {len(file_analysis['missing_files'])}")
    print()
    
    for filename, details in file_analysis["file_details"].items():
        if details["exists"]:
            print(f"  {filename}:")
            print(f"    Size: {details['size']:,} bytes")
            if "word_count" in details:
                print(f"    Words: {details['word_count']:,}")
                print(f"    Lines: {details['line_count']:,}")
        else:
            print(f"  {filename}: NOT FOUND")
    print()
    
    # Analyzer Results
    print("FILE ANALYZER TEST RESULTS:")
    if analyzer_results["analyzer_created"]:
        print(f"  Analyzer created successfully")
        print(f"  Supported files: {len(analyzer_results['supported_files'])}")
        print(f"  Unsupported files: {len(analyzer_results['unsupported_files'])}")
        print()
        
        for file_path, analysis in analyzer_results["file_analyses"].items():
            filename = Path(file_path).name
            if "error" not in analysis:
                print(f"  {filename}:")
                print(f"    Type: {analysis['detected_type']}")
                print(f"    Supported: {analysis['is_supported']}")
                print(f"    Size: {analysis['size_formatted']}")
            else:
                print(f"  {filename}: ANALYSIS FAILED - {analysis['error']}")
    else:
        print(f"  Analyzer creation failed: {analyzer_results.get('error', 'Unknown error')}")
    print()
    
    # Simulation Results
    print("PROCESSING SIMULATION RESULTS:")
    print(f"  Files processed: {len(simulation_results['processed_files'])}")
    print(f"  Total estimated chunks: {simulation_results['estimated_chunks']}")
    print(f"  Total content characters: {simulation_results['total_content_chars']:,}")
    print()
    
    for file_result in simulation_results["processed_files"]:
        filename = file_result["filename"]
        print(f"  {filename}:")
        
        if "error" not in file_result:
            print(f"    Steps completed: {len(file_result['steps_completed'])}")
            print(f"    Steps: {', '.join(file_result['steps_completed'])}")
            
            if "estimated_chunks" in file_result:
                print(f"    Estimated chunks: {file_result['estimated_chunks']}")
            
            if "word_count" in file_result:
                print(f"    Word count: {file_result['word_count']:,}")
            elif "file_size" in file_result:
                print(f"    File size: {file_result['file_size']:,} bytes")
        else:
            print(f"    ERROR: {file_result['error']}")
    print()
    
    # Summary
    print("DEMONSTRATION SUMMARY:")
    print(f"  Pipeline components tested: FileAnalyzer, Configuration")
    print(f"  Processing workflow simulated end-to-end")
    print(f"  All output follows AGENTS.md logging standards")
    print(f"  No emojis or decorative characters used")
    print("=" * 80)


def main() -> None:
    """Main entry point for the compliant pipeline demo."""
    logger.info("Starting document processing pipeline demonstration")
    
    try:
        # Set up paths
        data_dir = Path(__file__).parent.parent / "data"
        
        if not data_dir.exists():
            logger.error("Data directory not found: %s", data_dir.absolute())
            print(f"Error: Data directory not found at {data_dir.absolute()}")
            return
        
        # Create configuration
        config = create_demo_config()
        logger.info("Pipeline configuration created successfully")
        
        # Analyze files
        file_analysis = analyze_data_files(data_dir)
        
        if not file_analysis["existing_files"]:
            logger.warning("No files found for processing")
            print("Warning: No target files found in data directory")
            return
        
        # Convert to Path objects
        file_paths = [Path(fp) for fp in file_analysis["existing_files"]]
        
        # Test FileAnalyzer
        analyzer_results = test_file_analyzer(config, file_paths)
        
        # Simulate processing
        simulation_results = simulate_document_processing(file_paths)
        
        # Print results
        print_results(file_analysis, analyzer_results, simulation_results)
        
        logger.info("Pipeline demonstration completed successfully")
        
    except Exception as e:
        logger.error("Pipeline demonstration failed: %s", str(e))
        print(f"Error: Pipeline demonstration failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()