#!/usr/bin/env python3
"""
Test script for the enhanced DocumentProcessor with multi-file type support.
Demonstrates PDF, text, and markdown processing capabilities.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_processing.doc_processor import DocumentProcessor

def create_test_files():
    """Create sample test files for demonstration."""
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    # Create a sample text file
    text_file = test_dir / "sample.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("""This is a sample text file for testing the DocumentProcessor.
        
The processor should be able to handle this text file and chunk it appropriately.
It contains multiple paragraphs and sentences that will be processed and split
into intelligent chunks based on natural boundaries.

This is another paragraph to demonstrate the chunking capabilities.
The system should maintain context while creating manageable chunk sizes.""")
    
    # Create a sample markdown file
    markdown_file = test_dir / "sample.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write("""# Sample Markdown Document

This is a **sample markdown file** for testing the DocumentProcessor.

## Features

- Handles multiple file types
- Smart chunking with natural boundaries
- Comprehensive logging
- Metadata preservation

## Code Example

```python
processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
result = processor.run(sources=["file1.txt", "file2.md", "file3.pdf"])
```

## Conclusion

The DocumentProcessor provides a unified interface for processing various document types
while maintaining high-quality chunking and detailed processing logs.
""")
    
    return [str(text_file), str(markdown_file)]

def test_document_processor():
    """Test the DocumentProcessor with multiple file types."""
    
    print("=" * 80)
    print("🧪 TESTING UNIVERSAL DOCUMENT PROCESSOR")
    print("=" * 80)
    
    # Create test files
    print("📁 Creating test files...")
    test_files = create_test_files()
    print(f"✅ Created {len(test_files)} test files:")
    for file in test_files:
        print(f"   • {file}")
    
    # Initialize processor
    print("\n🔧 Initializing DocumentProcessor...")
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    
    # Process the documents
    print("\n🚀 Processing documents...")
    try:
        result = processor.run(sources=test_files)
        documents = result['documents']
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📊 Results:")
        print(f"   • Total chunks created: {len(documents)}")
        
        # Analyze results by type
        type_stats = {}
        for doc in documents:
            doc_type = doc.meta.get('source_type', 'unknown')
            type_stats[doc_type] = type_stats.get(doc_type, 0) + 1
        
        print(f"   • Chunk distribution:")
        for doc_type, count in type_stats.items():
            print(f"     - {doc_type}: {count} chunks")
        
        # Show sample chunks
        print(f"\n📝 Sample chunks:")
        for i, doc in enumerate(documents[:3], 1):
            chunk_id = doc.meta.get('chunk_id', 'unknown')
            content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content # type: ignore
            print(f"   {i}. [{chunk_id}] {content_preview}")
        
        if len(documents) > 3:
            print(f"   ... and {len(documents) - 3} more chunks")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise
    
    print("\n🎉 Test completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    test_document_processor()