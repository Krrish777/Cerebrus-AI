# Sample Markdown Document

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
