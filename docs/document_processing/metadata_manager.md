# Metadata Manager

## Table of Contents
- [Overview](#overview)
- [Class: MetadataManager](#class-metadatamanager)
- [Methods](#methods)
- [Metadata Structure](#metadata-structure)
- [Usage Examples](#usage-examples)
- [Validation](#validation)
- [Citation Generation](#citation-generation)
- [Integration Guide](#integration-guide)

---

## Overview

### Purpose
`metadata_manager.py` manages **comprehensive metadata** for documents and chunks throughout the processing pipeline. The `MetadataManager` class ensures consistent metadata structure, provides validation, and generates citation information for traceability.

### Key Features
- ✅ **Metadata enrichment** - Adds standardized fields to documents/chunks
- ✅ **Citation generation** - Creates citation metadata for source tracking
- ✅ **Validation** - Ensures metadata completeness and correctness
- ✅ **Serialization safety** - Ensures JSON-serializable metadata
- ✅ **Schema management** - Defines and enforces metadata structure
- ✅ **Source tracking** - Maintains document lineage and provenance

### Design Principles
Follows **AGENTS.md**:
- **Single Responsibility** - Only handles metadata management
- **Data Consistency** - Standardized metadata across all stages
- **Validation** - Defensive checks for data integrity

### When to Use
- Enriching documents with processing metadata
- Creating chunk-specific metadata
- Validating metadata completeness
- Generating citations for retrieval results
- Ensuring metadata schema compliance

---

## Class: MetadataManager

### Location
```python
from src.document_processing.metadata_manager import MetadataManager
```

### Initialization

```python
def __init__(self, config: PipelineConfig)
```

**Parameters:**
- `config` (PipelineConfig): Pipeline configuration containing metadata field names

**Example:**
```python
from src.document_processing.metadata_manager import MetadataManager
from src.document_processing.pipeline_config import get_pipeline_config

config = get_pipeline_config()
metadata_mgr = MetadataManager(config)
```

---

## Methods

### `enhance_metadata()`

**Enhance document metadata with standardized fields and processing information.**

```python
def enhance_metadata(self, document: Document, file_type: str) -> Document
```

**Parameters:**
- `document` (Document): Document to enhance
- `file_type` (str): Type of the source file ("PDF", "Text", "Markdown")

**Returns:**
- `Document`: New document with enhanced metadata

**Enhancements Added:**
1. **Standard metadata** - Source file, type, content metrics
2. **Processing metadata** - Timestamps, versions, pipeline info
3. **Citation metadata** - Source tracking, extraction details

**Example:**
```python
from haystack.dataclasses import Document

metadata_mgr = MetadataManager(config)

# Document from converter
doc = Document(
    id="doc_123",
    content="Document content...",
    meta={"file_path": "/data/document.pdf", "page_number": 1}
)

# Enhance metadata
enhanced_doc = metadata_mgr.enhance_metadata(doc, "PDF")

print(f"Original fields: {len(doc.meta)}")
print(f"Enhanced fields: {len(enhanced_doc.meta)}")
print(f"Source file: {enhanced_doc.meta['source_file']}")
print(f"Processing date: {enhanced_doc.meta['processing_date']}")
```

---

### `create_chunk_metadata()`

**Create comprehensive metadata specifically for document chunks.**

```python
def create_chunk_metadata(
    self, 
    chunk: Document, 
    parent_doc: Document,
    chunk_index: int, 
    boundaries: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `chunk` (Document): Chunk document
- `parent_doc` (Document): Original parent document
- `chunk_index` (int): Index of this chunk
- `boundaries` (Optional[Dict[str, Any]]): Boundary detection information

**Returns:**
- `Dict[str, Any]`: Chunk-specific metadata dictionary

**Metadata Added:**
- Chunk identification (ID, index)
- Parent document tracking
- Content metrics (word count, line count, size)
- Boundary information (if provided)
- Content hash

**Example:**
```python
from haystack.dataclasses import Document

metadata_mgr = MetadataManager(config)

# Parent document
parent = Document(
    id="doc_123",
    content="Long document...",
    meta={"source_file": "document.pdf", "source_type": "pdf"}
)

# Chunk
chunk = Document(
    id="chunk_0",
    content="First chunk content...",
    meta={}
)

# Create chunk metadata
chunk_meta = metadata_mgr.create_chunk_metadata(
    chunk=chunk,
    parent_doc=parent,
    chunk_index=0,
    boundaries={"found": True, "type": "paragraph", "start_char": 0, "end_char": 500}
)

print(f"Chunk metadata fields: {len(chunk_meta)}")
print(f"Chunk ID: {chunk_meta['chunk_id']}")
print(f"Parent ID: {chunk_meta['parent_document_id']}")
print(f"Boundary found: {chunk_meta.get('boundary_found', False)}")
```

---

### `validate_metadata()`

**Validate document metadata for completeness and correctness.**

```python
def validate_metadata(self, document: Document) -> Dict[str, Any]
```

**Parameters:**
- `document` (Document): Document to validate

**Returns:**
```python
{
    "valid": bool,                          # True if all checks pass
    "errors": List[str],                    # Critical errors
    "warnings": List[str],                  # Non-critical warnings
    "field_count": int,                     # Total metadata fields
    "required_fields_present": int          # Count of required fields present
}
```

**Validation Checks:**
1. Required fields presence
2. Non-empty required fields
3. Data type validation
4. JSON serializability

**Example:**
```python
metadata_mgr = MetadataManager(config)

doc = Document(
    id="doc_123",
    content="Content...",
    meta={"source_file": "document.pdf", "source_type": "pdf"}
)

validation = metadata_mgr.validate_metadata(doc)

if validation['valid']:
    print("✅ Metadata valid")
else:
    print(f"❌ Validation failed: {len(validation['errors'])} errors")
    for error in validation['errors']:
        print(f"  {error}")
```

---

### `extract_citations()`

**Extract citation information from a list of documents.**

```python
def extract_citations(self, documents: List[Document]) -> List[Dict[str, Any]]
```

**Parameters:**
- `documents` (List[Document]): List of documents

**Returns:**
- `List[Dict[str, Any]]`: List of citation dictionaries

**Citation Fields:**
- `source_file` - Original file path
- `document_type` - File type
- `document_id` - Document identifier
- `chunk_index` - Chunk index (if applicable)
- `chunk_id` - Chunk identifier (if applicable)

**Example:**
```python
metadata_mgr = MetadataManager(config)

# Documents with metadata
documents = [...]  # List of Document objects

# Extract citations
citations = metadata_mgr.extract_citations(documents)

for citation in citations:
    print(f"Source: {citation['source_file']}")
    print(f"Type: {citation['document_type']}")
    if 'chunk_index' in citation:
        print(f"Chunk: {citation['chunk_index']}")
```

---

### `get_metadata_schema()`

**Get the metadata schema definition.**

```python
def get_metadata_schema(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "version": str,
    "required_fields": List[str],
    "optional_fields": List[str],
    "field_descriptions": Dict[str, str]
}
```

**Example:**
```python
metadata_mgr = MetadataManager(config)

schema = metadata_mgr.get_metadata_schema()

print(f"Schema version: {schema['version']}")
print(f"\nRequired fields ({len(schema['required_fields'])}):")
for field in schema['required_fields']:
    print(f"  - {field}: {schema['field_descriptions'].get(field, 'No description')}")
```

---

### Private Methods

#### `_create_standard_metadata()`
```python
def _create_standard_metadata(self, document: Document, file_type: str) -> Dict[str, Any]
```
**Purpose:** Create standardized metadata fields  
**Returns:** Dictionary with source file, type, content metrics

#### `_create_processing_metadata()`
```python
def _create_processing_metadata(self) -> Dict[str, Any]
```
**Purpose:** Create processing-related metadata  
**Returns:** Dictionary with timestamps, versions

#### `_create_citation_metadata()`
```python
def _create_citation_metadata(self, document: Document, file_type: str) -> Dict[str, Any]
```
**Purpose:** Create citation and reference metadata  
**Returns:** Dictionary with citation information

#### `_clean_metadata()`
```python
def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]
```
**Purpose:** Clean and validate metadata for JSON serializability  
**Returns:** Cleaned metadata dictionary

---

## Metadata Structure

### Standard Metadata

```python
{
    "source_file": str,            # Path to source file
    "source_type": str,            # File type (pdf, text, markdown)
    "document_id": str,            # Unique document identifier
    "content_length": int,         # Content length in characters
    "has_content": bool            # Whether content is non-empty
}
```

### Processing Metadata

```python
{
    "processing_date": str,        # ISO timestamp
    "processing_timestamp": float, # Unix timestamp
    "processor_version": str,      # Processor version
    "pipeline_version": str,       # Pipeline version
    "metadata_schema_version": str # Schema version
}
```

### Citation Metadata

```python
{
    "citation": {
        "source_file": str,
        "document_type": str,
        "document_id": str,
        "extraction_method": str,
        "extraction_timestamp": str,
        "page_number": int          # For PDF files
    }
}
```

### Chunk-Specific Metadata

```python
{
    # From create_chunk_metadata()
    "chunk_id": str,               # Unique chunk identifier
    "chunk_index": int,            # Sequential index
    "parent_document_id": str,     # Parent document ID
    "word_count": int,             # Number of words
    "line_count": int,             # Number of lines
    "chunk_size": int,             # Content length
    "content_hash": str,           # MD5 hash (first 8 chars)
    
    # Boundary information (if provided)
    "boundary_found": bool,
    "boundary_type": str,
    "start_char": int,
    "end_char": int
}
```

---

## Usage Examples

### Example 1: Basic Metadata Enhancement

```python
from haystack.dataclasses import Document
from src.document_processing.metadata_manager import MetadataManager
from src.document_processing.pipeline_config import get_pipeline_config

config = get_pipeline_config()
metadata_mgr = MetadataManager(config)

# Create document
doc = Document(
    id="doc_1",
    content="Document content...",
    meta={"file_path": "/data/document.pdf"}
)

# Enhance metadata
enhanced = metadata_mgr.enhance_metadata(doc, "PDF")

# Check enhancements
print(f"Original fields: {list(doc.meta.keys())}")
print(f"Enhanced fields: {list(enhanced.meta.keys())}")
```

### Example 2: Creating Chunk Metadata

```python
metadata_mgr = MetadataManager(config)

# Parent document
parent = Document(
    id="doc_123",
    content="Long document text...",
    meta={
        "source_file": "/data/paper.pdf",
        "source_type": "pdf",
        "page_number": 5
    }
)

# Chunks
chunks = []  # Assume we have chunks from ChunkingService

for i, chunk in enumerate(chunks):
    # Create chunk metadata
    chunk_meta = metadata_mgr.create_chunk_metadata(
        chunk=chunk,
        parent_doc=parent,
        chunk_index=i
    )
    
    # Update chunk metadata
    chunk.meta.update(chunk_meta)
    
    print(f"Chunk {i}: {chunk_meta['word_count']} words, hash={chunk_meta['content_hash']}")
```

### Example 3: Metadata Validation

```python
def validate_documents(documents: List[Document]):
    """Validate metadata for all documents."""
    metadata_mgr = MetadataManager(config)
    
    all_valid = True
    for i, doc in enumerate(documents):
        validation = metadata_mgr.validate_metadata(doc)
        
        if not validation['valid']:
            all_valid = False
            print(f"❌ Document {i} validation failed:")
            for error in validation['errors']:
                print(f"  {error}")
        
        if validation['warnings']:
            print(f"⚠️  Document {i} warnings:")
            for warning in validation['warnings']:
                print(f"  {warning}")
    
    if all_valid:
        print("✅ All documents have valid metadata")
    
    return all_valid

# Use validation
is_valid = validate_documents(processed_documents)
```

### Example 4: Citation Extraction

```python
from src.vector_database.qdrant_db import QdrantDB

# Retrieve documents from vector DB
vector_db = QdrantDB(collection_name="documents")
search_results = vector_db.search("machine learning", top_k=5)

# Extract citations
metadata_mgr = MetadataManager(config)
citations = metadata_mgr.extract_citations(search_results)

# Display citations
print("Sources:")
for i, citation in enumerate(citations, 1):
    print(f"{i}. {citation['source_file']}")
    if 'chunk_index' in citation:
        print(f"   Chunk {citation['chunk_index']}")
    print(f"   Type: {citation['document_type']}")
```

### Example 5: Schema Inspection

```python
metadata_mgr = MetadataManager(config)

schema = metadata_mgr.get_metadata_schema()

print(f"Metadata Schema v{schema['version']}")
print(f"\n{'='*50}")

print(f"\nRequired Fields ({len(schema['required_fields'])}):")
for field in schema['required_fields']:
    desc = schema['field_descriptions'].get(field, 'No description')
    print(f"  {field:<20} - {desc}")

print(f"\nOptional Fields ({len(schema['optional_fields'])}):")
for field in schema['optional_fields']:
    desc = schema['field_descriptions'].get(field, 'No description')
    print(f"  {field:<20} - {desc}")
```

### Example 6: Complete Pipeline Integration

```python
from pathlib import Path
from src.document_processing.document_converter import DocumentConverter
from src.document_processing.chunking_service import ChunkingService

converter = DocumentConverter(config)
chunker = ChunkingService(config)
metadata_mgr = MetadataManager(config)

# Convert files
files = [Path("data/document.pdf")]
conversion_result = converter.convert_files("PDF", files)

# Chunk documents
chunking_result = chunker.chunk_documents(conversion_result['documents'])

# Enhance metadata for all chunks
final_chunks = []
for chunk in chunking_result['documents']:
    enhanced = metadata_mgr.enhance_metadata(chunk, "PDF")
    final_chunks.append(enhanced)

print(f"Processed {len(final_chunks)} chunks with enhanced metadata")

# Validate all chunks
for chunk in final_chunks:
    validation = metadata_mgr.validate_metadata(chunk)
    if not validation['valid']:
        print(f"❌ Chunk {chunk.meta['chunk_index']} has invalid metadata")
```

### Example 7: Custom Metadata Fields

```python
from haystack.dataclasses import Document

metadata_mgr = MetadataManager(config)

# Document with custom fields
doc = Document(
    id="doc_1",
    content="Content...",
    meta={
        "file_path": "/data/document.pdf",
        "author": "John Doe",           # Custom field
        "publication_date": "2024-01",  # Custom field
        "tags": ["ML", "AI"]            # Custom field
    }
)

# Enhance (preserves custom fields)
enhanced = metadata_mgr.enhance_metadata(doc, "PDF")

print("Custom fields preserved:")
print(f"  Author: {enhanced.meta.get('author')}")
print(f"  Publication date: {enhanced.meta.get('publication_date')}")
print(f"  Tags: {enhanced.meta.get('tags')}")

print("\nStandard fields added:")
print(f"  Source type: {enhanced.meta['source_type']}")
print(f"  Processing date: {enhanced.meta['processing_date']}")
```

---

## Validation

### Required Fields

```python
required_fields = [
    "source_file",    # Path to source file
    "source_type"     # File type
]
```

### Validation Rules

#### 1. Field Presence
```python
for field in required_fields:
    if field not in document.meta:
        errors.append(f"Missing required metadata field: {field}")
```

#### 2. Non-Empty Fields
```python
if field in document.meta and not document.meta[field]:
    warnings.append(f"Empty required metadata field: {field}")
```

#### 3. Type Validation
```python
type_validations = {
    "word_count": int,
    "line_count": int,
    "chunk_index": int
}

for field, expected_type in type_validations.items():
    if field in document.meta and document.meta[field] is not None:
        if not isinstance(document.meta[field], expected_type):
            errors.append(f"Invalid type for {field}: expected {expected_type.__name__}")
```

#### 4. JSON Serializability
```python
try:
    import json
    json.dumps(document.meta)
except (TypeError, ValueError) as e:
    errors.append(f"Metadata not JSON serializable: {str(e)}")
```

---

## Citation Generation

### Citation Structure

```python
{
    "source_file": str,           # Original file path
    "document_type": str,         # File type
    "document_id": str,           # Document identifier
    "extraction_method": str,     # How content was extracted
    "extraction_timestamp": str,  # ISO timestamp
    "page_number": int,           # For PDFs (optional)
    "chunk_index": int,           # For chunks (optional)
    "chunk_id": str              # For chunks (optional)
}
```

### Citation Example

```python
{
    "source_file": "/data/research_paper.pdf",
    "document_type": "pdf",
    "document_id": "doc_abc123",
    "extraction_method": "haystack_pipeline",
    "extraction_timestamp": "2024-01-15T10:30:45.123456",
    "page_number": 5,
    "chunk_index": 12,
    "chunk_id": "chunk_12_a1b2c3d4"
}
```

### Use Cases

#### 1. Search Result Citations
```python
# After vector search
citations = metadata_mgr.extract_citations(search_results)

# Display to user
for i, citation in enumerate(citations, 1):
    print(f"[{i}] {Path(citation['source_file']).name}, p. {citation.get('page_number', 'N/A')}")
```

#### 2. RAG Source Attribution
```python
# In RAG response
generated_answer = rag_system.generate(query)
citations = metadata_mgr.extract_citations(generated_answer.source_documents)

print(f"Answer: {generated_answer.text}")
print(f"\nSources:")
for citation in citations:
    print(f"  - {citation['source_file']} (chunk {citation['chunk_index']})")
```

---

## Integration Guide

### With Pipeline Orchestrator

```python
class DocumentPipelineOrchestrator:
    def _process_file_group(self, file_type, files):
        # ... conversion and chunking ...
        
        # Enhance metadata
        final_documents = []
        for document in chunking_result["documents"]:
            enhanced_doc = self.metadata_manager.enhance_metadata(document, file_type)
            final_documents.append(enhanced_doc)
        
        return {"documents": final_documents, "errors": []}
```

### With Chunking Service

```python
# After chunking, enhance chunk metadata
chunking_result = chunker.chunk_documents(documents)

for chunk in chunking_result['documents']:
    # Metadata manager adds additional fields
    enhanced = metadata_mgr.enhance_metadata(chunk, "PDF")
```

### Standalone Usage

```python
from src.document_processing.metadata_manager import MetadataManager

metadata_mgr = MetadataManager(config)

# Enhance any document
doc = Document(id="doc_1", content="...", meta={})
enhanced = metadata_mgr.enhance_metadata(doc, "Text")

# Validate metadata
validation = metadata_mgr.validate_metadata(enhanced)
```

---

## Configuration Reference

### Metadata Field Names

```yaml
metadata_fields:
  chunk_id: "chunk_id"
  chunk_index: "chunk_index"
  source_file: "source_file"
  source_type: "source_type"
  page_number: "page_number"
  content_hash: "content_hash"
  start_char: "start_char"
  end_char: "end_char"
  word_count: "word_count"
  line_count: "line_count"
  boundary_found: "boundary_found"
  boundary_type: "boundary_type"
  processing_date: "processing_date"
```

**Note:** Field names are configurable, allowing customization for different use cases.

---

## Dependencies

### Internal Dependencies
- `src.document_processing.pipeline_config.PipelineConfig`
- `src.core.logging.get_logger`

### External Dependencies (Haystack)
- `haystack.dataclasses.Document` - Document data structure

### Standard Library
- `typing` - Type hints
- `time` - Timestamps
- `datetime` - ISO timestamps
- `hashlib` - Content hashing (via chunking service)

---

## See Also
- [Overview](./overview.md) - Module architecture
- [Pipeline Orchestrator](./pipeline_orchestrator.md) - Main coordinator
- [Chunking Service](./chunking_service.md) - Document splitting
- [Pipeline Config](./pipeline_config.md) - Configuration management
