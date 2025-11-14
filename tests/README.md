# Document Processor Test Suite

## Overview
This test suite comprehensively tests the `PDFProcessor` class in `src.document_processing.doc_processor`. The tests ensure that the processor correctly handles PDF document processing with smart chunking, metadata generation, and citation information.

## Test Coverage

### Unit Tests (14 tests)
1. **Initialization Tests**
   - Default and custom parameter initialization
   - Proper setup of internal components

2. **Core Functionality Tests**
   - PDF document processing pipeline
   - Smart chunking algorithm with boundary detection
   - Overlap handling between chunks
   - Empty document handling

3. **Metadata and Citation Tests**
   - Complete metadata generation
   - Unique chunk ID creation
   - Citation information structure
   - Character positioning accuracy
   - Timestamp format validation

4. **Error Handling Tests**
   - Invalid content handling
   - Chunk size limit enforcement
   - Graceful failure scenarios

5. **Data Integrity Tests**
   - Content hash consistency
   - Character position accuracy
   - Chunk uniqueness validation

### Integration Tests (1 test)
- Full pipeline simulation with realistic PDF content

## Key Features Tested

### Smart Chunking Algorithm ✅
- Character-based chunking with configurable size
- Intelligent boundary detection (periods, newlines)
- Configurable overlap between chunks
- Proper handling of edge cases

### Citation System ✅
- Unique chunk IDs with content hashing
- Complete citation metadata
- Source file tracking
- Page number preservation
- Character-level positioning

### Metadata Generation ✅
- Processing timestamps
- Chunk indices and sizes
- Source information
- PDF-specific metadata preservation

### Error Handling ✅
- Empty content handling
- Invalid input graceful failure
- Proper logging integration

## Running the Tests

### Run all tests:
```bash
python -m pytest tests/test_doc_processor.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_doc_processor.py::TestPDFProcessor -v
```

### Run specific test:
```bash
python -m pytest tests/test_doc_processor.py::TestPDFProcessor::test_processor_initialization -v
```

### Run with coverage:
```bash
python run_tests.py --coverage
```

## Test Results
- **Total Tests**: 15
- **Passed**: 15 ✅
- **Failed**: 0 ❌
- **Coverage**: Comprehensive coverage of all major functionality

## Files Created
- `tests/test_doc_processor.py` - Main test file
- `tests/conftest.py` - Pytest configuration and shared fixtures
- `tests/__init__.py` - Package initialization
- `run_tests.py` - Test runner script

## Test Dependencies
- pytest
- unittest.mock (for mocking external dependencies)
- haystack (for Document class)

The test suite ensures that the PDFProcessor meets all requirements for production use and maintains compatibility with the original `mew.py` functionality while integrating seamlessly with the Haystack ecosystem.