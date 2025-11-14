import hashlib
import logging
from datetime import datetime
from src.core.logging import CustomLogger
from haystack import Pipeline, component, Document
from haystack.components.converters import PyPDFToDocument, TextFileToDocument, MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from typing import List, Dict, Any, Optional, Union
import os
import mimetypes
from pathlib import Path

# Configure logging with type safety
try:
    from src.core.logging import CustomLogger
    _logger_instance = CustomLogger()
    _temp_logger = _logger_instance.get_logger(__name__)
    if _temp_logger is not None:
        logger: logging.Logger = _temp_logger
    else:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Universal Document Processor that handles multiple file types (PDF, text, markdown)
    using FileTypeRouter for intelligent routing and comprehensive logging at every step.
    
    Supported formats:
    - PDF files (.pdf)
    - Text files (.txt, .text)
    - Markdown files (.md, .markdown)
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        logger.info("=" * 60)
        logger.info("INITIALIZING UNIVERSAL DOCUMENT PROCESSOR")
        logger.info("=" * 60)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"🔧 Configuration Parameters:")
        logger.info(f"   • Chunk Size: {chunk_size} characters")
        logger.info(f"   • Chunk Overlap: {chunk_overlap} characters")
        logger.info(f"   • Overlap Percentage: {(chunk_overlap/chunk_size)*100:.1f}%")
        
        # Initialize file type router with supported MIME types
        logger.info("🔧 Initializing FileTypeRouter...")
        try:
            self.file_type_router = FileTypeRouter(mime_types=[
                "application/pdf",      # PDF files
                "text/plain",          # Text files
                "text/markdown"         # Markdown files
            ])
            logger.info("✅ FileTypeRouter initialized successfully")
            logger.info("   • Supported types: PDF, Text, Markdown")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FileTypeRouter: {e}")
            raise
        
        # Initialize document converters
        logger.info("📦 Initializing document converters...")
        try:
            self.pdf_converter = PyPDFToDocument()
            logger.info("✅ PyPDFToDocument converter initialized")
            
            self.text_converter = TextFileToDocument()
            logger.info("✅ TextFileToDocument converter initialized")
            
            # Try to initialize markdown converter, fallback if dependencies missing
            try:
                self.markdown_converter = MarkdownToDocument()
                logger.info("✅ MarkdownToDocument converter initialized")
                self.markdown_supported = True
            except ImportError as e:
                logger.warning(f"⚠️  MarkdownToDocument not available: {e}")
                logger.info("   • Markdown files will be processed as text files")
                self.markdown_converter = None
                self.markdown_supported = False
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize document converters: {e}")
            raise
        
        # Initialize document joiner
        logger.info("🔗 Initializing DocumentJoiner...")
        try:
            self.document_joiner = DocumentJoiner()
            logger.info("✅ DocumentJoiner initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DocumentJoiner: {e}")
            raise
        
        # Initialize logger - use a shared instance to avoid multiple log files
        self._logger_instance = CustomLogger()
        self.logger = self._logger_instance.get_logger(f"{__name__}.DocumentProcessor")  # type: ignore
        
        logger.info("🚀 Universal Document Processor initialization completed successfully")
        logger.info("-" * 60)
        
    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]) -> Dict[str, List[Document]]:
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING UNIVERSAL DOCUMENT PROCESSING PIPELINE")
        logger.info("=" * 80)
        
        logger.info(f"📋 INPUT ANALYSIS:")
        logger.info(f"   • Number of sources: {len(sources)}")
        logger.info(f"   • Source files: {sources}")
        
        # Analyze file types
        file_type_stats = self._analyze_file_types(sources)
        
        for i, source in enumerate(sources, 1):
            file_type = self._detect_file_type(source)
            file_size = self._get_file_size(source)
            logger.info(f"   • Source {i}: '{os.path.basename(source)}'")
            logger.info(f"     - Full path: {source}")
            logger.info(f"     - Detected type: {file_type}")
            logger.info(f"     - File size: {file_size}")
        
        logger.info(f"\n📊 FILE TYPE DISTRIBUTION:")
        for file_type, count in file_type_stats.items():
            logger.info(f"   • {file_type}: {count} files")
        
        if not sources:
            logger.warning("⚠️  WARNING: No sources provided for document processing")
            logger.info("📤 RETURNING: Empty documents list")
            return {'documents': []}
        
        logger.info("\n🔀 STEP 1: FILE TYPE ROUTING")
        logger.info("-" * 40)
        logger.debug("🔧 Calling FileTypeRouter.run()...")
        
        try:
            start_time = datetime.now()
            routed_files = self.file_type_router.run(sources=sources) # type: ignore # type: ignore
            routing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ File routing completed successfully")
            logger.info(f"   • Routing time: {routing_time:.2f} seconds")
            
            # Log routing results
            pdf_files = routed_files.get('application/pdf', [])
            text_files = routed_files.get('text/plain', [])
            markdown_files = routed_files.get('text/markdown', [])
            
            logger.info(f"   • PDF files routed: {len(pdf_files)}")
            logger.info(f"   • Text files routed: {len(text_files)}")
            logger.info(f"   • Markdown files routed: {len(markdown_files)}")
            
            if pdf_files:
                logger.debug(f"     PDF files: {[os.path.basename(f) for f in pdf_files]}") # type: ignore
            if text_files:
                logger.debug(f"     Text files: {[os.path.basename(f) for f in text_files]}") # type: ignore
            if markdown_files:
                logger.debug(f"     Markdown files: {[os.path.basename(f) for f in markdown_files]}") # type: ignore
                
        except Exception as e:
            logger.error(f"❌ File routing failed: {e}")
            logger.error(f"   • Error type: {type(e).__name__}")
            raise
        
        # Process each file type
        all_documents = []
        
        logger.info("\n📖 STEP 2: DOCUMENT CONVERSION")
        logger.info("-" * 40)
        
        # Process PDF files
        if pdf_files:
            logger.info(f"\n🔴 Processing {len(pdf_files)} PDF files...")
            pdf_docs = self._process_pdf_files(pdf_files) # type: ignore
            all_documents.extend(pdf_docs)
            
        # Process text files
        if text_files:
            logger.info(f"\n📝 Processing {len(text_files)} text files...")
            text_docs = self._process_text_files(text_files) # type: ignore
            all_documents.extend(text_docs)
            
        # Process markdown files
        if markdown_files:
            logger.info(f"\n📑 Processing {len(markdown_files)} markdown files...")
            markdown_docs = self._process_markdown_files(markdown_files) # type: ignore
            all_documents.extend(markdown_docs)
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Universal document processing completed successfully")
        logger.info(f"📈 STATISTICS:")
        logger.info(f"   • Total files processed: {len(sources)}")
        logger.info(f"   • Total documents/chunks created: {len(all_documents)}")
        
        if all_documents:
            # Calculate statistics across all document types
            all_chunk_sizes = [len(doc.content) for doc in all_documents if doc.content]
            if all_chunk_sizes:
                avg_chunk_size = sum(all_chunk_sizes) / len(all_chunk_sizes)
                total_chars = sum(all_chunk_sizes)
                
                logger.info(f"   • Average chunk size: {avg_chunk_size:.1f} characters")
                logger.info(f"   • Chunk size range: {min(all_chunk_sizes)} - {max(all_chunk_sizes)} characters")
                logger.info(f"   • Total output characters: {total_chars:,}")
                
                # File type breakdown
                type_counts = {}
                for doc in all_documents:
                    source_type = doc.meta.get('source_type', 'unknown')
                    type_counts[source_type] = type_counts.get(source_type, 0) + 1
                
                logger.info(f"   • Document type breakdown:")
                for doc_type, count in type_counts.items():
                    logger.info(f"     - {doc_type}: {count} chunks")
        
        logger.info(f"🎯 RESULT: Returning {len(all_documents)} processed document chunks")
        logger.info("=" * 80)
        
        return {'documents': all_documents}
        logger.info(f"✅ PDF processing completed successfully")
        logger.info(f"📈 STATISTICS:")
        logger.info(f"   • Total documents processed: {len(pdf_docs['documents'])}")
        logger.info(f"   • Total input characters: {total_input_chars:,}")
        logger.info(f"   • Total chunks created: {len(all_chunks)}")
        
        if all_chunks:
            all_chunk_sizes = [len(chunk.content) for chunk in all_chunks]
            avg_chunk_size = sum(all_chunk_sizes) / len(all_chunks)
            compression_ratio = len(all_chunks) / len(pdf_docs['documents'])
            
            logger.info(f"   • Average chunk size: {avg_chunk_size:.1f} characters")
            logger.info(f"   • Chunk size range: {min(all_chunk_sizes)} - {max(all_chunk_sizes)} characters")
            logger.info(f"   • Compression ratio: {compression_ratio:.2f} chunks per document")
            logger.info(f"   • Total output characters: {sum(all_chunk_sizes):,}")
            
            # Character efficiency analysis
            char_efficiency = (sum(all_chunk_sizes) / total_input_chars) * 100 if total_input_chars > 0 else 0
            logger.info(f"   • Character retention: {char_efficiency:.1f}%")
            
            # Chunk ID summary
            chunk_ids = [chunk.meta.get('chunk_id', 'no_id') for chunk in all_chunks]
            logger.debug(f"📋 Generated chunk IDs: {chunk_ids[:5]}{'...' if len(chunk_ids) > 5 else ''}")
            
        logger.info(f"🎯 RESULT: Returning {len(all_chunks)} processed document chunks")
        logger.info("=" * 80)
        
        return {'documents': all_chunks}
    
    def _create_smart_chunks(self, document: Document) -> List[Document]:
        """Create smart chunks from any document type with comprehensive logging."""
        
        doc_name = document.meta.get('name', document.meta.get('source_file', 'unknown'))
        page_number = document.meta.get('page_number', 1)
        source_type = document.meta.get('source_type', 'unknown')
        
        logger.info(f"\n🔍 STARTING SMART CHUNKING")
        logger.info(f"   • Document: {doc_name}")
        logger.info(f"   • Type: {source_type}")
        if source_type == 'pdf':
            logger.info(f"   • Page: {page_number}")
        
        # Safely get text content
        text = document.content or ""
        source_file = document.meta.get('name', document.meta.get('source_file', 'unknown'))
        
        logger.info(f"📊 CONTENT ANALYSIS:")
        logger.info(f"   • Raw text length: {len(text):,} characters")
        logger.info(f"   • Source file: {source_file}")
        
        if not text.strip():
            logger.warning(f"⚠️  WARNING: No text content found")
            logger.warning(f"   • Page {page_number} of {source_file} is empty or whitespace only")
            return []
        
        # Text content statistics
        lines = text.split('\n')
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        logger.info(f"📈 TEXT STATISTICS:")
        logger.info(f"   • Lines: {len(lines):,}")
        logger.info(f"   • Words: {len(words):,}")
        logger.info(f"   • Sentences (approx): {len(sentences):,}")
        logger.info(f"   • Paragraphs (approx): {len(paragraphs):,}")
        logger.info(f"   • Average words per line: {len(words)/len(lines):.1f}" if len(lines) > 0 else "   • No lines found")
        
        # Preview text content
        text_preview = (text[:200] + "...") if len(text) > 200 else text
        logger.debug(f"📖 TEXT PREVIEW: '{text_preview}'")
        
        # Chunking configuration
        logger.info(f"\n⚙️ CHUNKING CONFIGURATION:")
        logger.info(f"   • Target chunk size: {self.chunk_size:,} characters")
        logger.info(f"   • Overlap: {self.chunk_overlap:,} characters ({(self.chunk_overlap/self.chunk_size)*100:.1f}%)")
        logger.info(f"   • Estimated chunks needed: {len(text) // self.chunk_size + 1}")
        
        chunks = []
        start = 0
        chunk_index = 0
        total_boundary_adjustments = 0
        
        logger.info(f"\n🔄 CHUNKING PROCESS:")
        logger.info("-" * 30)
        
        while start < len(text):
            logger.debug(f"\n🔸 CHUNK {chunk_index} CREATION:")
            logger.debug(f"   • Start position: {start:,} ({(start/len(text)*100):.1f}% through document)")
            
            # Calculate initial end position
            end = min(start + self.chunk_size, len(text))
            initial_end = end
            logger.debug(f"   • Initial end position: {end:,}")
            logger.debug(f"   • Initial chunk size: {end - start:,} characters")
            
            # Look for natural boundaries if not at end of text
            boundary_found = False
            boundary_type = "none"
            if end < len(text):
                logger.debug(f"   🔍 Searching for natural boundaries...")
                
                # Look for sentence endings
                last_period = text.rfind('.', start, end)
                last_exclamation = text.rfind('!', start, end)
                last_question = text.rfind('?', start, end)
                last_newline = text.rfind('\n', start, end)
                last_double_newline = text.rfind('\n\n', start, end)
                
                # Find the best boundary
                sentence_boundaries = [last_period, last_exclamation, last_question]
                best_sentence = max(sentence_boundaries)
                
                logger.debug(f"   📍 Boundary analysis:")
                logger.debug(f"      - Period: {last_period}")
                logger.debug(f"      - Exclamation: {last_exclamation}")
                logger.debug(f"      - Question: {last_question}")
                logger.debug(f"      - Line break: {last_newline}")
                logger.debug(f"      - Paragraph break: {last_double_newline}")
                
                # Choose the best boundary (prefer paragraph > sentence > line)
                boundary = -1
                min_chunk_size = start + int(self.chunk_size * 0.5)
                
                if last_double_newline > min_chunk_size:
                    boundary = last_double_newline + 2
                    boundary_type = "paragraph"
                elif best_sentence > min_chunk_size:
                    boundary = best_sentence + 1
                    boundary_type = "sentence"
                elif last_newline > min_chunk_size:
                    boundary = last_newline + 1
                    boundary_type = "line"
                
                if boundary > start:
                    end = boundary
                    boundary_found = True
                    total_boundary_adjustments += 1
                    adjustment = end - initial_end
                    
                    logger.debug(f"   ✅ Boundary found:")
                    logger.debug(f"      - Type: {boundary_type}")
                    logger.debug(f"      - New end position: {end:,}")
                    logger.debug(f"      - Adjustment: {adjustment:+} characters")
                    logger.debug(f"      - Final chunk size: {end - start:,} characters")
                else:
                    logger.debug(f"   ❌ No suitable boundary found (min size: {min_chunk_size})")
                    logger.debug(f"      - Using original end position: {end:,}")
            else:
                logger.debug(f"   📍 At end of document - using remaining text")
                
            # Extract and process chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                logger.debug(f"   📝 Chunk content:")
                logger.debug(f"      - Length: {len(chunk_text):,} characters")
                logger.debug(f"      - Words: {len(chunk_text.split()):,}")
                logger.debug(f"      - Lines: {len(chunk_text.split(chr(10))):,}")
                
                # Generate unique chunk ID
                content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                if source_type == 'pdf':
                    chunk_id = f"pdf_{chunk_index}_page{page_number}_{content_hash}"
                elif source_type == 'text':
                    chunk_id = f"text_{chunk_index}_{content_hash}"
                elif source_type == 'markdown':
                    chunk_id = f"md_{chunk_index}_{content_hash}"
                else:
                    chunk_id = f"doc_{chunk_index}_{content_hash}"
                
                logger.debug(f"   🔑 Generated chunk ID: {chunk_id}")
                
                # Create comprehensive metadata
                chunk_metadata = {
                    # Source information
                    'source_file': source_file,
                    'source_type': source_type,
                    'page_number': page_number if source_type == 'pdf' else None,
                    
                    # Chunk identification
                    'chunk_index': chunk_index,
                    'chunk_id': chunk_id,
                    'content_hash': content_hash,
                    
                    # Position information
                    'start_char': start,
                    'end_char': end - 1,
                    'chunk_size': len(chunk_text),
                    'char_range': f"{start}-{end - 1}",
                    
                    # Boundary information
                    'boundary_found': boundary_found,
                    'boundary_type': boundary_type,
                    'size_adjustment': end - initial_end,
                    
                    # Processing metadata
                    'processed_timestamp': datetime.now().isoformat(),
                    'processor_version': '3.0',
                    'chunk_overlap': self.chunk_overlap,
                    'target_chunk_size': self.chunk_size,
                    
                    # Citation information
                    'citation': {
                        'source_file': source_file,
                        'type': source_type,
                        'page_number': page_number if source_type == 'pdf' else None,
                        'char_range': f"{start}-{end - 1}",
                        'chunk_id': chunk_id,
                        'extraction_method': 'smart_boundary_chunking'
                    },
                    
                    # Content statistics
                    'word_count': len(chunk_text.split()),
                    'line_count': len(chunk_text.split('\n')),
                    'sentence_count': len([s for s in chunk_text.split('.') if s.strip()])
                }
                
                # Merge with original document metadata
                final_metadata = document.meta.copy()
                final_metadata.update(chunk_metadata)
                
                logger.debug(f"   📋 Metadata created with {len(final_metadata)} fields")
                
                # Create document chunk
                chunk_doc = Document(
                    content=chunk_text,
                    meta=final_metadata
                )
                
                chunks.append(chunk_doc)
                
                logger.info(f"✅ Chunk {chunk_index} created successfully")
                logger.info(f"   • Size: {len(chunk_text):,} characters")
                logger.info(f"   • Range: {start:,}-{end-1:,}")
                logger.info(f"   • Boundary: {boundary_type}")
                logger.info(f"   • Words: {len(chunk_text.split()):,}")
                
                chunk_index += 1
                
                # Calculate next start position with overlap
                if start + self.chunk_size - self.chunk_overlap >= len(text):
                    logger.debug(f"   🏁 Reached end of text - stopping chunking")
                    break
                    
                new_start = max(start + self.chunk_size - self.chunk_overlap, end)
                overlap_amount = end - new_start if end > new_start else 0
                
                logger.debug(f"   ➡️ Next chunk positioning:")
                logger.debug(f"      - Next start: {new_start:,}")
                logger.debug(f"      - Actual overlap: {overlap_amount} characters")
                logger.debug(f"      - Remaining text: {len(text) - new_start:,} characters")
                
                start = new_start
                
            else:
                logger.warning(f"   ⚠️ Empty chunk text after stripping - skipping")
                break
        
        # Chunking summary
        logger.info(f"\n📊 CHUNKING SUMMARY:")
        logger.info("-" * 30)
        logger.info(f"✅ Chunking completed for {source_file} (page {page_number})")
        logger.info(f"📈 RESULTS:")
        logger.info(f"   • Total chunks created: {len(chunks)}")
        logger.info(f"   • Boundary adjustments: {total_boundary_adjustments}")
        logger.info(f"   • Adjustment rate: {(total_boundary_adjustments/len(chunks)*100):.1f}%" if len(chunks) > 0 else "   • No chunks created")
        
        if chunks:
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            chunk_words = [len(chunk.content.split()) for chunk in chunks]
            
            logger.info(f"   • Size statistics:")
            logger.info(f"     - Average: {sum(chunk_sizes)/len(chunk_sizes):.1f} characters")
            logger.info(f"     - Range: {min(chunk_sizes)}-{max(chunk_sizes)} characters")
            logger.info(f"     - Total output: {sum(chunk_sizes):,} characters")
            logger.info(f"   • Word statistics:")
            logger.info(f"     - Average: {sum(chunk_words)/len(chunk_words):.1f} words per chunk")
            logger.info(f"     - Total words: {sum(chunk_words):,}")
            
            # Coverage analysis
            total_output_chars = sum(chunk_sizes)
            coverage = (total_output_chars / len(text)) * 100 if len(text) > 0 else 0
            logger.info(f"   • Text coverage: {coverage:.1f}%")
            
            # Efficiency metrics
            target_chunks = len(text) // self.chunk_size + 1
            efficiency = (len(chunks) / target_chunks) * 100 if target_chunks > 0 else 0
            logger.info(f"   • Chunking efficiency: {efficiency:.1f}%")
            
            chunk_ids = [chunk.meta['chunk_id'] for chunk in chunks]
            logger.debug(f"📋 Chunk IDs: {chunk_ids}")
        
        logger.info(f"🎯 Returning {len(chunks)} chunks for further processing\n")
        
        return chunks
    
    def _analyze_file_types(self, sources: List[str]) -> Dict[str, int]:
        """Analyze and count file types in the sources."""
        stats = {}
        for source in sources:
            file_type = self._detect_file_type(source)
            stats[file_type] = stats.get(file_type, 0) + 1
        return stats
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type based on extension and MIME type."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in ['.pdf']:
            return 'PDF'
        elif extension in ['.txt', '.text']:
            return 'Text'
        elif extension in ['.md', '.markdown']:
            return 'Markdown'
        else:
            # Try MIME type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                if 'pdf' in mime_type:
                    return 'PDF'
                elif 'text' in mime_type:
                    return 'Text'
                elif 'markdown' in mime_type:
                    return 'Markdown'
            return 'Unknown'
    
    def _get_file_size(self, file_path: str) -> str:
        """Get human-readable file size."""
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        except OSError:
            return "Unknown size"
    
    def _process_pdf_files(self, pdf_files: List[str]) -> List[Document]:
        """Process PDF files and return documents."""
        logger.info(f"📄 Converting {len(pdf_files)} PDF files...")
        start_time = datetime.now()
        
        try:
            pdf_docs = self.pdf_converter.run(sources=pdf_files) # type: ignore
            conversion_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ PDF conversion completed")
            logger.info(f"   • Conversion time: {conversion_time:.2f} seconds")
            logger.info(f"   • Documents extracted: {len(pdf_docs['documents'])}")
            
            # Process chunks
            all_chunks = []
            for doc in pdf_docs['documents']:
                chunks = self._create_smart_chunks(doc)
                all_chunks.extend(chunks)
                
            logger.info(f"   • Total PDF chunks: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            return []
    
    def _process_text_files(self, text_files: List[str]) -> List[Document]:
        """Process text files and return documents."""
        logger.info(f"📝 Converting {len(text_files)} text files...")
        start_time = datetime.now()
        
        try:
            text_docs = self.text_converter.run(sources=text_files) # type: ignore
            conversion_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Text conversion completed")
            logger.info(f"   • Conversion time: {conversion_time:.2f} seconds")
            logger.info(f"   • Documents extracted: {len(text_docs['documents'])}")
            
            # Process chunks
            all_chunks = []
            for doc in text_docs['documents']:
                chunks = self._create_smart_chunks(doc)
                all_chunks.extend(chunks)
                
            logger.info(f"   • Total text chunks: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ Text processing failed: {e}")
            return []
    
    def _process_markdown_files(self, markdown_files: List[str]) -> List[Document]:
        """Process markdown files and return documents."""
        logger.info(f"📑 Converting {len(markdown_files)} markdown files...")
        start_time = datetime.now()
        
        try:
            if self.markdown_supported and self.markdown_converter:
                # Use dedicated markdown converter
                markdown_docs = self.markdown_converter.run(sources=markdown_files) # type: ignore
                logger.info("✅ Markdown conversion completed using MarkdownToDocument")
            else:
                # Fallback to text converter
                logger.info("ℹ️  Using TextFileToDocument as fallback for markdown files")
                markdown_docs = self.text_converter.run(sources=markdown_files) # type: ignore
                # Update metadata to reflect markdown type
                for doc in markdown_docs['documents']:
                    doc.meta['source_type'] = 'markdown'
                    doc.meta['original_format'] = 'markdown'
                    doc.meta['processed_as'] = 'text'
            
            conversion_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Markdown conversion completed")
            logger.info(f"   • Conversion time: {conversion_time:.2f} seconds")
            logger.info(f"   • Documents extracted: {len(markdown_docs['documents'])}")
            
            # Process chunks
            all_chunks = []
            for doc in markdown_docs['documents']:
                # Ensure source_type is set for markdown docs
                if 'source_type' not in doc.meta:
                    doc.meta['source_type'] = 'markdown'
                chunks = self._create_smart_chunks(doc)
                all_chunks.extend(chunks)
                
            logger.info(f"   • Total markdown chunks: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ Markdown processing failed: {e}")
            return []