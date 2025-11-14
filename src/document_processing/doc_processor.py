import hashlib
import logging
from datetime import datetime
from core.logging import CustomLogger
from haystack import Pipeline, component, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from typing import List, Dict, Any, Optional, Union

# Configure logging with type safety
try:
    from core.logging import CustomLogger
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

class PDFProcessor:
    """
    Custom PDF Processor that converts, cleans, splits, and writes documents
    with comprehensive logging at every step.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        logger.info("=" * 60)
        logger.info("INITIALIZING PDF PROCESSOR")
        logger.info("=" * 60)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"🔧 Configuration Parameters:")
        logger.info(f"   • Chunk Size: {chunk_size} characters")
        logger.info(f"   • Chunk Overlap: {chunk_overlap} characters")
        logger.info(f"   • Overlap Percentage: {(chunk_overlap/chunk_size)*100:.1f}%")
        
        logger.debug("📦 Initializing PyPDFToDocument converter...")
        try:
            self.pdf_converter = PyPDFToDocument()
            logger.info("✅ PyPDFToDocument converter initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PyPDFToDocument: {e}")
            raise
        
        # Initialize logger - use a shared instance to avoid multiple log files
        self._logger_instance = CustomLogger()
        self.logger = self._logger_instance.get_logger(f"{__name__}.PDFProcessor")  # type: ignore
        
        logger.info("🚀 PDF Processor initialization completed successfully")
        logger.info("-" * 60)
        
    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]) -> Dict[str, List[Document]]:
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING PDF PROCESSING PIPELINE")
        logger.info("=" * 80)
        
        logger.info(f"📋 INPUT ANALYSIS:")
        logger.info(f"   • Number of sources: {len(sources)}")
        logger.info(f"   • Source files: {sources}")
        
        for i, source in enumerate(sources, 1):
            logger.info(f"   • Source {i}: '{source}'")
            logger.info(f"     - Type: {type(source).__name__}")
            logger.info(f"     - Length: {len(source)} characters")
        
        if not sources:
            logger.warning("⚠️  WARNING: No sources provided for PDF processing")
            logger.info("📤 RETURNING: Empty documents list")
            return {'documents': []}
        
        logger.info("\n📖 STEP 1: PDF CONVERSION")
        logger.info("-" * 40)
        logger.debug("🔧 Calling PyPDFToDocument.run()...")
        
        try:
            start_time = datetime.now()
            pdf_docs = self.pdf_converter.run(sources=sources)  # type: ignore
            conversion_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ PDF conversion completed successfully")
            logger.info(f"   • Conversion time: {conversion_time:.2f} seconds")
            logger.info(f"   • Raw documents extracted: {len(pdf_docs['documents'])}")
            
            # Detailed analysis of each converted document
            for i, doc in enumerate(pdf_docs['documents'], 1):
                doc_name = doc.meta.get('name', 'unknown')
                page_num = doc.meta.get('page_number', 'unknown')
                content_length = len(doc.content) if doc.content else 0
                
                logger.info(f"   📄 Document {i}:")
                logger.info(f"      - Name: {doc_name}")
                logger.info(f"      - Page: {page_num}")
                logger.info(f"      - Content length: {content_length:,} characters")
                logger.info(f"      - Metadata keys: {list(doc.meta.keys())}")
                
                if content_length > 0:
                    preview = (doc.content[:100] + "...") if len(doc.content) > 100 else doc.content
                    logger.debug(f"      - Content preview: '{preview}'")
                else:
                    logger.warning(f"      ⚠️  Empty content for document {i}")
        
        except Exception as e:
            logger.error(f"❌ PDF conversion failed: {e}")
            logger.error(f"   • Error type: {type(e).__name__}")
            raise
        
        logger.info("\n🔄 STEP 2: SMART CHUNKING PROCESS")
        logger.info("-" * 40)
        
        all_chunks = []
        total_input_chars = 0
        total_chunks_created = 0
        
        for doc_index, pdf_doc in enumerate(pdf_docs['documents'], 1):
            doc_name = pdf_doc.meta.get('name', 'unknown')
            page_num = pdf_doc.meta.get('page_number', 'unknown')
            content_length = len(pdf_doc.content) if pdf_doc.content else 0
            total_input_chars += content_length
            
            logger.info(f"\n📄 PROCESSING DOCUMENT {doc_index}/{len(pdf_docs['documents'])}")
            logger.info(f"   • Document: {doc_name}")
            logger.info(f"   • Page: {page_num}")
            logger.info(f"   • Content length: {content_length:,} characters")
            
            chunk_start_time = datetime.now()
            page_chunks = self._create_smart_chunks(pdf_doc)
            chunk_time = (datetime.now() - chunk_start_time).total_seconds()
            
            all_chunks.extend(page_chunks)
            total_chunks_created += len(page_chunks)
            
            logger.info(f"✅ Document {doc_index} processing completed:")
            logger.info(f"   • Chunks created: {len(page_chunks)}")
            logger.info(f"   • Processing time: {chunk_time:.2f} seconds")
            logger.info(f"   • Chunks/second: {len(page_chunks)/chunk_time:.1f}" if chunk_time > 0 else "   • Processing time: <0.01 seconds")
            logger.info(f"   • Cumulative total chunks: {len(all_chunks)}")
            
            if len(page_chunks) > 0:
                chunk_sizes = [len(chunk.content or "") for chunk in page_chunks]  # type: ignore
                logger.info(f"   • Chunk sizes - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.1f}")
        
        # Final summary
        processing_time = datetime.now()
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("=" * 80)
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
    
    def _create_smart_chunks(self, pdf_doc: Document) -> List[Document]:
        """Create smart chunks from PDF document with comprehensive logging."""
        
        doc_name = pdf_doc.meta.get('name', 'unknown')
        page_number = pdf_doc.meta.get('page_number', 1)
        
        logger.info(f"\n🔍 STARTING SMART CHUNKING")
        logger.info(f"   • Document: {doc_name}")
        logger.info(f"   • Page: {page_number}")
        
        # Safely get text content
        text = pdf_doc.content or ""
        source_file = pdf_doc.meta.get('name', 'unknown')
        
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
                chunk_id = f"pdf_{chunk_index}_page{page_number}_{content_hash}"
                
                logger.debug(f"   🔑 Generated chunk ID: {chunk_id}")
                
                # Create comprehensive metadata
                chunk_metadata = {
                    # Source information
                    'source_file': source_file,
                    'source_type': 'pdf',
                    'page_number': page_number,
                    
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
                    'processor_version': '2.0',
                    'chunk_overlap': self.chunk_overlap,
                    'target_chunk_size': self.chunk_size,
                    
                    # Citation information
                    'citation': {
                        'source_file': source_file,
                        'type': 'pdf',
                        'page_number': page_number,
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
                final_metadata = pdf_doc.meta.copy()
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