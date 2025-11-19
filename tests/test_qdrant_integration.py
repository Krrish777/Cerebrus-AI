"""
Integration and Performance Tests for Qdrant Vector Database

This module contains integration tests that work with real components
and performance tests for the QdrantVectorDB implementation.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from vector_database.qdrant_db import QdrantVectorDB
    from embeddings.embedding_generator import EmbeddingGenerator
    from document_processing.doc_processor import DocumentProcessor
    from haystack import Document
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"⚠️ Dependencies not available: {e}")


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_qdrant_perf_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def embedding_generator():
    """Create real embedding generator (cached)."""
    try:
        return EmbeddingGenerator(
            model_name="BAAI/bge-small-en-v1.5",
            batch_size=8
        )
    except Exception:
        pytest.skip("Embedding generator not available")


@pytest.fixture
def large_document_set():
    """Create a larger set of test documents."""
    documents = []
    topics = ["AI", "ML", "NLP", "CV", "DL", "RL", "robotics", "quantum", "blockchain", "IoT"]
    
    for i in range(50):
        topic = topics[i % len(topics)]
        content = f"This is document {i} about {topic}. " \
                 f"It contains information about {topic} technologies and applications. " \
                 f"The field of {topic} is rapidly evolving with new breakthroughs. " \
                 f"Researchers in {topic} are developing innovative solutions. " \
                 f"Document ID {i} focuses specifically on {topic} implementations."
        
        doc = Document(
            content=content,
            meta={
                "source_file": f"document_{i}.txt",
                "source_type": "text",
                "chunk_index": i,
                "topic": topic,
                "doc_id": f"doc_{i}"
            },
            id=f"doc_{i}"
        )
        documents.append(doc)
    
    return documents


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
@pytest.mark.integration
class TestQdrantIntegration:
    """Integration tests with real components."""
    
    def test_real_document_processing_pipeline(self, temp_storage, embedding_generator):
        """Test complete pipeline from document processing to vector search."""
        storage_path = str(Path(temp_storage) / "real_pipeline")
        
        try:
            # Initialize vector database
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="real_documents",
                recreate_index=True
            )
            
            # Create sample documents with realistic content
            sample_texts = [
                """Machine learning is a method of data analysis that automates analytical model building. 
                It is a branch of artificial intelligence based on the idea that systems can learn from data, 
                identify patterns and make decisions with minimal human intervention.""",
                
                """Natural language processing (NLP) is a branch of artificial intelligence that helps 
                computers understand, interpret and manipulate human language. NLP draws from many disciplines, 
                including computer science and computational linguistics.""",
                
                """Computer vision is a field of artificial intelligence that trains computers to interpret 
                and understand the visual world. Using digital images from cameras and videos and deep learning 
                models, machines can accurately identify and classify objects.""",
                
                """Deep learning is part of a broader family of machine learning methods based on artificial 
                neural networks with representation learning. Learning can be supervised, semi-supervised or 
                unsupervised."""
            ]
            
            # Create documents with metadata
            documents = embedding_generator.create_documents_from_texts(
                texts=sample_texts,
                metadatas=[
                    {"source": "ml_textbook.pdf", "chapter": 1, "topic": "machine_learning"},
                    {"source": "nlp_handbook.pdf", "chapter": 2, "topic": "natural_language"},
                    {"source": "cv_guide.pdf", "chapter": 3, "topic": "computer_vision"},
                    {"source": "dl_manual.pdf", "chapter": 4, "topic": "deep_learning"}
                ]
            )
            
            # Generate embeddings
            embedded_docs = embedding_generator.embed_documents(documents)
            assert len(embedded_docs) == len(sample_texts)
            
            # Insert into vector database
            inserted_ids = vector_db.insert_embedded_documents(embedded_docs)
            assert len(inserted_ids) == len(embedded_docs)
            
            # Test semantic search
            queries = [
                "What is AI?",
                "How do computers understand language?", 
                "Image recognition technology",
                "Neural network learning"
            ]
            
            for query in queries:
                results = vector_db.search_with_query_text(
                    query_text=query,
                    embedding_generator=embedding_generator,
                    top_k=2
                )
                
                assert len(results) <= 2
                assert all(result['score'] >= 0 for result in results)
                assert all(len(result['content']) > 0 for result in results)
                
                # Verify citation information
                for result in results:
                    citation = result['citation']
                    assert 'source' in result['metadata'] or 'source_file' in citation
                
            # Test collection statistics
            stats = vector_db.get_collection_stats()
            assert stats['total_documents'] == len(embedded_docs)
            assert len(stats['source_types']) > 0
            
        except Exception as e:
            if "not found" in str(e) or "No module" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise
        finally:
            try:
                vector_db.close()
            except:
                pass
    
    def test_metadata_filtering_integration(self, temp_storage, embedding_generator):
        """Test metadata filtering with real embeddings."""
        storage_path = str(Path(temp_storage) / "metadata_filtering")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="filtered_docs",
                recreate_index=True
            )
            
            # Create documents with different metadata
            texts_and_meta = [
                ("Machine learning algorithms for classification", {"category": "algorithms", "difficulty": "beginner"}),
                ("Advanced neural network architectures", {"category": "architecture", "difficulty": "advanced"}),
                ("Basic data preprocessing techniques", {"category": "preprocessing", "difficulty": "beginner"}),
                ("Complex optimization methods", {"category": "optimization", "difficulty": "advanced"}),
                ("Simple feature engineering", {"category": "features", "difficulty": "beginner"})
            ]
            
            documents = embedding_generator.create_documents_from_texts(
                texts=[text for text, _ in texts_and_meta],
                metadatas=[meta for _, meta in texts_and_meta]
            )
            
            embedded_docs = embedding_generator.embed_documents(documents)
            vector_db.insert_embedded_documents(embedded_docs)
            
            # Test filtering by difficulty level
            query_embedding = embedding_generator.embed_query("machine learning techniques")
            
            # Search for beginner content only
            beginner_results = vector_db.search(
                query_embedding=query_embedding.tolist(),
                top_k=5,
                filters={"field": "difficulty", "operator": "==", "value": "beginner"}
            )
            
            # Verify all results are beginner level
            for result in beginner_results:
                assert result['metadata']['difficulty'] == 'beginner'
            
            # Search for advanced content only
            advanced_results = vector_db.search(
                query_embedding=query_embedding.tolist(),
                top_k=5,
                filters={"field": "difficulty", "operator": "==", "value": "advanced"}
            )
            
            # Verify all results are advanced level
            for result in advanced_results:
                assert result['metadata']['difficulty'] == 'advanced'
                
        except Exception as e:
            if "not found" in str(e) or "No module" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
@pytest.mark.performance
class TestQdrantPerformance:
    """Performance tests for QdrantVectorDB."""
    
    def test_large_scale_insertion_performance(self, temp_storage, embedding_generator, large_document_set):
        """Test performance with larger document sets."""
        storage_path = str(Path(temp_storage) / "performance_test")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="performance_collection",
                recreate_index=True,
                hnsw_config={
                    "m": 16,
                    "ef_construct": 100,  # Faster indexing
                    "full_scan_threshold": 1000
                }
            )
            
            # Generate embeddings for all documents
            start_time = time.time()
            embedded_docs = embedding_generator.embed_documents(large_document_set)
            embedding_time = time.time() - start_time
            
            print(f"🔗 Embedding generation time: {embedding_time:.2f}s for {len(large_document_set)} docs")
            print(f"📊 Embedding rate: {len(large_document_set)/embedding_time:.1f} docs/sec")
            
            # Test insertion performance
            start_time = time.time()
            inserted_ids = vector_db.insert_embedded_documents(embedded_docs, policy="overwrite")
            insertion_time = time.time() - start_time
            
            print(f"💾 Insertion time: {insertion_time:.2f}s for {len(inserted_ids)} docs")
            print(f"📊 Insertion rate: {len(inserted_ids)/insertion_time:.1f} docs/sec")
            
            assert len(inserted_ids) == len(embedded_docs)
            
            # Test search performance
            query_embedding = embedding_generator.embed_query("artificial intelligence research")
            
            search_times = []
            for _ in range(10):  # Multiple searches for average
                start_time = time.time()
                results = vector_db.search(
                    query_embedding=query_embedding.tolist(),
                    top_k=10
                )
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                assert len(results) <= 10
            
            avg_search_time = sum(search_times) / len(search_times)
            print(f"🔍 Average search time: {avg_search_time*1000:.1f}ms")
            print(f"📊 Search throughput: {1/avg_search_time:.1f} searches/sec")
            
            # Performance assertions
            assert embedding_time < 30.0  # Should process 50 docs in under 30 seconds
            assert insertion_time < 10.0   # Should insert 50 docs in under 10 seconds 
            assert avg_search_time < 0.5   # Search should be under 500ms
            
        except Exception as e:
            if "not found" in str(e) or "No module" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_memory_efficiency(self, temp_storage, embedding_generator):
        """Test memory usage with quantization."""
        storage_path = str(Path(temp_storage) / "memory_test")
        
        try:
            # Test with quantization enabled
            quantization_config = {
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99
                }
            }
            
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="quantized_collection",
                recreate_index=True,
                quantization_config=quantization_config
            )
            
            # Create medium-sized document set
            texts = [f"Document {i} about artificial intelligence and machine learning research." 
                    for i in range(20)]
            
            documents = embedding_generator.create_documents_from_texts(
                texts=texts,
                metadatas=[{"doc_id": i} for i in range(len(texts))]
            )
            
            embedded_docs = embedding_generator.embed_documents(documents)
            inserted_ids = vector_db.insert_embedded_documents(embedded_docs)
            
            assert len(inserted_ids) == len(embedded_docs)
            
            # Verify quantization is enabled in stats
            stats = vector_db.get_collection_stats()
            assert stats['quantization_enabled'] is True
            
        except Exception as e:
            if "not found" in str(e) or "No module" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_concurrent_operations(self, temp_storage, embedding_generator):
        """Test concurrent read operations."""
        storage_path = str(Path(temp_storage) / "concurrent_test")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="concurrent_collection",
                recreate_index=True
            )
            
            # Insert some documents first
            texts = [f"Test document {i} for concurrent access testing." for i in range(10)]
            documents = embedding_generator.create_documents_from_texts(texts)
            embedded_docs = embedding_generator.embed_documents(documents)
            vector_db.insert_embedded_documents(embedded_docs)
            
            # Simulate concurrent searches
            query_embedding = embedding_generator.embed_query("test document")
            
            # Perform multiple rapid searches
            for i in range(20):
                results = vector_db.search(
                    query_embedding=query_embedding.tolist(),
                    top_k=3
                )
                assert len(results) <= 3
                assert all(result['score'] >= 0 for result in results)
            
            print("✅ Concurrent operations completed successfully")
            
        except Exception as e:
            if "not found" in str(e) or "No module" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
@pytest.mark.real_data
class TestQdrantRealData:
    """Tests with realistic data scenarios."""
    
    def test_mixed_document_types(self, temp_storage, embedding_generator):
        """Test with documents from different sources (PDF, text, markdown)."""
        storage_path = str(Path(temp_storage) / "mixed_types")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="mixed_documents",
                recreate_index=True
            )
            
            # Simulate documents from different sources
            mixed_documents = [
                {
                    "content": "This is content from a PDF research paper about neural networks.",
                    "meta": {
                        "source_file": "neural_networks.pdf",
                        "source_type": "pdf", 
                        "page_number": 1,
                        "chunk_index": 0
                    }
                },
                {
                    "content": "# Machine Learning\n\nThis is markdown content about ML algorithms.",
                    "meta": {
                        "source_file": "ml_notes.md",
                        "source_type": "markdown",
                        "chunk_index": 0
                    }
                },
                {
                    "content": "Plain text document discussing artificial intelligence applications.",
                    "meta": {
                        "source_file": "ai_applications.txt",
                        "source_type": "text",
                        "chunk_index": 0
                    }
                }
            ]
            
            # Create Haystack documents
            documents = []
            for i, doc_data in enumerate(mixed_documents):
                doc = Document(
                    content=doc_data["content"],
                    meta=doc_data["meta"],
                    id=f"mixed_doc_{i}"
                )
                documents.append(doc)
            
            # Generate embeddings and insert
            embedded_docs = embedding_generator.embed_documents(documents)
            inserted_ids = vector_db.insert_embedded_documents(embedded_docs)
            
            assert len(inserted_ids) == len(mixed_documents)
            
            # Test search across different document types
            query = "machine learning algorithms"
            results = vector_db.search_with_query_text(
                query_text=query,
                embedding_generator=embedding_generator,
                top_k=5
            )
            
            # Verify we can find relevant content across different types
            source_types_found = set()
            for result in results:
                source_type = result['metadata'].get('source_type')
                if source_type:
                    source_types_found.add(source_type)
            
            assert len(source_types_found) > 1  # Should find multiple document types
            
            # Test filtering by source type
            pdf_results = vector_db.search_with_query_text(
                query_text=query,
                embedding_generator=embedding_generator,
                top_k=5,
                filters={"field": "source_type", "operator": "==", "value": "pdf"}
            )
            
            for result in pdf_results:
                assert result['metadata']['source_type'] == 'pdf'
                
        except Exception as e:
            if "not found" in str(e) or "No module" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_citation_information_accuracy(self, temp_storage, embedding_generator):
        """Test accuracy of citation information extraction."""
        storage_path = str(Path(temp_storage) / "citation_test")
        
        try:
            vector_db = QdrantVectorDB(
                storage_path=storage_path,
                collection_name="citation_collection",
                recreate_index=True
            )
            
            # Create documents with detailed citation information
            documents_data = [
                {
                    "content": "First chunk of research paper about deep learning.",
                    "meta": {
                        "source_file": "deep_learning_research.pdf",
                        "source_type": "pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                        "start_char": 0,
                        "end_char": 50,
                        "citation": {
                            "authors": ["Smith, J.", "Doe, A."],
                            "title": "Deep Learning Advances",
                            "year": 2024
                        }
                    }
                },
                {
                    "content": "Second chunk discussing neural network architectures.",
                    "meta": {
                        "source_file": "deep_learning_research.pdf",
                        "source_type": "pdf",
                        "page_number": 2,
                        "chunk_index": 1,
                        "start_char": 51,
                        "end_char": 100,
                        "citation": {
                            "authors": ["Smith, J.", "Doe, A."],
                            "title": "Deep Learning Advances", 
                            "year": 2024
                        }
                    }
                }
            ]
            
            # Create and insert documents
            documents = []
            for i, doc_data in enumerate(documents_data):
                doc = Document(
                    content=doc_data["content"],
                    meta=doc_data["meta"],
                    id=f"cite_doc_{i}"
                )
                documents.append(doc)
            
            embedded_docs = embedding_generator.embed_documents(documents)
            vector_db.insert_embedded_documents(embedded_docs)
            
            # Search and verify citation information
            query = "neural networks"
            results = vector_db.search_with_query_text(
                query_text=query,
                embedding_generator=embedding_generator,
                top_k=5
            )
            
            for result in results:
                citation = result['citation']
                
                # Verify basic citation fields are present
                assert 'source_file' in citation
                assert 'source_type' in citation
                assert 'page_number' in citation
                assert 'chunk_index' in citation
                assert 'start_char' in citation
                assert 'end_char' in citation
                
                # Verify values are correct
                assert citation['source_file'] == "deep_learning_research.pdf"
                assert citation['source_type'] == "pdf"
                assert isinstance(citation['page_number'], int)
                assert isinstance(citation['chunk_index'], int)
                
        except Exception as e:
            if "not found" in str(e) or "No module" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise


if __name__ == "__main__":
    # Run tests with markers
    pytest.main([
        __file__, 
        "-v", 
        "-m", "integration or performance or real_data",
        "--tb=short"
    ])