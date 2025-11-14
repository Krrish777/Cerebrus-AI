"""
Embedding Generator for Cerebrus AI

This module provides embedding generation capabilities using Sentence Transformers
through Haystack components. It supports both document embedding for indexing
and query embedding for retrieval.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from haystack import Document # type: ignore
    from haystack.components.embedders import (
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder
    )
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    # Fallback Document class
    class Document:
        def __init__(self, content: str, meta: Dict[str, Any] = None): # type: ignore
            self.content = content
            self.meta = meta or {}
            self.embedding = None

# Import CustomLogger
from src.core.logging import CustomLogger

# Initialize logger
custom_logger = CustomLogger()
logger = custom_logger.get_logger(__name__)


@dataclass
class EmbeddedDocument:
    """Document with its embedding vector and metadata"""
    document: Document
    embedding: np.ndarray
    embedding_model: str
    embedding_dimension: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            'content': self.document.content,
            'meta': self.document.meta,
            'embedding': self.embedding.tolist(),
            'embedding_model': self.embedding_model,
            'embedding_dimension': self.embedding_dimension
        }


class EmbeddingGenerator:
    """
    Advanced embedding generator using Sentence Transformers via Haystack.
    
    This class provides a high-level interface for generating embeddings from
    both documents and queries using state-of-the-art transformer models.
    
    Features:
    - Document embedding for indexing pipelines
    - Query embedding for retrieval
    - Batch processing capabilities
    - Multiple model support
    - Metadata embedding
    - Error handling and logging
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        prefix: Optional[str] = None,
        meta_fields_to_embed: Optional[List[str]] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cpu', 'cuda', etc.)
            normalize_embeddings: Whether to normalize embedding vectors
            batch_size: Batch size for processing multiple documents
            prefix: Prefix to add to text before embedding (model-specific)
            meta_fields_to_embed: Metadata fields to include in embeddings
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.prefix = prefix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        
        # Initialize components
        self.document_embedder = None
        self.text_embedder = None
        self.embedding_dimension = None
        
        self._initialize_embedders()
    
    def _initialize_embedders(self):
        """Initialize Haystack embedding components"""
        if not HAYSTACK_AVAILABLE:
            raise ImportError(
                "Haystack is required for embedding generation. "
                "Install it with: pip install haystack-ai"
            )
        
        try:
            logger.info(f"🚀 Initializing embedding model: {self.model_name}") # type: ignore # type: ignore
            
            # Initialize document embedder
            embedder_kwargs = {
                'model': self.model_name,
                'normalize_embeddings': self.normalize_embeddings,
                'batch_size': self.batch_size
            }
            
            if self.device:
                embedder_kwargs['device'] = self.device
            if self.prefix:
                embedder_kwargs['prefix'] = self.prefix
            if self.meta_fields_to_embed:
                embedder_kwargs['meta_fields_to_embed'] = self.meta_fields_to_embed
            
            self.document_embedder = SentenceTransformersDocumentEmbedder(**embedder_kwargs) # type: ignore
            
            # Initialize text embedder (for queries)
            text_kwargs = {
                'model': self.model_name,
                'normalize_embeddings': self.normalize_embeddings
            }
            if self.device:
                text_kwargs['device'] = self.device
            if self.prefix:
                text_kwargs['prefix'] = self.prefix
            
            self.text_embedder = SentenceTransformersTextEmbedder(**text_kwargs) # type: ignore
            
            # Warm up the models
            logger.info("🔥 Warming up embedding models...") # type: ignore
            self.document_embedder.warm_up()
            self.text_embedder.warm_up()
            
            # Get embedding dimension
            test_result = self.text_embedder.run("test")
            self.embedding_dimension = len(test_result['embedding'])
            
            logger.info(f"✅ Embedding models initialized successfully") # type: ignore
            logger.info(f"📊 Model: {self.model_name}") # type: ignore
            logger.info(f"📏 Embedding dimension: {self.embedding_dimension}") # type: ignore
            logger.info(f"🖥️ Device: {self.device or 'auto'}") # type: ignore
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding models: {e}") # type: ignore
            raise
    
    def embed_documents(
        self, 
        documents: List[Document]
    ) -> List[EmbeddedDocument]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Haystack Document objects
            
        Returns:
            List of EmbeddedDocument objects with embeddings
        """
        if not documents:
            logger.warning("⚠️ No documents provided for embedding") # type: ignore
            return []
        
        logger.info(f"🔗 Generating embeddings for {len(documents)} documents") # type: ignore
        
        try:
            # Generate embeddings using Haystack
            result = self.document_embedder.run(documents) # type: ignore
            embedded_docs = result['documents']
            
            # Convert to EmbeddedDocument objects
            embedded_documents = []
            for doc in embedded_docs:
                if hasattr(doc, 'embedding') and doc.embedding:
                    embedding_array = np.array(doc.embedding, dtype=np.float32)
                    
                    embedded_doc = EmbeddedDocument(
                        document=doc,
                        embedding=embedding_array,
                        embedding_model=self.model_name,
                        embedding_dimension=self.embedding_dimension # type: ignore
                    )
                    embedded_documents.append(embedded_doc)
                else:
                    logger.warning(f"⚠️ Document has no embedding: {doc.content[:50]}...") # type: ignore
            
            logger.info(f"✅ Successfully generated {len(embedded_documents)} embeddings") # type: ignore
            return embedded_documents
            
        except Exception as e:
            logger.error(f"❌ Error generating document embeddings: {e}") # type: ignore
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Numpy array with query embedding
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")
        
        logger.info(f"🔍 Generating query embedding for: '{query[:50]}{'...' if len(query) > 50 else ''}'") # type: ignore
        
        try:
            result = self.text_embedder.run(query) # type: ignore
            embedding = np.array(result['embedding'], dtype=np.float32)
            
            logger.info(f"✅ Generated query embedding: {embedding.shape}") # type: ignore
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating query embedding: {e}") # type: ignore
            raise
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text strings.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of numpy arrays with embeddings
        """
        if not texts:
            return []
        
        logger.info(f"📝 Generating embeddings for {len(texts)} texts") # type: ignore
        
        # Convert texts to documents
        documents = [Document(content=text) for text in texts]
        
        # Generate embeddings
        embedded_docs = self.embed_documents(documents)
        
        # Extract embeddings
        embeddings = [doc.embedding for doc in embedded_docs]
        
        logger.info(f"✅ Generated {len(embeddings)} text embeddings") # type: ignore
        return embeddings
    
    def batch_embed_documents(
        self,
        document_batches: List[List[Document]]
    ) -> List[List[EmbeddedDocument]]:
        """
        Process multiple batches of documents.
        
        Args:
            document_batches: List of document batches
            
        Returns:
            List of embedded document batches
        """
        logger.info(f"📦 Processing {len(document_batches)} document batches") # type: ignore
        
        all_embedded_batches = []
        
        for i, batch in enumerate(document_batches):
            logger.info(f"📦 Processing batch {i+1}/{len(document_batches)} ({len(batch)} documents)") # type: ignore
            
            try:
                embedded_batch = self.embed_documents(batch)
                all_embedded_batches.append(embedded_batch)
                
            except Exception as e:
                logger.error(f"❌ Error processing batch {i+1}: {e}") # type: ignore
                # Continue with other batches
                all_embedded_batches.append([])
        
        total_docs = sum(len(batch) for batch in all_embedded_batches)
        logger.info(f"✅ Batch processing complete: {total_docs} documents embedded") # type: ignore
        
        return all_embedded_batches
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'device': self.device,
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.batch_size,
            'prefix': self.prefix,
            'meta_fields_to_embed': self.meta_fields_to_embed
        }
    
    def create_documents_from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        Create Haystack Document objects from text strings.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of Haystack Document objects
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadatas must match")
        
        documents = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            meta = meta.copy() if meta else {}
            meta['doc_index'] = i
            
            doc = Document(content=text, meta=meta)
            documents.append(doc)
        
        logger.info(f"📄 Created {len(documents)} documents from texts") # type: ignore
        return documents


# Convenience functions for easy usage
def create_embedding_generator(
    model_name: str = "BAAI/bge-small-en-v1.5",
    **kwargs
) -> EmbeddingGenerator:
    """
    Create an EmbeddingGenerator instance with common defaults.
    
    Args:
        model_name: Sentence transformer model name
        **kwargs: Additional parameters for EmbeddingGenerator
        
    Returns:
        Initialized EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model_name=model_name, **kwargs)


def embed_documents_simple(
    texts: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> List[np.ndarray]:
    """
    Simple function to embed a list of texts.
    
    Args:
        texts: List of text strings
        model_name: Model to use for embedding
        
    Returns:
        List of embedding vectors
    """
    embedder = create_embedding_generator(model_name)
    embeddings = embedder.embed_texts(texts)
    return embeddings


def embed_query_simple(
    query: str,
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> np.ndarray:
    """
    Simple function to embed a query string.
    
    Args:
        query: Query string
        model_name: Model to use for embedding
        
    Returns:
        Query embedding vector
    """
    embedder = create_embedding_generator(model_name)
    return embedder.embed_query(query)


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Embedding Generator")
    print("=" * 50)
    
    try:
        # Initialize embedding generator
        print("🚀 Creating embedding generator...")
        embedder = create_embedding_generator(
            model_name="BAAI/bge-small-en-v1.5",
            batch_size=16
        )
        
        # Test with sample texts
        sample_texts = [
            "Artificial intelligence is transforming the world.",
            "Machine learning enables computers to learn from data.",
            "Natural language processing helps computers understand text.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        print(f"\n📝 Creating documents from {len(sample_texts)} sample texts...")
        documents = embedder.create_documents_from_texts(
            texts=sample_texts,
            metadatas=[
                {"topic": "AI", "category": "overview"},
                {"topic": "ML", "category": "definition"},
                {"topic": "NLP", "category": "application"},
                {"topic": "DL", "category": "architecture"}
            ]
        )
        
        print(f"🔗 Generating document embeddings...")
        embedded_docs = embedder.embed_documents(documents)
        
        print(f"📊 Results:")
        for i, emb_doc in enumerate(embedded_docs):
            print(f"   📄 Document {i+1}:")
            print(f"      📝 Content: {emb_doc.document.content[:50]}...")
            print(f"      📏 Embedding shape: {emb_doc.embedding.shape}")
            print(f"      🏷️ Topic: {emb_doc.document.meta.get('topic', 'N/A')}")
        
        # Test query embedding
        query = "What is machine learning?"
        print(f"\n🔍 Generating query embedding for: '{query}'")
        query_embedding = embedder.embed_query(query)
        print(f"📏 Query embedding shape: {query_embedding.shape}")
        
        # Test similarity (simple dot product)
        print(f"\n🎯 Computing similarities:")
        for i, emb_doc in enumerate(embedded_docs):
            similarity = np.dot(query_embedding, emb_doc.embedding)
            print(f"   📄 Document {i+1}: {similarity:.4f}")
        
        # Show model info
        print(f"\n📋 Model Information:")
        info = embedder.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ Embedding generator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure to install Haystack: pip install haystack-ai")
