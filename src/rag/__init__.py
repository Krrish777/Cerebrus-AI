"""
RAG (Retrieval Augmented Generation) System.

Modular, production-ready RAG architecture with:
- Multiple provider support (InMemory, Elasticsearch, VectorDB)
- Configurable retrieval, ranking, and generation
- Pipeline-based orchestration
- Comprehensive citation tracking
- Type-safe configuration with Pydantic

Example Usage:
    from src.rag import RAGFactory
    
    # Create factory
    factory = RAGFactory.from_yaml("config/rag.yml")
    
    # Create components
    components = factory.create_pipeline_components()
    services = components['services']
    builder = components['builder']
    orchestrator = components['orchestrator']
    
    # Add documents
    services['ingestion'].add_documents(documents)
    
    # Build pipeline
    retriever = factory.create_retriever()
    ranker = factory.create_ranker()
    pipeline = builder.build_rag_pipeline(retriever, ranker)
    
    # Generate response
    result = orchestrator.execute_rag(pipeline, "What is Python?")
    print(result.response)
    print(result.get_citation_summary())
"""

from src.rag.config import RAGConfig
from src.rag.models import RAGResult, SearchResult
from src.rag.factories import RAGFactory
from src.rag.providers import (
    InMemoryRetrieverProvider,
    ElasticsearchRetrieverProvider,
    VectorDatabaseRetrieverProvider,
    FastEmbedRankerProvider,
    GeminiGeneratorProvider
)
from src.rag.services import (
    DocumentIngestionService,
    RetrievalService,
    RankingService,
    GenerationService,
    ContextBuilderService,
    CitationService,
    SearchService
)
from src.rag.pipeline import PipelineBuilder, PipelineOrchestrator
from src.rag.utils import PromptTemplateManager, ResultFormatter

__all__ = [
    # Main entry point
    "RAGFactory",
    
    # Configuration
    "RAGConfig",
    
    # Models
    "RAGResult",
    "SearchResult",
    
    # Providers
    "InMemoryRetrieverProvider",
    "ElasticsearchRetrieverProvider",
    "VectorDatabaseRetrieverProvider",
    "FastEmbedRankerProvider",
    "GeminiGeneratorProvider",
    
    # Services
    "DocumentIngestionService",
    "RetrievalService",
    "RankingService",
    "GenerationService",
    "ContextBuilderService",
    "CitationService",
    "SearchService",
    
    # Pipeline
    "PipelineBuilder",
    "PipelineOrchestrator",
    
    # Utilities
    "PromptTemplateManager",
    "ResultFormatter",
]
