"""
RAG factory.
Provides dependency injection and component wiring for RAG system.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.core.logging import get_logger
from src.rag.config import RAGConfig
from src.rag.providers import (
    InMemoryRetrieverProvider,
    ElasticsearchRetrieverProvider,
    VectorDatabaseRetrieverProvider,
    FastEmbedRankerProvider,
    GeminiGeneratorProvider,
    RetrieverProvider,
    RankerProvider,
    GeneratorProvider
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
from src.rag.utils import PromptTemplateManager

logger = get_logger(__name__)


class RAGFactory:
    """
    Factory for creating and wiring RAG components with dependency injection.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG factory.
        
        Args:
            config: RAG configuration (if None, loads from default path)
        """
        if config is None:
            config_path = Path("config/rag.yml")
            if config_path.exists():
                config = RAGConfig.from_yaml(config_path)
            else:
                logger.warning("No config file found, using defaults")
                config = RAGConfig()
        
        self.config = config
        
        # Component cache
        self._document_store = None
        self._retriever = None
        self._ranker = None
        self._generator = None
        self._prompt_manager = None
        
        logger.info("Initialized RAGFactory")
    
    def create_document_store(self) -> Any:
        """
        Create document store based on configuration.
        
        Returns:
            Document store instance
        """
        if self._document_store is not None:
            return self._document_store
        
        provider = self.config.document_store.provider
        
        if provider == "inmemory":
            logger.info("Creating InMemoryDocumentStore")
            self._document_store = InMemoryDocumentStore()
            
        elif provider == "elasticsearch":
            logger.info("Creating ElasticsearchDocumentStore")
            try:
                from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
                
                es_config = self.config.document_store.elasticsearch
                self._document_store = ElasticsearchDocumentStore(
                    hosts=es_config.hosts,
                    index=es_config.index
                )
            except ImportError:
                logger.error("Elasticsearch not available, falling back to InMemory")
                self._document_store = InMemoryDocumentStore()
                
        elif provider == "vectordb":
            logger.info("Creating VectorDB document store")
            from src.vector_database import VectorDatabase, VectorDatabaseConfig
            
            # Load VectorDB config
            vectordb_config_path = Path("config/vectordb.yml")
            if vectordb_config_path.exists():
                vectordb_config = VectorDatabaseConfig.from_yaml(vectordb_config_path)
            else:
                # Create default config
                vectordb_config = VectorDatabaseConfig()
            
            self._document_store = VectorDatabase(vectordb_config)
        
        else:
            raise ValueError(f"Unknown document store provider: {provider}")
        
        return self._document_store
    
    def create_retriever(self) -> RetrieverProvider:
        """
        Create retriever based on configuration.
        
        Returns:
            Retriever provider
        """
        if self._retriever is not None:
            return self._retriever
        
        provider = self.config.retrieval.provider
        top_k = self.config.retrieval.top_k
        doc_store = self.create_document_store()
        
        if provider == "inmemory_bm25":
            logger.info("Creating InMemoryRetrieverProvider")
            self._retriever = InMemoryRetrieverProvider(doc_store, top_k=top_k)
            
        elif provider == "elasticsearch_bm25":
            logger.info("Creating ElasticsearchRetrieverProvider")
            fuzziness = self.config.retrieval.bm25.fuzziness
            self._retriever = ElasticsearchRetrieverProvider(
                doc_store,
                top_k=top_k,
                fuzziness=fuzziness
            )
            
        elif provider == "vectordb":
            logger.info("Creating VectorDatabaseRetrieverProvider")
            # Need embedding model - use embeddings module
            from src.embeddings import EmbeddingGenerator
            
            embedding_model = EmbeddingGenerator()
            self._retriever = VectorDatabaseRetrieverProvider(
                doc_store,
                embedding_model,
                top_k=top_k
            )
        
        else:
            raise ValueError(f"Unknown retrieval provider: {provider}")
        
        return self._retriever
    
    def create_ranker(self) -> Optional[RankerProvider]:
        """
        Create ranker based on configuration.
        
        Returns:
            Ranker provider or None if disabled
        """
        if not self.config.ranking.enabled:
            logger.info("Ranking disabled")
            return None
        
        if self._ranker is not None:
            return self._ranker
        
        provider = self.config.ranking.provider
        top_k = self.config.ranking.top_k
        
        if provider == "fastembed":
            logger.info("Creating FastEmbedRankerProvider")
            ranker_config = self.config.ranking.fastembed
            self._ranker = FastEmbedRankerProvider(
                model_name=ranker_config.model_name,
                top_k=top_k,
                batch_size=ranker_config.batch_size
            )
        
        else:
            logger.warning(f"Unknown ranker provider: {provider}, using FastEmbed")
            self._ranker = FastEmbedRankerProvider(top_k=top_k)
        
        return self._ranker
    
    def create_generator(self) -> GeneratorProvider:
        """
        Create generator based on configuration.
        
        Returns:
            Generator provider
        """
        if self._generator is not None:
            return self._generator
        
        provider = self.config.generation.provider
        
        if provider == "gemini":
            logger.info("Creating GeminiGeneratorProvider")
            gemini_config = self.config.generation.gemini
            
            self._generator = GeminiGeneratorProvider(
                model=gemini_config.model,
                api_key_env=gemini_config.api_key_env,
                fallback_models=gemini_config.fallback_models,
                temperature=gemini_config.temperature,
                max_tokens=gemini_config.max_tokens,
                top_p=gemini_config.top_p
            )
        
        else:
            raise ValueError(f"Unknown generation provider: {provider}")
        
        return self._generator
    
    def create_prompt_manager(self) -> PromptTemplateManager:
        """
        Create prompt template manager.
        
        Returns:
            PromptTemplateManager instance
        """
        if self._prompt_manager is not None:
            return self._prompt_manager
        
        prompts_config = self.config.prompts
        
        self._prompt_manager = PromptTemplateManager(
            system_prompt_file=prompts_config.system_prompt_file,
            custom_prompts_file=prompts_config.custom_prompts_file,
            template_engine=prompts_config.template_engine
        )
        
        logger.info("Created PromptTemplateManager")
        return self._prompt_manager
    
    def create_services(self) -> Dict[str, Any]:
        """
        Create all RAG services with proper dependency injection.
        
        Returns:
            Dictionary of service instances
        """
        logger.info("Creating RAG services")
        
        # Create providers
        doc_store = self.create_document_store()
        retriever = self.create_retriever()
        ranker = self.create_ranker()
        generator = self.create_generator()
        
        # Create services
        ingestion_service = DocumentIngestionService(doc_store)
        retrieval_service = RetrievalService(retriever)
        ranking_service = RankingService(ranker, enabled=self.config.ranking.enabled)
        generation_service = GenerationService(generator)
        
        context_builder = ContextBuilderService(self.config.context)
        citation_service = CitationService(self.config.citation)
        
        search_service = SearchService(retrieval_service, ranking_service)
        
        # Warm up if configured
        if self.config.performance.warm_up_on_init:
            logger.info("Warming up components")
            retrieval_service.warm_up()
            ranking_service.warm_up()
            generation_service.warm_up()
        
        services = {
            'ingestion': ingestion_service,
            'retrieval': retrieval_service,
            'ranking': ranking_service,
            'generation': generation_service,
            'context_builder': context_builder,
            'citation': citation_service,
            'search': search_service
        }
        
        logger.info("All services created successfully")
        return services
    
    def create_pipeline_components(self) -> Dict[str, Any]:
        """
        Create pipeline builder and orchestrator.
        
        Returns:
            Dictionary with pipeline components
        """
        services = self.create_services()
        
        pipeline_builder = PipelineBuilder()
        
        pipeline_orchestrator = PipelineOrchestrator(
            context_builder=services['context_builder'],
            citation_service=services['citation'],
            generation_service=services['generation']
        )
        
        return {
            'builder': pipeline_builder,
            'orchestrator': pipeline_orchestrator,
            'services': services
        }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGFactory":
        """
        Create factory from YAML config file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            RAGFactory instance
        """
        config = RAGConfig.from_yaml(Path(config_path))
        return cls(config)
