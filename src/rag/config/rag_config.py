"""
Pydantic configuration models for RAG system.
Provides type-safe configuration with validation and YAML loading.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import yaml

from src.core.logging import get_logger

logger = get_logger(__name__)


class SystemConfig(BaseModel):
    """System-level configuration."""
    name: str = "Cerebrus RAG System"
    version: str = "2.0.0"
    environment: Literal["development", "production"] = "development"


class InMemoryConfig(BaseModel):
    """InMemory document store configuration."""
    embedding_similarity_function: Literal["cosine", "dot_product", "euclidean"] = "cosine"


class ElasticsearchConfig(BaseModel):
    """Elasticsearch document store configuration."""
    hosts: List[str] = Field(default_factory=lambda: ["http://localhost:9200"])
    index: str = "cerebrus_documents"
    timeout: int = 30
    verify_certs: bool = False


class VectorDBConfig(BaseModel):
    """VectorDB document store configuration."""
    provider: Literal["qdrant", "chromadb", "pinecone"] = "qdrant"
    collection_name: str = "cerebrus_rag"
    embedding_dim: int = 384


class DocumentStoreConfig(BaseModel):
    """Document store configuration with provider selection."""
    provider: Literal["inmemory", "elasticsearch", "vectordb"] = "inmemory"
    inmemory: InMemoryConfig = Field(default_factory=InMemoryConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)


class BM25Config(BaseModel):
    """BM25 retrieval configuration."""
    fuzziness: Literal["AUTO", "0", "1", "2"] = "AUTO"


class VectorDBRetrievalConfig(BaseModel):
    """VectorDB retrieval configuration."""
    similarity_metric: Literal["cosine", "dot_product", "euclidean"] = "cosine"
    filter_strategy: Literal["pre_filter", "post_filter"] = "pre_filter"


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    provider: Literal["inmemory_bm25", "elasticsearch_bm25", "vectordb"] = "inmemory_bm25"
    top_k: int = Field(default=20, ge=1, le=100)
    bm25: BM25Config = Field(default_factory=BM25Config)
    vectordb: VectorDBRetrievalConfig = Field(default_factory=VectorDBRetrievalConfig)


class FastEmbedRankingConfig(BaseModel):
    """FastEmbed ranker configuration."""
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    batch_size: int = Field(default=32, ge=1)


class CohereRankingConfig(BaseModel):
    """Cohere ranker configuration."""
    model: str = "rerank-english-v2.0"


class SentenceTransformersRankingConfig(BaseModel):
    """Sentence Transformers ranker configuration."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RankingConfig(BaseModel):
    """Ranking configuration."""
    enabled: bool = True
    provider: Literal["fastembed", "cohere", "sentencetransformers"] = "fastembed"
    top_k: int = Field(default=8, ge=1, le=50)
    fastembed: FastEmbedRankingConfig = Field(default_factory=FastEmbedRankingConfig)
    cohere: CohereRankingConfig = Field(default_factory=CohereRankingConfig)
    sentencetransformers: SentenceTransformersRankingConfig = Field(default_factory=SentenceTransformersRankingConfig)


class GeminiConfig(BaseModel):
    """Google Gemini generator configuration."""
    model: str = "gemini-2.0-flash"
    api_key_env: str = "GEMINI_API_KEY"
    fallback_models: List[str] = Field(default_factory=lambda: [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-pro"
    ])
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class OpenAIConfig(BaseModel):
    """OpenAI generator configuration."""
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class AnthropicConfig(BaseModel):
    """Anthropic generator configuration."""
    model: str = "claude-3-5-sonnet-20241022"
    api_key_env: str = "ANTHROPIC_API_KEY"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class GenerationConfig(BaseModel):
    """Generation configuration."""
    provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)


class ContextConfig(BaseModel):
    """Context building configuration."""
    max_documents: int = Field(default=8, ge=1, le=20)
    include_metadata: bool = True
    metadata_fields: List[str] = Field(default_factory=lambda: [
        "source_file", "page_number", "source_type", "timestamp"
    ])
    format: Literal["numbered", "markdown", "plain"] = "numbered"
    max_context_length: int = Field(default=4000, ge=100, le=32000)
    truncation_strategy: Literal["start", "middle", "end"] = "middle"


class CitationConfig(BaseModel):
    """Citation configuration."""
    enabled: bool = True
    style: Literal["numeric", "author_year", "footnote"] = "numeric"
    include_scores: bool = True
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    preview_length: int = Field(default=200, ge=50, le=500)


class PromptsConfig(BaseModel):
    """Prompts configuration."""
    system_prompt_file: str = "config/prompts/default_system_prompt.txt"
    custom_prompts_file: str = "config/prompts/custom_prompts.yml"
    template_engine: Literal["jinja2", "string"] = "jinja2"


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    warm_up_on_init: bool = True
    cache_embeddings: bool = True
    batch_processing: bool = True
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)
    request_timeout: int = Field(default=60, ge=10, le=300)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["structured", "plain"] = "structured"
    include_timestamps: bool = True
    log_performance_metrics: bool = True
    log_sources: bool = True


class RAGConfig(BaseModel):
    """Complete RAG system configuration."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    document_store: DocumentStoreConfig = Field(default_factory=DocumentStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    citation: CitationConfig = Field(default_factory=CitationConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @field_validator("system", "document_store", "retrieval", "ranking", "generation", mode="before")
    @classmethod
    def ensure_dict(cls, v):
        """Ensure nested configs are dicts for proper parsing."""
        if v is None:
            return {}
        return v
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "RAGConfig":
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                logger.warning(f"Empty config file at {config_path}, using defaults")
                return cls()
            
            config = cls(**config_data)
            logger.info(f"Loaded RAG configuration from {config_path}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return cls()
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_path}: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    @classmethod
    def from_yaml_path(cls, config_path: str) -> "RAGConfig":
        """Load configuration from YAML file path string."""
        return cls.from_yaml(Path(config_path))
    
    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    self.model_dump(exclude_none=True),
                    f,
                    default_flow_style=False,
                    sort_keys=False
                )
            
            logger.info(f"Saved RAG configuration to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {output_path}: {e}")
            raise
