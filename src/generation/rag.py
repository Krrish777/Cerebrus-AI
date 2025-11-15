"""
Advanced RAG system using Haystack components with Elasticsearch BM25, FastEmbed ranking, and Google Gemini.
Simplified implementation focusing on core functionality.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from haystack import Pipeline, Document
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

try:
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Enhanced result object for RAG generation with comprehensive citation tracking"""
    query: str
    response: str
    sources_used: List[Dict[str, Any]]
    retrieval_count: int
    ranking_count: int
    generation_tokens: Optional[int] = None
    
    def get_citation_summary(self) -> str:
        """Generate a formatted summary of all sources used in the response"""
        if not self.sources_used:
            return "No sources cited"
        
        source_summary = []
        for source in self.sources_used:
            source_info = f"• {source.get('source_file', 'Unknown')} ({source.get('source_type', 'unknown')})"
            if source.get('page_number'):
                source_info += f" - Page {source['page_number']}"
            if source.get('relevance_score'):
                source_info += f" [Score: {source['relevance_score']:.3f}]"
            source_summary.append(source_info)
        
        return "\n".join(source_summary)
    
    def get_performance_summary(self) -> str:
        """Get performance metrics summary"""
        return f"Retrieved: {self.retrieval_count} documents, Ranked: {self.ranking_count} documents"


class ElasticsearchRAGGenerator:
    """
    RAG system using Elasticsearch BM25 for retrieval, FastEmbed for ranking,
    and Google Gemini for response generation.
    """
    
    def __init__(
        self,
        elasticsearch_host: str = "localhost:9200",
        elasticsearch_index: str = "cerebrus_documents",
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        ranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        retrieval_top_k: int = 20,
        ranking_top_k: int = 8,
        fuzziness: str = "AUTO"
    ):
        """Initialize the RAG system."""
        self.elasticsearch_host = elasticsearch_host
        self.elasticsearch_index = elasticsearch_index
        self.retrieval_top_k = retrieval_top_k
        self.ranking_top_k = ranking_top_k
        self.fuzziness = fuzziness
        
        # Initialize document store - Force InMemory for development
        use_elasticsearch = False
        
        # For development purposes, skip Elasticsearch and use InMemory directly
        logger.info("Using InMemoryDocumentStore for development (Elasticsearch disabled)")
        self.document_store = InMemoryDocumentStore()
        
        # Uncomment below to try Elasticsearch (requires running ES instance)
        # try:
        #     if ELASTICSEARCH_AVAILABLE:
        #         # Try with proper URL format
        #         if not elasticsearch_host.startswith(('http://', 'https://')):
        #             elasticsearch_host = f"http://{elasticsearch_host}"
        #         
        #         self.document_store = ElasticsearchDocumentStore(
        #             hosts=[elasticsearch_host],
        #             index=elasticsearch_index
        #         )
        #         logger.info(f"Connected to Elasticsearch at {elasticsearch_host}")
        #         use_elasticsearch = True
        #     else:
        #         raise ImportError("Elasticsearch not available")
        # except Exception as e:
        #     logger.warning(f"Failed to connect to Elasticsearch: {e}")
        #     logger.info("Using InMemoryDocumentStore for development")
        #     self.document_store = InMemoryDocumentStore()
        #     use_elasticsearch = False
        
        # Initialize retriever - Always use InMemory for development
        self.retriever = InMemoryBM25Retriever(
            document_store=self.document_store,
            top_k=retrieval_top_k
        )
        
        # Uncomment below to use Elasticsearch retriever (requires ES)
        # if use_elasticsearch and ELASTICSEARCH_AVAILABLE:
        #     self.retriever = ElasticsearchBM25Retriever(
        #         document_store=self.document_store,
        #         fuzziness=fuzziness,
        #         top_k=retrieval_top_k
        #     )
        # else:
        #     self.retriever = InMemoryBM25Retriever(
        #         document_store=self.document_store,
        #         top_k=retrieval_top_k
        #     )
        
        # Initialize ranker
        self.ranker = FastembedRanker(
            model_name=ranker_model,
            top_k=ranking_top_k
        )
        
        # Initialize Google Gemini
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter.")
        
        # Use supported model names from Haystack documentation
        supported_models = [
            "gemini-2.0-flash",  # Latest default model
            "gemini-1.5-pro-latest",  # Fallback option
            "gemini-1.5-flash-latest",  # Alternative Flash model
            "gemini-pro"  # Final fallback
        ]
        
        # If user provided an old model name, update it
        if model_name == "gemini-1.5-flash":
            model_name = "gemini-2.0-flash"
            logger.info(f"Updated deprecated model name to {model_name}")
        
        # Try to initialize with the specified model, with fallbacks
        for attempt_model in [model_name] + [m for m in supported_models if m != model_name]:
            try:
                self.generator = GoogleGenAIChatGenerator(
                    model=attempt_model,
                    api_key=Secret.from_token(api_key)
                )
                logger.info(f"✅ Successfully initialized GoogleGenAIChatGenerator with model: {attempt_model}")
                break
            except Exception as model_error:
                logger.warning(f"Failed to initialize with {attempt_model}: {model_error}")
                if attempt_model == supported_models[-1]:  # Last model in list
                    raise ValueError(f"Failed to initialize with any supported Gemini model. Last error: {model_error}")
                continue
        
        # Initialize prompt builder
        self.chat_prompt_builder = ChatPromptBuilder()
        
        # Build the pipeline
        self._build_pipeline()
        
        # Warm up components with robust error handling
        self._warm_up_components()
        
        logger.info(f"ElasticsearchRAGGenerator initialized with {model_name}")
    
    def _build_pipeline(self):
        """Build the complete RAG pipeline"""
        self.pipeline = Pipeline()
        
        # Add components
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("ranker", self.ranker)
        self.pipeline.add_component("prompt_builder", self.chat_prompt_builder)
        self.pipeline.add_component("generator", self.generator)
        
        # Connect components
        self.pipeline.connect("retriever.documents", "ranker.documents")
        
        logger.info("RAG pipeline built successfully")
    
    def _warm_up_components(self):
        """Warm up all components with proper error handling"""
        try:
            # Warm up ranker (FastembedRanker)
            if hasattr(self.ranker, 'warm_up'):
                self.ranker.warm_up()
                logger.info("Ranker warmed up successfully")
                
            logger.info("All components warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Component warm-up failed: {e}")
            logger.info("System will continue with cold start - first queries may be slower")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the document store."""
        try:
            haystack_docs = []
            for doc_data in documents:
                # Handle different document formats
                if isinstance(doc_data, dict):
                    content = doc_data.get('content', '')
                    meta = doc_data.get('metadata', {})
                elif hasattr(doc_data, 'document'):
                    # EmbeddedDocument structure
                    content = doc_data.document.content
                    meta = doc_data.document.meta or {}
                elif hasattr(doc_data, 'content'):
                    # Direct document
                    content = doc_data.content
                    meta = getattr(doc_data, 'meta', {}) or getattr(doc_data, 'metadata', {})
                else:
                    continue
                
                haystack_doc = Document(content=content, meta=meta)
                haystack_docs.append(haystack_doc)
            
            self.document_store.write_documents(haystack_docs)
            logger.info(f"Successfully added {len(haystack_docs)} documents to document store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def generate_response(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        custom_prompt_template: Optional[str] = None
    ) -> RAGResult:
        """Generate a response using the RAG pipeline."""
        if not query.strip():
            return RAGResult(
                query=query,
                response="Please provide a valid question.",
                sources_used=[],
                retrieval_count=0,
                ranking_count=0
            )
        
        try:
            logger.info(f"Generating response for: '{query[:50]}...'")
            
            # Ranker should already be warmed up from initialization
            # If not, it will be handled gracefully by the ranker itself
            
            # Step 1: Retrieve documents
            # Note: InMemoryBM25Retriever may not support filters parameter
            try:
                retrieval_result = self.retriever.run(
                    query=query,
                    filters=filters or {}
                )
            except TypeError:
                # Fallback for retrievers that don't support filters
                retrieval_result = self.retriever.run(query=query)
            retrieved_docs = retrieval_result.get("documents", [])
            
            if not retrieved_docs:
                return RAGResult(
                    query=query,
                    response="I couldn't find any relevant information to answer your question.",
                    sources_used=[],
                    retrieval_count=0,
                    ranking_count=0
                )
            
            # Step 2: Rank documents
            try:
                ranking_result = self.ranker.run(
                    query=query,
                    documents=retrieved_docs
                )
            except Exception as rank_error:
                logger.warning(f"Ranking failed: {rank_error}, using retrieved docs directly")
                ranking_result = {"documents": retrieved_docs}
            ranked_docs = ranking_result.get("documents", [])
            
            # Step 3: Build context and prompt
            context = self._build_context(ranked_docs)
            
            # Prepare chat messages
            system_prompt = custom_prompt_template or self._get_default_system_prompt()
            messages = [
                ChatMessage.from_system(system_prompt),
                ChatMessage.from_user(f"Context: {context}\n\nQuestion: {query}")
            ]
            
            # Step 4: Generate response
            generation_result = self.generator.run(messages=messages)
            replies = generation_result.get("replies", [])
            
            if not replies:
                response_text = "I couldn't generate a response. Please try rephrasing your question."
            else:
                response_text = replies[0].text or "No response generated."  # Use .text for ChatMessage
            
            # Step 5: Extract sources info
            sources_info = self._extract_sources_info(ranked_docs)
            
            result = RAGResult(
                query=query,
                response=response_text,
                sources_used=sources_info,
                retrieval_count=len(retrieved_docs),
                ranking_count=len(ranked_docs)
            )
            
            logger.info(f"Response generated successfully using {len(sources_info)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources_used=[],
                retrieval_count=0,
                ranking_count=0
            )
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt with citation instructions"""
        return """You are an AI assistant that answers questions based on provided source material. Follow these rules:

1. For each factual claim in your answer, include citation references [1], [2], etc.
2. Only use information from the provided context - do not add external knowledge
3. If you cannot find relevant information in the context, say so clearly
4. Be precise and accurate in your citations
5. When multiple sources support the same point, list all relevant citations
6. Provide comprehensive and well-structured answers

The context documents are numbered and you should reference them accordingly."""
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.meta or {}
            source_info = f"Source: {meta.get('source_file', 'Unknown')}"
            if meta.get('page_number'):
                source_info += f" (Page {meta['page_number']})"
            
            context_parts.append(f"[{i}] {doc.content}\n{source_info}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources_info(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources_info = []
        
        for i, doc in enumerate(documents, 1):
            meta = doc.meta or {}
            content = doc.content or ""
            
            source_info = {
                'reference': f"[{i}]",
                'source_file': meta.get('source_file', 'Unknown Source'),
                'source_type': meta.get('source_type', 'unknown'),
                'page_number': meta.get('page_number'),
                'content_preview': content[:200] + "..." if len(content) > 200 else content
            }
            
            # Add relevance score if available
            if hasattr(doc, 'score'):
                source_info['relevance_score'] = doc.score
            
            sources_info.append(source_info)
        
        return sources_info
    
    def search_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents without generating a response"""
        try:
            search_top_k = top_k or self.ranking_top_k
            
            # Retrieve documents
            try:
                retrieval_result = self.retriever.run(
                    query=query,
                    filters=filters or {}
                )
            except TypeError:
                # Fallback for retrievers that don't support filters
                retrieval_result = self.retriever.run(query=query)
            documents = retrieval_result.get("documents", [])
            
            if documents and len(documents) > 0:
                # Rank documents
                try:
                    ranking_result = self.ranker.run(
                        query=query,
                        documents=documents
                    )
                    ranked_docs = ranking_result.get("documents", [])[:search_top_k]
                except Exception as rank_error:
                    logger.warning(f"Ranking failed: {rank_error}, using retrieved docs directly")
                    ranked_docs = documents[:search_top_k]
            else:
                ranked_docs = []
            
            # Format results
            results = []
            for i, doc in enumerate(ranked_docs):
                meta = doc.meta or {}
                result_info = {
                    'rank': i + 1,
                    'content': doc.content or "",
                    'source_file': meta.get('source_file', 'Unknown'),
                    'source_type': meta.get('source_type', 'unknown'),
                    'page_number': meta.get('page_number'),
                    'score': getattr(doc, 'score', 0.0),
                    'metadata': meta
                }
                results.append(result_info)
            
            logger.info(f"Found {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the document store"""
        try:
            return self.document_store.count_documents()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0


def create_rag_generator(
    elasticsearch_host: str = "localhost:9200",
    gemini_api_key: Optional[str] = None,
    **kwargs
) -> ElasticsearchRAGGenerator:
    """Factory function to create a configured RAG generator."""
    # Set default model if not provided in kwargs
    if 'model_name' not in kwargs:
        kwargs['model_name'] = "gemini-2.0-flash"
    
    return ElasticsearchRAGGenerator(
        elasticsearch_host=elasticsearch_host,
        gemini_api_key=gemini_api_key,
        **kwargs
    )


if __name__ == "__main__":
    """Example usage and testing"""
    import sys
    
    # Check for required environment variables
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    try:
        # Initialize RAG generator
        rag_generator = create_rag_generator(
            elasticsearch_host="localhost:9200",
            gemini_api_key=gemini_key,
            model_name="gemini-2.0-flash"
        )
        
        print("RAG Generator initialized successfully!")
        print(f"Document count: {rag_generator.get_document_count()}")
        
        # Add sample documents
        sample_docs = [
            {
                'content': "Artificial Intelligence (AI) is transforming industries worldwide through machine learning and automation.",
                'metadata': {'source_file': 'ai_overview.pdf', 'source_type': 'pdf', 'page_number': 1}
            },
            {
                'content': "Deep learning uses neural networks with multiple layers to solve complex problems in computer vision and NLP.",
                'metadata': {'source_file': 'deep_learning.pdf', 'source_type': 'pdf', 'page_number': 2}
            }
        ]
        
        rag_generator.add_documents(sample_docs)
        
        # Test query
        test_query = "What is artificial intelligence and how does it work?"
        print(f"\nTesting query: {test_query}")
        
        result = rag_generator.generate_response(test_query)
        
        print(f"\nQuery: {result.query}")
        print(f"Response: {result.response}")
        print(f"\nPerformance: {result.get_performance_summary()}")
        print(f"\nSources Used ({len(result.sources_used)}):")
        print(result.get_citation_summary())
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline example: {e}")
        print(f"Error: {e}")
