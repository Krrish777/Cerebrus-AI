"""
Data models for RAG system results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    """Enhanced result object for RAG generation with comprehensive citation tracking."""
    
    query: str
    response: str
    sources_used: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0
    ranking_count: int = 0
    generation_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_citation_summary(self) -> str:
        """Generate a formatted summary of all sources used in the response."""
        if not self.sources_used:
            return "No sources cited"
        
        source_summary = []
        for source in self.sources_used:
            source_info = f"• {source.get('source_file', 'Unknown')} ({source.get('source_type', 'unknown')})"
            if source.get('page_number'):
                source_info += f" - Page {source['page_number']}"
            if source.get('relevance_score') is not None:
                source_info += f" [Score: {source['relevance_score']:.3f}]"
            source_summary.append(source_info)
        
        return "\n".join(source_summary)
    
    def get_performance_summary(self) -> str:
        """Get performance metrics summary."""
        parts = [
            f"Retrieved: {self.retrieval_count} documents",
            f"Ranked: {self.ranking_count} documents"
        ]
        
        if self.generation_tokens:
            parts.append(f"Tokens: {self.generation_tokens}")
        
        return ", ".join(parts)
    
    def get_sources_by_type(self, source_type: str) -> List[Dict[str, Any]]:
        """Filter sources by type."""
        return [
            source for source in self.sources_used
            if source.get('source_type') == source_type
        ]
    
    def get_top_sources(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get top N sources by relevance score."""
        sorted_sources = sorted(
            self.sources_used,
            key=lambda x: x.get('relevance_score', 0.0),
            reverse=True
        )
        return sorted_sources[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'query': self.query,
            'response': self.response,
            'sources_used': self.sources_used,
            'retrieval_count': self.retrieval_count,
            'ranking_count': self.ranking_count,
            'generation_tokens': self.generation_tokens,
            'metadata': self.metadata,
            'citation_summary': self.get_citation_summary(),
            'performance_summary': self.get_performance_summary()
        }
    
    def __str__(self) -> str:
        """String representation of the result."""
        lines = [
            f"Query: {self.query}",
            f"Response: {self.response}",
            "",
            "Sources:",
            self.get_citation_summary(),
            "",
            "Performance:",
            self.get_performance_summary()
        ]
        return "\n".join(lines)
