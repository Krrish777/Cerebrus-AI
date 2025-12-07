"""
Data model for search-only operations (retrieval without generation).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Result object for search operations without generation."""
    
    query: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0
    ranking_count: int = 0
    filters_applied: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_document_summary(self) -> str:
        """Generate a formatted summary of retrieved documents."""
        if not self.documents:
            return "No documents found"
        
        summary_lines = []
        for i, doc in enumerate(self.documents, 1):
            doc_info = f"{i}. {doc.get('source_file', 'Unknown')}"
            if doc.get('page_number'):
                doc_info += f" (Page {doc['page_number']})"
            if doc.get('score') is not None:
                doc_info += f" - Score: {doc['score']:.3f}"
            
            # Add content preview
            content = doc.get('content', '')
            preview = content[:100] + "..." if len(content) > 100 else content
            doc_info += f"\n   Preview: {preview}"
            
            summary_lines.append(doc_info)
        
        return "\n".join(summary_lines)
    
    def get_performance_summary(self) -> str:
        """Get performance metrics summary."""
        parts = [
            f"Retrieved: {self.retrieval_count} documents"
        ]
        
        if self.ranking_count > 0:
            parts.append(f"Ranked: {self.ranking_count} documents")
        
        if self.filters_applied:
            parts.append(f"Filters: {len(self.filters_applied)}")
        
        return ", ".join(parts)
    
    def get_documents_by_type(self, source_type: str) -> List[Dict[str, Any]]:
        """Filter documents by type."""
        return [
            doc for doc in self.documents
            if doc.get('source_type') == source_type
        ]
    
    def get_top_documents(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N documents by score."""
        sorted_docs = sorted(
            self.documents,
            key=lambda x: x.get('score', 0.0),
            reverse=True
        )
        return sorted_docs[:n]
    
    def get_documents_above_threshold(self, threshold: float) -> List[Dict[str, Any]]:
        """Get documents with score above threshold."""
        return [
            doc for doc in self.documents
            if doc.get('score', 0.0) >= threshold
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'query': self.query,
            'documents': self.documents,
            'retrieval_count': self.retrieval_count,
            'ranking_count': self.ranking_count,
            'filters_applied': self.filters_applied,
            'metadata': self.metadata,
            'performance_summary': self.get_performance_summary()
        }
    
    def __str__(self) -> str:
        """String representation of the result."""
        lines = [
            f"Query: {self.query}",
            f"Found {len(self.documents)} documents",
            "",
            "Documents:",
            self.get_document_summary(),
            "",
            "Performance:",
            self.get_performance_summary()
        ]
        return "\n".join(lines)
    
    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.documents)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get document by index."""
        return self.documents[index]
    
    def __iter__(self):
        """Iterate over documents."""
        return iter(self.documents)
