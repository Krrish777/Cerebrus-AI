"""
Citation service.
Extracts and formats source citations from documents.
"""

from typing import Any, Dict, List

from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.rag.config import CitationConfig

logger = get_logger(__name__)


class CitationService:
    """Service for extracting and formatting citations."""
    
    def __init__(self, config: CitationConfig):
        """
        Initialize citation service.
        
        Args:
            config: Citation configuration
        """
        self.config = config
        logger.info(f"Initialized CitationService with style={config.style}")
    
    def extract_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract citation information from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of citation dictionaries
        """
        if not self.config.enabled or not documents:
            return []
        
        citations = []
        
        for i, doc in enumerate(documents, 1):
            # Apply score threshold
            if hasattr(doc, 'score') and doc.score is not None:
                if doc.score < self.config.score_threshold:
                    continue
            
            citation = self._build_citation(doc, reference_number=i)
            citations.append(citation)
        
        logger.debug(f"Extracted {len(citations)} citations from {len(documents)} documents")
        return citations
    
    def _build_citation(self, doc: Document, reference_number: int) -> Dict[str, Any]:
        """
        Build citation dictionary for a document.
        
        Args:
            doc: Document to cite
            reference_number: Reference number
            
        Returns:
            Citation dictionary
        """
        meta = doc.meta or {}
        content = doc.content or ""
        
        citation = {
            'reference': self._format_reference(reference_number),
            'source_file': meta.get('source_file', 'Unknown Source'),
            'source_type': meta.get('source_type', 'unknown'),
            'page_number': meta.get('page_number'),
            'content_preview': self._create_preview(content)
        }
        
        # Add score if configured
        if self.config.include_scores and hasattr(doc, 'score'):
            citation['relevance_score'] = doc.score
        
        # Add additional metadata
        if 'timestamp' in meta:
            citation['timestamp'] = meta['timestamp']
        if 'author' in meta:
            citation['author'] = meta['author']
        if 'title' in meta:
            citation['title'] = meta['title']
        
        return citation
    
    def _format_reference(self, number: int) -> str:
        """Format reference based on citation style."""
        if self.config.style == "numeric":
            return f"[{number}]"
        elif self.config.style == "footnote":
            return f"^{number}"
        else:  # author_year (would need author/year in metadata)
            return f"[{number}]"
    
    def _create_preview(self, content: str) -> str:
        """Create content preview."""
        max_len = self.config.preview_length
        
        if len(content) <= max_len:
            return content
        
        return content[:max_len] + "..."
    
    def format_citation_summary(self, citations: List[Dict[str, Any]]) -> str:
        """
        Format citations as a summary string.
        
        Args:
            citations: List of citation dictionaries
            
        Returns:
            Formatted citation summary
        """
        if not citations:
            return "No sources cited"
        
        summary_lines = []
        
        for citation in citations:
            line = f"• {citation.get('source_file', 'Unknown')}"
            
            if citation.get('source_type'):
                line += f" ({citation['source_type']})"
            
            if citation.get('page_number'):
                line += f" - Page {citation['page_number']}"
            
            if self.config.include_scores and citation.get('relevance_score') is not None:
                line += f" [Score: {citation['relevance_score']:.3f}]"
            
            summary_lines.append(line)
        
        return "\n".join(summary_lines)
