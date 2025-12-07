"""
Result formatting utilities.
Format RAG results in various output formats.
"""

import json
from typing import Any, Dict

from src.core.logging import get_logger
from src.rag.models import RAGResult, SearchResult

logger = get_logger(__name__)


class ResultFormatter:
    """Formatter for RAG results."""
    
    @staticmethod
    def format_as_text(result: RAGResult) -> str:
        """
        Format RAG result as plain text.
        
        Args:
            result: RAG result
            
        Returns:
            Formatted text
        """
        lines = [
            f"Query: {result.query}",
            "",
            "Response:",
            result.response,
            "",
            "Sources:",
            result.get_citation_summary(),
            "",
            "Performance:",
            result.get_performance_summary()
        ]
        
        return "\n".join(lines)
    
    @staticmethod
    def format_as_json(result: RAGResult, indent: int = 2) -> str:
        """
        Format RAG result as JSON.
        
        Args:
            result: RAG result
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        return json.dumps(result.to_dict(), indent=indent, ensure_ascii=False)
    
    @staticmethod
    def format_as_markdown(result: RAGResult) -> str:
        """
        Format RAG result as Markdown.
        
        Args:
            result: RAG result
            
        Returns:
            Markdown formatted string
        """
        lines = [
            f"# Query: {result.query}",
            "",
            "## Response",
            "",
            result.response,
            "",
            "## Sources",
            ""
        ]
        
        # Add sources as list
        for source in result.sources_used:
            source_line = f"- **{source.get('source_file', 'Unknown')}**"
            if source.get('page_number'):
                source_line += f" (Page {source['page_number']})"
            if source.get('relevance_score') is not None:
                source_line += f" - Score: {source['relevance_score']:.3f}"
            lines.append(source_line)
        
        lines.extend([
            "",
            "## Performance Metrics",
            "",
            f"- Retrieved: {result.retrieval_count} documents",
            f"- Ranked: {result.ranking_count} documents"
        ])
        
        if result.generation_tokens:
            lines.append(f"- Tokens: {result.generation_tokens}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_search_as_text(result: SearchResult) -> str:
        """
        Format search result as plain text.
        
        Args:
            result: Search result
            
        Returns:
            Formatted text
        """
        return str(result)
    
    @staticmethod
    def format_search_as_json(result: SearchResult, indent: int = 2) -> str:
        """
        Format search result as JSON.
        
        Args:
            result: Search result
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        return json.dumps(result.to_dict(), indent=indent, ensure_ascii=False)
    
    @staticmethod
    def format_search_as_markdown(result: SearchResult) -> str:
        """
        Format search result as Markdown.
        
        Args:
            result: Search result
            
        Returns:
            Markdown formatted string
        """
        lines = [
            f"# Search Results for: {result.query}",
            "",
            f"Found {len(result.documents)} documents",
            ""
        ]
        
        for i, doc in enumerate(result.documents, 1):
            lines.append(f"## Document {i}")
            lines.append("")
            
            if doc.get('source_file'):
                lines.append(f"**Source:** {doc['source_file']}")
            if doc.get('page_number'):
                lines.append(f"**Page:** {doc['page_number']}")
            if doc.get('score') is not None:
                lines.append(f"**Score:** {doc['score']:.3f}")
            
            lines.append("")
            
            content = doc.get('content', '')
            preview = content[:200] + "..." if len(content) > 200 else content
            lines.append(f"**Preview:** {preview}")
            lines.append("")
        
        return "\n".join(lines)
