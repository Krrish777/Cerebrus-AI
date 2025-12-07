"""
Context builder service.
Builds formatted context strings from documents for LLM prompts.
"""

from typing import List

from haystack.dataclasses import Document

from src.core.logging import get_logger
from src.rag.config import ContextConfig

logger = get_logger(__name__)


class ContextBuilderService:
    """Service for building context from documents."""
    
    def __init__(self, config: ContextConfig):
        """
        Initialize context builder service.
        
        Args:
            config: Context configuration
        """
        self.config = config
        logger.info(f"Initialized ContextBuilderService with format={config.format}")
    
    def build_context(self, documents: List[Document]) -> str:
        """
        Build formatted context string from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            logger.debug("No documents to build context from")
            return ""
        
        # Limit documents
        documents = documents[:self.config.max_documents]
        
        # Format based on configuration
        if self.config.format == "numbered":
            context = self._build_numbered_context(documents)
        elif self.config.format == "markdown":
            context = self._build_markdown_context(documents)
        else:  # plain
            context = self._build_plain_context(documents)
        
        # Apply length limit
        context = self._apply_length_limit(context)
        
        logger.debug(f"Built context of length {len(context)} from {len(documents)} documents")
        return context
    
    def _build_numbered_context(self, documents: List[Document]) -> str:
        """Build numbered context format."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Document content
            content = doc.content or ""
            
            # Metadata
            meta_str = ""
            if self.config.include_metadata:
                meta_str = self._format_metadata(doc.meta or {})
            
            # Combine
            if meta_str:
                context_parts.append(f"[{i}] {content}\n{meta_str}")
            else:
                context_parts.append(f"[{i}] {content}")
        
        return "\n\n".join(context_parts)
    
    def _build_markdown_context(self, documents: List[Document]) -> str:
        """Build markdown context format."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.content or ""
            
            # Add as markdown section
            section = f"## Source {i}\n\n{content}"
            
            # Add metadata
            if self.config.include_metadata:
                meta_str = self._format_metadata(doc.meta or {})
                if meta_str:
                    section += f"\n\n*{meta_str}*"
            
            context_parts.append(section)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_plain_context(self, documents: List[Document]) -> str:
        """Build plain context format."""
        context_parts = []
        
        for doc in documents:
            content = doc.content or ""
            context_parts.append(content)
        
        return "\n\n".join(context_parts)
    
    def _format_metadata(self, meta: dict) -> str:
        """Format metadata for context."""
        if not meta:
            return ""
        
        parts = []
        
        for field in self.config.metadata_fields:
            if field in meta and meta[field]:
                value = meta[field]
                parts.append(f"{field}: {value}")
        
        return " | ".join(parts) if parts else ""
    
    def _apply_length_limit(self, context: str) -> str:
        """Apply context length limit with truncation strategy."""
        if len(context) <= self.config.max_context_length:
            return context
        
        logger.warning(
            f"Context length {len(context)} exceeds limit {self.config.max_context_length}, "
            f"truncating with strategy: {self.config.truncation_strategy}"
        )
        
        max_len = self.config.max_context_length
        
        if self.config.truncation_strategy == "start":
            return context[:max_len] + "..."
        elif self.config.truncation_strategy == "end":
            return "..." + context[-max_len:]
        else:  # middle
            half = max_len // 2
            return context[:half] + "\n\n...[truncated]...\n\n" + context[-half:]
