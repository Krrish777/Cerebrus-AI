"""
Web Scraping Module for Cerebrus AI

This module provides comprehensive web scraping capabilities using Firecrawl API
for intelligent content extraction from websites.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urljoin
import time
from datetime import datetime

try:
    from firecrawl import FirecrawlApp
except ImportError:
    try:
        from firecrawl import Firecrawl as FirecrawlApp
    except ImportError:
        raise ImportError("firecrawl package not found. Install with: pip install firecrawl-py")

from haystack import Document
from ..core.logging import CustomLogger

# Initialize logger with fallback
def get_safe_logger():
    """Get a logger instance that's guaranteed to work."""
    try:
        custom_logger = CustomLogger()
        logger = custom_logger.get_logger(__name__)
        if logger and hasattr(logger, 'info'):
            return logger
    except Exception:
        pass
    
    # Fallback to basic logging
    import logging
    basic_logger = logging.getLogger(__name__)
    if not basic_logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    return basic_logger

logger = get_safe_logger()


@dataclass
class WebPageData:
    """Represents scraped web page data with additional metadata"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class WebScraper:
    """
    Advanced web scraper using Firecrawl API for intelligent content extraction.
    
    Features:
    - Intelligent content chunking
    - Metadata preservation
    - Batch processing
    - Error handling and retries
    - URL validation
    - Content preview
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the WebScraper with Firecrawl API.
        
        Args:
            api_key: Firecrawl API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key required. Set FIRECRAWL_API_KEY environment variable "
                "or pass api_key parameter. Get your free key at: https://firecrawl.dev"
            )
        
        try:
            self.app = FirecrawlApp(api_key=self.api_key)
            logger.info("✅ WebScraper initialized with Firecrawl") # type: ignore
        except Exception as e:
            logger.error(f"❌ Failed to initialize Firecrawl: {e}")
            raise
    
    def scrape_url(
        self,
        url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        wait_for_results: int = 30
    ) -> List[Document]:
        """
        Scrape a single URL and return Haystack Documents.
        
        Args:
            url: URL to scrape
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            wait_for_results: Timeout in seconds
            
        Returns:
            List of Haystack Document objects
        """
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")
        
        logger.info(f"🌐 Scraping URL: {url}")
        
        try:
            # Configure scraping parameters
            scrape_params = {
                'formats': ['markdown', 'html'],
                'timeout': wait_for_results * 1000
            }
            
            # Perform scraping with error handling for different Firecrawl versions
            try:
                result = self.app.scrape(url)
            except Exception as e:
                logger.error(f"Scraping failed with method 'scrape': {e}")
                raise
            page_data = self._process_firecrawl_result(result, url)
            
            # Create Haystack documents
            documents = self._create_documents_from_web_content(
                page_data, 
                chunk_size, 
                chunk_overlap
            )
            
            logger.info(f"✅ Successfully scraped {url}: {len(documents)} documents created")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error scraping URL {url}: {str(e)}")
            raise
    
    def _process_firecrawl_result(self, result: Any, url: str) -> WebPageData:
        """Process Firecrawl API result into structured data."""
        try:
            # Handle different Firecrawl response formats
            content = ""
            metadata_dict = {}
            
            if isinstance(result, dict):
                # Direct dictionary response
                content = result.get('markdown', '') or result.get('content', '')
                metadata_dict = result.get('metadata', {})
            elif hasattr(result, 'get'):
                # Object with get method
                content = result.get('markdown', '') or result.get('content', '')
                metadata_dict = result.get('metadata', {})
            elif hasattr(result, 'markdown'):
                # Object with markdown attribute
                content = getattr(result, 'markdown', '') or getattr(result, 'content', '')
                metadata_dict = getattr(result, 'metadata', {})
            else:
                # Fallback
                content = str(result) if result else ''
                metadata_dict = {}
            
            # Handle metadata - convert to dict if it's an object
            if not isinstance(metadata_dict, dict):
                if hasattr(metadata_dict, '__dict__'):
                    metadata_dict = metadata_dict.__dict__
                else:
                    # Extract common attributes
                    metadata_dict = {
                        'title': getattr(metadata_dict, 'title', ''),
                        'description': getattr(metadata_dict, 'description', ''),
                        'language': getattr(metadata_dict, 'language', 'en')
                    }
            
            metadata = {
                'scraped_at': datetime.now().isoformat(),
                'original_url': url,
                'title': metadata_dict.get('title', ''),
                'description': metadata_dict.get('description', ''),
                'keywords': metadata_dict.get('keywords', []),
                'language': metadata_dict.get('language', 'en'),
                'word_count': len(content.split()) if content else 0,
                'character_count': len(content) if content else 0,
                'domain': urlparse(url).netloc,
                'content_type': 'web_page',
                'source_type': 'web_scraping'
            }
            
            return WebPageData(
                url=url,
                title=metadata['title'] or f"Web Page - {metadata['domain']}",
                content=content,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing Firecrawl result: {str(e)}")
            return WebPageData(
                url=url,
                title=f"Error - {urlparse(url).netloc}",
                content="",
                metadata={
                    'error': str(e), 
                    'scraped_at': datetime.now().isoformat(),
                    'content_type': 'web_page_error',
                    'source_type': 'web_scraping'
                },
                success=False,
                error=str(e)
            )
    
    def _create_documents_from_web_content(
        self,
        page_data: WebPageData,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        """
        Create Haystack Documents from web page content with intelligent chunking.
        
        Args:
            page_data: Processed web page data
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            
        Returns:
            List of Haystack Document objects
        """
        if not page_data.success or not page_data.content.strip():
            logger.warning(f"⚠️ No content to process for {page_data.url}")
            return []
        
        documents = []
        content = page_data.content
        start = 0
        chunk_index = 0
        
        # Smart chunking with content-aware boundaries
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at logical boundaries
            if end < len(content):
                # First try: double newline (paragraph break)
                last_double_newline = content.rfind('\n\n', start, end)
                if last_double_newline > start + chunk_size * 0.3:
                    end = last_double_newline + 2
                else:
                    # Second try: single newline
                    last_newline = content.rfind('\n', start, end)
                    if last_newline > start + chunk_size * 0.5:
                        end = last_newline + 1
                    else:
                        # Third try: sentence end
                        last_period = content.rfind('.', start, end)
                        if last_period > start + chunk_size * 0.5:
                            end = last_period + 1
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                # Create metadata for this chunk
                chunk_metadata = page_data.metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'chunk_character_start': start,
                    'chunk_character_end': end - 1,
                    'url_fragment': f"{page_data.url}#chunk-{chunk_index}",
                    'chunk_word_count': len(chunk_text.split()),
                    'chunk_character_count': len(chunk_text)
                })
                
                # Create Haystack Document
                document = Document(
                    content=chunk_text,
                    meta=chunk_metadata
                )
                
                documents.append(document)
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = max(start + chunk_size - chunk_overlap, end)
        
        logger.info(f"📄 Created {len(documents)} documents from {page_data.url}")
        return documents
    
    def batch_scrape_urls(
        self,
        urls: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        delay_between_requests: float = 1.0
    ) -> List[Document]:
        """
        Scrape multiple URLs and return all documents.
        
        Args:
            urls: List of URLs to scrape
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            delay_between_requests: Delay between requests to avoid rate limiting
            
        Returns:
            Flattened list of all Haystack Documents
        """
        all_documents = []
        successful_urls = 0
        
        logger.info(f"🚀 Starting batch scraping of {len(urls)} URLs") # type: ignore
        
        for i, url in enumerate(urls):
            try:
                logger.info(f"📥 Processing URL {i+1}/{len(urls)}: {url}") # type: ignore
                documents = self.scrape_url(url, chunk_size, chunk_overlap)
                all_documents.extend(documents)
                successful_urls += 1
                
                logger.info(f"✅ Successfully scraped {url}: {len(documents)} documents") # type: ignore
                
                # Rate limiting delay
                if i < len(urls) - 1:
                    time.sleep(delay_between_requests)
                    
            except Exception as e:
                logger.error(f"❌ Failed to scrape {url}: {str(e)}")
                continue
        
        logger.info(f"📊 Batch scraping complete: {len(all_documents)} total documents from {successful_urls}/{len(urls)} URLs") # type: ignore
        return all_documents
    
    def get_url_preview(self, url: str) -> Dict[str, Any]:
        """
        Get a preview of URL content without full scraping.
        
        Args:
            url: URL to preview
            
        Returns:
            Dictionary with preview information
        """
        logger.info(f"👁️ Getting preview for: {url}") # type: ignore
        
        try:
            result = self.app.scrape(url)
            
            # Handle different Firecrawl response formats
            content = ""
            metadata_dict = {}
            
            if isinstance(result, dict):
                content = result.get('markdown', '') or result.get('content', '')
                metadata_dict = result.get('metadata', {})
            elif hasattr(result, 'get'):
                content = result.get('markdown', '') or result.get('content', '') # type: ignore
                metadata_dict = result.get('metadata', {})
            elif hasattr(result, 'markdown'):
                content = getattr(result, 'markdown', '') or getattr(result, 'content', '')
                metadata_dict = getattr(result, 'metadata', {})
            else:
                content = str(result) if result else ''
                metadata_dict = {}
            
            # Handle metadata dict conversion
            if not isinstance(metadata_dict, dict):
                if hasattr(metadata_dict, '__dict__'):
                    metadata_dict = metadata_dict.__dict__
                else:
                    metadata_dict = {
                        'title': getattr(metadata_dict, 'title', ''),
                        'description': getattr(metadata_dict, 'description', ''),
                        'language': getattr(metadata_dict, 'language', 'unknown')
                    }
            
            preview_info = {
                'url': url,
                'title': metadata_dict.get('title', ''),
                'description': metadata_dict.get('description', ''),
                'word_count': len(content.split()) if content else 0,
                'character_count': len(content) if content else 0,
                'domain': urlparse(url).netloc,
                'content_preview': content[:500] + '...' if len(content) > 500 else content,
                'language': metadata_dict.get('language', 'unknown'),
                'estimated_chunks': max(1, len(content) // 1000) if content else 0,
                'success': True
            }
            
            logger.info(f"✅ Preview generated for {url}")
            return preview_info
            
        except Exception as e:
            logger.error(f"❌ Error getting URL preview: {str(e)}")
            return {
                'url': url,
                'error': str(e),
                'success': False
            }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False

    def run(self, sources: List[str], **kwargs) -> Dict[str, List[Document]]:
        """
        Haystack-compatible run method for pipeline integration.
        
        Args:
            sources: List of URLs to scrape
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'documents' key containing all scraped documents
        """
        chunk_size = kwargs.get('chunk_size', 1000)
        chunk_overlap = kwargs.get('chunk_overlap', 100)
        delay = kwargs.get('delay_between_requests', 1.0)
        
        documents = self.batch_scrape_urls(
            urls=sources,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            delay_between_requests=delay
        )
        
        return {"documents": documents}


# Utility functions for easy import
def create_web_scraper(api_key: Optional[str] = None) -> WebScraper:
    """Create a WebScraper instance with error handling."""
    try:
        return WebScraper(api_key=api_key)
    except Exception as e:
        logger.error(f"❌ Failed to create WebScraper: {e}")
        raise


def scrape_single_url(url: str, api_key: Optional[str] = None, **kwargs) -> List[Document]:
    """Convenience function to scrape a single URL."""
    scraper = create_web_scraper(api_key)
    return scraper.scrape_url(url, **kwargs)


def scrape_multiple_urls(urls: List[str], api_key: Optional[str] = None, **kwargs) -> List[Document]:
    """Convenience function to scrape multiple URLs."""
    scraper = create_web_scraper(api_key)
    return scraper.batch_scrape_urls(urls, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("🌐 Testing Web Scraper")
    print("=" * 50)
    
    try:
        scraper = create_web_scraper()
        
        # Test single URL
        test_url = "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
        print(f"📋 Getting preview for: {test_url}")
        
        preview = scraper.get_url_preview(test_url)
        if preview.get('success'):
            print(f"   ✅ Title: {preview['title']}")
            print(f"   📊 Word Count: {preview['word_count']}")
            print(f"   📄 Estimated Chunks: {preview['estimated_chunks']}")
        else:
            print(f"   ❌ Preview failed: {preview.get('error')}")
        
        # Test scraping
        print(f"\n🚀 Scraping: {test_url}")
        documents = scraper.scrape_url(test_url)
        
        print(f"\n📊 Scraping Results:")
        print(f"   📄 Generated {len(documents)} documents")
        
        # Show sample documents
        for i, doc in enumerate(documents[:2]):
            print(f"\n📖 Document {i+1}:")
            content_preview = doc.content[:200] if doc.content else "No content"
            print(f"   📝 Content Preview: {content_preview}...")
            print(f"   🌐 URL: {doc.meta.get('original_url', 'N/A')}")
            print(f"   📊 Word Count: {doc.meta.get('chunk_word_count', 'N/A')}")
            print(f"   🏷️ Domain: {doc.meta.get('domain', 'N/A')}")
        
        print(f"\n✅ Web scraper test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure FIRECRAWL_API_KEY is set in your environment or .env file")