"""
Simple Web Scraper for Cerebrus AI

A working web scraper implementation using requests and BeautifulSoup
as a backup to Firecrawl integration.
"""

import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path

from haystack import Document
from ..core.logging import CustomLogger


class SimpleWebScraper:
    """
    Simple web scraper using requests and BeautifulSoup.
    This serves as a backup when Firecrawl is not available.
    """
    
    def __init__(self):
        """Initialize the SimpleWebScraper."""
        try:
            self.custom_logger = CustomLogger()
            self.logger = self.custom_logger.get_logger(__name__)
            if self.logger is None:
                # Fallback to basic logging
                import logging
                self.logger = logging.getLogger(__name__)
                logging.basicConfig(level=logging.INFO)
        except Exception:
            import logging
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.logger.info("✅ SimpleWebScraper initialized")
    
    def scrape_url(
        self,
        url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> List[Document]:
        """
        Scrape a URL and return Haystack Documents.
        
        Args:
            url: URL to scrape
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            
        Returns:
            List of Haystack Document objects
        """
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")
        
        self.logger.info(f"🌐 Scraping URL: {url}")
        
        try:
            # Fetch the page
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract metadata
            title = ""
            if soup.title:
                title = soup.title.get_text().strip()
            
            description = ""
            desc_tag = soup.find("meta", {"name": "description"})
            if desc_tag:
                description = desc_tag.get("content", "")
            
            # Create metadata
            metadata = {
                'url': url,
                'title': title,
                'description': description,
                'domain': urlparse(url).netloc,
                'scraped_at': datetime.now().isoformat(),
                'content_type': 'web_page',
                'source_type': 'simple_scraper',
                'word_count': len(text.split()),
                'character_count': len(text)
            }
            
            # Create chunks
            documents = self._create_chunks(text, metadata, chunk_size, chunk_overlap)
            
            self.logger.info(f"✅ Successfully scraped {url}: {len(documents)} documents created")
            return documents
            
        except Exception as e:
            self.logger.error(f"❌ Error scraping URL {url}: {str(e)}")
            raise
    
    def _create_chunks(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        """Create chunked documents from text."""
        if not text.strip():
            self.logger.warning("⚠️ No content to process")
            return []
        
        documents = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at word boundaries
            if end < len(text):
                # Find last space within reasonable distance
                last_space = text.rfind(' ', start, end)
                if last_space > start + chunk_size * 0.7:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'chunk_start': start,
                    'chunk_end': end,
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
        
        self.logger.info(f"📄 Created {len(documents)} chunks")
        return documents
    
    def batch_scrape_urls(
        self,
        urls: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        delay_between_requests: float = 1.0
    ) -> List[Document]:
        """Scrape multiple URLs."""
        all_documents = []
        successful_urls = 0
        
        self.logger.info(f"🚀 Starting batch scraping of {len(urls)} URLs")
        
        for i, url in enumerate(urls):
            try:
                self.logger.info(f"📥 Processing URL {i+1}/{len(urls)}: {url}")
                documents = self.scrape_url(url, chunk_size, chunk_overlap)
                all_documents.extend(documents)
                successful_urls += 1
                
                if i < len(urls) - 1:
                    import time
                    time.sleep(delay_between_requests)
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to scrape {url}: {str(e)}")
                continue
        
        self.logger.info(f"📊 Batch scraping complete: {len(all_documents)} total documents from {successful_urls}/{len(urls)} URLs")
        return all_documents
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False
    
    def run(self, sources: List[str], **kwargs) -> Dict[str, List[Document]]:
        """Haystack-compatible run method."""
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


# For now, use the simple scraper until Firecrawl is properly set up
WebScraper = SimpleWebScraper

def create_web_scraper() -> SimpleWebScraper:
    """Create a WebScraper instance."""
    return SimpleWebScraper()


if __name__ == "__main__":
    # Test the simple web scraper
    print("🌐 Testing Simple Web Scraper")
    print("=" * 50)
    
    try:
        scraper = create_web_scraper()
        
        # Test single URL
        test_url = "https://example.com"
        print(f"🔗 Testing URL: {test_url}")
        
        documents = scraper.scrape_url(test_url)
        
        print(f"📊 Scraping Results:")
        print(f"   📄 Generated {len(documents)} documents")
        
        # Show sample documents
        for i, doc in enumerate(documents[:2]):
            print(f"\n📖 Document {i+1}:")
            content_preview = doc.content[:200] if doc.content else "No content"
            print(f"   📝 Content: {content_preview}...")
            print(f"   🌐 URL: {doc.meta.get('url', 'N/A')}")
            print(f"   📊 Words: {doc.meta.get('chunk_word_count', 'N/A')}")
            print(f"   🏷️ Domain: {doc.meta.get('domain', 'N/A')}")
        
        print(f"\n✅ Simple web scraper test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()