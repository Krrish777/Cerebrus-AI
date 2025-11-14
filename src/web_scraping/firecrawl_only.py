"""
Simple Web Scraper for Cerebrus AI - Firecrawl Only
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

try:
    from firecrawl import FirecrawlApp
except ImportError:
    from firecrawl import Firecrawl as FirecrawlApp

try:
    from haystack import Document
except ImportError:
    # Simple document class if Haystack is not available
    class Document:
        def __init__(self, content: str, meta: Dict[str, Any] = None):
            self.content = content
            self.meta = meta or {}


@dataclass
class ScrapedContent:
    """Simple container for scraped web content"""
    url: str
    title: str
    content: str
    word_count: int
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class SimpleWebScraper:
    """
    Simple web scraper using only Firecrawl API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Firecrawl API key"""
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key required. Set FIRECRAWL_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.app = FirecrawlApp(api_key=self.api_key)
    
    def scrape_url(self, url: str, chunk_size: int = 1000) -> List[Document]:
        """
        Scrape a single URL and return Document objects with chunks
        
        Args:
            url: URL to scrape
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of Document objects
        """
        print(f"🌐 Scraping: {url}")
        
        try:
            # Scrape with Firecrawl
            result = self.app.scrape(url)
            
            # Extract content and metadata
            content = getattr(result, 'markdown', '') or getattr(result, 'content', '')
            metadata_obj = getattr(result, 'metadata', None)
            
            # Process metadata
            title = ""
            if metadata_obj:
                title = getattr(metadata_obj, 'title', '')
            
            # Create base metadata
            base_metadata = {
                'url': url,
                'title': title,
                'domain': urlparse(url).netloc,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split()) if content else 0,
                'source': 'firecrawl'
            }
            
            # Create chunks
            documents = []
            if content:
                chunks = self._create_chunks(content, chunk_size)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'chunk_count': len(chunks),
                        'chunk_size': len(chunk)
                    })
                    
                    documents.append(Document(content=chunk, meta=chunk_metadata))
            
            print(f"   ✅ Created {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            print(f"   ❌ Error scraping {url}: {e}")
            return []
    
    def scrape_multiple(self, urls: List[str], chunk_size: int = 1000) -> List[Document]:
        """Scrape multiple URLs"""
        all_documents = []
        
        print(f"🚀 Scraping {len(urls)} URLs...")
        
        for url in urls:
            documents = self.scrape_url(url, chunk_size)
            all_documents.extend(documents)
        
        print(f"📊 Total: {len(all_documents)} documents from {len(urls)} URLs")
        return all_documents
    
    def get_preview(self, url: str) -> ScrapedContent:
        """Get a preview of URL content"""
        try:
            result = self.app.scrape(url)
            
            content = getattr(result, 'markdown', '') or getattr(result, 'content', '')
            metadata_obj = getattr(result, 'metadata', None)
            
            title = ""
            if metadata_obj:
                title = getattr(metadata_obj, 'title', '')
            
            return ScrapedContent(
                url=url,
                title=title,
                content=content[:500] + "..." if len(content) > 500 else content,
                word_count=len(content.split()) if content else 0,
                metadata={
                    'domain': urlparse(url).netloc,
                    'full_length': len(content),
                    'estimated_chunks': max(1, len(content) // 1000)
                },
                success=True
            )
            
        except Exception as e:
            return ScrapedContent(
                url=url,
                title="Error",
                content="",
                word_count=0,
                metadata={},
                success=False,
                error=str(e)
            )
    
    def _create_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Create text chunks with smart boundary detection"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at paragraph boundary
            if end < len(text):
                # Look for double newline (paragraph break)
                last_para = text.rfind('\n\n', start, end)
                if last_para > start + chunk_size * 0.3:
                    end = last_para + 2
                else:
                    # Look for single newline
                    last_newline = text.rfind('\n', start, end)
                    if last_newline > start + chunk_size * 0.5:
                        end = last_newline + 1
                    else:
                        # Look for sentence end
                        last_period = text.rfind('.', start, end)
                        if last_period > start + chunk_size * 0.5:
                            end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks


# Convenience functions
def create_scraper(api_key: Optional[str] = None) -> SimpleWebScraper:
    """Create a SimpleWebScraper instance"""
    return SimpleWebScraper(api_key=api_key)


def scrape_url(url: str, api_key: Optional[str] = None, chunk_size: int = 1000) -> List[Document]:
    """Quick function to scrape a single URL"""
    scraper = create_scraper(api_key)
    return scraper.scrape_url(url, chunk_size)


if __name__ == "__main__":
    # Test the scraper
    print("🧪 Testing Simple Web Scraper")
    print("=" * 40)
    
    try:
        scraper = create_scraper()
        
        # Test with example.com
        test_url = "https://example.com"
        print(f"📋 Preview: {test_url}")
        
        preview = scraper.get_preview(test_url)
        if preview.success:
            print(f"   ✅ Title: {preview.title}")
            print(f"   📊 Words: {preview.word_count}")
            print(f"   📝 Preview: {preview.content[:100]}...")
        
        # Test scraping
        documents = scraper.scrape_url(test_url, chunk_size=300)
        print(f"📄 Generated {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            print(f"   📖 Doc {i+1}: {len(doc.content)} chars")
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")