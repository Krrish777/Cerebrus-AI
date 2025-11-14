#!/usr/bin/env python3
"""
Cerebrus AI Web Scraper - Usage Examples
"""

import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from src.web_scraping.firecrawl_only import create_scraper

def example_basic_scraping():
    """Basic web scraping example"""
    print("🌐 Example 1: Basic Web Scraping")
    print("-" * 40)
    
    # Create scraper
    scraper = create_scraper()
    
    # URLs to scrape
    urls = [
        "https://example.com",
        "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
    ]
    
    for url in urls:
        print(f"\n🔍 Analyzing: {url}")
        
        # Get preview
        preview = scraper.get_preview(url)
        print(f"   📋 Title: {preview.title}")
        print(f"   📊 Words: {preview.word_count}")
        print(f"   📄 Est. Chunks: {preview.metadata['estimated_chunks']}")
        
        # Scrape with chunking
        documents = scraper.scrape_url(url, chunk_size=1000)
        print(f"   ✅ Created {len(documents)} chunks")

def example_batch_scraping():
    """Batch scraping example"""
    print("\n\n🚀 Example 2: Batch Scraping")
    print("-" * 40)
    
    scraper = create_scraper()
    
    urls = [
        "https://example.com",
        "https://httpbin.org/html"
    ]
    
    # Batch scrape all URLs
    all_documents = scraper.scrape_multiple(urls, chunk_size=800)
    
    print(f"\n📊 Summary:")
    print(f"   🌐 URLs processed: {len(urls)}")
    print(f"   📄 Total documents: {len(all_documents)}")
    
    # Group by domain
    by_domain = {}
    for doc in all_documents:
        domain = doc.meta.get('domain', 'unknown')
        by_domain[domain] = by_domain.get(domain, 0) + 1
    
    for domain, count in by_domain.items():
        print(f"   📍 {domain}: {count} documents")

def example_custom_chunking():
    """Custom chunking strategy example"""
    print("\n\n✂️ Example 3: Custom Chunking")
    print("-" * 40)
    
    scraper = create_scraper()
    
    url = "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
    
    # Try different chunk sizes
    chunk_sizes = [500, 1000, 2000]
    
    for size in chunk_sizes:
        documents = scraper.scrape_url(url, chunk_size=size)
        print(f"   📏 Chunk size {size}: {len(documents)} documents")

def main():
    """Run all examples"""
    print("🎯 Cerebrus AI Web Scraper Examples")
    print("=" * 50)
    
    try:
        example_basic_scraping()
        example_batch_scraping()
        example_custom_chunking()
        
        print(f"\n\n✅ All examples completed successfully!")
        print(f"🎉 Ready to scrape the web with Cerebrus AI!")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("Make sure FIRECRAWL_API_KEY is set in your .env file")

if __name__ == "__main__":
    main()