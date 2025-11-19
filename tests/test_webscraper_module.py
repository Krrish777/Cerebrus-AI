#!/usr/bin/env python3
"""
Test script for the Cerebrus AI Web Scraper Module
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the web scraper
from src.web_scraping.web_scraper import create_web_scraper

def test_web_scraper_module():
    """Test the web scraper module functionality"""
    print("🌐 Testing Cerebrus AI Web Scraper Module")
    print("=" * 50)
    
    try:
        # Create web scraper instance
        print("🚀 Creating web scraper...")
        scraper = create_web_scraper()
        print("✅ Web scraper created successfully!")
        
        # Test URL
        test_url = "https://example.com"
        print(f"\n📋 Getting preview for: {test_url}")
        
        # Get preview
        preview = scraper.get_url_preview(test_url)
        if preview.get('success'):
            print(f"   ✅ Title: {preview['title']}")
            print(f"   📊 Word Count: {preview['word_count']}")
            print(f"   📄 Estimated Chunks: {preview['estimated_chunks']}")
            print(f"   🌍 Domain: {preview['domain']}")
            print(f"   📝 Content Preview: {preview['content_preview'][:100]}...")
        else:
            print(f"   ❌ Preview failed: {preview.get('error')}")
            return
        
        # Test scraping
        print(f"\n🚀 Scraping: {test_url}")
        documents = scraper.scrape_url(test_url, chunk_size=500, chunk_overlap=50)
        
        print("\n📊 Scraping Results:")
        print(f"   📄 Generated {len(documents)} documents")
        
        # Show sample documents
        for i, doc in enumerate(documents[:2]):
            print(f"\n📖 Document {i+1}:")
            content_preview = doc.content[:150] if doc.content else "No content"
            print(f"   📝 Content: {content_preview}...")
            print(f"   🌐 URL: {doc.meta.get('original_url', 'N/A')}")
            print(f"   📊 Word Count: {doc.meta.get('chunk_word_count', 'N/A')}")
            print(f"   🏷️ Domain: {doc.meta.get('domain', 'N/A')}")
            print(f"   📄 Chunk Index: {doc.meta.get('chunk_index', 'N/A')}")
        
        print("\n✅ Web scraper test completed successfully!")
        print("🎉 The web scraper module is ready to use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_web_scraper_module()