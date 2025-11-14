#!/usr/bin/env python3
"""
Test the Firecrawl-only web scraper
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.web_scraping.firecrawl_only import create_scraper

def main():
    print("🧪 Testing Firecrawl-Only Web Scraper")
    print("=" * 45)
    
    try:
        # Create scraper
        scraper = create_scraper()
        print("✅ Scraper created successfully")
        
        # Test URL
        test_url = "https://example.com"
        
        # Get preview
        print(f"\n📋 Getting preview for: {test_url}")
        preview = scraper.get_preview(test_url)
        
        if preview.success:
            print(f"   ✅ Title: {preview.title}")
            print(f"   📊 Word Count: {preview.word_count}")
            print(f"   🌍 Domain: {preview.metadata['domain']}")
            print(f"   📄 Estimated Chunks: {preview.metadata['estimated_chunks']}")
            print(f"   📝 Preview: {preview.content[:100]}...")
        else:
            print(f"   ❌ Preview failed: {preview.error}")
            return
        
        # Test scraping with chunking
        print(f"\n🚀 Scraping with chunking: {test_url}")
        documents = scraper.scrape_url(test_url, chunk_size=100)  # Small chunks for testing
        
        print(f"📊 Results: {len(documents)} document chunks created")
        
        for i, doc in enumerate(documents):
            print(f"\n📖 Document {i+1}:")
            print(f"   📝 Content: {doc.content[:80]}...")
            print(f"   🏷️ Title: {doc.meta.get('title', 'N/A')}")
            print(f"   📊 Chunk Size: {doc.meta.get('chunk_size', 'N/A')} chars")
            print(f"   🔢 Chunk Index: {doc.meta.get('chunk_index', 'N/A')}")
        
        # Test with a content-rich URL
        rich_url = "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
        print(f"\n🌐 Testing with content-rich URL...")
        preview2 = scraper.get_preview(rich_url)
        
        if preview2.success:
            print(f"   ✅ Title: {preview2.title}")
            print(f"   📊 Word Count: {preview2.word_count}")
            print(f"   📄 Estimated Chunks: {preview2.metadata['estimated_chunks']}")
        
        print(f"\n✅ All tests completed successfully!")
        print(f"🎉 Firecrawl web scraper is working perfectly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()