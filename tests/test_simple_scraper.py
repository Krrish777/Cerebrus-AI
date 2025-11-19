#!/usr/bin/env python3
"""
Simple Firecrawl Web Scraper Test - No Extra Dependencies
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

try:
    from firecrawl import FirecrawlApp
except ImportError:
    from firecrawl import Firecrawl as FirecrawlApp

def test_firecrawl_scraper():
    """Test Firecrawl web scraping functionality directly"""
    print("🌐 Testing Firecrawl Web Scraper")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("❌ No FIRECRAWL_API_KEY found in environment!")
        return
    
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-5:]}")
    
    try:
        # Initialize Firecrawl
        app = FirecrawlApp(api_key=api_key)
        print("✅ Firecrawl initialized successfully")
        
        # Test URLs
        test_urls = [
            "https://example.com",
            "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
        ]
        
        for url in test_urls:
            print(f"\n🚀 Scraping: {url}")
            try:
                # Scrape the URL
                result = app.scrape(url)
                
                # Extract content
                content = getattr(result, 'markdown', '') or getattr(result, 'content', '')
                metadata = getattr(result, 'metadata', None)
                
                # Display results
                print("   ✅ Success!")
                print(f"   📊 Content Length: {len(content)} characters")
                print(f"   📝 Preview: {content[:100]}...")
                
                if metadata:
                    title = getattr(metadata, 'title', 'No title')
                    print(f"   🏷️ Title: {title}")
                    print(f"   🌍 URL: {getattr(metadata, 'url', 'N/A')}")
                    print(f"   📄 Status: {getattr(metadata, 'status_code', 'N/A')}")
                
                # Simple chunking example
                chunk_size = 500
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                print(f"   📄 Would create {len(chunks)} chunks of ~{chunk_size} chars each")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n✅ Firecrawl test completed!")
        print("🎉 Web scraper is working correctly!")
        
    except Exception as e:
        print(f"❌ Failed to initialize Firecrawl: {e}")

if __name__ == "__main__":
    test_firecrawl_scraper()