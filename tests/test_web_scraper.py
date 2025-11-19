#!/usr/bin/env python3
"""
Simple Web Scraper Test - Direct Firecrawl Integration

This test helps us understand the correct Firecrawl response format
and create a working web scraper.
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded environment variables from .env file")

# Load the environment
load_env_file()

# Check API key
api_key = os.getenv("FIRECRAWL_API_KEY")
if not api_key:
    print("❌ No FIRECRAWL_API_KEY found in environment!")
    print("Please set FIRECRAWL_API_KEY in your .env file or environment")
    print("Get a free API key at: https://firecrawl.dev")
    sys.exit(1)

print("✅ API key found")
print("🌐 Testing Direct Web Scraping with Firecrawl")
print("=" * 60)

try:
    from firecrawl import FirecrawlApp
    print("✅ FirecrawlApp imported successfully")
except ImportError as e:
    print(f"❌ Failed to import FirecrawlApp: {e}")
    print("Install with: pip install firecrawl-py")
    sys.exit(1)

def test_web_scraping():
    """Test direct web scraping with Firecrawl."""
    print("\n🧪 Testing Direct Web Scraping")
    print("=" * 50)
    
    test_url = "https://example.com"
    print(f"🔗 Testing URL: {test_url}")
    
    try:
        # Initialize Firecrawl
        app = FirecrawlApp(api_key=api_key)
        print("✅ Firecrawl app initialized")
        
        # Test scraping
        print("🚀 Starting scrape...")
        result = app.scrape(test_url)
        
        print("✅ Scraping completed!")
        print(f"📊 Result type: {type(result)}")
        
        # Analyze result structure
        if isinstance(result, dict):
            print(f"📋 Result keys: {list(result.keys())}")
            
            # Extract content
            if 'markdown' in result:
                content = result['markdown']
                print("📄 Content type: markdown")
                print(f"📏 Content length: {len(content)} characters")
                print(f"📝 Content preview: {content[:200]}...")
            elif 'content' in result:
                content = result['content'] 
                print("📄 Content type: content")
                print(f"📏 Content length: {len(content)} characters")
                print(f"📝 Content preview: {content[:200]}...")
            
            # Extract metadata
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"🏷️ Metadata type: {type(metadata)}")
                if isinstance(metadata, dict):
                    print(f"🏷️ Metadata keys: {list(metadata.keys())}")
                    print(f"📊 Title: {metadata.get('title', 'No title')}")
        else:
            print(f"🔍 Result has attributes: {dir(result)}")
            
            # Try to access common attributes
            if hasattr(result, 'markdown'):
                content = result.markdown
                print(f"📄 Markdown content length: {len(content)}")
                print(f"📝 Content preview: {content[:200]}...")
                
            if hasattr(result, 'metadata'):
                print(f"🏷️ Metadata: {result.metadata}")
        
        # Test with a more interesting URL
        print("\n🔗 Testing with content-rich URL...")
        blog_url = "https://firecrawl.dev"
        result2 = app.scrape(blog_url)
        
        if isinstance(result2, dict):
            content2 = result2.get('markdown', '') or result2.get('content', '')
            print(f"📄 Content length: {len(content2)} characters")
            print(f"🔤 Word count: {len(content2.split())}")
            
            metadata2 = result2.get('metadata', {})
            if isinstance(metadata2, dict):
                print(f"📰 Title: {metadata2.get('title', 'No title')}")
                print(f"📝 Description: {metadata2.get('description', 'No description')}")
        
        print("\n✅ Web scraping test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_web_scraping()
        if success:
            print("\n🎉 Firecrawl integration working correctly!")
        else:
            print("\n💥 Firecrawl integration needs debugging")
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")