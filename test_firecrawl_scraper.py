#!/usr/bin/env python3
"""
Test the Firecrawl Web Scraper with proper error handling
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded environment variables from .env file")

def test_web_scraper():
    """Test the Firecrawl web scraper"""
    print("🌐 Testing Firecrawl Web Scraper")
    print("=" * 60)
    
    # Load environment
    load_env_file()
    
    # Check API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("❌ No FIRECRAWL_API_KEY found in environment!")
        return False
    
    print("✅ Firecrawl API key found")
    
    try:
        # Import the web scraper
        from web_scraping.web_scraper import create_web_scraper
        
        print("📦 Creating web scraper...")
        scraper = create_web_scraper(api_key=api_key)
        print("✅ Web scraper created successfully")
        
        # Test with a simple URL
        test_url = "https://example.com"
        print(f"🔗 Testing URL: {test_url}")
        
        # Get preview first
        print("👁️ Getting URL preview...")
        preview = scraper.get_url_preview(test_url)
        
        if preview.get('success'):
            print(f"✅ Preview successful:")
            print(f"   📰 Title: {preview.get('title', 'N/A')}")
            print(f"   📊 Word Count: {preview.get('word_count', 'N/A')}")
            print(f"   📄 Estimated Chunks: {preview.get('estimated_chunks', 'N/A')}")
            
            # Test full scraping
            print(f"\n🚀 Full scraping of {test_url}...")
            documents = scraper.scrape_url(test_url)
            
            print(f"📊 Scraping Results:")
            print(f"   📄 Generated {len(documents)} documents")
            
            if documents:
                # Show first document
                doc = documents[0]
                print(f"\n📖 Sample Document:")
                content_preview = doc.content[:200] if doc.content else "No content"
                print(f"   📝 Content: {content_preview}...")
                print(f"   🌐 URL: {doc.meta.get('original_url', 'N/A')}")
                print(f"   📊 Word Count: {doc.meta.get('chunk_word_count', 'N/A')}")
                print(f"   🏷️ Domain: {doc.meta.get('domain', 'N/A')}")
            
            print(f"\n✅ Web scraper test completed successfully!")
            return True
            
        else:
            print(f"❌ Preview failed: {preview.get('error')}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure firecrawl-py is installed: pip install firecrawl-py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_web_scraper()
    if success:
        print("\n🎉 Firecrawl web scraper is working correctly!")
    else:
        print("\n💥 Firecrawl web scraper needs attention")
    
    sys.exit(0 if success else 1)