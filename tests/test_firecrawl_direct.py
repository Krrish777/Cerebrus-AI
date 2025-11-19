#!/usr/bin/env python3
"""
Test Firecrawl Web Scraper

Direct test of the Firecrawl web scraper with your API key.
"""

import os
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
        return True
    return False

# Load the environment
if load_env_file():
    api_key = os.getenv("FIRECRAWL_API_KEY")
    print(f"🔑 API key loaded: {api_key[:10]}..." if api_key else "❌ No API key found")
else:
    print("❌ No .env file found")

# Test Firecrawl scraper
try:
    from firecrawl import FirecrawlApp
    print("✅ FirecrawlApp imported")
    
    if api_key:
        app = FirecrawlApp(api_key=api_key)
        print("✅ Firecrawl app initialized")
        
        # Test basic scraping
        print("🌐 Testing scrape...")
        test_url = "https://example.com"
        result = app.scrape_url(test_url)
        print(f"✅ Scrape result type: {type(result)}")
        print(f"📄 Result: {str(result)[:200]}...")
        
    else:
        print("❌ Cannot test without API key")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()