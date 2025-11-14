#!/usr/bin/env python3
"""
Test script to understand Firecrawl API response format
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from firecrawl import FirecrawlApp
except ImportError:
    from firecrawl import Firecrawl as FirecrawlApp

def test_firecrawl_api():
    """Test Firecrawl API and examine response format"""
    
    # Get API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    print(f"🔑 API key: {api_key[:10]}...{api_key[-5:] if api_key else 'Not found'}")
    
    if not api_key:
        print("❌ No API key found!")
        return
    
    try:
        # Initialize Firecrawl
        app = FirecrawlApp(api_key=api_key)
        print("✅ Firecrawl app initialized")
        
        # Test URL
        test_url = "https://example.com"
        print(f"🌐 Testing with URL: {test_url}")
        
        # Test scrape method
        result = app.scrape(test_url)
        print(f"\n📊 Result type: {type(result)}")
        print(f"📊 Result: {result}")
        
        # Check if it's a dictionary
        if isinstance(result, dict):
            print(f"\n📋 Dictionary keys: {list(result.keys())}")
            if 'data' in result:
                data = result['data']
                print(f"📋 Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Check if it has attributes
        if hasattr(result, '__dict__'):
            print(f"\n📋 Object attributes: {list(result.__dict__.keys())}")
        
        print("\n✅ API test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_firecrawl_api()