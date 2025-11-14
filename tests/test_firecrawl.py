"""
Test Firecrawl API to understand the correct interface
"""

import os
from firecrawl import FirecrawlApp

# Load API key
api_key = os.getenv("FIRECRAWL_API_KEY")
if not api_key:
    print("Please set FIRECRAWL_API_KEY environment variable")
    exit(1)

try:
    app = FirecrawlApp(api_key=api_key)
    
    # Test basic scraping
    result = app.scrape_url("https://example.com")
    print("Method works: scrape_url")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
except AttributeError as e:
    print(f"scrape_url failed: {e}")
    try:
        # Try alternative method
        result = app.scrape("https://example.com")
        print("Method works: scrape") 
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    except Exception as e2:
        print(f"scrape failed: {e2}")

except Exception as e:
    print(f"General error: {e}")