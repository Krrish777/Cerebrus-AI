#!/usr/bin/env python3
"""
Check Gemini API quota and available models
"""

import os
from dotenv import load_dotenv
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if the API key is working and what models are available"""
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
        
    print(f"✅ API Key found: {gemini_api_key[:10]}...")
    
    # Test different models to see which ones work  
    models_to_test = [
        "gemini-2.0-flash",  # Latest default model
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest", 
        "gemini-pro"
    ]
    
    working_models = []
    
    for model in models_to_test:
        print(f"\n🧪 Testing model: {model}")
        try:
            generator = GoogleGenAIChatGenerator(
                model=model,
                api_key=Secret.from_token(gemini_api_key)
            )
            
            # Try a simple generation
            messages = [ChatMessage.from_user("Say hello")]
            response = generator.run(messages=messages)
            
            if response and response.get("replies"):
                answer = response["replies"][0].text
                print(f"✅ Model {model} works! Response: {answer[:50]}...")
                working_models.append(model)
            else:
                print(f"❌ Model {model} returned empty response")
                
        except Exception as e:
            print(f"❌ Model {model} failed: {str(e)}")
            
        # Add delay between tests
        import time
        time.sleep(5)
    
    print(f"\n📊 Summary:")
    print(f"Working models: {working_models}")
    print(f"Total working: {len(working_models)}")
    
    if working_models:
        print(f"✅ Recommend using: {working_models[0]}")
    else:
        print("❌ No models are working. Check your API key or quota.")
    
    return len(working_models) > 0

if __name__ == "__main__":
    print("🔍 Checking Gemini API Status")
    print("=" * 40)
    check_api_key()