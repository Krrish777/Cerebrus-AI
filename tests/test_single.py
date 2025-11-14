#!/usr/bin/env python3
"""
Simple single API call test to check if new key works
"""

import os
from dotenv import load_dotenv
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

load_dotenv()

def test_single_call():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    print(f"✅ Testing API key: {gemini_api_key[:12]}...")
    
    try:
        # Try the most basic model
        generator = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",  # Use the working model
            api_key=Secret.from_token(gemini_api_key)
        )
        
        messages = [ChatMessage.from_user("Hello, reply with just 'Hi'")]
        print("🔍 Making single API call...")
        response = generator.run(messages=messages)
        
        if response and response.get("replies"):
            answer = response["replies"][0].text
            print(f"✅ SUCCESS! Response: {answer}")
            return True
        else:
            print("❌ Empty response received")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_single_call()