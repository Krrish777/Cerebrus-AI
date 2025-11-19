#!/usr/bin/env python3
"""
Agora AI Diagnostics and Fix Script

This script will help diagnose and fix the 400 Bad Request error when creating Agora agents.
"""

import os
import json
import requests
from src.conversational_ai.agora_ai import *
from dotenv import load_dotenv

load_dotenv()

def diagnose_agora_credentials():
    """Test Agora API credentials and connectivity"""
    
    print("🔍 Diagnosing Agora API Credentials...")
    print("=" * 60)
    
    # Load credentials from environment
    app_id = os.getenv("AGORA_APP_ID")
    customer_id = os.getenv("AGORA_CUSTOMER_ID") 
    customer_secret = os.getenv("AGORA_CUSTOMER_SECRET")
    rtc_token = os.getenv("AGORA_RTC_TOKEN")
    
    print("📋 Loaded Credentials:")
    print(f"   • App ID: {app_id}")
    print(f"   • Customer ID: {customer_id}")
    print(f"   • Customer Secret: {'*' * (len(customer_secret) if customer_secret else 0)}")
    print(f"   • RTC Token: {rtc_token[:20]}..." if rtc_token else "   • RTC Token: Not set")
    print()
    
    if not all([app_id, customer_id, customer_secret]):
        print("❌ Missing required credentials!")
        return False
    
    # Create credentials object
    credentials = AgoraCredentials(
        app_id=app_id,
        customer_id=customer_id,
        customer_secret=customer_secret,
        rtc_token=rtc_token
    )
    
    # Test API connectivity by listing agents
    print("🔌 Testing API Connectivity...")
    try:
        url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/agents"
        headers = {
            "Authorization": credentials.get_auth_header(),
            "Content-Type": "application/json"
        }
        
        print(f"   • Endpoint: {url}")
        print(f"   • Authorization: {credentials.get_auth_header()}")
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"   • Response Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ API connectivity successful!")
            data = response.json()
            print(f"   • Active agents: {data.get('data', {}).get('count', 0)}")
        elif response.status_code == 401:
            print("   ❌ Authentication failed - check customer_id and customer_secret")
            return False
        elif response.status_code == 404:
            print("   ❌ App ID not found - check AGORA_APP_ID")
            return False
        else:
            print(f"   ⚠️ Unexpected response: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    print()
    return True


def test_agent_creation():
    """Test creating an agent with minimal configuration"""
    
    print("🤖 Testing Agent Creation...")
    print("=" * 60)
    
    # Load credentials
    credentials = AgoraCredentials(
        app_id=os.getenv("AGORA_APP_ID"),
        customer_id=os.getenv("AGORA_CUSTOMER_ID"),
        customer_secret=os.getenv("AGORA_CUSTOMER_SECRET"),
        rtc_token=os.getenv("AGORA_RTC_TOKEN")
    )
    
    # Create minimal configuration for testing
    test_config = {
        "name": f"test_agent_{int(time.time())}",
        "properties": {
            "channel": "test_channel_diagnostic",
            "token": credentials.rtc_token or "006bd7a58fc67a74713b925d43a1cf2382bIAA...",  # Placeholder
            "agent_rtc_uid": "0",
            "remote_rtc_uids": ["*"],
            "enable_string_uid": False,
            "idle_timeout": 120,
            "llm": {
                "url": "https://api.openai.com/v1/chat/completions",
                "api_key": os.getenv("OPENAI_API_KEY", "test_key"),
                "system_messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant."
                    }
                ],
                "greeting_message": "Hello!",
                "failure_message": "Sorry, I can't help with that.",
                "max_history": 10,
                "params": {
                    "model": "gpt-4o-mini"
                }
            },
            "tts": {
                "vendor": "elevenlabs",
                "voice_id": "pNInz6obpgDQGcFmaJgB",
                "api_key": os.getenv("ELEVENLABS_API_KEY", "test_key")
            },
            "asr": {
                "language": "en-US",
                "vendor": "ares"
            }
        }
    }
    
    print("📤 Sending test request...")
    print(f"Payload: {json.dumps(test_config, indent=2)}")
    print()
    
    url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{credentials.app_id}/join"
    headers = {
        "Authorization": credentials.get_auth_header(),
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=test_config, timeout=30)
        
        print("📥 Response:")
        print(f"   • Status Code: {response.status_code}")
        print(f"   • Headers: {dict(response.headers)}")
        
        if response.content:
            try:
                response_data = response.json()
                print(f"   • Response Data: {json.dumps(response_data, indent=2)}")
            except:
                print(f"   • Response Text: {response.text}")
        
        if response.status_code == 200:
            print("   ✅ Agent created successfully!")
            return True
        elif response.status_code == 400:
            print("   ❌ 400 Bad Request - Invalid payload format")
            if response.content:
                error_details = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                print(f"   Error details: {error_details}")
        elif response.status_code == 401:
            print("   ❌ 401 Unauthorized - Authentication failed") 
        elif response.status_code == 403:
            print("   ❌ 403 Forbidden - Insufficient permissions")
        else:
            print(f"   ❌ Unexpected error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False
        
    return False


def suggest_fixes():
    """Suggest potential fixes based on common issues"""
    
    print("💡 Suggested Fixes:")
    print("=" * 60)
    
    print("1. 🔑 RTC Token Issues:")
    print("   • Generate a new RTC token for your channel")
    print("   • Ensure token hasn't expired")
    print("   • Use channel name that matches your token")
    print()
    
    print("2. 🔐 API Key Issues:")
    print("   • Verify ElevenLabs API key is valid")
    print("   • Check OpenAI API key has sufficient credits")
    print("   • Ensure all API keys have proper permissions")
    print()
    
    print("3. 🌐 Network Issues:")
    print("   • Check if webhook server is accessible from internet")
    print("   • Ensure webhook URL returns proper responses")
    print("   • Consider using ngrok for local development")
    print()
    
    print("4. 📝 Payload Issues:")
    print("   • Simplify configuration to minimal required fields")
    print("   • Remove optional parameters that might cause validation errors")
    print("   • Use standard voice IDs and model names")
    print()
    
    print("5. 🔄 Alternative Approach:")
    print("   • Use OpenAI directly instead of custom LLM webhook")
    print("   • Test with Microsoft Azure TTS first")
    print("   • Start with basic configuration, then add features")


if __name__ == "__main__":
    import time
    
    print("🚀 Agora AI Diagnostic Tool")
    print("=" * 60)
    print()
    
    # Step 1: Check credentials
    if diagnose_agora_credentials():
        print()
        
        # Step 2: Test agent creation
        if not test_agent_creation():
            print()
            suggest_fixes()
    else:
        print("\n❌ Fix credentials first before proceeding!")
        suggest_fixes()