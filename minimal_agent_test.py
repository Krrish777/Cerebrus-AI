#!/usr/bin/env python3
"""
Minimal Agora Agent Creation Test

This script creates a minimal agent configuration based on the exact Agora API specification.
"""

import os
import json
import requests
from dotenv import load_dotenv
import time

load_dotenv()

def create_minimal_agent():
    """Create agent with absolute minimal configuration following Agora docs exactly"""
    
    # Load credentials
    app_id = os.getenv("AGORA_APP_ID")
    customer_id = os.getenv("AGORA_CUSTOMER_ID") 
    customer_secret = os.getenv("AGORA_CUSTOMER_SECRET")
    rtc_token = os.getenv("AGORA_RTC_TOKEN")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY", "test_key")
    
    # Create auth header
    import base64
    credentials_str = f"{customer_id}:{customer_secret}"
    encoded_credentials = base64.b64encode(credentials_str.encode()).decode()
    auth_header = f"Basic {encoded_credentials}"
    
    # Minimal payload following the exact Agora documentation
    payload = {
        "name": f"cerebrus_agent_{int(time.time())}",
        "properties": {
            "channel": "cerebrus_test_channel",
            "token": rtc_token,
            "agent_rtc_uid": "0",
            "remote_rtc_uids": ["*"],
            "enable_string_uid": False,
            "idle_timeout": 120,
            "llm": {
                "url": "https://api.openai.com/v1/chat/completions",
                "api_key": openai_key,
                "system_messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful chatbot."
                    }
                ],
                "greeting_message": "Hello, how can I help you?",
                "failure_message": "Sorry, I don't know how to answer this question.",
                "max_history": 10,
                "params": {
                    "model": "gpt-4o-mini"
                }
            },
            "asr": {
                "language": "en-US"
            },
            "tts": {
                "vendor": "elevenlabs",
                "voice_id": "pNInz6obpgDQGcFmaJgB",
                "api_key": elevenlabs_key
            }
        }
    }
    
    print("🧪 Testing Minimal Agent Configuration")
    print("=" * 50)
    print(f"Agent Name: {payload['name']}")
    print(f"Channel: {payload['properties']['channel']}")
    print()
    
    # Make request
    url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/join"
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }
    
    print("📤 Sending request...")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Payload size: {len(json.dumps(payload))} characters")
    print()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.content:
            try:
                response_data = response.json()
                print("Response Data:")
                print(json.dumps(response_data, indent=2))
            except:
                print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Agent created successfully!")
            agent_id = response_data.get('agent_id')
            if agent_id:
                print(f"🆔 Agent ID: {agent_id}")
                return agent_id
        else:
            print("❌ FAILED! See response above for details.")
            
    except Exception as e:
        print(f"❌ Request Exception: {e}")
    
    return None


def test_with_microsoft_tts():
    """Test with Microsoft TTS instead of ElevenLabs"""
    
    # Load credentials
    app_id = os.getenv("AGORA_APP_ID")
    customer_id = os.getenv("AGORA_CUSTOMER_ID") 
    customer_secret = os.getenv("AGORA_CUSTOMER_SECRET")
    rtc_token = os.getenv("AGORA_RTC_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY", "test_key")
    
    # Create auth header
    import base64
    credentials_str = f"{customer_id}:{customer_secret}"
    encoded_credentials = base64.b64encode(credentials_str.encode()).decode()
    auth_header = f"Basic {encoded_credentials}"
    
    # Payload with Microsoft TTS (as shown in Agora examples)
    payload = {
        "name": f"cerebrus_ms_tts_{int(time.time())}",
        "properties": {
            "channel": "cerebrus_test_channel_ms",
            "token": rtc_token,
            "agent_rtc_uid": "0",
            "remote_rtc_uids": ["*"],
            "enable_string_uid": False,
            "idle_timeout": 120,
            "llm": {
                "url": "https://api.openai.com/v1/chat/completions",
                "api_key": openai_key,
                "system_messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful chatbot."
                    }
                ],
                "greeting_message": "Hello, how can I help you?",
                "failure_message": "Sorry, I don't know how to answer this question.",
                "max_history": 10,
                "params": {
                    "model": "gpt-4o-mini"
                }
            },
            "asr": {
                "language": "en-US"
            },
            "tts": {
                "vendor": "microsoft",
                "params": {
                    "key": "test_microsoft_key",
                    "region": "eastus",
                    "voice_name": "en-US-AndrewMultilingualNeural"
                }
            }
        }
    }
    
    print("🧪 Testing with Microsoft TTS")
    print("=" * 50)
    print(f"Agent Name: {payload['name']}")
    print(f"Channel: {payload['properties']['channel']}")
    print(f"TTS: Microsoft Azure")
    print()
    
    # Make request
    url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/join"
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }
    
    print("📤 Sending request...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.content:
            try:
                response_data = response.json()
                print("Response Data:")
                print(json.dumps(response_data, indent=2))
            except:
                print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print("✅ SUCCESS with Microsoft TTS!")
            return response_data.get('agent_id')
        else:
            print("❌ FAILED with Microsoft TTS")
            
    except Exception as e:
        print(f"❌ Request Exception: {e}")
    
    return None


if __name__ == "__main__":
    print("🚀 Agora Agent Creation Test")
    print("=" * 60)
    print()
    
    # Test 1: Try minimal ElevenLabs configuration
    agent_id = create_minimal_agent()
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Try Microsoft TTS if first test fails
    if not agent_id:
        agent_id = test_with_microsoft_tts()
    
    if agent_id:
        print(f"\n🎉 SUCCESS! Agent created with ID: {agent_id}")
    else:
        print("\n😞 Both tests failed. Check the error messages above for clues.")