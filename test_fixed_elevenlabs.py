#!/usr/bin/env python3
"""
Test the fixed ElevenLabs TTS configuration
"""

import os
import json
import requests
from dotenv import load_dotenv
import time

load_dotenv()

def test_fixed_elevenlabs():
    """Test the fixed ElevenLabs configuration"""
    
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
    
    # Fixed payload with corrected ElevenLabs format
    payload = {
        "name": f"cerebrus_elevenlabs_fixed_{int(time.time())}",
        "properties": {
            "channel": "cerebrus_test_channel_fixed",
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
                "language": "en-US",
                "vendor": "ares"
            },
            "tts": {
                "vendor": "elevenlabs",
                "params": {
                    "voice_id": "pNInz6obpgDQGcFmaJgB",
                    "model_id": "eleven_flash_v2_5",
                    "api_key": elevenlabs_key
                }
            }
        }
    }
    
    print("🧪 Testing Fixed ElevenLabs Configuration")
    print("=" * 50)
    print(f"Agent Name: {payload['name']}")
    print(f"Channel: {payload['properties']['channel']}")
    print(f"TTS: ElevenLabs (Fixed Format)")
    print(f"Voice ID: {payload['properties']['tts']['params']['voice_id']}")
    print()
    
    # Make request
    url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/join"
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }
    
    print("📤 Sending request...")
    print("TTS Configuration:")
    print(json.dumps(payload['properties']['tts'], indent=2))
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
            print("✅ SUCCESS! ElevenLabs TTS is working!")
            agent_id = response_data.get('agent_id')
            print(f"🆔 Agent ID: {agent_id}")
            print(f"📊 Status: {response_data.get('status')}")
            return agent_id
        else:
            print("❌ FAILED! Check error details above.")
            
    except Exception as e:
        print(f"❌ Request Exception: {e}")
    
    return None


if __name__ == "__main__":
    print("🚀 Testing Fixed ElevenLabs Configuration")
    print("=" * 60)
    print()
    
    agent_id = test_fixed_elevenlabs()
    
    if agent_id:
        print(f"\n🎉 SUCCESS! ElevenLabs agent created: {agent_id}")
        print("\n✅ The 400 error has been fixed!")
        print("🔧 Issue was: TTS parameters need to be nested under 'params' object")
    else:
        print("\n😞 Still having issues. Check the error details above.")