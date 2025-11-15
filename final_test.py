#!/usr/bin/env python3
"""
Final test with the correct ElevenLabs format
"""

import os
import json
import requests
from dotenv import load_dotenv
import time

load_dotenv()

def test_final_elevenlabs():
    """Test the corrected ElevenLabs configuration"""
    
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
    
    # Final corrected payload with proper ElevenLabs format
    payload = {
        "name": f"cerebrus_final_test_{int(time.time())}",
        "properties": {
            "channel": "cerebrus_final_channel",
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
                        "content": "You are Cerebrus AI, a helpful assistant."
                    }
                ],
                "greeting_message": "Hello! I'm Cerebrus AI. How can I help you?",
                "failure_message": "I'm sorry, I don't have information about that.",
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
                    "key": elevenlabs_key,  # 'key' not 'api_key'
                    "voice_id": "pNInz6obpgDQGcFmaJgB",
                    "model_id": "eleven_flash_v2_5",
                    "sample_rate": 24000
                }
            }
        }
    }
    
    print("🚀 Final ElevenLabs Configuration Test")
    print("=" * 50)
    print(f"Agent Name: {payload['name']}")
    print(f"Channel: {payload['properties']['channel']}")
    print(f"TTS: ElevenLabs (Corrected)")
    print(f"Voice ID: {payload['properties']['tts']['params']['voice_id']}")
    print()
    
    # Make request
    url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/join"
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }
    
    print("📤 Sending request...")
    print("Corrected TTS Configuration:")
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
            print("🎉 SUCCESS! ElevenLabs + Ares ASR + Gemini LLM working!")
            agent_id = response_data.get('agent_id')
            print(f"🆔 Agent ID: {agent_id}")
            print(f"📊 Status: {response_data.get('status')}")
            print(f"🕐 Created: {response_data.get('create_ts')}")
            print()
            print("✅ Your configuration is now working:")
            print("   • LLM: OpenAI (fallback for testing)")
            print("   • TTS: ElevenLabs")
            print("   • ASR: Ares")
            print()
            print("🔄 Next: Update to use webhook for RAG integration")
            return agent_id
        else:
            print("❌ Still failed - check error details above.")
            
    except Exception as e:
        print(f"❌ Request Exception: {e}")
    
    return None


if __name__ == "__main__":
    print("🎯 Final Agora Configuration Test")
    print("=" * 60)
    print()
    
    agent_id = test_final_elevenlabs()
    
    if agent_id:
        print(f"\n🏆 VICTORY! Agent successfully created: {agent_id}")
        print("\n🎊 The 400 Bad Request error has been resolved!")
        print("\n📋 Summary of fixes applied:")
        print("   1. Fixed ElevenLabs TTS parameter format")
        print("   2. Used 'key' instead of 'api_key' for ElevenLabs")
        print("   3. Added required 'sample_rate' parameter")
        print("   4. Corrected ASR vendor configuration")
        print("   5. Simplified LLM configuration")
    else:
        print("\n💭 If this test fails, the issue might be:")
        print("   • Invalid API keys")
        print("   • Network connectivity")
        print("   • Agora service issues")