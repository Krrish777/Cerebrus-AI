#!/usr/bin/env python3
"""
Test script to verify Agora AI payload format
"""

import os
import json
from src.conversational_ai.agora_ai import *

def test_agora_payload():
    """Test the payload format that will be sent to Agora API"""
    
    print("🧪 Testing Agora AI Payload Format...")
    print("-" * 50)
    
    # Create test credentials
    credentials = AgoraCredentials(
        app_id="bd7a58fc67a74713b925d43a1cf2382b",  # Your app ID
        customer_id=os.getenv("AGORA_CUSTOMER_ID", "test_customer"),
        customer_secret=os.getenv("AGORA_CUSTOMER_SECRET", "test_secret"),
        rtc_token="test_token"
    )
    
    # Create agent configuration
    agent_config = create_default_agent_config(
        name="test_agent_payload",
        channel="test_channel",
        llm_api_key=os.getenv("GEMINI_API_KEY", "test_llm_key"),
        tts_api_key=os.getenv("ELEVENLABS_API_KEY", "test_tts_key")
    )
    
    # Create Agora AI instance
    agora_ai = AgoraConversationalAI(credentials=credentials)
    
    # Create the properties that would be sent to the API
    properties = {
        "channel": agent_config.channel,
        "token": credentials.rtc_token,
        "agent_rtc_uid": agent_config.agent_rtc_uid,
        "remote_rtc_uids": agent_config.remote_rtc_uids,
        "enable_string_uid": agent_config.enable_string_uid,
        "idle_timeout": agent_config.idle_timeout,
        
        # LLM Configuration
        "llm": {
            "url": agent_config.llm.url,  # Use direct URL for testing
            "api_key": agent_config.llm.api_key,
            "system_messages": [
                {
                    "role": "system",
                    "content": agent_config.llm.system_message
                }
            ],
            "greeting_message": agent_config.llm.greeting_message,
            "failure_message": agent_config.llm.failure_message,
            "max_history": agent_config.llm.max_history,
            "params": {
                "model": agent_config.llm.model
            }
        },
        
        # ASR Configuration
        "asr": agent_config.asr.get_asr_config(),
        
        # TTS Configuration
        "tts": agent_config.tts.get_vendor_config(),
        
        # Advanced Features
        "advanced_features": {
            "enable_aivad": True
        },
        
        # Turn Detection Configuration (optional)
        "turn_detection": {
            "interrupt_mode": agent_config.turn_detection.interrupt_mode
        }
    }
    
    # Request payload
    payload = {
        "name": agent_config.name,
        "properties": properties
    }
    
    print("📋 Generated Payload:")
    print(json.dumps(payload, indent=2))
    print()
    
    print("🔍 Key Configuration Details:")
    print(f"   • App ID: {credentials.app_id}")
    print(f"   • Agent Name: {agent_config.name}")
    print(f"   • Channel: {agent_config.channel}")
    print(f"   • LLM URL: {agent_config.llm.url}")
    print(f"   • LLM Model: {agent_config.llm.model}")
    print(f"   • TTS Vendor: {agent_config.tts.vendor}")
    print(f"   • TTS Voice ID: {agent_config.tts.voice_id}")
    print(f"   • ASR Vendor: {agent_config.asr.vendor}")
    print(f"   • ASR Language: {agent_config.asr.language}")
    print()
    
    print("🔗 API Endpoint:")
    endpoint = f"/api/conversational-ai-agent/v2/projects/{credentials.app_id}/join"
    print(f"   POST https://api.agora.io{endpoint}")
    print()
    
    print("🔐 Authorization Header:")
    print(f"   {credentials.get_auth_header()}")
    print()
    
    # Validate required fields
    print("✅ Validation Checks:")
    required_fields = ['name', 'properties']
    for field in required_fields:
        if field in payload:
            print(f"   ✓ {field}: Present")
        else:
            print(f"   ✗ {field}: Missing")
    
    required_properties = ['channel', 'token', 'llm', 'tts', 'asr']
    for field in required_properties:
        if field in properties:
            print(f"   ✓ properties.{field}: Present")
        else:
            print(f"   ✗ properties.{field}: Missing")
    
    print()
    print("🎯 Potential Issues to Check:")
    print("   1. Ensure RTC token is valid and not expired")
    print("   2. Verify API credentials (customer_id, customer_secret)")
    print("   3. Check if channel name follows Agora naming conventions")
    print("   4. Confirm LLM URL is accessible from Agora servers")
    print("   5. Validate ElevenLabs API key and voice_id")
    print("   6. Ensure Gemini API key has proper permissions")

if __name__ == "__main__":
    test_agora_payload()