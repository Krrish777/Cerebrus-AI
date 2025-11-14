#!/usr/bin/env python3
"""
Setup script for AssemblyAI API key configuration.
"""

import os

def setup_api_key():
    """Interactive setup for AssemblyAI API key."""
    print("🔑 ASSEMBLYAI API KEY SETUP")
    print("=" * 40)
    
    # Check if key already exists
    existing_key = os.getenv("ASSEMBLYAI_API_KEY")
    if existing_key:
        print(f"✅ API key already set: {existing_key[:10]}...{existing_key[-4:]}")
        response = input("Do you want to update it? (y/N): ").lower().strip()
        if response != 'y':
            return
    
    print("\n📋 To get your API key:")
    print("1. Go to: https://www.assemblyai.com/")
    print("2. Sign up/Login to your account")
    print("3. Go to your dashboard")
    print("4. Copy your API key")
    
    print("\n" + "=" * 40)
    api_key = input("Enter your AssemblyAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    if len(api_key) < 10:
        print("❌ API key seems too short. Please check and try again.")
        return
    
    # Show commands to set environment variable
    print("\n💡 To set your API key, run one of these commands:")
    print(f"PowerShell: $env:ASSEMBLYAI_API_KEY='{api_key}'")
    print(f"CMD: set ASSEMBLYAI_API_KEY={api_key}")
    
    print("\n✅ Copy and run the appropriate command for your shell.")
    print("Then run: python tests/test_audio_real_api.py")

if __name__ == "__main__":
    setup_api_key()