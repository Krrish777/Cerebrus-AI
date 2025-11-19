#!/usr/bin/env python3
"""
Cerebrus AI Voice Chat Launcher

This script starts both the webhook server and the Streamlit app for one-click voice conversations.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_webhook_server():
    """Check if webhook server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_webhook_server():
    """Start the webhook server in the background"""
    print("🚀 Starting webhook server...")
    
    # Start webhook server
    webhook_cmd = [
        sys.executable, "-m", "uvicorn",
        "src.conversational_ai.webhook_server_fixed:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    webhook_process = subprocess.Popen(
        webhook_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for webhook server to start
    print("⏳ Waiting for webhook server to start...")
    for i in range(10):
        if check_webhook_server():
            print("✅ Webhook server started successfully!")
            return webhook_process
        time.sleep(1)
        print(f"   Checking... ({i+1}/10)")
    
    print("⚠️ Webhook server may not have started properly, but continuing...")
    return webhook_process

def start_streamlit_app():
    """Start the Streamlit app"""
    print("🎨 Starting Streamlit voice chat interface...")
    
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/conversational_ai/simple_voice_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ]
    
    streamlit_process = subprocess.Popen(streamlit_cmd)
    return streamlit_process

def main():
    """Main launcher function"""
    
    print("🎙️ Cerebrus AI Voice Chat Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src/conversational_ai/simple_voice_app.py").exists():
        print("❌ Please run this script from the Cerebrus AI root directory")
        sys.exit(1)
    
    # Check environment variables
    required_vars = [
        "AGORA_APP_ID",
        "AGORA_CUSTOMER_ID", 
        "AGORA_CUSTOMER_SECRET",
        "ELEVENLABS_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        sys.exit(1)
    
    print("✅ Environment variables loaded")
    
    processes = []
    
    try:
        # Start webhook server if not running
        if not check_webhook_server():
            webhook_process = start_webhook_server()
            processes.append(("webhook", webhook_process))
        else:
            print("✅ Webhook server already running")
        
        # Start Streamlit app
        streamlit_process = start_streamlit_app()
        processes.append(("streamlit", streamlit_process))
        
        print("\n🎉 Cerebrus AI Voice Chat is ready!")
        print("=" * 50)
        print("🌐 Streamlit App: http://localhost:8501")
        print("🔗 Webhook Server: http://localhost:8000")
        print("📋 Health Check: http://localhost:8000/health")
        print("\n💡 Click 'Start Voice Chat' in the web interface to begin!")
        print("\n⚡ Press Ctrl+C to stop all services")
        
        # Wait for processes
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    print(f"⚠️ {name} process stopped unexpectedly")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Cerebrus AI Voice Chat...")
        
        # Stop all processes
        for name, process in processes:
            try:
                print(f"   Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        print("✅ All services stopped. Goodbye!")

if __name__ == "__main__":
    main()