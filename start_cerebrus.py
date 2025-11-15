"""
Cerebrus AI - Complete Startup Script

This script helps you start the complete Cerebrus AI system with voice capabilities.
It will start both the Streamlit app and the webhook server for Agora integration.
"""

import subprocess
import sys
import time
import os
from pathlib import Path
import argparse


def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("📋 Please copy .env.template to .env and configure your API keys")
        return False
    
    # Check for required packages
    try:
        import streamlit
        import uvicorn
        import requests
        import psutil
        print("✅ Required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Please run: pip install streamlit uvicorn requests psutil")
        return False
    
    print("✅ Environment check passed")
    return True


def start_webhook_server(port=8000, host="0.0.0.0"):
    """Start the LLM webhook server"""
    print(f"🚀 Starting webhook server on {host}:{port}...")
    
    try:
        # Start webhook server in background
        webhook_process = subprocess.Popen([
            sys.executable, "manage_webhook.py", "start",
            "--port", str(port),
            "--host", host
        ])
        
        # Wait a moment for it to start
        time.sleep(3)
        
        # Check if it's running
        import requests
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Webhook server started successfully at http://{host}:{port}")
                return webhook_process
            else:
                print(f"❌ Webhook server error: {response.status_code}")
                return None
        except:
            print("❌ Webhook server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting webhook server: {e}")
        return None


def start_streamlit_app(port=8501, host="192.168.1.38"):
    """Start the Streamlit application"""
    print(f"🎨 Starting Streamlit app on {host}:{port}...")
    
    try:
        # Start Streamlit app
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", str(port),
            "--server.address", host,
            "--browser.gatherUsageStats", "false"
        ])
        
        print(f"✅ Streamlit app started at http://{host}:{port}")
        return streamlit_process
        
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {e}")
        return None


def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="Start Cerebrus AI Complete System")
    parser.add_argument("--webhook-port", type=int, default=8000, help="Webhook server port")
    parser.add_argument("--streamlit-port", type=int, default=8501, help="Streamlit app port")
    parser.add_argument("--host", default="192.168.1.38", help="Host address for Streamlit")
    parser.add_argument("--webhook-host", default="0.0.0.0", help="Host for webhook server")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip environment check")
    
    args = parser.parse_args()
    
    print("🧠 Cerebrus AI - Complete System Startup")
    print("=" * 50)
    
    # Check environment
    if not args.skip_env_check and not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return 1
    
    processes = []
    
    try:
        # Start webhook server
        webhook_process = start_webhook_server(args.webhook_port, args.webhook_host)
        if webhook_process:
            processes.append(("Webhook Server", webhook_process))
        else:
            print("⚠️ Continuing without webhook server (voice features will not work)")
        
        # Start Streamlit app
        streamlit_process = start_streamlit_app(args.streamlit_port, args.host)
        if streamlit_process:
            processes.append(("Streamlit App", streamlit_process))
        else:
            print("❌ Failed to start Streamlit app")
            return 1
        
        # Show status
        print("\n🎉 Cerebrus AI System Started Successfully!")
        print("=" * 50)
        print(f"🎨 Streamlit App: http://{args.host}:{args.streamlit_port}")
        if webhook_process:
            print(f"🔗 Webhook Server: http://{args.webhook_host}:{args.webhook_port}")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 50)
        
        # Wait for user interrupt
        try:
            while True:
                time.sleep(1)
                # Check if processes are still running
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"⚠️ {name} has stopped unexpectedly")
        
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Cerebrus AI system...")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    finally:
        # Clean up processes
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except:
                try:
                    process.kill()
                    print(f"⚡ {name} force stopped")
                except:
                    print(f"⚠️ Could not stop {name}")
        
        print("👋 Cerebrus AI system shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())