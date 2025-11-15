"""
Webhook Server Management Script

This script provides easy management of the Agora LLM webhook server.
It can start, stop, restart, and check the status of the webhook server.
"""

import os
import sys
import subprocess
import time
import signal
import psutil
import requests
from pathlib import Path
import argparse
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Initialize logger
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class WebhookServerManager:
    """Manager for the webhook server process"""
    
    def __init__(self, port: int = 8000, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.process: Optional[subprocess.Popen] = None
        self.pid_file = Path("webhook_server.pid")
    
    def start(self, reload: bool = False) -> bool:
        """Start the webhook server"""
        
        if self.is_running():
            logger.warning(f"🔄 Webhook server is already running on {self.host}:{self.port}")
            return True
        
        logger.info(f"🚀 Starting webhook server on {self.host}:{self.port}")
        
        try:
            # Prepare command
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.conversational_ai.webhook_server_fixed:app",
                "--host", self.host,
                "--port", str(self.port),
                "--log-level", "info"
            ]
            
            if reload:
                cmd.append("--reload")
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent.parent.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(self.process.pid))
            
            # Wait a moment and check if it started successfully
            time.sleep(2)
            
            if self.process.poll() is None:  # Still running
                if self.health_check():
                    logger.info(f"✅ Webhook server started successfully")
                    logger.info(f"🌐 Server available at: http://{self.host}:{self.port}")
                    logger.info(f"📊 Health check: http://{self.host}:{self.port}/health")
                    logger.info(f"🎙️ Webhook endpoint: http://{self.host}:{self.port}/llm-webhook")
                    return True
                else:
                    logger.warning("⚠️ Server started but health check failed")
                    return False
            else:
                logger.error(f"❌ Failed to start webhook server")
                self.cleanup_pid()
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting webhook server: {e}")
            self.cleanup_pid()
            return False
    
    def stop(self) -> bool:
        """Stop the webhook server"""
        
        logger.info("🛑 Stopping webhook server...")
        
        try:
            # Try to get PID from file
            pid = self.get_pid_from_file()
            
            if pid:
                try:
                    process = psutil.Process(pid)
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                        logger.info("✅ Webhook server stopped gracefully")
                    except psutil.TimeoutExpired:
                        # Force kill if necessary
                        process.kill()
                        logger.info("⚡ Webhook server force stopped")
                    
                    self.cleanup_pid()
                    return True
                    
                except psutil.NoSuchProcess:
                    logger.info("ℹ️ Process not found (already stopped)")
                    self.cleanup_pid()
                    return True
                    
            else:
                logger.info("ℹ️ No PID file found (server not running)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error stopping webhook server: {e}")
            return False
    
    def restart(self, reload: bool = False) -> bool:
        """Restart the webhook server"""
        logger.info("🔄 Restarting webhook server...")
        
        if self.stop():
            time.sleep(1)
            return self.start(reload=reload)
        return False
    
    def status(self) -> dict:
        """Get server status"""
        
        pid = self.get_pid_from_file()
        running = self.is_running()
        healthy = False
        
        if running:
            healthy = self.health_check()
        
        return {
            "running": running,
            "healthy": healthy,
            "pid": pid,
            "host": self.host,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}"
        }
    
    def is_running(self) -> bool:
        """Check if server is running"""
        
        pid = self.get_pid_from_file()
        if not pid:
            return False
        
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
    
    def health_check(self) -> bool:
        """Perform health check"""
        
        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/health",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Health check passed: {data.get('status')}")
                logger.info(f"🧠 RAG system: {data.get('rag_system')}")
                return True
            else:
                logger.warning(f"⚠️ Health check failed with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ Health check failed: Connection refused")
            return False
        except requests.exceptions.Timeout:
            logger.warning("⚠️ Health check failed: Timeout")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Health check failed: {e}")
            return False
    
    def get_pid_from_file(self) -> Optional[int]:
        """Get PID from file"""
        
        try:
            if self.pid_file.exists():
                with open(self.pid_file, 'r') as f:
                    return int(f.read().strip())
        except:
            pass
        return None
    
    def cleanup_pid(self):
        """Clean up PID file"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
        except:
            pass
    
    def logs(self, lines: int = 50):
        """Show server logs (if running with log output)"""
        
        if not self.is_running():
            logger.warning("Server is not running")
            return
        
        # For now, just show status since we're running as subprocess
        status = self.status()
        logger.info(f"Server Status: {status}")
        
        # Test endpoint
        try:
            response = requests.get(f"http://{self.host}:{self.port}/")
            logger.info(f"Root endpoint response: {response.json()}")
        except Exception as e:
            logger.error(f"Error testing endpoint: {e}")


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="Manage Agora LLM Webhook Server")
    parser.add_argument("command", choices=["start", "stop", "restart", "status", "logs"], 
                       help="Command to execute")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--lines", type=int, default=50, help="Number of log lines to show")
    
    args = parser.parse_args()
    
    manager = WebhookServerManager(port=args.port, host=args.host)
    
    if args.command == "start":
        success = manager.start(reload=args.reload)
        if not success:
            sys.exit(1)
    
    elif args.command == "stop":
        success = manager.stop()
        if not success:
            sys.exit(1)
    
    elif args.command == "restart":
        success = manager.restart(reload=args.reload)
        if not success:
            sys.exit(1)
    
    elif args.command == "status":
        status = manager.status()
        print(f"\n📊 Webhook Server Status:")
        print(f"Running: {'✅ Yes' if status['running'] else '❌ No'}")
        print(f"Healthy: {'✅ Yes' if status['healthy'] else '❌ No'}")
        print(f"PID: {status['pid'] or 'N/A'}")
        print(f"Address: {status['url']}")
        
        if status['running'] and status['healthy']:
            print(f"\n🎙️ Webhook endpoint: {status['url']}/llm-webhook")
            print(f"📊 Health check: {status['url']}/health")
        print()
    
    elif args.command == "logs":
        manager.logs(lines=args.lines)


if __name__ == "__main__":
    main()