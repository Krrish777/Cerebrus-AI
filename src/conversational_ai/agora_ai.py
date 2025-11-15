"""
Agora Conversational AI Integration for Cerebrus AI

This module provides comprehensive integration with Agora's Conversational AI Engine,
enabling real-time voice interactions with AI agents that can access your RAG knowledge base.

Features:
- Ultra-low latency voice conversations (650ms response time)
- Custom LLM integration with your RAG system
- Advanced audio features (VAD, noise reduction, interruption handling)
- Multi-platform support (Web, iOS, Android)
- Resilient to network issues
"""

import os
import base64
import json
import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
import requests
import websockets
from urllib.parse import urljoin

from src.core.logging import CustomLogger

# Initialize logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


@dataclass
class AgoraCredentials:
    """Agora API credentials"""
    app_id: str
    customer_id: str
    customer_secret: str
    rtc_token: Optional[str] = None
    
    def get_auth_header(self) -> str:
        """Generate Base64-encoded authorization header"""
        credentials = f"{self.customer_id}:{self.customer_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"


@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    url: str = "https://api.openai.com/v1/chat/completions"  # Default OpenAI-compatible endpoint
    api_key: str = os.getenv("OPENAI_API_KEY", "")  # Default to OpenAI for fallback
    model: str = "gpt-4o-mini"  # Default OpenAI model
    system_message: str = "You are Cerebrus AI, a helpful AI assistant with access to a comprehensive knowledge base. Provide accurate, conversational responses optimized for voice interaction. Keep responses concise and natural."
    greeting_message: str = "Hello! I'm Cerebrus AI. How can I help you explore your knowledge base today?"
    failure_message: str = "I apologize, but I don't have information about that in my current knowledge base. Could you try asking something else?"
    max_history: int = 10
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech (following official Agora API structure)"""
    vendor: str = "elevenlabs"  # elevenlabs, microsoft, openai
    
    # ElevenLabs specific parameters
    api_key: str = os.getenv("ELEVENLABS_API_KEY", os.getenv("ELEVEN_LABS_API_KEY", ""))
    voice_id: str = "pNInz6obpgDQGcFmaJgB"  # Adam - Natural, professional voice
    model_id: str = "eleven_flash_v2_5"  # Fast, high-quality model
    stability: float = 0.5
    similarity_boost: float = 0.8
    style: float = 0.2
    use_speaker_boost: bool = True
    
    # Microsoft Azure specific parameters  
    azure_api_key: str = os.getenv("AZURE_TTS_API_KEY", "")
    azure_region: str = "eastus"
    voice_name: str = "en-US-JennyNeural"  # For Microsoft Azure
    
    # OpenAI specific parameters
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    openai_model: str = "tts-1"
    
    # Common parameters
    speed: float = 1.0
    
    def get_vendor_config(self) -> Dict[str, Any]:
        """Get vendor-specific configuration"""
        if self.vendor == "elevenlabs":
            config = {
                "vendor": "elevenlabs",
                "params": {
                    "voice_id": self.voice_id,
                    "model_id": self.model_id,
                    "sample_rate": 24000  # Standard sample rate for ElevenLabs
                }
            }
            # ElevenLabs uses 'key' not 'api_key'
            if self.api_key:
                config["params"]["key"] = self.api_key
            return config
            
        elif self.vendor == "microsoft":
            config = {
                "vendor": "microsoft",
                "params": {
                    "voice_name": self.voice_name,
                    "rate": f"+{int((self.speed - 1) * 100)}%" if self.speed != 1.0 else "+0%"
                }
            }
            # Add required parameters for Microsoft Azure
            if self.azure_api_key:
                config["params"]["key"] = self.azure_api_key  # Use 'key' not 'api_key'
            if self.azure_region:
                config["params"]["region"] = self.azure_region
            return config
            
        elif self.vendor == "openai":
            config = {
                "vendor": "openai",
                "params": {
                    "voice": self.openai_voice,
                    "model": self.openai_model,
                    "speed": self.speed
                }
            }
            # Only add API key if provided
            if self.openai_api_key:
                config["params"]["api_key"] = self.openai_api_key
            return config
        else:
            raise ValueError(f"Unsupported TTS vendor: {self.vendor}")


@dataclass
class ASRConfig:
    """Configuration for Automatic Speech Recognition"""
    language: str = "en-US"
    vendor: str = "ares"  # Ares ASR vendor
    
    def get_asr_config(self) -> Dict[str, Any]:
        """Get ASR configuration in Agora format"""
        config: Dict[str, Any] = {
            "language": self.language
        }
        
        # Add vendor if it's supported (Ares is a valid vendor)
        if self.vendor == "ares":
            config["vendor"] = "ares"
            # Note: Ares ASR typically doesn't require additional API keys 
            # as it's integrated into Agora's infrastructure
        
        return config


@dataclass
class TurnDetectionConfig:
    """Configuration for turn detection and interruption handling"""
    interrupt_mode: str = "interrupt"  # interrupt, append, ignore
    interrupt_duration_ms: int = 500
    silence_duration_ms: int = 800
    threshold: float = 0.3
    eagerness: float = 0.5
    enable_interrupt: bool = True


@dataclass
class AgentConfig:
    """Complete agent configuration"""
    name: str
    channel: str
    agent_rtc_uid: str = "0"
    remote_rtc_uids: List[str] = field(default_factory=lambda: ["*"])
    enable_string_uid: bool = False
    idle_timeout: int = 120
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)
    enable_metrics: bool = True
    enable_data_channel: bool = True
    enable_error_messages: bool = True


class AgoraConversationalAI:
    """
    Main class for Agora Conversational AI integration with Cerebrus AI RAG system
    """
    
    def __init__(
        self,
        credentials: AgoraCredentials,
        rag_generator=None,
        base_url: str = "https://api.agora.io"
    ):
        """
        Initialize Agora Conversational AI
        
        Args:
            credentials: Agora API credentials
            rag_generator: Optional RAG generator for knowledge base queries
            base_url: Agora API base URL
        """
        self.credentials = credentials
        self.rag_generator = rag_generator
        self.base_url = base_url
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        
        # API endpoints
        self.endpoints = {
            "join": f"/api/conversational-ai-agent/v2/projects/{credentials.app_id}/join",
            "leave": f"/api/conversational-ai-agent/v2/projects/{credentials.app_id}/agents/{{agent_id}}/leave",
            "query": f"/api/conversational-ai-agent/v2/projects/{credentials.app_id}/agents/{{agent_id}}/query",
            "update": f"/api/conversational-ai-agent/v2/projects/{credentials.app_id}/agents/{{agent_id}}/update",
            "list": f"/api/conversational-ai-agent/v2/projects/{credentials.app_id}/agents"
        }
        
        logger.info("🤖 Agora Conversational AI initialized")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make authenticated request to Agora API"""
        
        url = urljoin(self.base_url, endpoint)
        headers = {
            "Authorization": self.credentials.get_auth_header(),
            "Content-Type": "application/json"
        }
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, **kwargs)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data, **kwargs)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            if response.content:
                return response.json()
            else:
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Agora API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error details: {error_detail}")
                except:
                    pass
            raise
    
    def _create_custom_llm_handler(self, agent_config: AgentConfig) -> str:
        """
        Create a custom LLM webhook URL that integrates with RAG system
        
        In a production environment, this would be a webhook endpoint on your server
        that receives LLM requests and processes them with your RAG system.
        """
        # Use the local webhook server for RAG integration
        webhook_port = os.getenv("WEBHOOK_PORT", "8000")
        webhook_url = f"http://localhost:{webhook_port}/llm-webhook/{agent_config.name}"
        
        logger.info(f"📡 Custom LLM webhook URL: {webhook_url}")
        return webhook_url
    
    async def start_agent(
        self,
        agent_config: AgentConfig,
        use_rag: bool = True
    ) -> str:
        """
        Start a conversational AI agent
        
        Args:
            agent_config: Agent configuration
            use_rag: Whether to integrate with RAG system
            
        Returns:
            Agent ID for the started agent
        """
        
        logger.info(f"🚀 Starting Agora AI agent: {agent_config.name}")
        logger.info(f"   • Channel: {agent_config.channel}")
        logger.info(f"   • RAG Integration: {use_rag}")
        
        # Prepare agent properties
        properties = {
            "channel": agent_config.channel,
            "token": self.credentials.rtc_token,
            "agent_rtc_uid": agent_config.agent_rtc_uid,
            "remote_rtc_uids": agent_config.remote_rtc_uids,
            "enable_string_uid": agent_config.enable_string_uid,
            "idle_timeout": agent_config.idle_timeout,
            
            # LLM Configuration
            "llm": {
                "url": self._create_custom_llm_handler(agent_config) if use_rag else agent_config.llm.url,
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
        
        try:
            response = self._make_request("POST", self.endpoints["join"], payload)
            
            agent_id = response.get("agent_id")
            if not agent_id:
                raise ValueError("No agent_id returned from Agora API")
            
            # Store agent information
            self.active_agents[agent_id] = {
                "config": agent_config,
                "started_at": datetime.now(),
                "status": response.get("status", "UNKNOWN"),
                "use_rag": use_rag
            }
            
            logger.info(f"✅ Agent started successfully")
            logger.info(f"   • Agent ID: {agent_id}")
            logger.info(f"   • Status: {response.get('status')}")
            logger.info(f"   • Create Time: {response.get('create_ts')}")
            
            return agent_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start agent: {e}")
            raise
    
    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a running conversational AI agent
        
        Args:
            agent_id: ID of the agent to stop
            
        Returns:
            True if successful, False otherwise
        """
        
        logger.info(f"🛑 Stopping Agora AI agent: {agent_id}")
        
        try:
            endpoint = self.endpoints["leave"].format(agent_id=agent_id)
            response = self._make_request("POST", endpoint)
            
            # Remove from active agents
            if agent_id in self.active_agents:
                agent_info = self.active_agents[agent_id]
                duration = datetime.now() - agent_info["started_at"]
                logger.info(f"   • Agent ran for: {duration}")
                del self.active_agents[agent_id]
            
            logger.info("✅ Agent stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop agent: {e}")
            return False
    
    async def query_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Query the status of a running agent"""
        
        try:
            endpoint = self.endpoints["query"].format(agent_id=agent_id)
            response = self._make_request("GET", endpoint)
            
            logger.debug(f"Agent {agent_id} status: {response}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to query agent status: {e}")
            raise
    
    async def update_agent_config(
        self,
        agent_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update configuration of a running agent"""
        
        logger.info(f"🔄 Updating agent configuration: {agent_id}")
        
        try:
            endpoint = self.endpoints["update"].format(agent_id=agent_id)
            response = self._make_request("PUT", endpoint, updates)
            
            logger.info("✅ Agent configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update agent: {e}")
            return False
    
    async def list_active_agents(self) -> List[Dict[str, Any]]:
        """List all active agents"""
        
        try:
            response = self._make_request("GET", self.endpoints["list"])
            
            agents = response.get("agents", [])
            logger.info(f"📊 Found {len(agents)} active agents")
            
            return agents
            
        except Exception as e:
            logger.error(f"❌ Failed to list agents: {e}")
            return []
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get local information about an agent"""
        return self.active_agents.get(agent_id)
    
    def get_all_local_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all locally tracked agents"""
        return self.active_agents.copy()
    
    async def stop_all_agents(self) -> int:
        """Stop all active agents"""
        
        logger.info("🛑 Stopping all active agents...")
        
        stopped_count = 0
        for agent_id in list(self.active_agents.keys()):
            try:
                if await self.stop_agent(agent_id):
                    stopped_count += 1
            except Exception as e:
                logger.error(f"Failed to stop agent {agent_id}: {e}")
        
        logger.info(f"✅ Stopped {stopped_count} agents")
        return stopped_count


class AgoraWebhookHandler:
    """
    Webhook handler for custom LLM integration with RAG system
    
    This would be implemented as a FastAPI or Flask server in production
    to handle LLM requests from Agora and integrate with your RAG system.
    """
    
    def __init__(self, rag_generator=None):
        self.rag_generator = rag_generator
        logger.info("🔗 Agora Webhook Handler initialized")
    
    async def handle_llm_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle LLM request from Agora and generate response using RAG system
        
        Args:
            request_data: LLM request data from Agora
            
        Returns:
            LLM response in Agora-compatible format
        """
        
        try:
            # Extract user message from Agora request
            messages = request_data.get("messages", [])
            user_message = ""
            
            for message in messages:
                if message.get("role") == "user":
                    user_message = message.get("content", "")
                    break
            
            if not user_message:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "I didn't receive a message. Could you please repeat your question?"
                            }
                        }
                    ]
                }
            
            # Generate response using RAG system
            if self.rag_generator:
                logger.info(f"🧠 Processing RAG query: {user_message[:100]}...")
                rag_result = self.rag_generator.generate_response(user_message)
                response_text = rag_result.response
                
                # Add source information if available
                if rag_result.sources_used:
                    sources_info = "\n\nSources: " + ", ".join([
                        source.get("title", "Unknown")
                        for source in rag_result.sources_used[:3]
                    ])
                    response_text += sources_info
            else:
                response_text = f"I received your message: {user_message}. However, I don't have access to a knowledge base to provide a detailed answer."
            
            # Return in OpenAI-compatible format
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        }
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing LLM request: {e}")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'm sorry, I encountered an error processing your request. Please try again."
                        }
                    }
                ]
            }


def create_agora_conversational_ai(
    app_id: str,
    customer_id: str,
    customer_secret: str,
    rtc_token: Optional[str] = None,
    rag_generator=None
) -> AgoraConversationalAI:
    """
    Factory function to create Agora Conversational AI instance
    
    Args:
        app_id: Agora App ID
        customer_id: Agora Customer ID
        customer_secret: Agora Customer Secret
        rtc_token: Optional RTC token for authentication
        rag_generator: Optional RAG generator for knowledge base integration
        
    Returns:
        Configured AgoraConversationalAI instance
    """
    
    credentials = AgoraCredentials(
        app_id=app_id,
        customer_id=customer_id,
        customer_secret=customer_secret,
        rtc_token=rtc_token
    )
    
    return AgoraConversationalAI(
        credentials=credentials,
        rag_generator=rag_generator
    )


def create_default_agent_config(
    name: str,
    channel: str,
    llm_api_key: str = "",
    tts_api_key: str = "",
    system_message: Optional[str] = None
) -> AgentConfig:
    """
    Create a default agent configuration with sensible defaults
    
    Args:
        name: Agent name
        channel: Agora channel name
        llm_api_key: LLM API key (Gemini, etc.)
        tts_api_key: TTS API key (ElevenLabs, etc.)
        system_message: Custom system message
        
    Returns:
        Configured AgentConfig
    """
    
    default_system_message = (
        "You are Cerebrus AI, an intelligent assistant with access to a comprehensive knowledge base. "
        "You can answer questions, provide explanations, and help users find information from uploaded documents, "
        "web content, and other sources. Be helpful, accurate, and cite your sources when possible. "
        "Keep responses conversational and engaging for voice interaction."
    )
    
    llm_config = LLMConfig(
        api_key=llm_api_key,
        system_message=system_message or default_system_message,
        greeting_message="Hello! I'm Cerebrus AI. How can I help you explore your knowledge base today?",
        failure_message="I'm sorry, I don't have information about that in my current knowledge base."
    )
    
    tts_config = TTSConfig(
        api_key=tts_api_key or os.getenv("ELEVENLABS_API_KEY", ""),
        vendor="elevenlabs",
        voice_id="pNInz6obpgDQGcFmaJgB"  # Adam - Natural, professional voice
    )
    
    asr_config = ASRConfig(
        vendor="ares"
    )
    
    return AgentConfig(
        name=name,
        channel=channel,
        llm=llm_config,
        tts=tts_config,
        asr=asr_config
    )


if __name__ == "__main__":
    # Example usage
    async def test_agora_ai():
        """Test Agora Conversational AI integration"""
        
        # Load credentials from environment
        app_id = os.getenv("AGORA_APP_ID")
        customer_id = os.getenv("AGORA_CUSTOMER_ID")
        customer_secret = os.getenv("AGORA_CUSTOMER_SECRET")
        rtc_token = os.getenv("AGORA_RTC_TOKEN")
        
        if not all([app_id, customer_id, customer_secret]):
            logger.error("Missing Agora credentials in environment variables")
            logger.error("Please set AGORA_APP_ID, AGORA_CUSTOMER_ID, and AGORA_CUSTOMER_SECRET")
            return
        
        # Create Agora AI instance
        agora_ai = create_agora_conversational_ai(
            app_id=app_id,  # type: ignore
            customer_id=customer_id,  # type: ignore
            customer_secret=customer_secret,  # type: ignore
            rtc_token=rtc_token
        )
        
        # Create agent configuration
        agent_config = create_default_agent_config(
            name="test_agent",
            channel="test_channel",
            llm_api_key=os.getenv("OPENAI_API_KEY", ""),
            tts_api_key=os.getenv("AZURE_TTS_API_KEY", "")
        )
        
        try:
            # Start agent
            agent_id = await agora_ai.start_agent(agent_config, use_rag=False)
            
            # Query status
            status = await agora_ai.query_agent_status(agent_id)
            logger.info(f"Agent status: {status}")
            
            # Wait for a bit
            await asyncio.sleep(10)
            
            # Stop agent
            await agora_ai.stop_agent(agent_id)
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_agora_ai())