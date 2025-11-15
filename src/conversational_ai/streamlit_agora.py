"""
Agora Conversational AI Streamlit Integration

This module provides a Streamlit interface for managing Agora Conversational AI agents
and integrating them with the Cerebrus AI RAG system.
"""

import streamlit as st
import asyncio
import os
import time
import requests
import subprocess
from typing import Dict, List, Optional
from datetime import datetime
import json

from src.conversational_ai.agora_ai import (
    AgoraConversationalAI,
    AgentConfig,
    LLMConfig,
    TTSConfig,
    ASRConfig,
    TurnDetectionConfig,
    AgoraCredentials,
    create_agora_conversational_ai,
    create_default_agent_config
)
from src.generation.rag import ElasticsearchRAGGenerator
from src.core.logging import CustomLogger

# Initialize logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


class AgoraStreamlitInterface:
    """Streamlit interface for Agora Conversational AI"""
    
    def __init__(self, rag_generator=None):
        self.rag_generator = rag_generator
        
        # Initialize session state
        if 'agora_ai' not in st.session_state:
            st.session_state.agora_ai = None
        if 'agora_agents' not in st.session_state:
            st.session_state.agora_agents = {}
        if 'agora_configured' not in st.session_state:
            st.session_state.agora_configured = False
    
    def render_configuration_section(self):
        """Render Agora configuration section"""
        
        st.header("🎙️ Agora Conversational AI Configuration")
        
        with st.expander("📋 Setup Instructions", expanded=not st.session_state.agora_configured):
            st.markdown("""
            **Prerequisites:**
            1. Create an Agora account at [console.agora.io](https://console.agora.io)
            2. Create a new project and enable Conversational AI
            3. Get your App ID, Customer ID, and Customer Secret
            4. Generate RTC tokens for authentication
            5. Set up LLM (OpenAI) and TTS/ASR (Azure) API keys
            
            **Environment Variables:**
            ```
            AGORA_APP_ID=your_app_id
            AGORA_CUSTOMER_ID=your_customer_id
            AGORA_CUSTOMER_SECRET=your_customer_secret
            AGORA_RTC_TOKEN=your_rtc_token
            ELEVEN_LABS_API_KEY=your_elevenlabs_api_key
            GEMINI_API_KEY=your_gemini_api_key
            ARES_ASR_API_KEY=your_ares_asr_key (optional)
            ARES_ASR_REGION=us-east-1 (optional)
            ```
            """)
        
        # Configuration form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔑 Agora Credentials")
            app_id = st.text_input(
                "App ID",
                value=os.getenv("AGORA_APP_ID", ""),
                type="password",
                help="Your Agora project App ID"
            )
            customer_id = st.text_input(
                "Customer ID",
                value=os.getenv("AGORA_CUSTOMER_ID", ""),
                type="password",
                help="Your Agora Customer ID for API authentication"
            )
            customer_secret = st.text_input(
                "Customer Secret",
                value=os.getenv("AGORA_CUSTOMER_SECRET", ""),
                type="password",
                help="Your Agora Customer Secret for API authentication"
            )
            rtc_token = st.text_input(
                "RTC Token (Optional)",
                value=os.getenv("AGORA_RTC_TOKEN", ""),
                type="password",
                help="RTC token for channel authentication"
            )
        
        with col2:
            st.subheader("🤖 AI Service Keys")
            llm_api_key = st.text_input(
                "Gemini API Key",
                value=os.getenv("GEMINI_API_KEY", ""),
                type="password",
                help="Google Gemini API key for language model"
            )
            tts_api_key = st.text_input(
                "ElevenLabs API Key",
                value=os.getenv("ELEVENLABS_API_KEY", os.getenv("ELEVEN_LABS_API_KEY", "")),
                type="password",
                help="ElevenLabs API key for text-to-speech"
            )
            asr_api_key = st.text_input(
                "Ares ASR API Key",
                value=os.getenv("ARES_ASR_API_KEY", ""),
                type="password",
                help="Ares ASR API key for speech recognition (optional)"
            )
            asr_region = st.text_input(
                "Ares ASR Region",
                value=os.getenv("ARES_ASR_REGION", "us-east-1"),
                help="Ares ASR region (e.g., us-east-1, eu-west-1)"
            )
        
        # Initialize button
        if st.button("🚀 Initialize Agora Conversational AI", type="primary"):
            if not all([app_id, customer_id, customer_secret]):
                st.error("❌ Please provide at least App ID, Customer ID, and Customer Secret")
                return False
            
            try:
                with st.spinner("Initializing Agora Conversational AI..."):
                    # Create Agora AI instance
                    st.session_state.agora_ai = create_agora_conversational_ai(
                        app_id=app_id,
                        customer_id=customer_id,
                        customer_secret=customer_secret,
                        rtc_token=rtc_token if rtc_token else None,
                        rag_generator=self.rag_generator
                    )
                    
                    # Store API keys for agent creation
                    st.session_state.llm_api_key = llm_api_key
                    st.session_state.tts_api_key = tts_api_key
                    st.session_state.asr_api_key = asr_api_key
                    st.session_state.asr_region = asr_region
                    st.session_state.agora_configured = True
                
                st.success("✅ Agora Conversational AI initialized successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to initialize Agora AI: {e}")
                logger.error(f"Agora AI initialization failed: {e}")
                return False
        
        # Webhook Server Management Section
        st.header("🔗 LLM Webhook Server")
        
        webhook_col1, webhook_col2, webhook_col3 = st.columns([2, 1, 1])
        
        with webhook_col1:
            webhook_url = st.text_input(
                "Webhook Server URL",
                value="http://localhost:8000",
                help="URL where your LLM webhook server is running"
            )
        
        with webhook_col2:
            if st.button("🚀 Start Server", help="Start the LLM webhook server"):
                try:
                    import time
                    
                    # Start webhook server
                    result = subprocess.run([
                        "python", "manage_webhook.py", "start", "--reload"
                    ], capture_output=True, text=True, cwd=os.getcwd())
                    
                    if result.returncode == 0:
                        st.success("✅ Webhook server started!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to start server: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with webhook_col3:
            if st.button("🛑 Stop Server", help="Stop the LLM webhook server"):
                try:
                    import time
                    
                    result = subprocess.run([
                        "python", "manage_webhook.py", "stop"
                    ], capture_output=True, text=True, cwd=os.getcwd())
                    
                    if result.returncode == 0:
                        st.success("✅ Webhook server stopped!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to stop server: {result.stderr}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        # Server Status Check
        try:
            response = requests.get(f"{webhook_url}/health", timeout=3)
            if response.status_code == 200:
                health_data = response.json()
                st.success(f"✅ Webhook server is running at {webhook_url}")
                
                # Show webhook endpoints
                st.info(f"""
                **Available Endpoints:**
                - 🎙️ LLM Webhook: `{webhook_url}/llm-webhook`
                - 📊 Health Check: `{webhook_url}/health`
                - 🔍 RAG Status: `{webhook_url}/rag/status`
                
                **Status:** {health_data.get('status', 'unknown')}
                **RAG System:** {health_data.get('rag_system', 'unknown')}
                """)
            else:
                st.error(f"❌ Webhook server error: {response.status_code}")
        except requests.exceptions.RequestException:
            st.warning(f"⚠️ Webhook server not responding at {webhook_url}")
            st.markdown("""
            **To start the webhook server:**
            1. Click "🚀 Start Server" above, or
            2. Run manually: `python manage_webhook.py start`
            
            **The webhook server is required for LLM integration with Agora agents.**
            """)
        except Exception as e:
            st.warning(f"⚠️ Cannot check server status: {e}")
        
        return st.session_state.agora_configured
    
    def render_agent_management(self):
        """Render agent management interface"""
        
        if not st.session_state.agora_configured:
            st.warning("⚠️ Please configure Agora Conversational AI first")
            return
        
        st.header("🤖 AI Voice Agents")
        
        # Agent status overview
        active_agents = st.session_state.agora_agents
        
        if active_agents:
            st.subheader("📊 Active Agents")
            
            for agent_id, agent_info in active_agents.items():
                with st.expander(f"🎙️ {agent_info['name']} ({agent_id[:8]}...)", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Channel:** {agent_info['channel']}")
                        st.write(f"**Status:** {agent_info.get('status', 'Unknown')}")
                        st.write(f"**Started:** {agent_info['started_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with col2:
                        st.write(f"**RAG Integration:** {'✅' if agent_info.get('use_rag') else '❌'}")
                        if 'duration' in agent_info:
                            st.write(f"**Duration:** {agent_info['duration']}")
                    
                    with col3:
                        if st.button(f"🛑 Stop", key=f"stop_{agent_id}"):
                            self._stop_agent(agent_id)
        else:
            st.info("📭 No active agents. Create one below!")
        
        st.divider()
        
        # Create new agent
        st.subheader("➕ Create New Voice Agent")
        
        with st.form("create_agent_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                agent_name = st.text_input(
                    "Agent Name",
                    value=f"agent_{datetime.now().strftime('%H%M%S')}",
                    help="Unique name for this agent"
                )
                channel_name = st.text_input(
                    "Channel Name",
                    value=f"cerebrus_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    help="Agora channel where the agent will operate"
                )
                use_rag = st.checkbox(
                    "🧠 Enable RAG Integration",
                    value=True,
                    help="Connect agent to your knowledge base"
                )
            
            with col2:
                voice_name = st.selectbox(
                    "ElevenLabs Voice",
                    [
                        ("pNInz6obpgDQGcFmaJgB", "Adam - Natural, Professional"),
                        ("21m00Tcm4TlvDq8ikWAM", "Rachel - Clear, Friendly"),
                        ("AZnzlk1XvdvUeBnXmlld", "Domi - Confident, Engaging"),
                        ("EXAVITQu4vr4xnSDxMaL", "Bella - Warm, Approachable"),
                        ("ErXwobaYiN019PkySvjV", "Antoni - Sophisticated, Calm"),
                        ("MF3mGyEYCl7XYWbV9V6O", "Elli - Energetic, Young"),
                        ("TxGEqnHWrfWFTfGW9XjX", "Josh - Deep, Authoritative"),
                        ("VR6AewLTigWG4xSOukaG", "Arnold - Strong, Commanding"),
                        ("pqHfZKP75CvOlQylNhV4", "Bill - Friendly, Trustworthy"),
                        ("yoZ06aMxZJJ28mfd3POQ", "Sam - Natural, Conversational")
                    ],
                    format_func=lambda x: x[1],  # Show the description
                    help="Choose the ElevenLabs voice for responses"
                )
                
                # Extract just the voice ID for use later
                voice_id = voice_name[0] if isinstance(voice_name, tuple) else voice_name
                
                interrupt_mode = st.selectbox(
                    "Interrupt Mode",
                    ["interrupt", "append", "ignore"],
                    help="How the agent handles user interruptions"
                )
                
                idle_timeout = st.slider(
                    "Idle Timeout (seconds)",
                    min_value=60,
                    max_value=600,
                    value=120,
                    help="Agent will stop after this period of inactivity"
                )
            
            # Advanced configuration
            with st.expander("⚙️ Advanced Configuration"):
                custom_system_message = st.text_area(
                    "Custom System Message",
                    value="You are Cerebrus AI, an intelligent voice assistant with access to a comprehensive knowledge base. Provide helpful, accurate responses in a conversational tone suitable for voice interaction.",
                    help="Custom instructions for the AI agent"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    temperature = st.slider(
                        "Response Creativity",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="Higher values make responses more creative"
                    )
                    max_tokens = st.slider(
                        "Max Response Length",
                        min_value=100,
                        max_value=3000,
                        value=1000,
                        step=100,
                        help="Maximum tokens in AI responses"
                    )
                
                with col2:
                    silence_duration = st.slider(
                        "Silence Detection (ms)",
                        min_value=300,
                        max_value=2000,
                        value=800,
                        step=100,
                        help="Silence duration before agent responds"
                    )
                    interrupt_duration = st.slider(
                        "Interrupt Threshold (ms)",
                        min_value=200,
                        max_value=1000,
                        value=500,
                        step=50,
                        help="Time before allowing interruptions"
                    )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Create & Start Agent", type="primary")
            
            if submitted:
                if not agent_name or not channel_name:
                    st.error("❌ Please provide agent name and channel name")
                else:
                    self._create_agent(
                        agent_name=agent_name,
                        channel_name=channel_name,
                        use_rag=use_rag,
                        voice_id=voice_id,
                        interrupt_mode=interrupt_mode,
                        idle_timeout=idle_timeout,
                        custom_system_message=custom_system_message,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        silence_duration=silence_duration,
                        interrupt_duration=interrupt_duration
                    )
    
    def _create_agent(
        self,
        agent_name: str,
        channel_name: str,
        use_rag: bool = True,
        voice_id: str = "pNInz6obpgDQGcFmaJgB",
        interrupt_mode: str = "interrupt",
        idle_timeout: int = 120,
        custom_system_message: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        silence_duration: int = 800,
        interrupt_duration: int = 500
    ):
        """Create and start a new Agora AI agent"""
        
        try:
            with st.spinner(f"Creating agent: {agent_name}..."):
                # Create agent configuration
                agent_config = create_default_agent_config(
                    name=agent_name,
                    channel=channel_name,
                    llm_api_key=st.session_state.get("llm_api_key", ""),
                    tts_api_key=st.session_state.get("tts_api_key", ""),
                    asr_api_key=st.session_state.get("asr_api_key", ""),
                    asr_region=st.session_state.get("asr_region", "eastus"),
                    system_message=custom_system_message
                )
                
                # Customize configuration
                agent_config.idle_timeout = idle_timeout
                agent_config.llm.temperature = temperature
                agent_config.llm.max_tokens = max_tokens
                agent_config.tts.voice_id = voice_id  # Use the extracted voice_id
                agent_config.turn_detection.interrupt_mode = interrupt_mode
                agent_config.turn_detection.silence_duration_ms = silence_duration
                agent_config.turn_detection.interrupt_duration_ms = interrupt_duration
                
                # Start agent asynchronously
                async def start_agent_async():
                    return await st.session_state.agora_ai.start_agent(agent_config, use_rag=use_rag)
                
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                agent_id = loop.run_until_complete(start_agent_async())
                loop.close()
                
                # Store agent info
                st.session_state.agora_agents[agent_id] = {
                    "name": agent_name,
                    "channel": channel_name,
                    "started_at": datetime.now(),
                    "use_rag": use_rag,
                    "config": agent_config,
                    "status": "RUNNING"
                }
                
                st.success(f"✅ Agent '{agent_name}' created successfully!")
                st.info(f"🎙️ **Agent ID:** {agent_id}")
                st.info(f"📡 **Channel:** {channel_name}")
                st.info(f"🧠 **RAG Enabled:** {'Yes' if use_rag else 'No'}")
                
                # Instructions
                st.markdown("""
                ### 🎯 How to Connect:
                1. **Download the Agora Sample App** or integrate Agora RTC SDK into your app
                2. **Join the channel:** `{}`
                3. **Use App ID:** `{}`
                4. **Start talking!** The AI agent will respond with voice
                
                ### ✨ Features:
                - 🗣️ **Natural voice conversations** with ultra-low latency
                - 🧠 **RAG-powered responses** from your knowledge base
                - 🎛️ **Interrupt handling** - you can interrupt the AI anytime
                - 🔊 **High-quality voice synthesis** with Azure TTS
                - 🎯 **Context awareness** from uploaded documents
                """.format(channel_name, st.session_state.agora_ai.credentials.app_id))
                
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Failed to create agent: {e}")
            logger.error(f"Agent creation failed: {e}")
    
    def _stop_agent(self, agent_id: str):
        """Stop a running agent"""
        
        try:
            agent_info = st.session_state.agora_agents.get(agent_id)
            if not agent_info:
                st.error("❌ Agent not found")
                return
            
            with st.spinner(f"Stopping agent: {agent_info['name']}..."):
                # Stop agent asynchronously
                async def stop_agent_async():
                    return await st.session_state.agora_ai.stop_agent(agent_id)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(stop_agent_async())
                loop.close()
                
                if success:
                    # Calculate duration
                    duration = datetime.now() - agent_info['started_at']
                    st.session_state.agora_agents[agent_id]['duration'] = str(duration).split('.')[0]
                    
                    # Remove from active agents
                    del st.session_state.agora_agents[agent_id]
                    
                    st.success(f"✅ Agent '{agent_info['name']}' stopped successfully!")
                    st.info(f"⏱️ **Runtime:** {duration}")
                    st.rerun()
                else:
                    st.error("❌ Failed to stop agent")
                
        except Exception as e:
            st.error(f"❌ Error stopping agent: {e}")
            logger.error(f"Agent stop failed: {e}")
    
    def render_agent_statistics(self):
        """Render agent usage statistics"""
        
        if not st.session_state.agora_configured:
            return
        
        st.header("📊 Voice Agent Analytics")
        
        active_count = len(st.session_state.agora_agents)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🤖 Active Agents", active_count)
        
        with col2:
            rag_enabled = sum(1 for agent in st.session_state.agora_agents.values() if agent.get('use_rag'))
            st.metric("🧠 RAG-Enabled", rag_enabled)
        
        with col3:
            if active_count > 0:
                avg_runtime = "Running"
            else:
                avg_runtime = "N/A"
            st.metric("⏱️ Status", avg_runtime)
        
        with col4:
            total_channels = len(set(agent['channel'] for agent in st.session_state.agora_agents.values()))
            st.metric("📡 Channels", total_channels)
        
        # Agent details table
        if st.session_state.agora_agents:
            st.subheader("🔍 Agent Details")
            
            agent_data = []
            for agent_id, agent_info in st.session_state.agora_agents.items():
                runtime = datetime.now() - agent_info['started_at']
                runtime_str = str(runtime).split('.')[0]
                
                agent_data.append({
                    "Agent ID": agent_id[:12] + "...",
                    "Name": agent_info['name'],
                    "Channel": agent_info['channel'],
                    "RAG": "✅" if agent_info.get('use_rag') else "❌",
                    "Runtime": runtime_str,
                    "Status": agent_info.get('status', 'Unknown')
                })
            
            st.dataframe(agent_data, use_container_width=True)
    
    def render_integration_guide(self):
        """Render integration guide"""
        
        st.header("📚 Integration Guide")
        
        tab1, tab2, tab3 = st.tabs(["🏗️ Setup", "📱 Client Integration", "🎯 Best Practices"])
        
        with tab1:
            st.markdown("""
            ## 🏗️ Complete Setup Guide
            
            ### 1. Agora Console Setup
            ```bash
            # 1. Visit https://console.agora.io
            # 2. Create new project
            # 3. Enable Conversational AI service
            # 4. Get App ID, Customer ID, Customer Secret
            # 5. Generate RTC tokens
            ```
            
            ### 2. API Keys Setup
            ```bash
            # OpenAI for LLM
            export OPENAI_API_KEY="sk-..."
            
            # Azure for Speech Services
            export AZURE_TTS_API_KEY="your_tts_key"
            export AZURE_ASR_API_KEY="your_asr_key"
            
            # Agora Credentials
            export AGORA_APP_ID="your_app_id"
            export AGORA_CUSTOMER_ID="your_customer_id"
            export AGORA_CUSTOMER_SECRET="your_customer_secret"
            ```
            
            ### 3. Environment Configuration
            ```python
            # Add to your .env file
            AGORA_APP_ID=your_app_id_here
            AGORA_CUSTOMER_ID=your_customer_id_here
            AGORA_CUSTOMER_SECRET=your_customer_secret_here
            OPENAI_API_KEY=your_openai_key_here
            AZURE_TTS_API_KEY=your_azure_tts_key_here
            AZURE_ASR_API_KEY=your_azure_asr_key_here
            ```
            """)
        
        with tab2:
            st.markdown("""
            ## 📱 Client Application Integration
            
            ### Web Integration (JavaScript)
            ```javascript
            import AgoraRTC from "agora-rtc-sdk-ng";
            
            // Initialize Agora client
            const client = AgoraRTC.createClient({mode: "rtc", codec: "vp8"});
            
            // Join channel with AI agent
            await client.join(
                "YOUR_APP_ID",
                "CHANNEL_NAME", 
                "TOKEN",
                USER_ID
            );
            
            // Enable microphone
            const microphone = await AgoraRTC.createMicrophoneAudioTrack();
            await client.publish(microphone);
            
            // Listen for AI agent audio
            client.on("user-published", async (user, mediaType) => {
                if (mediaType === "audio") {
                    await client.subscribe(user, mediaType);
                    user.audioTrack.play();
                }
            });
            ```
            
            ### Mobile Integration (React Native)
            ```jsx
            import {RtcEngine} from 'react-native-agora';
            
            // Initialize engine
            const engine = await RtcEngine.create('YOUR_APP_ID');
            
            // Join channel
            await engine.joinChannel(
                'TOKEN',
                'CHANNEL_NAME',
                null,
                USER_ID
            );
            
            // Enable audio
            await engine.enableAudio();
            ```
            
            ### Flutter Integration
            ```dart
            import 'package:agora_rtc_engine/agora_rtc_engine.dart';
            
            // Initialize engine
            RtcEngine engine = createAgoraRtcEngine();
            await engine.initialize(RtcEngineContext(
                appId: 'YOUR_APP_ID',
            ));
            
            // Join channel
            await engine.joinChannel(
                token: 'TOKEN',
                channelId: 'CHANNEL_NAME',
                uid: USER_ID,
                options: ChannelMediaOptions(),
            );
            ```
            """)
        
        with tab3:
            st.markdown("""
            ## 🎯 Best Practices
            
            ### Voice Interaction Design
            - **Keep responses concise** - Users prefer shorter voice responses
            - **Use natural language** - Avoid technical jargon
            - **Provide clear next steps** - Guide user conversation flow
            - **Handle interruptions gracefully** - Allow natural conversation patterns
            
            ### Performance Optimization
            - **Use appropriate timeout values** - Balance responsiveness with battery life
            - **Implement proper error handling** - Network issues are common in voice apps
            - **Optimize for different devices** - Test on various mobile devices
            - **Monitor latency metrics** - Keep response times under 1 second
            
            ### Security Considerations
            - **Use temporary tokens** - Rotate RTC tokens regularly
            - **Implement proper authentication** - Verify user identity
            - **Sanitize RAG responses** - Filter sensitive information
            - **Monitor usage patterns** - Detect unusual activity
            
            ### RAG Integration Tips
            - **Keep context relevant** - Only include pertinent information
            - **Cite sources when possible** - Help users understand information origin
            - **Handle "no results" gracefully** - Provide helpful alternatives
            - **Update knowledge base regularly** - Keep information current
            """)


def render_agora_interface(rag_generator=None):
    """
    Render the complete Agora Conversational AI interface in Streamlit
    
    Args:
        rag_generator: Optional RAG generator for knowledge base integration
    """
    
    interface = AgoraStreamlitInterface(rag_generator)
    
    st.title("🎙️ Agora Conversational AI")
    st.markdown("*Ultra-low latency voice conversations with AI agents powered by your knowledge base*")
    
    # Configuration section
    configured = interface.render_configuration_section()
    
    if configured:
        # Create tabs for different features
        tab1, tab2, tab3 = st.tabs(["🤖 Agents", "📊 Analytics", "📚 Guide"])
        
        with tab1:
            interface.render_agent_management()
        
        with tab2:
            interface.render_agent_statistics()
        
        with tab3:
            interface.render_integration_guide()
    
    return interface


if __name__ == "__main__":
    # Test the interface
    render_agora_interface()