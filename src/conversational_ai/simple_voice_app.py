"""
Simplified Agora Conversational AI Streamlit App

One-click voice conversation interface with RAG integration for chatting with your documents.
"""

import streamlit as st
import asyncio
import os
import time
import requests
import tempfile
from typing import Optional
from datetime import datetime
from pathlib import Path

from src.conversational_ai.agora_ai import (
    AgoraConversationalAI,
    AgentConfig,
    LLMConfig,
    TTSConfig,
    ASRConfig,
    TurnDetectionConfig,
    AgoraCredentials
)

# Page configuration
st.set_page_config(
    page_title="Cerebrus AI Voice Chat",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        text-align: center;
    }
    
    .agent-running {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    .agent-stopped {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    
    .big-button {
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 50px !important;
        margin: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


class SimplifiedAgoraInterface:
    """Simplified one-click Agora interface"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'agora_ai' not in st.session_state:
            st.session_state.agora_ai = None
        if 'current_agent_id' not in st.session_state:
            st.session_state.current_agent_id = None
        if 'agent_status' not in st.session_state:
            st.session_state.agent_status = "stopped"
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'webhook_running' not in st.session_state:
            st.session_state.webhook_running = False
    
    def check_prerequisites(self) -> bool:
        """Check if all required environment variables are set"""
        required_vars = [
            "AGORA_APP_ID",
            "AGORA_CUSTOMER_ID", 
            "AGORA_CUSTOMER_SECRET",
            "AGORA_RTC_TOKEN",
            "ELEVENLABS_API_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            with st.expander("🔧 Setup Instructions"):
                st.markdown("""
                Please set these environment variables in your `.env` file:
                
                ```
                AGORA_APP_ID=your_app_id_here
                AGORA_CUSTOMER_ID=your_customer_id_here  
                AGORA_CUSTOMER_SECRET=your_customer_secret_here
                AGORA_RTC_TOKEN=your_rtc_token_here
                ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
                OPENAI_API_KEY=your_openai_api_key_here (optional, for fallback)
                ```
                
                **How to get these credentials:**
                1. **Agora**: Sign up at [console.agora.io](https://console.agora.io)
                2. **ElevenLabs**: Get API key from [elevenlabs.io](https://elevenlabs.io)
                3. **OpenAI**: Get API key from [platform.openai.com](https://platform.openai.com)
                """)
            return False
        
        return True
    
    def upload_documents(self) -> bool:
        """Handle document upload to RAG system"""
        st.markdown("### 📄 Upload Documents")
        st.markdown("Upload documents to chat with using voice. Supported: PDF, TXT, DOCX, MD")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            help="Upload documents to add to your knowledge base"
        )
        
        if uploaded_files and st.button("📤 Upload & Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                success_count = 0
                total_count = len(uploaded_files)
                
                for uploaded_file in uploaded_files:
                    try:
                        # Save file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Process document through RAG system
                        documents = [{"content": uploaded_file.getvalue().decode('utf-8', errors='ignore'), "title": uploaded_file.name}]
                        
                        # Send to webhook server for processing
                        response = requests.post(
                            "http://localhost:8000/rag/add-documents",
                            json=documents,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            success_count += 1
                            st.success(f"✅ {uploaded_file.name} processed successfully")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                        
                        # Clean up
                        os.unlink(tmp_file_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                
                if success_count > 0:
                    st.success(f"🎉 Successfully processed {success_count}/{total_count} documents!")
                    st.info("💡 You can now start a voice conversation to ask questions about your documents.")
                    return True
                else:
                    st.error("❌ No documents were processed successfully.")
        
        return False
        """Check if the webhook server is running"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=3)
            if response.status_code == 200:
                st.session_state.webhook_running = True
                return True
        except:
            pass
        
        st.session_state.webhook_running = False
        return False
    
    def create_agora_instance(self) -> Optional[AgoraConversationalAI]:
        """Create Agora AI instance with credentials"""
        try:
            credentials = AgoraCredentials(
                app_id=os.getenv("AGORA_APP_ID"),
                customer_id=os.getenv("AGORA_CUSTOMER_ID"),
                customer_secret=os.getenv("AGORA_CUSTOMER_SECRET"),
                rtc_token=os.getenv("AGORA_RTC_TOKEN")
            )
            
            agora_ai = AgoraConversationalAI(credentials=credentials)
            return agora_ai
            
        except Exception as e:
            st.error(f"❌ Failed to create Agora instance: {e}")
            return None
    
    def create_agent_config(self, use_rag: bool = True) -> AgentConfig:
        """Create optimized agent configuration"""
        
        # Generate unique channel name
        timestamp = int(time.time())
        channel_name = f"cerebrus_voice_{timestamp}"
        
        # LLM Configuration - Always use webhook for RAG when available
        if use_rag and st.session_state.webhook_running:
            llm_config = LLMConfig(
                url="http://localhost:8000/llm-webhook/cerebrus",
                api_key="webhook_key",  # Placeholder since we're using our webhook
                model="gpt-4o-mini",
                system_message="You are Cerebrus AI, an intelligent assistant with access to uploaded documents. Answer questions based on the knowledge base and provide helpful, accurate responses for voice interaction."
            )
            st.info("🧠 Using RAG system - you can ask questions about your uploaded documents!")
        else:
            llm_config = LLMConfig(
                url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model="gpt-4o-mini",
                system_message="You are Cerebrus AI, a helpful assistant. Provide conversational responses optimized for voice interaction."
            )
            st.warning("⚠️ Using basic mode - upload documents and start webhook server for RAG functionality.")
        
        # TTS Configuration (using our validated format)
        tts_config = TTSConfig(
            vendor="elevenlabs",
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam - professional voice
            model_id="eleven_flash_v2_5"
        )
        
        # ASR Configuration (using validated format) 
        asr_config = ASRConfig(
            vendor="ares",
            language="en-US"
        )
        
        # Turn detection for natural conversation
        turn_detection = TurnDetectionConfig(
            interrupt_mode="interrupt",
            interrupt_duration_ms=500,
            silence_duration_ms=800
        )
        
        return AgentConfig(
            name=f"cerebrus_agent_{timestamp}",
            channel=channel_name,
            llm=llm_config,
            tts=tts_config,
            asr=asr_config,
            turn_detection=turn_detection,
            idle_timeout=300  # 5 minutes
        )
    
    async def start_voice_conversation(self, use_rag: bool = True):
        """Start a voice conversation with one click"""
        
        if not st.session_state.agora_ai:
            st.session_state.agora_ai = self.create_agora_instance()
            
        if not st.session_state.agora_ai:
            st.error("❌ Failed to initialize Agora AI")
            return
        
        try:
            # Create agent configuration
            agent_config = self.create_agent_config(use_rag)
            
            # Start the agent
            with st.spinner("🚀 Starting voice conversation..."):
                agent_id = await st.session_state.agora_ai.start_agent(
                    agent_config, 
                    use_rag=use_rag
                )
            
            if agent_id:
                st.session_state.current_agent_id = agent_id
                st.session_state.agent_status = "running"
                st.success(f"✅ Voice conversation started! Agent ID: {agent_id}")
                st.info("📱 **Join the conversation:**")
                st.code(f"""
Channel: {agent_config.channel}
App ID: {os.getenv('AGORA_APP_ID')}
Token: {os.getenv('AGORA_RTC_TOKEN')}

Use the Agora Web SDK or mobile app to join this channel and start talking!
                """)
            else:
                st.error("❌ Failed to start agent")
                
        except Exception as e:
            st.error(f"❌ Error starting conversation: {e}")
    
    async def stop_voice_conversation(self):
        """Stop the current voice conversation"""
        
        if not st.session_state.current_agent_id:
            st.warning("⚠️ No active conversation to stop")
            return
        
        try:
            with st.spinner("🛑 Stopping voice conversation..."):
                success = await st.session_state.agora_ai.stop_agent(
                    st.session_state.current_agent_id
                )
            
            if success:
                st.session_state.current_agent_id = None
                st.session_state.agent_status = "stopped"
                st.success("✅ Voice conversation stopped")
            else:
                st.error("❌ Failed to stop conversation")
                
        except Exception as e:
            st.error(f"❌ Error stopping conversation: {e}")
    
    def render_main_interface(self):
        """Render the main one-click interface"""
        
        # Header
        st.markdown('<h1 class="main-header">🎙️ Cerebrus AI Voice Chat</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                One-click voice conversations with your AI assistant
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return
        
        # Status section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            webhook_status = self.check_webhook_status()
            if webhook_status:
                st.markdown("""
                <div class="status-card agent-running">
                    <h4>🚀 Webhook Server</h4>
                    <p>RAG System Ready</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card agent-stopped">
                    <h4>⏸️ Webhook Server</h4>
                    <p>RAG System Offline</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.agent_status == "running":
                st.markdown("""
                <div class="status-card agent-running">
                    <h4>🗣️ Voice Agent</h4>
                    <p>Ready for Conversation</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card agent-stopped">
                    <h4>😴 Voice Agent</h4>
                    <p>Waiting to Start</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            agora_ready = bool(os.getenv("AGORA_APP_ID"))
            if agora_ready:
                st.markdown("""
                <div class="status-card agent-running">
                    <h4>📡 Agora Ready</h4>
                    <p>Credentials Loaded</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card agent-stopped">
                    <h4>🔑 Agora Setup</h4>
                    <p>Credentials Missing</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Main action buttons
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Start Voice Chat (RAG)", 
                        disabled=(st.session_state.agent_status == "running"),
                        help="Start voice conversation with RAG knowledge base"):
                asyncio.run(self.start_voice_conversation(use_rag=True))
        
        with col2:
            if st.button("💬 Start Voice Chat (Basic)", 
                        disabled=(st.session_state.agent_status == "running"),
                        help="Start voice conversation with basic OpenAI"):
                asyncio.run(self.start_voice_conversation(use_rag=False))
        
        with col3:
            if st.button("🛑 Stop Conversation", 
                        disabled=(st.session_state.agent_status != "running"),
                        help="Stop the current voice conversation"):
                asyncio.run(self.stop_voice_conversation())
        
        with col4:
            if st.button("🔄 Refresh Status", help="Check system status"):
                st.rerun()
        
        # Current session info
        if st.session_state.current_agent_id:
            with st.expander("📊 Current Session", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Agent ID:** `{st.session_state.current_agent_id}`")
                    st.write(f"**Status:** {st.session_state.agent_status}")
                with col2:
                    st.write(f"**Started:** {datetime.now().strftime('%H:%M:%S')}")
                    st.write(f"**RAG Enabled:** {'Yes' if st.session_state.webhook_running else 'No'}")
        
        # Instructions
        with st.expander("📱 How to Join Voice Conversation"):
            st.markdown("""
            Once you start a voice conversation, you can join it using:
            
            **Option 1: Web Browser**
            1. Use the Agora Web SDK demo
            2. Enter the Channel and App ID shown above
            3. Click "Join" and start speaking
            
            **Option 2: Mobile App**
            1. Use any Agora RTC-enabled app
            2. Enter the channel details
            3. Join and start your voice conversation
            
            **Option 3: Custom Integration**
            1. Use the provided channel name and token
            2. Integrate with your own Agora RTC client
            3. Join the channel and communicate with Cerebrus AI
            
            **Voice Commands:**
            - Just speak naturally - the AI will respond
            - The AI has access to your knowledge base (if RAG is enabled)
            - Say "goodbye" or "stop" to end the conversation
            """)


def main():
    """Main Streamlit application"""
    
    # Initialize the interface
    interface = SimplifiedAgoraInterface()
    
    # Render the main interface
    interface.render_main_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Powered by Cerebrus AI • Agora Conversational AI • ElevenLabs TTS • Ares ASR</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()