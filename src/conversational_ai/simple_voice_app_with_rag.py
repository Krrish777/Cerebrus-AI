"""
Simplified Agora Conversational AI Streamlit App

One-click voice conversation interface with RAG integration for chatting with your documents.
"""

import streamlit as st
import asyncio
import os
import time
import requests
import json
import tempfile
import sys
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.conversational_ai.agora_ai import (
    AgoraConversationalAI,
    AgentConfig,
    LLMConfig,
    TTSConfig,
    ASRConfig,
    TurnDetectionConfig,
    AgoraCredentials,
    create_default_agent_config
)

# NOTE: page config is provided by the main app (`main.py`).
# Do not call `st.set_page_config` here so this module can be imported
# and its `render_voice_interface` function can be used inside the main Streamlit app.

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
    
    .upload-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class SimplifiedAgoraInterface:
    """Simplified one-click Agora interface with RAG integration"""
    
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
        if 'documents_uploaded' not in st.session_state:
            st.session_state.documents_uploaded = 0
    
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
                st.markdown(f"""
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
    
    def check_webhook_status(self) -> bool:
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
                        # For simple text files, we can process directly
                        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.md'):
                            content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                        else:
                            # For other files, we'll treat them as text for now
                            content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                        
                        # Create document structure
                        documents = [{
                            "content": content,
                            "title": uploaded_file.name,
                            "metadata": {
                                "filename": uploaded_file.name,
                                "uploaded_at": datetime.now().isoformat()
                            }
                        }]
                        
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
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                
                if success_count > 0:
                    st.session_state.documents_uploaded += success_count
                    st.success(f"🎉 Successfully processed {success_count}/{total_count} documents!")
                    st.info("💡 You can now start a voice conversation to ask questions about your documents.")
                    return True
                else:
                    st.error("❌ No documents were processed successfully.")
        
        # Show uploaded documents count
        if st.session_state.documents_uploaded > 0:
            st.info(f"📚 {st.session_state.documents_uploaded} documents in knowledge base")
        
        return False
    
    def create_agora_instance(self) -> Optional[AgoraConversationalAI]:
        """Create Agora AI instance with credentials"""
        try:
            credentials = AgoraCredentials(
                app_id=os.getenv("AGORA_APP_ID", ""),
                customer_id=os.getenv("AGORA_CUSTOMER_ID", ""),
                customer_secret=os.getenv("AGORA_CUSTOMER_SECRET", ""),
                rtc_token=os.getenv("AGORA_RTC_TOKEN", "")
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
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
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
            interrupt_duration_ms=500
        )
        
        # Create complete agent config
        return AgentConfig(
            name=f"cerebrus_voice_{timestamp}",
            channel=channel_name,
            llm=llm_config,
            tts=tts_config,
            asr=asr_config,
            turn_detection=turn_detection,
            idle_timeout=300  # 5 minutes
        )
    
    def render_agent_status(self):
        """Render current agent status"""
        if st.session_state.current_agent_id:
            if st.session_state.agent_status == "running":
                st.markdown("""
                <div class="status-card agent-running">
                    <h4>🎙️ Voice Agent Active</h4>
                    <p>Ready for conversation!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card agent-stopped">
                    <h4>⏸️ Voice Agent Stopped</h4>
                    <p>Click start to begin</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card agent-stopped">
                <h4>😴 No Active Agent</h4>
                <p>Ready to start conversation</p>
            </div>
            """, unsafe_allow_html=True)
    
    async def start_voice_conversation(self, use_rag: bool = True):
        """Start a new voice conversation"""
        
        try:
            with st.spinner("🚀 Starting voice conversation..."):
                # Create Agora instance if not exists
                if not st.session_state.agora_ai:
                    st.session_state.agora_ai = self.create_agora_instance()
                
                if not st.session_state.agora_ai:
                    st.error("❌ Failed to initialize Agora AI")
                    return
                
                # Create agent configuration
                agent_config = self.create_agent_config(use_rag=use_rag)
                
                # Start the agent
                agent_id = await st.session_state.agora_ai.start_agent(
                    agent_config, 
                    use_rag=use_rag
                )
                
                # Update session state
                st.session_state.current_agent_id = agent_id
                st.session_state.agent_status = "running"
                
                st.success("✅ Voice conversation started!")
                st.info(f"🆔 Agent ID: {agent_id}")
                
                # Display connection info
                st.code(f"""
Channel: {agent_config.channel}
App ID: {os.getenv('AGORA_APP_ID')}
Token: {os.getenv('AGORA_RTC_TOKEN')}

Use the Agora Web SDK or mobile app to join this channel and start talking!
                """)
                
        except Exception as e:
            st.error(f"❌ Error starting conversation: {e}")
    
    def start_voice_conversation_sync(self, use_rag: bool = True):
        """Synchronous wrapper for starting voice conversation"""
        asyncio.run(self.start_voice_conversation(use_rag))
    
    def stop_voice_conversation_sync(self):
        """Synchronous wrapper for stopping voice conversation"""
        asyncio.run(self.stop_voice_conversation())
    
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
    
    def show_agent_details(self):
        """Show detailed agent information"""
        if st.session_state.current_agent_id:
            with st.expander("🔍 Agent Details", expanded=True):
                st.write(f"**Agent ID**: {st.session_state.current_agent_id}")
                st.write(f"**Status**: {st.session_state.agent_status}")
                st.write(f"**RAG Enabled**: {'Yes' if st.session_state.webhook_running else 'No'}")
                st.write(f"**Documents in KB**: {st.session_state.documents_uploaded}")
        else:
            st.info("No active agent to display details for.")
    
    def render_main_interface(self):
        """Render the main application interface"""
        
        # Header
        st.markdown('<h1 class="main-header">🎙️ Cerebrus AI Voice Chat</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Talk to your documents with AI-powered voice conversations</p>', unsafe_allow_html=True)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return
        
        # Create two columns: Document Upload | Voice Chat
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📚 Document Management")
            
            # Check webhook status
            webhook_running = self.check_webhook_status()
            if webhook_running:
                st.success("✅ RAG system is running")
                
                # Document upload section
                with st.container():
                    self.upload_documents()
                
                # RAG status
                with st.expander("🔍 RAG System Status"):
                    try:
                        response = requests.get("http://localhost:8000/rag/status", timeout=3)
                        if response.status_code == 200:
                            rag_status = response.json()
                            st.json(rag_status)
                        else:
                            st.error("Failed to get RAG status")
                    except:
                        st.error("Could not connect to RAG system")
            else:
                st.error("❌ Webhook server not running")
                with st.expander("📋 Setup Instructions"):
                    st.markdown("""
                    **To enable RAG functionality:**
                    
                    1. **Start the webhook server**:
                    ```bash
                    python -m uvicorn src.conversational_ai.webhook_server_fixed:app --host 0.0.0.0 --port 8000
                    ```
                    
                    2. **Or use the launcher**:
                    ```bash
                    python launcher.py
                    ```
                    
                    3. **Refresh this page**
                    4. **Upload your documents**
                    5. **Start voice conversation**
                    """)
        
        with col2:
            st.markdown("### 🎙️ Voice Conversation")
            
            # Agent status display
            self.render_agent_status()
            
            # Main action buttons
            if st.session_state.agent_status == "stopped":
                st.markdown("#### Start Conversation")
                
                col_rag, col_basic = st.columns(2)
                
                with col_rag:
                    if st.button(
                        "🧠 Start with RAG", 
                        type="primary", 
                        help="Use uploaded documents",
                        use_container_width=True,
                        disabled=not webhook_running
                    ):
                        self.start_voice_conversation_sync(use_rag=True)
                
                with col_basic:
                    if st.button(
                        "💬 Start Basic Chat", 
                        help="General conversation",
                        use_container_width=True
                    ):
                        self.start_voice_conversation_sync(use_rag=False)
                        
            else:
                st.markdown("#### Manage Conversation")
                
                col_stop, col_info = st.columns(2)
                
                with col_stop:
                    if st.button(
                        "⏹️ Stop Conversation", 
                        type="secondary",
                        use_container_width=True
                    ):
                        self.stop_voice_conversation_sync()
                
                with col_info:
                    if st.button(
                        "📊 Show Details", 
                        use_container_width=True
                    ):
                        self.show_agent_details()
        
        # Instructions section
        with st.expander("📖 How to Use"):
            st.markdown("""
            ### Quick Start Guide:
            
            1. **Upload Documents** (Left Panel):
               - Upload PDF, TXT, DOCX, or MD files
               - Click "Upload & Process Documents"
               - Wait for processing to complete
            
            2. **Start Voice Chat** (Right Panel):
               - Click "Start with RAG" to use your documents
               - Or "Start Basic Chat" for general conversation
               - Wait for agent to initialize
            
            3. **Join the Conversation**:
               - Use Agora Web SDK demo or mobile app
               - Enter the channel and App ID displayed
               - Start speaking naturally
            
            4. **Voice Interaction**:
               - Ask questions about your uploaded documents
               - Speak clearly and wait for AI responses
               - The AI responds with natural voice synthesis
            
            ### Tips:
            - Ensure microphone permissions are enabled
            - Use a stable internet connection
            - Upload relevant documents before starting RAG chat
            - Say "goodbye" to end the conversation naturally
            """)


def render_voice_interface(rag_generator: Optional[object] = None):
    """Render the voice chat UI from the main app.

    Call this from the main Streamlit app and pass the initialized
    `rag_generator` (optional). This keeps `simple_voice_app_with_rag.py`
    importable without running a standalone Streamlit page.
    """
    # If a RAG generator is provided by the main app, store it in session
    if rag_generator is not None:
        st.session_state.rag_generator = rag_generator

    app = SimplifiedAgoraInterface()
    app.render_main_interface()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🚀 Powered by Cerebrus AI • Agora • ElevenLabs • Ares ASR • OpenAI</p>
    </div>
    """, unsafe_allow_html=True)