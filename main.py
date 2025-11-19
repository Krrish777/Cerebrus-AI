#!/usr/bin/env python3
"""
Cerebrus AI - Advanced RAG System with Streamlit Interface
Based on mew.py with modern RAG implementation using Haystack components
"""

import streamlit as st
import os
import re
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Import our modules
from src.generation.rag import create_rag_generator
from src.document_processing.doc_processor import DocumentProcessor
from src.audio_processing.yt_audio_transcriber import create_youtube_transcription_pipeline
from src.web_scraping.firecrawl_only import SimpleWebScraper
from src.conversational_ai.simple_voice_app_with_rag import render_voice_interface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_interactive_citations(response_text: str, sources_used: List[Dict[str, Any]]) -> str:
    """Create interactive citations for response text"""
    logger.info(f"Processing interactive citations for {len(sources_used)} sources")

    citation_map = {}
    for i, source in enumerate(sources_used, 1):
        citation_map[i] = {
            'title': source.get('title', 'Unknown Source'),
            'url': source.get('url', ''),
            'content': source.get('content', '')[:200] + "...",
            'source_type': source.get('source_type', 'document')
        }
    
    def replace_citation(match):
        citation_num = int(match.group(1))
        if citation_num in citation_map:
            source_info = citation_map[citation_num]
            tooltip_content = f"<strong>{source_info['title']}</strong><br/>{source_info['content']}"
            return f"""
            <span class="citation-number" title="{source_info['title']}">
                [{citation_num}]
                <span class="citation-tooltip">
                    <div class="tooltip-source">{source_info['title']}</div>
                    <div class="tooltip-content">{source_info['content']}</div>
                </span>
            </span>
            """
        return match.group(0)
    
    # Replace all citation patterns [1], [2], etc.
    interactive_text = re.sub(r'\[(\d+)\]', replace_citation, response_text)
    return interactive_text

# Streamlit page configuration
st.set_page_config(
    page_title="Cerebrus AI - RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 20px;
    }
    
    .source-item {
        background: #2d3748;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #4299e1;
    }
    
    .source-title {
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    .source-meta {
        font-size: 12px;
        color: #a0aec0;
    }
    
    .chat-message {
        background: #2d3748;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    
    .user-message {
        background: #4299e1;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: #2d3748;
        margin-right: 20%;
        border-left: 3px solid #48bb78;
    }
    
    .citation-number {
        background: #4299e1;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        cursor: pointer;
        display: inline-block;
        margin: 0 2px;
        position: relative;
        transition: all 0.2s ease;
    }
    
    .citation-number:hover {
        background: #3182ce;
        transform: scale(1.1);
    }
    
    .upload-area {
        border: 2px dashed #4a5568;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        background: #1a202c;
        margin: 20px 0;
    }
    
    .upload-text {
        color: #a0aec0;
        font-size: 16px;
        margin-bottom: 20px;
    }
    
    .stButton > button {
        background: #4299e1;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 24px;
        font-weight: 500;
    }
    
    .source-count {
        background: #4a5568;
        color: #ffffff;
        border-radius: 12px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'rag_generator' not in st.session_state:
        st.session_state.rag_generator = None
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if 'show_source_dialog' not in st.session_state:
        st.session_state.show_source_dialog = False
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

def reset_chat():
    """Reset the chat history and session"""
    st.session_state.chat_history = []
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.success("Chat reset successfully!")
    st.rerun()

def initialize_rag_system():
    """Initialize the RAG system with error handling"""
    if st.session_state.system_initialized and st.session_state.rag_generator:
        return True
    
    try:
        with st.spinner("🔧 Initializing Cerebrus AI RAG System..."):
            # Check for required API keys
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                st.error("❌ GEMINI_API_KEY not found in environment variables")
                return False
            
            # Initialize RAG generator
            st.session_state.rag_generator = create_rag_generator(
                gemini_api_key=gemini_key,
                model_name="gemini-2.0-flash",
                ranking_top_k=8,
                retrieval_top_k=20
            )
            
            st.session_state.system_initialized = True
            st.success("✅ RAG System initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        logger.error(f"RAG initialization failed: {e}")
        return False

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to RAG system"""
    if not uploaded_files:
        return
    
    try:
        doc_processor = DocumentProcessor()
        documents = []
        
        for uploaded_file in uploaded_files:
            st.info(f"📄 Processing: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Process the document using DocumentProcessor.run() method
            try:
                result = doc_processor.run(sources=[temp_path])
                processed_docs = result.get('documents', [])
            except Exception as proc_error:
                st.warning(f"⚠️ Using fallback processing for {uploaded_file.name}: {proc_error}")
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                processed_docs = [{'content': content, 'meta': {}}]
            
            # Add metadata
            for doc in processed_docs:
                if isinstance(doc, dict):
                    doc['meta'] = doc.get('meta', {})
                    doc['meta'].update({
                        'filename': uploaded_file.name,
                        'source_type': 'uploaded_file',
                        'uploaded_at': datetime.now().isoformat(),
                        'file_size': len(uploaded_file.getvalue())
                    })
                    documents.append(doc)
                else:
                    # Handle Document objects
                    doc_dict = {
                        'content': doc.content if hasattr(doc, 'content') else str(doc),
                        'meta': {
                            'filename': uploaded_file.name,
                            'source_type': 'uploaded_file',
                            'uploaded_at': datetime.now().isoformat(),
                            'file_size': len(uploaded_file.getvalue())
                        }
                    }
                    documents.append(doc_dict)
            
            # Cleanup temp file
            os.remove(temp_path)
        
        # Add to RAG system
        if documents and st.session_state.rag_generator:
            success = st.session_state.rag_generator.add_documents(documents)
            if success:
                st.session_state.sources.extend(documents)
                st.success(f"✅ Added {len(documents)} documents to knowledge base")
            else:
                st.error("❌ Failed to add documents to knowledge base")
        
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        logger.error(f"File processing failed: {e}")

def process_urls(urls_text):
    """Process URLs and add to RAG system"""
    if not urls_text.strip():
        return
    
    try:
        firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
        if not firecrawl_key:
            st.error("❌ FIRECRAWL_API_KEY not found in environment variables")
            return
        
        # Parse URLs
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        if not urls:
            st.warning("⚠️ No valid URLs found")
            return
        
        scraper = SimpleWebScraper(api_key=firecrawl_key)
        documents = []
        
        for url in urls:
            st.info(f"🌐 Scraping: {url}")
            scraped_docs = scraper.scrape_url(url)
            
            # Add metadata
            for doc in scraped_docs:
                if hasattr(doc, 'meta'):
                    doc.meta['source_type'] = 'web_scraping'
                    doc.meta['scraped_at'] = datetime.now().isoformat()
                documents.extend([{'content': doc.content, 'meta': doc.meta}])
        
        # Add to RAG system
        if documents and st.session_state.rag_generator:
            success = st.session_state.rag_generator.add_documents(documents)
            if success:
                st.session_state.sources.extend(documents)
                st.success(f"✅ Added {len(documents)} documents from {len(urls)} URLs")
            else:
                st.error("❌ Failed to add web content to knowledge base")
                
    except Exception as e:
        st.error(f"❌ Error processing URLs: {str(e)}")
        logger.error(f"URL processing failed: {e}")

def process_youtube_video(youtube_url):
    """Process YouTube video and add transcription to RAG system"""
    if not youtube_url.strip():
        return
    
    try:
        assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not assemblyai_key:
            st.error("❌ ASSEMBLYAI_API_KEY not found in environment variables")
            return
        
        st.info(f"🎥 Processing YouTube video: {youtube_url}")
        
        # Initialize YouTube transcriber
        transcriber = create_youtube_transcription_pipeline(
            assemblyai_api_key=assemblyai_key,
            enable_advanced_features=True
        )
        
        # Process the video
        try:
            result = transcriber.run(sources=[youtube_url])
            documents = result.get('documents', [])
        except Exception as yt_error:
            st.warning(f"YouTube transcriber error: {yt_error}. Trying alternative approach...")
            # Alternative: create basic document
            documents = [{
                'content': f"YouTube video: {youtube_url}",
                'meta': {
                    'source_type': 'youtube_video',
                    'url': youtube_url,
                    'processed_at': datetime.now().isoformat(),
                    'note': 'Transcription failed, placeholder created'
                }
            }]
        
        # Convert to our format
        processed_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                processed_docs.append(doc)
            else:
                processed_docs.append({
                    'content': doc.content if hasattr(doc, 'content') else str(doc),
                    'meta': doc.meta if hasattr(doc, 'meta') else {
                        'source_type': 'youtube_video',
                        'url': youtube_url,
                        'processed_at': datetime.now().isoformat()
                    }
                })
        
        # Add to RAG system
        if processed_docs and st.session_state.rag_generator:
            success = st.session_state.rag_generator.add_documents(processed_docs)
            if success:
                st.session_state.sources.extend(processed_docs)
                st.success(f"✅ Added YouTube transcription ({len(processed_docs)} segments)")
            else:
                st.error("❌ Failed to add YouTube content to knowledge base")
                
    except Exception as e:
        st.error(f"❌ Error processing YouTube video: {str(e)}")
        logger.error(f"YouTube processing failed: {e}")

def process_text_input(text_content):
    """Process direct text input and add to RAG system"""
    if not text_content.strip():
        return
    
    try:
        document = {
            'content': text_content,
            'meta': {
                'source_type': 'manual_input',
                'title': f"Text Input {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'added_at': datetime.now().isoformat(),
                'word_count': len(text_content.split())
            }
        }
        
        if st.session_state.rag_generator:
            success = st.session_state.rag_generator.add_documents([document])
            if success:
                st.session_state.sources.append(document)
                st.success("✅ Text added to knowledge base")
            else:
                st.error("❌ Failed to add text to knowledge base")
                
    except Exception as e:
        st.error(f"❌ Error processing text: {str(e)}")
        logger.error(f"Text processing failed: {e}")

def render_sources_sidebar():
    """Render the sources sidebar"""
    with st.sidebar:
        st.header("📚 Knowledge Base")
        
        if st.session_state.sources:
            st.markdown(f'<span class="source-count">{len(st.session_state.sources)} sources</span>', 
                       unsafe_allow_html=True)
            
            # Display sources
            for i, source in enumerate(st.session_state.sources[-10:], 1):  # Show last 10
                meta = source.get('meta', {})
                title = meta.get('title', meta.get('filename', f"Source {i}"))
                source_type = meta.get('source_type', 'unknown')
                
                st.markdown(f"""
                <div class="source-item">
                    <div class="source-title">{title}</div>
                    <div class="source-meta">Type: {source_type}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No sources added yet")
        
        # Add source button
        if st.button("➕ Add Sources"):
            st.session_state.show_source_dialog = True

def render_source_upload_dialog():
    """Render the source upload dialog"""
    if st.session_state.show_source_dialog:
        st.header("📁 Add Knowledge Sources")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Files", "🌐 URLs", "🎥 YouTube", "📝 Text"])
        
        with tab1:
            st.subheader("Upload Files")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'docx', 'doc'],
                accept_multiple_files=True
            )
            
            if st.button("Upload Files", key="upload_files"):
                if uploaded_files:
                    process_uploaded_files(uploaded_files)
                else:
                    st.warning("Please select files to upload")
        
        with tab2:
            st.subheader("Add URLs")
            urls_text = st.text_area(
                "Enter URLs (one per line)",
                placeholder="https://example.com\nhttps://another-site.com"
            )
            
            if st.button("Scrape URLs", key="scrape_urls"):
                process_urls(urls_text)
        
        with tab3:
            st.subheader("YouTube Video")
            youtube_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            
            if st.button("Process Video", key="process_youtube"):
                process_youtube_video(youtube_url)
        
        with tab4:
            st.subheader("Direct Text Input")
            text_content = st.text_area(
                "Enter text content",
                placeholder="Paste your text here...",
                height=200
            )
            
            if st.button("Add Text", key="add_text"):
                process_text_input(text_content)
        
        if st.button("Close", key="close_dialog"):
            st.session_state.show_source_dialog = False

def render_chat_interface():
    """Render the main chat interface"""
    st.header("💬 Chat with Cerebrus AI")
    
    # Chat history
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Process citations if present
                content = message["content"]
                if message.get("sources"):
                    content = create_interactive_citations(content, message["sources"])
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Cerebrus AI:</strong> {content}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a question...",
            placeholder="What would you like to know?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", key="send_query")
    
    # Process user query
    if send_button and user_query.strip():
        if not st.session_state.system_initialized:
            st.error("Please initialize the system first")
            return
        
        if not st.session_state.rag_generator:
            st.error("RAG system not available")
            return
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        try:
            with st.spinner("🤔 Thinking..."):
                # Generate response
                result = st.session_state.rag_generator.generate_response(user_query)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result.response,
                    "sources": result.sources_used
                })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            logger.error(f"Response generation failed: {e}")

def main():
    """Main application function"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Cerebrus AI - Advanced RAG System</h1>', 
                unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.system_initialized:
        if not initialize_rag_system():
            st.stop()
    
    # Sidebar
    render_sources_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create tabs for different features
        tab1, tab2 = st.tabs(["💬 Text Chat", "🎙️ Voice Chat"])
        
        with tab1:
            # Chat interface
            render_chat_interface()
        
        with tab2:
            # Agora Conversational AI interface (voice)
            render_voice_interface(st.session_state.rag_generator)
    
    with col2:
        # Control panel
        st.header("🎛️ Controls")
        
        if st.button("🔄 Reset Chat"):
            reset_chat()
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.sources:
            st.metric("📚 Sources", len(st.session_state.sources))
        
        if st.session_state.chat_history:
            st.metric("💬 Messages", len(st.session_state.chat_history))
        
        # System status
        st.markdown("### System Status")
        if st.session_state.system_initialized:
            st.success("🟢 RAG System Online")
        else:
            st.error("🔴 RAG System Offline")
    
    # Source upload dialog
    if st.session_state.show_source_dialog:
        render_source_upload_dialog()

if __name__ == "__main__":
    main()