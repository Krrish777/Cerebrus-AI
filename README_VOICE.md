# Cerebrus AI 🧠🎙️

**Ultra-low latency voice conversations with AI agents powered by your knowledge base**

Cerebrus AI combines Retrieval-Augmented Generation (RAG) with Agora Conversational AI to create intelligent voice assistants that can answer questions from your documents with natural speech interaction.

## ✨ Features

### 📚 Document Processing & RAG
- **Multi-format support**: PDF, TXT, DOCX, and more
- **Intelligent chunking**: Smart text segmentation for optimal retrieval
- **Vector embeddings**: Advanced semantic search capabilities  
- **Elasticsearch integration**: Scalable document storage with InMemory fallback

### 🎙️ Voice Conversational AI
- **Ultra-low latency**: ~650ms voice response time
- **Natural interruptions**: Speak anytime, AI handles interruptions gracefully
- **Multi-language support**: Voice synthesis and recognition in multiple languages
- **RAG integration**: Voice responses powered by your knowledge base
- **Real-time analytics**: Monitor conversation performance and usage

### 🔧 Easy Integration
- **Streamlit interface**: User-friendly web interface for management
- **RESTful webhooks**: Easy integration with existing systems
- **Multi-platform SDKs**: iOS, Android, Web, Unity support
- **Real-time channels**: Group voice conversations with multiple participants

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd cerebrus-ai

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your API keys
```

### 2. API Keys Setup

#### Required Services:
- **Agora Account**: [console.agora.io](https://console.agora.io) - For voice infrastructure
- **OpenAI/Gemini**: API key for LLM responses
- **Azure Speech**: TTS/ASR services for voice processing

#### Edit `.env` file:
```env
AGORA_APP_ID=your_app_id
AGORA_CUSTOMER_ID=your_customer_id  
AGORA_CUSTOMER_SECRET=your_customer_secret
OPENAI_API_KEY=your_openai_key
AZURE_TTS_API_KEY=your_azure_tts_key
AZURE_ASR_API_KEY=your_azure_asr_key
```

### 3. Launch Complete System

```bash
# Option 1: Use startup script (recommended)
python start_cerebrus.py

# Option 2: Manual startup
# Terminal 1: Start webhook server
python manage_webhook.py start --reload

# Terminal 2: Start Streamlit app
streamlit run main.py --server.address 192.168.1.38 --server.port 8501
```

### 4. Access the Interface

- **Streamlit App**: http://192.168.1.38:8501
- **Webhook Server**: http://localhost:8000

## 📖 Usage Guide

### Document Processing
1. **Upload documents** via the Streamlit interface
2. **Process documents** - they'll be automatically chunked and indexed
3. **Test queries** to verify your knowledge base is working

### Voice Agent Setup
1. **Configure Agora credentials** in the Voice Chat tab
2. **Start the webhook server** (handles LLM requests from Agora)
3. **Create voice agents** with custom configurations
4. **Join voice channels** to start conversations

### Voice Conversation Flow
```
User speaks → Agora ASR → Text query → RAG system → 
LLM response → Agora TTS → Voice response
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   RAG System    │    │  Agora Voice    │
│   Interface     │◄──►│  (Knowledge     │◄──►│   Platform      │
│                 │    │   Base)         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │  Elasticsearch  │    │   Webhook       │
│   Processing    │    │   (Optional)    │    │   Server        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎛️ Configuration Options

### Agent Configuration
- **Voice selection**: Choose from multiple neural voices
- **Interrupt handling**: Control how AI responds to interruptions
- **Timeout settings**: Configure idle timeouts
- **RAG integration**: Enable/disable knowledge base access
- **Custom system messages**: Personalize AI personality

### Voice Settings
- **Language**: Multi-language voice synthesis
- **Voice style**: Various speaking styles and emotions
- **Speech rate**: Adjust speaking speed
- **Audio quality**: Configure bitrate and sample rate

## 🔌 Integration Examples

### Python SDK
```python
from src.conversational_ai.agora_ai import create_agora_conversational_ai

# Initialize Agora AI
agora_ai = create_agora_conversational_ai(
    app_id="your_app_id",
    customer_id="your_customer_id", 
    customer_secret="your_customer_secret",
    rag_generator=your_rag_generator
)

# Create voice agent
agent_config = create_default_agent_config()
agent_id = await agora_ai.start_agent(agent_config, use_rag=True)
```

### Webhook Integration
```python
# Your webhook endpoint receives LLM requests from Agora
@app.post("/llm-webhook")
async def process_voice_query(request: LLMRequest):
    # Process with RAG system
    response = rag_generator.generate_response(request.messages[-1].content)
    return {"choices": [{"message": {"role": "assistant", "content": response}}]}
```

### JavaScript/Web Integration
```javascript
import { RtmClient } from 'agora-rtm-sdk'

const client = new RtmClient({
  appId: 'your_app_id'
})

// Join voice channel
await client.login(token)
await client.join(channelName)
```

## 📊 Monitoring & Analytics

### Real-time Metrics
- **Response latency**: Track voice response times
- **Agent uptime**: Monitor agent availability
- **Conversation volume**: Track usage patterns
- **Error rates**: Monitor system health

### Performance Optimization
- **Cache frequently accessed documents**
- **Optimize chunk sizes** for your content type  
- **Use appropriate timeout values**
- **Monitor webhook server performance**

## 🛠️ Advanced Features

### Custom Voice Models
```python
# Configure custom TTS voice
tts_config = TTSConfig(
    voice_name="en-US-AriaNeural",
    style="conversational",
    rate="0.9",
    pitch="+2Hz"
)
```

### Advanced RAG Configuration  
```python
# Fine-tune retrieval parameters
rag_generator = create_rag_generator(
    ranking_top_k=10,      # Number of top results to re-rank
    retrieval_top_k=25,    # Initial retrieval pool size
    chunk_size=500,        # Document chunk size
    chunk_overlap=50       # Overlap between chunks
)
```

### Webhook Customization
```python
# Add custom processing to webhook
@app.post("/llm-webhook/{agent_name}")
async def custom_webhook(agent_name: str, request: LLMRequest):
    # Add custom logic here
    if agent_name == "support_agent":
        # Route to support knowledge base
        pass
    elif agent_name == "sales_agent":
        # Route to product information
        pass
    
    return await process_with_rag(request)
```

## 🔍 Troubleshooting

### Common Issues

#### Webhook Server Not Starting
```bash
# Check if port is in use
netstat -an | findstr :8000

# Start with different port
python manage_webhook.py start --port 8001
```

#### Voice Agent Not Responding
1. **Check webhook URL** in agent configuration
2. **Verify API keys** are correctly configured
3. **Test webhook endpoint** manually
4. **Check Agora console** for agent status

#### RAG System Issues
1. **Verify documents are uploaded** and processed
2. **Check Elasticsearch connection** (or InMemory fallback)
3. **Test queries** in text mode first
4. **Check API key limits**

### Debug Mode
```bash
# Start with debug logging
python start_cerebrus.py --debug

# Check webhook server logs
python manage_webhook.py logs
```

## 📚 API Reference

### REST Endpoints

#### Webhook Server
- `GET /health` - Health check
- `POST /llm-webhook` - Main LLM processing endpoint
- `GET /rag/status` - RAG system status
- `POST /rag/add-documents` - Add documents to knowledge base

#### Agent Management  
- `POST /agents/{name}/create` - Create new agent
- `DELETE /agents/{name}` - Stop agent
- `GET /agents/{name}/status` - Get agent status
- `GET /agents/{name}/analytics` - Get usage analytics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Agora.io** - Real-time voice infrastructure
- **OpenAI/Google** - Large Language Models  
- **Azure Speech Services** - Voice synthesis and recognition
- **Elasticsearch** - Vector search capabilities
- **Streamlit** - Beautiful web interfaces

---

**Built with ❤️ for intelligent voice interactions**