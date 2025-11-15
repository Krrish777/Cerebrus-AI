"""
FastAPI Webhook Server for Agora Conversational AI

This server handles LLM requests from Agora and integrates them with the Cerebrus AI RAG system.
It provides a webhook endpoint that Agora can call to get AI responses powered by your knowledge base.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

# Try to import RAG generator, make it optional
try:
    from src.generation.rag import create_rag_generator
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG system not available: {e}")
    RAG_AVAILABLE = False
    
    def create_rag_generator(*args, **kwargs):
        """Placeholder when RAG is not available"""
        return None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Cerebrus AI - Agora LLM Webhook",
    description="Webhook server for Agora Conversational AI integration with RAG system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG generator
rag_generator = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG generator on startup"""
    global rag_generator
    try:
        logger.info("🚀 Initializing Agora LLM Webhook Server...")
        logger.info("📡 Server ready to receive LLM requests from Agora")
        
        if RAG_AVAILABLE:
            logger.info("🧠 RAG system available and will be initialized on first request")
        else:
            logger.warning("⚠️ RAG system not available - responses will be basic")
        
        logger.info("✅ Webhook server startup complete")
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Webhook server shutting down...")


class ChatMessage(BaseModel):
    """Chat message model for OpenAI-compatible format"""
    role: str
    content: str


class LLMRequest(BaseModel):
    """LLM request model from Agora"""
    messages: List[ChatMessage]
    model: Optional[str] = "gpt-4o-mini"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False


class LLMResponse(BaseModel):
    """LLM response model for Agora"""
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_generator
    
    logger.info("🚀 Starting Agora LLM Webhook Server")
    logger.info("🔧 Initializing RAG system...")
    
    try:
        # Check for required API keys
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            logger.error("❌ GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY is required")
        
        # Initialize RAG generator
        rag_generator = create_rag_generator(
            gemini_api_key=gemini_key,
            model_name="gemini-2.0-flash",
            ranking_top_k=8,
            retrieval_top_k=20
        )
        
        logger.info("✅ RAG system initialized successfully")
        logger.info("🎙️ Agora LLM webhook server ready")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Cerebrus AI - Agora LLM Webhook Server",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "rag_system": "ready" if rag_generator else "not_initialized"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Cerebrus AI - Agora LLM Webhook",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "llm_webhook": "/llm-webhook/{agent_name}",
            "rag_status": "/rag/status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global rag_generator
    
    # Check RAG system status
    rag_status = "not_initialized"
    rag_error = None
    
    try:
        if rag_generator is None:
            # Try to initialize RAG generator for health check
            rag_generator = create_rag_generator()
            rag_status = "ready" if rag_generator else "failed"
        else:
            rag_status = "ready"
    except Exception as e:
        rag_status = "error"
        rag_error = str(e)
        logger.warning(f"RAG system health check failed: {e}")
    
    response = {
        "status": "healthy",
        "rag_system": rag_status,
        "timestamp": datetime.now().isoformat(),
        "service": "Cerebrus AI - Agora LLM Webhook",
        "version": "1.0.0"
    }
    
    if rag_error:
        response["rag_error"] = rag_error
    
    return response


@app.get("/rag/status")
async def rag_status():
    """RAG system status endpoint"""
    global rag_generator
    
    if not RAG_AVAILABLE:
        return {
            "status": "not_available",
            "initialized": False,
            "error": "RAG dependencies not installed",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        if rag_generator is None:
            # Try to initialize
            logger.info("Initializing RAG generator for status check...")
            rag_generator = create_rag_generator()
        
        if rag_generator:
            return {
                "status": "ready",
                "type": type(rag_generator).__name__,
                "initialized": True,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "not_available",
                "initialized": False,
                "error": "RAG generator could not be created",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error checking RAG status: {e}")
        return {
            "status": "error",
            "initialized": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/llm-webhook/{agent_name}")
async def llm_webhook(agent_name: str, request: LLMRequest) -> LLMResponse:
    """
    Main LLM webhook endpoint for Agora Conversational AI
    
    This endpoint receives LLM requests from Agora and processes them using the RAG system.
    
    Args:
        agent_name: Name of the Agora agent making the request
        request: LLM request data from Agora
        
    Returns:
        LLM response in OpenAI-compatible format
    """
    
    logger.info(f"🎙️ Received LLM request from agent: {agent_name}")
    
    try:
        # Validate RAG system
        if not rag_generator:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized"
            )
        
        # Extract user message from conversation
        user_message = ""
        conversation_context = []
        
        for message in request.messages:
            conversation_context.append(f"{message.role}: {message.content}")
            if message.role == "user":
                user_message = message.content
        
        if not user_message:
            logger.warning("No user message found in request")
            return LLMResponse(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I didn't receive a clear message. Could you please repeat your question?"
                        }
                    }
                ]
            )
        
        logger.info(f"📝 Processing query: {user_message[:100]}...")
        
        # Generate response using RAG system
        rag_result = rag_generator.generate_response(user_message)
        response_text = rag_result.response
        
        # Enhance response for voice interaction
        # Make it more conversational and suitable for voice
        if rag_result.sources_used:
            # Add source information in a voice-friendly way
            source_count = len(rag_result.sources_used)
            if source_count == 1:
                response_text += f" This information comes from {rag_result.sources_used[0].get('title', 'your knowledge base')}."
            elif source_count <= 3:
                source_titles = [source.get('title', 'a document') for source in rag_result.sources_used[:3]]
                response_text += f" This information is based on {', '.join(source_titles[:-1])} and {source_titles[-1]}."
            else:
                response_text += f" This information comes from {source_count} sources in your knowledge base."
        
        # Optimize for voice - keep responses concise and conversational
        if len(response_text) > 500:
            # For very long responses, add a natural break
            sentences = response_text.split('. ')
            if len(sentences) > 3:
                # Keep first few sentences and add a natural transition
                short_response = '. '.join(sentences[:3]) + '. '
                short_response += "Would you like me to elaborate on any particular aspect?"
                response_text = short_response
        
        # Log successful processing
        logger.info(f"✅ Generated response ({len(response_text)} chars)")
        logger.info(f"📚 Used {len(rag_result.sources_used)} sources")
        
        # Return OpenAI-compatible response
        return LLMResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    }
                }
            ],
            usage={
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_message.split()) + len(response_text.split())
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing LLM request: {e}")
        
        # Return error response in proper format
        error_message = "I'm sorry, I encountered an error while processing your request. Please try again."
        
        return LLMResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": error_message
                    }
                }
            ]
        )


@app.post("/llm-webhook")
async def llm_webhook_default(request: LLMRequest) -> LLMResponse:
    """Default LLM webhook endpoint (without agent name)"""
    return await llm_webhook("default", request)


@app.get("/agents/{agent_name}/status")
async def get_agent_status(agent_name: str):
    """Get status of a specific agent"""
    return {
        "agent_name": agent_name,
        "status": "active",
        "rag_system": "ready" if rag_generator else "not_initialized",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/agents/{agent_name}/test")
async def test_agent_llm(agent_name: str, message: str = "Hello, can you help me?"):
    """Test LLM functionality for an agent"""
    
    test_request = LLMRequest(
        messages=[
            ChatMessage(role="system", content="You are a helpful AI assistant."),
            ChatMessage(role="user", content=message)
        ]
    )
    
    return await llm_webhook(agent_name, test_request)


@app.get("/rag/status")
async def rag_status():
    """Get RAG system status"""
    if not rag_generator:
        return {"status": "not_initialized", "message": "RAG system not available"}
    
    return {
        "status": "ready",
        "message": "RAG system is operational",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/rag/add-documents")
async def add_documents(documents: List[Dict[str, Any]], background_tasks: BackgroundTasks):
    """Add documents to the RAG system"""
    
    if not rag_generator:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    def add_docs_background():
        try:
            success = rag_generator.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to RAG system: {success}")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
    
    background_tasks.add_task(add_docs_background)
    
    return {
        "message": f"Queued {len(documents)} documents for processing",
        "status": "accepted"
    }


# Development server configuration
def create_webhook_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
) -> None:
    """
    Create and run the webhook server
    
    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    
    logger.info(f"🚀 Starting webhook server on {host}:{port}")
    
    uvicorn.run(
        "src.conversational_ai.webhook_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Run the server
    create_webhook_server(
        host="0.0.0.0",
        port=int(os.getenv("WEBHOOK_PORT", "8000")),
        reload=True
    )