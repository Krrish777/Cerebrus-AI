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
    create_rag_generator = None

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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Cerebrus AI - Agora LLM Webhook",
        "status": "running",
        "version": "1.0.0",
        "rag_available": RAG_AVAILABLE,
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
    rag_status = "not_available" if not RAG_AVAILABLE else "not_initialized"
    rag_error = None
    
    if RAG_AVAILABLE and create_rag_generator:
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
        "rag_available": RAG_AVAILABLE,
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
        if rag_generator is None and create_rag_generator:
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


def generate_fallback_response(user_message: str) -> str:
    """Generate a fallback response when RAG is not available"""
    return f"I understand you're asking about: '{user_message}'. However, I'm currently running in basic mode without access to the knowledge base. Please check with your system administrator about the RAG system status."


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
        
        response_text = ""
        
        # Try to use RAG system if available
        if RAG_AVAILABLE and create_rag_generator:
            try:
                # Initialize RAG generator if not already done
                global rag_generator
                if rag_generator is None:
                    logger.info("Initializing RAG generator...")
                    rag_generator = create_rag_generator()
                
                if rag_generator:
                    # Generate response using RAG system
                    rag_result = rag_generator.generate_response(user_message)
                    response_text = rag_result.response
                    
                    # Enhance response for voice interaction
                    # Make it more conversational and suitable for voice
                    if hasattr(rag_result, 'sources_used') and rag_result.sources_used:
                        # Add source information in a voice-friendly way
                        source_count = len(rag_result.sources_used)
                        if source_count == 1:
                            source_title = rag_result.sources_used[0].get('title', 'your knowledge base') if isinstance(rag_result.sources_used[0], dict) else 'your knowledge base'
                            response_text += f" This information comes from {source_title}."
                        elif source_count <= 3:
                            source_titles = []
                            for source in rag_result.sources_used[:3]:
                                if isinstance(source, dict):
                                    source_titles.append(source.get('title', 'a document'))
                                else:
                                    source_titles.append('a document')
                            
                            if len(source_titles) > 1:
                                response_text += f" This information is based on {', '.join(source_titles[:-1])} and {source_titles[-1]}."
                        else:
                            response_text += f" This information comes from {source_count} sources in your knowledge base."
                    
                    logger.info(f"✅ Generated RAG response ({len(response_text)} chars)")
                else:
                    response_text = generate_fallback_response(user_message)
                    logger.warning("RAG generator could not be initialized, using fallback")
                    
            except Exception as e:
                logger.error(f"RAG system error: {e}")
                response_text = generate_fallback_response(user_message)
        else:
            response_text = generate_fallback_response(user_message)
            logger.info("Using fallback response (RAG not available)")
        
        # Optimize for voice - keep responses concise and conversational
        if len(response_text) > 500:
            # For very long responses, add a natural break
            sentences = response_text.split('. ')
            if len(sentences) > 3:
                # Keep first few sentences and add a natural transition
                short_response = '. '.join(sentences[:3]) + '. '
                short_response += "Would you like me to elaborate on any particular aspect?"
                response_text = short_response
        
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
        "src.conversational_ai.webhook_server_fixed:app",
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