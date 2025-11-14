#!/usr/bin/env python3
"""
Test script for RAG system with InMemoryDocumentStore
This forces the system to use InMemoryDocumentStore directly.
"""

import os
import sys
from typing import List
from dataclasses import dataclass

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import required modules
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("Error: GEMINI_API_KEY not found in environment variables")
    sys.exit(1)

print(f"✅ Using Gemini API key: {gemini_api_key[:10]}...")

@dataclass
class RAGResult:
    """Data class for RAG results"""
    answer: str
    query: str
    retrieved_documents: List[Document]
    context_used: str
    sources: List[str]

class InMemoryRAGGenerator:
    """RAG Generator using InMemoryDocumentStore"""
    
    def __init__(self):
        # Initialize document store
        print("🔧 Initializing InMemoryDocumentStore...")
        self.document_store = InMemoryDocumentStore()
        
        # Initialize components
        print("🔧 Initializing retriever...")
        self.retriever = InMemoryBM25Retriever(document_store=self.document_store)
        
        print("🔧 Initializing ranker...")
        self.ranker = FastembedRanker(
            model_name="Xenova/ms-marco-MiniLM-L-6-v2",
            top_k=5
        )
        
        # Build prompt template
        self.prompt_template = """
        Answer the question based ONLY on the provided context. If the context doesn't contain enough information to answer the question, say "I cannot answer based on the provided context."
        
        Context:
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %}
        
        Question: {{ query }}
        Answer: 
        """
        
        print("🔧 Initializing prompt builder...")
        self.prompt_builder = PromptBuilder(template=self.prompt_template)
        
        print("🔧 Initializing Google Gemini generator...")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.generator = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",  # Use the verified working model
            api_key=Secret.from_token(gemini_api_key)
        )
        
        # Create pipeline (without LLM - we'll call it separately)
        print("🔧 Creating RAG pipeline...")
        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("ranker", self.ranker)  
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        
        # Connect pipeline 
        self.pipeline.connect("retriever", "ranker")
        self.pipeline.connect("ranker", "prompt_builder")
        
        print("✅ RAG system initialized successfully!")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the document store"""
        print(f"📚 Adding {len(documents)} documents to document store...")
        self.document_store.write_documents(documents)
        print(f"✅ Documents added. Total documents: {len(self.document_store.filter_documents())}")
    
    def generate_response(self, query: str, top_k: int = 10) -> RAGResult:
        """Generate a response using the RAG pipeline"""
        print(f"🔍 Processing query: {query}")
        
        try:
            # Run the first part of the pipeline (retrieval + ranking + prompt building)
            response = self.pipeline.run({
                "retriever": {"query": query, "top_k": top_k},
                "ranker": {"query": query},
                "prompt_builder": {"query": query}
            })
            
            # Debug: print the response structure
            print(f"🔍 Pipeline response keys: {list(response.keys())}")
            for key in response.keys():
                print(f"  {key}: {list(response[key].keys())}")
            
            # Get the generated prompt
            prompt = response["prompt_builder"]["prompt"]
            retrieved_docs = response["retriever"]["documents"] if "retriever" in response else []
            ranked_docs = response["ranker"]["documents"] if "ranker" in response else []
            
            # Convert prompt to ChatMessage and run the generator
            messages = [ChatMessage.from_user(prompt)]
            llm_response = self.generator.run(messages=messages)
            
            # Extract results
            answer = llm_response["replies"][0].text  # Use .text instead of .content
            
            # Build context and sources
            context_used = "\n\n".join([doc.content for doc in ranked_docs])
            sources = [doc.meta.get("source", "Unknown") for doc in ranked_docs if doc.meta]
            
            return RAGResult(
                answer=answer,
                query=query,
                retrieved_documents=ranked_docs,
                context_used=context_used,
                sources=sources
            )
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return RAGResult(
                answer=f"Error: {str(e)}",
                query=query,
                retrieved_documents=[],
                context_used="",
                sources=[]
            )

def create_sample_documents() -> List[Document]:
    """Create sample documents for testing"""
    documents = [
        Document(
            content="Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            meta={"source": "python_basics.txt", "type": "programming"}
        ),
        Document(
            content="Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn and make predictions from data.",
            meta={"source": "ml_intro.txt", "type": "ai"}
        ),
        Document(
            content="The Haystack framework is an open-source Python framework for building search systems and question-answering applications using state-of-the-art NLP models.",
            meta={"source": "haystack_info.txt", "type": "framework"}
        ),
        Document(
            content="ElasticSearch is a distributed search and analytics engine built on Apache Lucene. It's commonly used for log analytics, full-text search, and data analysis.",
            meta={"source": "elasticsearch_overview.txt", "type": "database"}
        ),
        Document(
            content="Google's Gemini is a family of large language models that can understand and generate human-like text, making it useful for various AI applications.",
            meta={"source": "gemini_info.txt", "type": "ai"}
        )
    ]
    
    print(f"📄 Created {len(documents)} sample documents")
    return documents

def main():
    """Main test function"""
    print("🚀 Starting InMemory RAG System Test")
    print("=" * 50)
    
    # Initialize RAG system
    rag = InMemoryRAGGenerator()
    
    # Add sample documents
    documents = create_sample_documents()
    rag.add_documents(documents)
    
    # Test queries
    test_queries = [
        "What is Python?",
        "Tell me about machine learning", 
        "What is Haystack framework?",
        "How does Google Gemini work?"
    ]
    
    print("\n🧪 Testing RAG System")
    print("=" * 50)
    
    import time
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {query}")
        print("-" * 40)
        
        # Add delay between API calls to avoid rate limiting
        if i > 1:
            print("⏳ Waiting 10 seconds to avoid rate limits...")
            time.sleep(10)
        
        result = rag.generate_response(query)
        
        print(f"💬 Answer: {result.answer}")
        print(f"📚 Sources: {', '.join(result.sources)}")
        print(f"🔍 Documents retrieved: {len(result.retrieved_documents)}")
        
        if result.retrieved_documents:
            print("📄 Top retrieved document:")
            top_doc = result.retrieved_documents[0]
            content = top_doc.content or ""
            print(f"   Content: {content[:100]}...")
            print(f"   Source: {top_doc.meta.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()