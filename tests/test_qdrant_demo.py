"""
Demo Script for Qdrant Vector Database with Cerebrus AI

This script demonstrates how to use the QdrantVectorDB implementation
with real documents and embeddings.

Usage:
    python test_qdrant_demo.py

Requirements:
    - qdrant-haystack
    - haystack-ai
    - sentence-transformers
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_database.qdrant_db import QdrantVectorDB, create_qdrant_vector_db
from embeddings.embedding_generator import EmbeddingGenerator
from core.logging import CustomLogger


def main():
    """Main demo function."""
    
    # Initialize logger
    logger = CustomLogger().get_logger(__name__)
    
    print("🚀 Qdrant Vector Database Demo")
    print("=" * 50)
    
    try:
        # Step 1: Initialize components
        print("\n📋 Step 1: Initializing Components")
        print("-" * 30)
        
        # Create embedding generator
        print("🔗 Creating embedding generator...")
        embedding_generator = EmbeddingGenerator(
            model_name="BAAI/bge-small-en-v1.5",
            batch_size=4
        )
        print("✅ Embedding generator ready")
        
        # Create vector database
        print("💾 Creating Qdrant vector database...")
        vector_db = create_qdrant_vector_db(
            storage_path="./storage/demo_qdrant_db",
            collection_name="demo_collection",
            embedding_dim=384,
            recreate_index=True
        )
        print("✅ Qdrant vector database ready")
        
        # Step 2: Prepare sample documents
        print("\n📋 Step 2: Preparing Sample Documents")
        print("-" * 30)
        
        sample_documents = [
            {
                "content": """
                Machine Learning is a subset of artificial intelligence (AI) that provides systems 
                the ability to automatically learn and improve from experience without being 
                explicitly programmed. Machine learning focuses on the development of computer 
                programs that can access data and use it to learn for themselves.
                """,
                "metadata": {
                    "source": "ml_textbook.pdf",
                    "chapter": "Introduction",
                    "topic": "machine_learning",
                    "difficulty": "beginner"
                }
            },
            {
                "content": """
                Natural Language Processing (NLP) is a branch of artificial intelligence that 
                helps computers understand, interpret and manipulate human language. NLP draws 
                from many disciplines, including computer science and computational linguistics, 
                in its pursuit to fill the gap between human communication and computer understanding.
                """,
                "metadata": {
                    "source": "nlp_handbook.pdf",
                    "chapter": "Fundamentals",
                    "topic": "natural_language_processing",
                    "difficulty": "intermediate"
                }
            },
            {
                "content": """
                Computer Vision is a field of artificial intelligence that trains computers to 
                interpret and understand the visual world. Using digital images from cameras 
                and videos and deep learning models, machines can accurately identify and 
                classify objects — and then react to what they "see."
                """,
                "metadata": {
                    "source": "cv_guide.pdf",
                    "chapter": "Introduction",
                    "topic": "computer_vision",
                    "difficulty": "intermediate"
                }
            },
            {
                "content": """
                Deep Learning is part of a broader family of machine learning methods based on 
                artificial neural networks with representation learning. Learning can be supervised, 
                semi-supervised or unsupervised. Deep learning architectures such as deep neural 
                networks have been applied to fields including computer vision, natural language processing.
                """,
                "metadata": {
                    "source": "dl_manual.pdf",
                    "chapter": "Neural Networks",
                    "topic": "deep_learning", 
                    "difficulty": "advanced"
                }
            },
            {
                "content": """
                Reinforcement Learning is an area of machine learning concerned with how software 
                agents ought to take actions in an environment in order to maximize the notion of 
                cumulative reward. Reinforcement learning is one of three basic machine learning 
                paradigms, alongside supervised learning and unsupervised learning.
                """,
                "metadata": {
                    "source": "rl_textbook.pdf",
                    "chapter": "Fundamentals",
                    "topic": "reinforcement_learning",
                    "difficulty": "advanced"
                }
            }
        ]
        
        print(f"📄 Created {len(sample_documents)} sample documents")
        
        # Step 3: Generate embeddings
        print("\n📋 Step 3: Generating Embeddings")
        print("-" * 30)
        
        # Create Haystack documents
        documents = embedding_generator.create_documents_from_texts(
            texts=[doc["content"] for doc in sample_documents],
            metadatas=[doc["metadata"] for doc in sample_documents]
        )
        
        print(f"🔗 Generating embeddings for {len(documents)} documents...")
        embedded_docs = embedding_generator.embed_documents(documents)
        print(f"✅ Generated {len(embedded_docs)} embeddings")
        
        # Step 4: Insert into vector database
        print("\n📋 Step 4: Storing in Vector Database")
        print("-" * 30)
        
        print("💾 Inserting documents into Qdrant...")
        inserted_ids = vector_db.insert_embedded_documents(embedded_docs)
        print(f"✅ Inserted {len(inserted_ids)} documents")
        
        # Step 5: Demonstrate search capabilities
        print("\n📋 Step 5: Testing Search Capabilities")
        print("-" * 30)
        
        # Test queries
        test_queries = [
            "What is artificial intelligence?",
            "How do computers understand images?", 
            "Explain neural networks and deep learning",
            "What is reinforcement learning?",
            "How does machine learning work?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")
            
            results = vector_db.search_with_query_text(
                query_text=query,
                embedding_generator=embedding_generator,
                top_k=3
            )
            
            print(f"📊 Found {len(results)} results:")
            
            for j, result in enumerate(results, 1):
                print(f"   {j}. Score: {result['score']:.4f}")
                print(f"      Topic: {result['metadata'].get('topic', 'Unknown')}")
                print(f"      Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"      Content: {result['content'][:100].strip()}...")
                print()
        
        # Step 6: Demonstrate filtered search
        print("\n📋 Step 6: Testing Filtered Search")
        print("-" * 30)
        
        query = "machine learning techniques"
        
        # Search for beginner content only
        print(f"🔍 Searching for beginner-level content about: '{query}'")
        beginner_results = vector_db.search_with_query_text(
            query_text=query,
            embedding_generator=embedding_generator,
            top_k=3,
            filters={"field": "difficulty", "operator": "==", "value": "beginner"}
        )
        
        print(f"📊 Beginner results ({len(beginner_results)}):")
        for result in beginner_results:
            print(f"   • {result['metadata'].get('topic')} (Difficulty: {result['metadata'].get('difficulty')})")
        
        # Search for advanced content only
        print(f"\n🔍 Searching for advanced-level content about: '{query}'")
        advanced_results = vector_db.search_with_query_text(
            query_text=query,
            embedding_generator=embedding_generator,
            top_k=3,
            filters={"field": "difficulty", "operator": "==", "value": "advanced"}
        )
        
        print(f"📊 Advanced results ({len(advanced_results)}):")
        for result in advanced_results:
            print(f"   • {result['metadata'].get('topic')} (Difficulty: {result['metadata'].get('difficulty')})")
        
        # Step 7: Collection statistics
        print("\n📋 Step 7: Collection Statistics")
        print("-" * 30)
        
        stats = vector_db.get_collection_stats()
        print("📊 Collection Statistics:")
        for key, value in stats.items():
            if key != 'hnsw_config':  # Skip complex config for readability
                print(f"   • {key}: {value}")
        
        # Step 8: Demonstrate citation information
        print("\n📋 Step 8: Citation Information")
        print("-" * 30)
        
        query = "neural networks"
        results = vector_db.search_with_query_text(
            query_text=query,
            embedding_generator=embedding_generator,
            top_k=2
        )
        
        print(f"🔍 Citation info for query: '{query}'")
        for i, result in enumerate(results, 1):
            citation = result['citation']
            print(f"   {i}. Source: {citation.get('source_file', 'N/A')}")
            print(f"      Type: {citation.get('source_type', 'N/A')}")
            print(f"      Chunk: {citation.get('chunk_index', 'N/A')}")
        
        print("\n✅ Demo completed successfully!")
        print("🎉 Qdrant Vector Database is working correctly!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install qdrant-haystack sentence-transformers")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")
        
    finally:
        try:
            vector_db.close()
            print("🔒 Database connection closed")
        except:
            pass


if __name__ == "__main__":
    main()