"""
Simple test script for Qdrant Vector Database

This script tests the Qdrant implementation by running it directly
as a Python script to avoid import issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_qdrant_basic():
    """Basic test of Qdrant functionality."""
    
    print("🧪 Testing Qdrant Vector Database")
    print("=" * 50)
    
    try:
        # Test imports
        from vector_database.qdrant_db import QdrantVectorDB, create_qdrant_vector_db
        print("✅ Qdrant imports successful")
        
        # Test initialization
        vector_db = create_qdrant_vector_db(
            storage_path="./storage/simple_test_qdrant",
            collection_name="simple_test",
            embedding_dim=384,
            recreate_index=True
        )
        print("✅ Qdrant database initialized")
        
        # Test collection stats (empty)
        stats = vector_db.get_collection_stats()
        print(f"📊 Initial stats: {stats['total_documents']} documents")
        
        # Test close
        vector_db.close()
        print("✅ Database closed successfully")
        
        print("\n🎉 Basic Qdrant test completed successfully!")
        return True
        
    except ImportError as e:
        if "qdrant" in str(e).lower():
            print(f"⚠️ Qdrant dependencies not installed: {e}")
            print("💡 Install with: pip install qdrant-haystack")
            return False
        else:
            print(f"❌ Import error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_manual_data():
    """Test with manually created data to avoid embedding dependencies."""
    
    print("\n🧪 Testing with Manual Data")
    print("=" * 30)
    
    try:
        from vector_database.qdrant_db import QdrantVectorDB, Document
        import numpy as np
        
        # Create vector database
        vector_db = QdrantVectorDB(
            storage_path="./storage/manual_test_qdrant",
            collection_name="manual_test",
            embedding_dim=3,  # Small dimension for testing
            recreate_index=True
        )
        
        # Create test documents with manual embeddings
        test_docs = [
            Document(
                content="Test document 1 about AI",
                meta={"topic": "AI", "source": "test1.txt"},
                id="doc_1"
            ),
            Document(
                content="Test document 2 about ML",
                meta={"topic": "ML", "source": "test2.txt"},
                id="doc_2"
            )
        ]
        
        # Add manual embeddings
        test_docs[0].embedding = [0.1, 0.2, 0.3]
        test_docs[1].embedding = [0.4, 0.5, 0.6]
        
        # Test insertion
        inserted_ids = vector_db.insert_documents(test_docs)
        print(f"✅ Inserted {len(inserted_ids)} documents")
        
        # Test search
        query_vector = [0.1, 0.2, 0.3]  # Similar to first document
        results = vector_db.search(query_vector, top_k=2)
        print(f"✅ Search found {len(results)} results")
        
        if results:
            print(f"   Best match: {results[0]['content'][:30]}...")
            print(f"   Score: {results[0]['score']:.4f}")
        
        # Test stats
        stats = vector_db.get_collection_stats()
        print(f"📊 Final stats: {stats['total_documents']} documents")
        
        vector_db.close()
        print("✅ Manual data test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Manual data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Qdrant Vector Database Tests")
    print("=" * 60)
    
    # Run basic test
    basic_success = test_qdrant_basic()
    
    # Run manual data test if basic test passed
    manual_success = False
    if basic_success:
        manual_success = test_with_manual_data()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 20)
    print(f"Basic test: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Manual data test: {'✅ PASS' if manual_success else '❌ FAIL'}")
    
    if basic_success and manual_success:
        print("\n🎉 All tests passed! Qdrant Vector Database is working correctly.")
    elif basic_success:
        print("\n⚠️ Basic functionality works, but manual data test failed.")
    else:
        print("\n❌ Basic test failed. Check dependencies and installation.")
    
    print("\n💡 Next steps:")
    print("   1. Install qdrant-haystack if not installed")
    print("   2. Run integration tests with real embeddings")
    print("   3. Test with document processing pipeline")