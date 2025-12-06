"""
Example usage of the refactored embedding module.

This script demonstrates how to use the new embedding architecture
with configuration files and dependency injection.
"""

from pathlib import Path

from src.embeddings import EmbedderFactory, create_documents_from_texts


def example_document_embedding():
    """Example: Embed documents using the factory."""
    print("=" * 60)
    print("Example 1: Document Embedding")
    print("=" * 60)

    # Create document embedder from configuration
    doc_embedder = EmbedderFactory.create_document_embedder()

    # Prepare some sample texts
    texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps computers understand text.",
    ]

    # Create documents from texts
    documents = create_documents_from_texts(
        texts=texts,
        metadatas=[
            {"topic": "AI", "category": "overview"},
            {"topic": "ML", "category": "definition"},
            {"topic": "NLP", "category": "application"},
        ],
    )

    # Generate embeddings
    embedded_docs = doc_embedder.embed(documents)

    # Display results
    print(f"\nEmbedded {len(embedded_docs)} documents:")
    for i, emb_doc in enumerate(embedded_docs):
        print(f"\n  Document {i+1}:")
        print(f"    Content: {emb_doc.content[:50]}...")
        print(f"    Model: {emb_doc.embedding_model}")
        print(f"    Dimension: {emb_doc.embedding_dimension}")
        print(f"    Topic: {emb_doc.metadata.get('topic', 'N/A')}")


def example_query_embedding():
    """Example: Embed a query using the factory."""
    print("\n" + "=" * 60)
    print("Example 2: Query Embedding")
    print("=" * 60)

    # Create query embedder from configuration
    query_embedder = EmbedderFactory.create_query_embedder()

    # Embed a query
    query = "What is machine learning?"
    query_embedding = query_embedder.embed(query)

    # Display results
    print(f"\nQuery: '{query}'")
    print(f"Embedding shape: {query_embedding.shape}")
    print(f"First 5 values: {query_embedding[:5]}")

    # Get model info
    model_info = query_embedder.get_model_info()
    print(f"\nModel Info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")


def example_batch_processing():
    """Example: Process documents in batches."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)

    # Create batch processor from configuration
    batch_processor = EmbedderFactory.create_batch_processor()

    # Create a larger set of documents
    texts = [f"This is document number {i} about various topics." for i in range(50)]
    documents = create_documents_from_texts(texts)

    # Process in batches (batch size from config)
    embedded_docs = batch_processor.process_documents_in_batches(documents)

    print(f"\nProcessed {len(embedded_docs)} documents in batches")
    print(f"First document: {embedded_docs[0].content[:50]}...")


def example_custom_config():
    """Example: Use custom configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)

    # Load configuration from a specific path
    config_path = Path("config/embeddings.yml")

    # Create embedder with custom config
    doc_embedder = EmbedderFactory.create_document_embedder(config_path=config_path)

    # Create a simple document
    texts = ["This is a test document."]
    documents = create_documents_from_texts(texts)

    # Embed
    embedded_docs = doc_embedder.embed(documents)

    print(f"\nUsed configuration from: {config_path}")
    print(f"Embedded {len(embedded_docs)} document(s)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Embedding Module Usage Examples")
    print("=" * 60)

    try:
        example_document_embedding()
        example_query_embedding()
        example_batch_processing()
        example_custom_config()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as error:
        print(f"\nError: {error}")
        print("\nMake sure:")
        print("  1. Haystack is installed: pip install haystack-ai")
        print("  2. Configuration file exists: config/embeddings.yml")
        print("  3. Virtual environment is activated")
