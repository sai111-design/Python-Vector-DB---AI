#!/usr/bin/env python3
"""
Day 7 Task: Vector DB Overview with Chroma
Task: Run Chroma locally, insert and retrieve embeddings

Requirements:
- pip install chromadb sentence-transformers

This script demonstrates:
1. Setting up Chroma vector database locally
2. Creating a collection
3. Inserting documents with embeddings
4. Retrieving similar documents using vector search
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import os

def main():
    """Main function to demonstrate Day 7 vector database operations"""
    print("=" * 60)
    print("DAY 7 TASK: VECTOR DB OVERVIEW WITH CHROMA")
    print("=" * 60)

    # Step 1: Initialize Chroma client (runs locally)
    print("\n1. Setting up Chroma locally...")
    client = chromadb.PersistentClient(path="./chroma_db")
    print("✓ Chroma client initialized with persistent storage")

    # Step 2: Create or get collection
    print("\n2. Creating collection...")
    collection_name = "day7_collection"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"✓ Retrieved existing collection: {collection_name}")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"✓ Created new collection: {collection_name}")

    # Step 3: Prepare sample documents
    print("\n3. Preparing sample documents...")
    documents = [
        "Python is a versatile programming language used in data science and AI.",
        "Machine learning algorithms can learn patterns from data automatically.",
        "Vector databases store high-dimensional vectors for similarity search.",
        "ChromaDB is an open-source embedding database for AI applications.",
        "Natural language processing enables computers to understand human text.",
        "Deep learning uses neural networks with multiple hidden layers.",
        "Embeddings convert text into numerical vectors that capture meaning.",
        "Retrieval-augmented generation combines search with language models.",
        "FastAPI is a modern Python framework for building web APIs quickly.",
        "Semantic search finds relevant content based on meaning, not just keywords."
    ]

    document_ids = [f"doc_{i}" for i in range(len(documents))]
    print(f"✓ Prepared {len(documents)} sample documents")

    # Step 4: Initialize embedding model
    print("\n4. Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ SentenceTransformer model loaded")

    # Step 5: Generate embeddings and insert into Chroma
    print("\n5. Generating embeddings and inserting into Chroma...")
    embeddings = model.encode(documents)

    # Convert numpy arrays to lists for Chroma
    embeddings_list = embeddings.tolist()

    # Add documents to collection
    collection.add(
        embeddings=embeddings_list,
        documents=documents,
        ids=document_ids
    )
    print(f"✓ Inserted {len(documents)} documents with embeddings")
    print(f"✓ Collection size: {collection.count()} documents")
    print(f"✓ Embedding dimension: {len(embeddings_list[0])}")

    # Step 6: Demonstrate retrieval with queries
    print("\n6. Testing retrieval with sample queries...")

    test_queries = [
        "What is machine learning?",
        "How do vector databases work?",
        "Python programming for AI"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: '{query}'")

        # Generate query embedding
        query_embedding = model.encode([query]).tolist()

        # Search for similar documents
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )

        # Display results
        for j, (doc_id, document, distance) in enumerate(
            zip(results['ids'][0], results['documents'][0], results['distances'][0])
        ):
            print(f"      {j+1}. [{doc_id}] (distance: {distance:.4f})")
            print(f"         {document}")

    # Step 7: Verify embeddings are stored and retrievable
    print("\n7. Verifying stored embeddings...")

    # Retrieve all data including embeddings
    all_data = collection.get(include=['embeddings', 'documents'])

    print(f"✓ Retrieved {len(all_data['documents'])} documents")
    print(f"✓ Each embedding has {len(all_data['embeddings'][0])} dimensions")
    print(f"✓ Sample embedding (first 5 values): {all_data['embeddings'][0][:5]}")

    # Step 8: Demonstrate additional operations
    print("\n8. Additional operations...")

    # Count documents
    total_docs = collection.count()
    print(f"✓ Total documents in collection: {total_docs}")

    # Get specific document by ID
    specific_doc = collection.get(ids=["doc_0"], include=['documents', 'embeddings'])
    print(f"✓ Retrieved specific document: {specific_doc['documents'][0]}")

    print("\n" + "=" * 60)
    print("✅ DAY 7 TASK COMPLETED SUCCESSFULLY!")
    print("✅ Chroma vector database setup, insertion, and retrieval demonstrated")
    print("=" * 60)

if __name__ == "__main__":
    main()
