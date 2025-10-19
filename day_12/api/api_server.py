#!/usr/bin/env python3
"""
Day 12 - RAG Pipeline FastAPI Server
===================================
REST API endpoints for the RAG pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import asyncio
import time

from main import RAGPipeline, Document, RAGResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Day 12 - Mini RAG Pipeline API",
    description="REST API for Retrieval-Augmented Generation pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None

# Pydantic models for API
class DocumentInput(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}

class QueryInput(BaseModel):
    text: str
    max_results: int = 5

class QueryResponse(BaseModel):
    query_id: str
    query_text: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    pipeline_stats: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global pipeline
    try:
        logger.info("Initializing RAG Pipeline...")
        pipeline = RAGPipeline(
            collection_name="api_knowledge_base",
            llm_provider="openai"
        )

        # Add sample documents
        from main import create_sample_knowledge_base
        sample_docs = create_sample_knowledge_base()
        pipeline.add_documents(sample_docs)

        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Day 12 - Mini RAG Pipeline API",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "documents": "/documents",
            "stats": "/stats"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        pipeline_stats=pipeline.get_pipeline_stats()
    )

@app.post("/query", response_model=QueryResponse)
async def query_rag(query: QueryInput):
    """Query the RAG pipeline"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        logger.info(f"Processing query: {query.text}")
        response = pipeline.query(query.text, query.max_results)

        return QueryResponse(
            query_id=response.query.id,
            query_text=response.query.text,
            answer=response.generated_answer,
            retrieved_documents=[
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in response.retrieved_docs
            ],
            processing_time=response.processing_time,
            metadata=response.metadata
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/documents")
async def add_documents(documents: List[DocumentInput]):
    """Add documents to the knowledge base"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        doc_objects = [
            Document(id=doc.id, content=doc.content, metadata=doc.metadata)
            for doc in documents
        ]

        results = pipeline.add_documents(doc_objects)
        logger.info(f"Added {results['successful']} documents")

        return {
            "message": f"Successfully processed {len(documents)} documents",
            "successful": results["successful"],
            "failed": results["failed"],
            "processing_time": results["processing_time"]
        }

    except Exception as e:
        logger.error(f"Document addition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document addition failed: {str(e)}")

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document (mock endpoint for demo)"""
    # This would typically query the vector store for the document
    return {
        "message": "Document retrieval not implemented in this demo",
        "doc_id": doc_id,
        "suggestion": "Use /query endpoint to search for relevant documents"
    }

@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return pipeline.get_pipeline_stats()

@app.get("/history")
async def get_query_history():
    """Get query history"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return {
        "total_queries": len(pipeline.query_history),
        "recent_queries": [
            {
                "query_id": response.query.id,
                "query_text": response.query.text,
                "answer_preview": response.generated_answer[:100] + "...",
                "processing_time": response.processing_time,
                "documents_retrieved": len(response.retrieved_docs)
            }
            for response in pipeline.query_history[-10:]  # Last 10 queries
        ]
    }

@app.delete("/reset")
async def reset_pipeline():
    """Reset the pipeline (for testing)"""
    global pipeline
    try:
        logger.info("Resetting RAG Pipeline...")
        pipeline = RAGPipeline(
            collection_name=f"api_knowledge_base_{int(time.time())}",
            llm_provider="openai"
        )

        return {"message": "Pipeline reset successfully"}

    except Exception as e:
        logger.error(f"Pipeline reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# Development endpoints
@app.get("/demo")
async def demo_endpoint():
    """Demo endpoint with sample queries"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    sample_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of deep learning?"
    ]

    results = []
    for query_text in sample_queries:
        response = pipeline.query(query_text, max_results=2)
        results.append({
            "query": query_text,
            "answer": response.generated_answer,
            "documents_retrieved": len(response.retrieved_docs),
            "processing_time": response.processing_time
        })

    return {
        "message": "Demo queries processed",
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
