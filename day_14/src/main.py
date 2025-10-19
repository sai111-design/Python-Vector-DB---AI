import os

# Force mock mode to avoid model downloads
if os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true":
    class MockSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts):
            import numpy as np
            if isinstance(texts, str):
                return np.random.rand(384)
            return [np.random.rand(384) for _ in texts]
    # Replace SentenceTransformer globally
    import sys
    import types
    mock_module = types.ModuleType('sentence_transformers')
    mock_module.SentenceTransformer = MockSentenceTransformer
    sys.modules['sentence_transformers'] = mock_module
#!/usr/bin/env python3
"""
Day 14 - Production RAG Pipeline API
===================================
Production-ready RAG pipeline with monitoring, health checks, and scaling
"""

import os
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration')
ACTIVE_REQUESTS = Gauge('rag_active_requests', 'Active requests')
PIPELINE_QUERIES = Counter('rag_pipeline_queries_total', 'Total pipeline queries')
PIPELINE_ERRORS = Counter('rag_pipeline_errors_total', 'Pipeline errors')

# Load SentenceTransformer model
model_path = os.getenv("MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(model_path)


# Real RAG Pipeline with FAISS
import numpy as np
import faiss

class RealRAGPipeline:
    def __init__(self, model, docs=None):
        self.model = model
        self.query_count = 0
        self.error_count = 0
        # Use provided docs or default
        if docs is None:
            docs = [
                "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval and generation.",
                "FAISS is a library for efficient similarity search and clustering of dense vectors.",
                "SentenceTransformers provide easy methods to compute dense vector representations for sentences."
            ]
        self.docs = docs
        self.embeddings = np.vstack([self.model.encode(doc) for doc in self.docs])
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    async def query(self, text: str, max_results: int = 5):
        self.query_count += 1
        start = time.time()
        query_vec = np.array([self.model.encode(text)])
        D, I = self.index.search(query_vec, min(max_results, len(self.docs)))
        retrieved = []
        for idx, dist in zip(I[0], D[0]):
            retrieved.append({
                "id": f"doc_{idx}",
                "content": self.docs[idx],
                "score": float(1.0 / (1.0 + dist))
            })
        answer = f"Relevant info: {retrieved[0]['content']}" if retrieved else "No relevant document found."
        return {
            "query": text,
            "answer": answer,
            "retrieved_docs": retrieved,
            "processing_time": time.time() - start,
            "metadata": {"status": "success", "mock": False}
        }

    def get_stats(self):
        return {
            "total_queries": self.query_count,
            "error_count": self.error_count,
            "uptime": time.time() - start_time
        }

# Global variables
pipeline: Optional[MockRAGPipeline] = None
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global pipeline

    logger.info("🚀 Starting RAG Pipeline API")


    # Initialize real pipeline
    try:
        pipeline = RealRAGPipeline(model)
        logger.info("✅ Real RAG Pipeline initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise

    yield

    # Cleanup
    logger.info("🔄 Shutting down RAG Pipeline API")

# FastAPI app
app = FastAPI(
    title="Day 14 - Production RAG Pipeline API",
    description="Production-ready RAG pipeline with monitoring and scaling",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Request/Response models
class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Query text")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to return")

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: float
    version: str
    environment: str

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        return response

    except Exception as e:
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise

    finally:
        ACTIVE_REQUESTS.dec()

# Dependency for rate limiting (mock implementation)
async def rate_limit_check():
    """Mock rate limiting check"""
    # In production, implement Redis-based rate limiting
    pass

# Health check endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": "Day 14 - Production RAG Pipeline API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": time.time()
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check"""
    global pipeline


    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    uptime = time.time() - start_time

    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        uptime=uptime,
        version="1.0.0",
        environment=os.getenv("APP_ENV", "development")
    )

@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Kubernetes readiness probe"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"status": "ready"}

@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": time.time()}

# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(prometheus_client.REGISTRY)

# Main API endpoints
@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(rate_limit_check)
):
    """Process RAG query"""
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        logger.info(f"Processing query: {request.text[:100]}...")

        # Record query attempt
        PIPELINE_QUERIES.inc()

        # Process query
        start_time = time.time()
        result = await pipeline.query(request.text, request.max_results)
        processing_time = time.time() - start_time

        # Background task for analytics (mock)
        background_tasks.add_task(log_query_analytics, request.text, processing_time)

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            retrieved_docs=result["retrieved_docs"],
            processing_time=result["processing_time"],
            metadata=result["metadata"]
        )
    except Exception as e:
        PIPELINE_ERRORS.inc()
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """Get pipeline statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    stats = pipeline.get_stats()

    return {
        "pipeline_stats": stats,
        "system_stats": {
            "uptime": time.time() - start_time,
            "environment": os.getenv("APP_ENV", "development"),
            "version": "1.0.0"
        }
    }

# Background task functions
async def log_query_analytics(query: str, processing_time: float):
    """Log query analytics (mock implementation)"""
    logger.info(f"Analytics: Query processed in {processing_time:.3f}s")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": request.url.path}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Additional utility endpoints
@app.post("/admin/reload", tags=["Admin"])
async def reload_pipeline():
    """Reload pipeline (admin endpoint)"""
    global pipeline

    try:
        logger.info("Reloading RAG pipeline...")
        pipeline = MockRAGPipeline()
        return {"status": "reloaded", "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

@app.get("/version", tags=["Info"])
async def get_version():
    """Get API version information"""
    return {
        "version": "1.0.0",
        "build_date": os.getenv("BUILD_DATE", "unknown"),
        "commit_hash": os.getenv("VCS_REF", "unknown"),
        "environment": os.getenv("APP_ENV", "development")
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=os.getenv("APP_ENV") == "development"
    )
