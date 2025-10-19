import os
import time
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import List, Dict, Any

from database.connection import DatabaseConnection, init_database, get_database_info
from database.vector_ops import VectorOperations
from models.schemas import (
    DocumentCreate, DocumentResponse, DocumentWithDistance, DocumentUpdate,
    SearchQuery, SearchResponse, BatchInsertRequest, BatchInsertResponse,
    CollectionStats, HealthResponse, DatabaseInfo, IndexCreationRequest,
    IndexCreationResponse, NearestNeighborQuery, NearestNeighborResponse,
    TableCreationRequest, TableCreationResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global vector operations instance
vector_ops: VectorOperations = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global vector_ops
    try:
        logger.info("Initializing pgvector PostgreSQL application...")

        # Initialize database
        init_database()

        # Create vector operations instance
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/vectordb")
        vector_ops = VectorOperations(database_url)
        vector_ops.connect()

        # Create default table
        vector_ops.create_embeddings_table()

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    try:
        if vector_ops:
            vector_ops.disconnect()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

app = FastAPI(
    title="pgvector PostgreSQL Vector Database API",
    description="FastAPI application for vector similarity search using PostgreSQL and pgvector",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_vector_ops() -> VectorOperations:
    """Dependency to get vector operations instance"""
    if vector_ops is None:
        raise HTTPException(status_code=500, detail="Vector operations not initialized")
    return vector_ops

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "pgvector PostgreSQL Vector Database API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_info = get_database_info()

        # Check vector operations
        vector_ops_status = vector_ops is not None and vector_ops._connection is not None

        return HealthResponse(
            status="healthy" if db_info["pgvector_enabled"] and vector_ops_status else "unhealthy",
            database_connection=True,
            pgvector_enabled=db_info["pgvector_enabled"],
            embedding_model_loaded=vector_ops_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database_connection=False,
            pgvector_enabled=False,
            embedding_model_loaded=False
        )

@app.get("/database/info", response_model=DatabaseInfo)
async def get_db_info():
    """Get database information"""
    try:
        info = get_database_info()
        return DatabaseInfo(
            postgresql_version=info["postgresql_version"],
            pgvector_enabled=info["pgvector_enabled"],
            connection_status="connected"
        )
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        raise HTTPException(status_code=500, detail=f"Database info error: {str(e)}")

@app.post("/documents", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    background_tasks: BackgroundTasks,
    ops: VectorOperations = Depends(get_vector_ops)
):
    """Insert a document with vector embedding"""
    try:
        success = ops.insert_document(
            document_id=document.document_id,
            content=document.content,
            metadata=document.metadata
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to insert document")

        # Get the inserted document
        doc_info = ops.get_document(document.document_id)
        if not doc_info:
            raise HTTPException(status_code=500, detail="Document inserted but retrieval failed")

        return DocumentResponse(
            id=1,  # In real implementation, get from database
            document_id=doc_info["document_id"],
            content=doc_info["content"],
            metadata=doc_info["metadata"],
            embedding_dimension=ops.embedding_dimension,
            created_at=doc_info["created_at"],
            updated_at=doc_info["updated_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(
    query: SearchQuery,
    ops: VectorOperations = Depends(get_vector_ops)
):
    """Search for similar documents"""
    try:
        start_time = time.time()

        results = ops.search_similar_documents(
            query_text=query.query,
            limit=query.limit,
            distance_metric=query.distance_metric.value,
            metadata_filter=query.metadata_filter
        )

        search_time = (time.time() - start_time) * 1000

        # Format results
        formatted_results = []
        for result in results:
            doc_result = DocumentWithDistance(
                id=1,  # In real implementation, get from database
                document_id=result["document_id"],
                content=result["content"] if query.include_content else "",
                metadata=result["metadata"],
                embedding_dimension=ops.embedding_dimension,
                created_at=result["created_at"],
                updated_at=None,
                distance=result["distance"]
            )
            formatted_results.append(doc_result)

        return SearchResponse(
            query=query.query,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time, 2),
            distance_metric=query.distance_metric
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    ops: VectorOperations = Depends(get_vector_ops)
):
    """Get a specific document by ID"""
    try:
        doc_info = ops.get_document(document_id)

        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

        return DocumentResponse(
            id=1,  # In real implementation, get from database
            document_id=doc_info["document_id"],
            content=doc_info["content"],
            metadata=doc_info["metadata"],
            embedding_dimension=ops.embedding_dimension,
            created_at=doc_info["created_at"],
            updated_at=doc_info["updated_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    ops: VectorOperations = Depends(get_vector_ops)
):
    """Delete a document"""
    try:
        # Check if document exists
        doc_info = ops.get_document(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

        success = ops.delete_document(document_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")

        return {"message": f"Document '{document_id}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion error: {str(e)}")

@app.post("/documents/batch", response_model=BatchInsertResponse)
async def batch_insert_documents(
    request: BatchInsertRequest,
    background_tasks: BackgroundTasks,
    ops: VectorOperations = Depends(get_vector_ops)
):
    """Batch insert multiple documents"""
    try:
        start_time = time.time()

        # Convert to format expected by vector_ops
        documents = []
        for doc in request.documents:
            documents.append({
                "document_id": doc.document_id,
                "content": doc.content,
                "metadata": doc.metadata
            })

        results = ops.batch_insert_documents(documents)
        processing_time = (time.time() - start_time) * 1000

        total_docs = len(request.documents)
        successful = results["successful"]
        failed = results["failed"]
        success_rate = (successful / total_docs) * 100 if total_docs > 0 else 0

        return BatchInsertResponse(
            total_documents=total_docs,
            successful_inserts=successful,
            failed_inserts=failed,
            success_rate=round(success_rate, 2),
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch insert error: {str(e)}")

@app.get("/collection/stats", response_model=CollectionStats)
async def get_collection_statistics(
    ops: VectorOperations = Depends(get_vector_ops)
):
    """Get collection statistics"""
    try:
        stats = ops.get_collection_stats()
        return CollectionStats(
            total_documents=stats["total_documents"],
            avg_content_length=stats["avg_content_length"],
            first_document=stats["first_document"],
            last_document=stats["last_document"],
            embedding_dimension=stats["embedding_dimension"],
            model_name=stats["model_name"]
        )

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@app.post("/indexes/create", response_model=IndexCreationResponse)
async def create_vector_index(
    request: IndexCreationRequest,
    ops: VectorOperations = Depends(get_vector_ops)
):
    """Create vector index for improved search performance"""
    try:
        start_time = time.time()

        ops.create_vector_index(
            table_name=request.table_name,
            index_type=request.index_type.value
        )

        creation_time = (time.time() - start_time) * 1000

        return IndexCreationResponse(
            message=f"{request.index_type.value.upper()} index created successfully",
            table_name=request.table_name,
            index_type=request.index_type,
            creation_time_ms=round(creation_time, 2)
        )

    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Index creation error: {str(e)}")

@app.post("/documents/{document_id}/neighbors", response_model=NearestNeighborResponse)
async def find_nearest_neighbors(
    document_id: str,
    query: NearestNeighborQuery,
    ops: VectorOperations = Depends(get_vector_ops)
):
    """Find nearest neighbors of a specific document"""
    try:
        # Get the source document
        source_doc = ops.get_document(document_id)
        if not source_doc:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

        # Search for similar documents
        results = ops.search_similar_documents(
            query_text=source_doc["content"],
            limit=query.limit + (1 if query.exclude_self else 0),
            distance_metric=query.distance_metric.value
        )

        # Filter out the source document if requested
        neighbors = []
        for result in results:
            if query.exclude_self and result["document_id"] == document_id:
                continue

            neighbor = DocumentWithDistance(
                id=1,  # In real implementation, get from database
                document_id=result["document_id"],
                content=result["content"],
                metadata=result["metadata"],
                embedding_dimension=ops.embedding_dimension,
                created_at=result["created_at"],
                updated_at=None,
                distance=result["distance"]
            )
            neighbors.append(neighbor)

            if len(neighbors) >= query.limit:
                break

        return NearestNeighborResponse(
            source_document_id=document_id,
            neighbors=neighbors,
            total_neighbors=len(neighbors),
            distance_metric=query.distance_metric
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Nearest neighbor search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Neighbor search error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
