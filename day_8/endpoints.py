from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
import time

from database import VectorDatabase, get_database
from models import (
    DocumentCreate, DocumentResponse, DocumentRetrieved,
    SearchQuery, SearchResponse, CollectionInfo, ErrorResponse
)

router = APIRouter()

@router.post("/insert", response_model=DocumentResponse)
async def insert_document(
    document: DocumentCreate,
    background_tasks: BackgroundTasks,
    db: VectorDatabase = Depends(get_database)
):
    try:
        success = db.insert_document(
            doc_id=document.doc_id,
            text=document.text,
            metadata=document.metadata
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to insert document with ID: {document.doc_id}"
            )

        # Get embedding dimension for response
        doc_info = db.get_document(document.doc_id)
        embedding_dim = doc_info["embedding_dimension"] if doc_info else 384  # Default for all-MiniLM-L6-v2

        return DocumentResponse(
            doc_id=document.doc_id,
            text=document.text,
            metadata=document.metadata,
            embedding_dimension=embedding_dim
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search_query: SearchQuery,
    db: VectorDatabase = Depends(get_database)
):
    try:
        start_time = time.time()

        results = db.search_documents(
            query=search_query.query,
            n_results=search_query.n_results,
            metadata_filter=search_query.metadata_filter
        )

        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Format results
        formatted_results = []
        for i, doc_id in enumerate(results["ids"]):
            doc_result = DocumentRetrieved(
                doc_id=doc_id,
                text=results["documents"][i],
                metadata=results["metadatas"][i] if results["metadatas"] else None,
                distance=results["distances"][i] if search_query.include_distances and results["distances"] else None
            )
            formatted_results.append(doc_result)

        return SearchResponse(
            query=search_query.query,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    db: VectorDatabase = Depends(get_database)
):
    try:
        doc_info = db.get_document(doc_id)

        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{doc_id}' not found"
            )

        return {
            "doc_id": doc_info["id"],
            "text": doc_info["document"],
            "metadata": doc_info["metadata"],
            "embedding_dimension": doc_info["embedding_dimension"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document: {str(e)}"
        )

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    db: VectorDatabase = Depends(get_database)
):
    try:
        # Check if document exists
        doc_info = db.get_document(doc_id)
        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{doc_id}' not found"
            )

        success = db.delete_document(doc_id)

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document with ID: {doc_id}"
            )

        return {"message": f"Document '{doc_id}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Delete operation failed: {str(e)}"
        )

@router.get("/collection/info", response_model=CollectionInfo)
async def get_collection_info(db: VectorDatabase = Depends(get_database)):
    try:
        info = db.get_collection_info()
        return CollectionInfo(
            name=info["name"],
            document_count=info["document_count"],
            metadata=info["metadata"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection info: {str(e)}"
        )

@router.post("/batch/insert")
async def batch_insert_documents(
    documents: List[DocumentCreate],
    background_tasks: BackgroundTasks,
    db: VectorDatabase = Depends(get_database)
):
    try:
        if len(documents) > 100:
            raise HTTPException(
                status_code=400,
                detail="Batch size cannot exceed 100 documents"
            )

        # Convert to required format
        doc_list = []
        for doc in documents:
            doc_list.append({
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata
            })

        results = db.batch_insert_documents(doc_list)
        successful = sum(results)
        failed = len(results) - successful

        return {
            "message": f"Batch insert completed",
            "total_documents": len(documents),
            "successful_inserts": successful,
            "failed_inserts": failed,
            "success_rate": round((successful / len(documents)) * 100, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch insert failed: {str(e)}"
        )
