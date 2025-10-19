from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class DocumentBase(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the document")

class DocumentCreate(DocumentBase):
    doc_id: str = Field(..., min_length=1, max_length=100, description="Unique document identifier")

    @validator('doc_id')
    def validate_doc_id(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('doc_id must contain only alphanumeric characters, hyphens, and underscores')
        return v

class DocumentResponse(DocumentBase):
    doc_id: str
    embedding_dimension: int
    created_at: datetime = Field(default_factory=datetime.now)

class DocumentRetrieved(BaseModel):
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    n_results: int = Field(5, ge=1, le=50, description="Number of results to return")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filter")
    include_distances: bool = Field(True, description="Include similarity distances in results")

class SearchResponse(BaseModel):
    query: str
    results: List[DocumentRetrieved]
    total_results: int
    search_time_ms: float

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
