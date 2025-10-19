from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"

class IndexType(str, Enum):
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"

class DocumentBase(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=255, description="Unique document identifier")
    content: str = Field(..., min_length=1, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")

    @validator('document_id')
    def validate_document_id(cls, v):
        if not v.replace('_', '').replace('-', '').replace('.', '').isalnum():
            raise ValueError('document_id can only contain alphanumeric characters, hyphens, underscores, and dots')
        return v

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    content: Optional[str] = Field(None, min_length=1, description="Updated document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated document metadata")

class DocumentResponse(DocumentBase):
    id: int
    embedding_dimension: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class DocumentWithDistance(DocumentResponse):
    distance: float = Field(..., description="Distance/similarity score")

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Search query text")
    limit: int = Field(5, ge=1, le=100, description="Number of results to return")
    distance_metric: DistanceMetric = Field(DistanceMetric.COSINE, description="Distance metric for similarity")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filter")
    include_content: bool = Field(True, description="Include document content in results")

class SearchResponse(BaseModel):
    query: str
    results: List[DocumentWithDistance]
    total_results: int
    search_time_ms: float
    distance_metric: DistanceMetric

class BatchInsertRequest(BaseModel):
    documents: List[DocumentCreate] = Field(..., min_items=1, max_items=1000, description="Documents to insert")

class BatchInsertResponse(BaseModel):
    total_documents: int
    successful_inserts: int
    failed_inserts: int
    success_rate: float
    processing_time_ms: float

class CollectionStats(BaseModel):
    total_documents: int
    avg_content_length: float
    first_document: Optional[datetime]
    last_document: Optional[datetime]
    embedding_dimension: int
    model_name: str

class DatabaseInfo(BaseModel):
    postgresql_version: str
    pgvector_enabled: bool
    connection_status: str

class HealthResponse(BaseModel):
    status: str
    database_connection: bool
    pgvector_enabled: bool
    embedding_model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class IndexCreationRequest(BaseModel):
    table_name: str = Field("document_embeddings", description="Target table name")
    index_type: IndexType = Field(IndexType.HNSW, description="Type of vector index")

class IndexCreationResponse(BaseModel):
    message: str
    table_name: str
    index_type: IndexType
    creation_time_ms: float

class NearestNeighborQuery(BaseModel):
    document_id: str = Field(..., description="Source document ID")
    limit: int = Field(5, ge=1, le=50, description="Number of nearest neighbors")
    distance_metric: DistanceMetric = Field(DistanceMetric.COSINE, description="Distance metric")
    exclude_self: bool = Field(True, description="Exclude the source document from results")

class NearestNeighborResponse(BaseModel):
    source_document_id: str
    neighbors: List[DocumentWithDistance]
    total_neighbors: int
    distance_metric: DistanceMetric

class VectorDimensions(BaseModel):
    dimension: int = Field(..., ge=1, le=2000, description="Vector dimension size")

class TableCreationRequest(BaseModel):
    table_name: str = Field(..., min_length=1, max_length=63, description="Table name")
    dimension: int = Field(384, ge=1, le=2000, description="Vector dimension")

    @validator('table_name')
    def validate_table_name(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('table_name can only contain alphanumeric characters and underscores')
        return v

class TableCreationResponse(BaseModel):
    message: str
    table_name: str
    dimension: int
    creation_time_ms: float
