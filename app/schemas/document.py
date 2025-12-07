from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    chunks_created: int
    vectors_stored: int
    user_id: str
    message: str = "Document processed successfully"


class DocumentDeleteRequest(BaseModel):
    """Request model for document deletion."""
    document_id: str


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""
    document_id: str
    deleted: bool
    message: str


class TextIngestionRequest(BaseModel):
    """Request model for text ingestion."""
    text: str = Field(..., min_length=1)
    source_name: str = "direct_input"
    metadata: Optional[Dict[str, Any]] = None


class DocumentSource(BaseModel):
    """Source document information."""
    filename: Optional[str]
    score: float
    text_preview: str


class NamespaceStats(BaseModel):
    """Statistics for a user's namespace."""
    namespace: str
    vector_count: int
    total_index_vectors: int
    total_documents: int = 0
    query_count: int = 0
