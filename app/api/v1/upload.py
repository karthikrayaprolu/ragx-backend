from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional
from app.rag.ingestion import ingestion_service
from app.services.vector_db import pinecone_service
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    TextIngestionRequest,
    NamespaceStats
)
from app.api.v1.auth import get_current_user_id
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    """
    Upload and process a document for RAG.
    
    Supports: PDF, TXT, CSV, XLSX, MD files.
    The document will be chunked, embedded, and stored in the user's namespace.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    content_type = file.content_type or "application/octet-stream"
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process document
        result = await ingestion_service.process_document(
            user_id=user_id,
            file_content=content,
            filename=file.filename,
            file_type=content_type
        )
        
        return DocumentUploadResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document")


@router.post("/text", response_model=DocumentUploadResponse)
async def upload_text(
    request: TextIngestionRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Upload and process raw text for RAG.
    
    The text will be chunked, embedded, and stored in the user's namespace.
    """
    try:
        result = await ingestion_service.process_text(
            user_id=user_id,
            text=request.text,
            source_name=request.source_name,
            metadata=request.metadata
        )
        
        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=result["source"],
            chunks_created=result["chunks_created"],
            vectors_stored=result["vectors_stored"],
            user_id=result["user_id"]
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail="Failed to process text")


@router.delete("/document", response_model=DocumentDeleteResponse)
async def delete_document(
    request: DocumentDeleteRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Delete a document and all its embeddings.
    """
    try:
        deleted = await ingestion_service.delete_document(user_id, request.document_id)
        
        return DocumentDeleteResponse(
            document_id=request.document_id,
            deleted=deleted,
            message="Document deleted successfully" if deleted else "Failed to delete document"
        )
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.delete("/all")
async def delete_all_documents(user_id: str = Depends(get_current_user_id)):
    """
    Delete all documents for the current user.
    """
    try:
        deleted = pinecone_service.delete_user_namespace(user_id)
        
        # Delete from MongoDB (Documents registry)
        from app.services.document_service import document_service
        await document_service.delete_all_documents(user_id)
        
        return {
            "deleted": deleted,
            "message": "All documents deleted successfully" if deleted else "Failed to delete documents"
        }
    
    except Exception as e:
        logger.error(f"Error deleting all documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete documents")


@router.get("/stats", response_model=NamespaceStats)
async def get_stats(user_id: str = Depends(get_current_user_id)):
    """
    Get statistics about the user's stored documents.
    """
    try:
        # Get vector stats from Pinecone
        pinecone_stats = pinecone_service.get_namespace_stats(user_id)
        
        # Get document count from MongoDB
        from app.services.document_service import document_service
        total_documents = await document_service.get_document_count(user_id)
        
        # Get query count from Chat History
        from app.services.chat_history import chat_history_service
        query_count = await chat_history_service.get_total_queries(user_id)
        
        return NamespaceStats(
            namespace=pinecone_stats["namespace"],
            vector_count=pinecone_stats["vector_count"],
            total_index_vectors=pinecone_stats["total_index_vectors"],
            total_documents=total_documents,
            query_count=query_count
        )
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.get("/documents")
async def get_user_documents(user_id: str = Depends(get_current_user_id)):
    """
    Get all uploaded documents for the current user.
    """
    try:
        from app.services.document_service import document_service
        documents = await document_service.get_user_documents(user_id)
        
        return {
            "documents": documents,
            "total": len(documents)
        }
    
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to get documents")