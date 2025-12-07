from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.rag.embeddings import embedding_service
from app.services.vector_db import pinecone_service
from app.utils.parsers import DocumentParser
import uuid
import logging

logger = logging.getLogger(__name__)


from app.services.document_service import document_service

class DocumentIngestionService:
    """Service for ingesting and processing documents into vector embeddings."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.parser = DocumentParser()
    
    async def process_document(
        self,
        user_id: str,
        file_content: bytes,
        filename: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document and store embeddings in Pinecone.
        """
        # Parse document to text
        text = self.parser.parse(file_content, file_type)
        
        if not text.strip():
            raise ValueError("No text content could be extracted from the document")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Generate embeddings for all chunks
        embeddings = embedding_service.generate_embeddings(chunks)
        
        # Prepare vectors with metadata
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{document_id}_{i}"
            vector_metadata = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk[:1000],  # Store first 1000 chars for retrieval
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": vector_metadata
            })
        
        # Upsert to Pinecone with user namespace
        upserted_count = pinecone_service.upsert_embeddings(user_id, vectors)
        
        # Register document in MongoDB
        await document_service.create_document(
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            metadata=metadata
        )
        
        logger.info(f"Ingested document {filename} with {upserted_count} vectors for user {user_id}")
        
        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "vectors_stored": upserted_count,
            "user_id": user_id
        }
    
    async def process_text(
        self,
        user_id: str,
        text: str,
        source_name: str = "direct_input",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process raw text and store embeddings in Pinecone.
        """
        if not text.strip():
            raise ValueError("Cannot process empty text")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Generate embeddings for all chunks
        embeddings = embedding_service.generate_embeddings(chunks)
        
        # Prepare vectors with metadata
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{document_id}_{i}"
            vector_metadata = {
                "document_id": document_id,
                "source": source_name,
                "chunk_index": i,
                "text": chunk[:1000],
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": vector_metadata
            })
        
        # Upsert to Pinecone with user namespace
        upserted_count = pinecone_service.upsert_embeddings(user_id, vectors)
        
        # Register document in MongoDB
        await document_service.create_document(
            user_id=user_id,
            document_id=document_id,
            filename=source_name,
            file_type="text/plain",
            metadata=metadata
        )
        
        return {
            "document_id": document_id,
            "source": source_name,
            "chunks_created": len(chunks),
            "vectors_stored": upserted_count,
            "user_id": user_id
        }
    
    async def delete_document(self, user_id: str, document_id: str) -> bool:
        """
        Delete all vectors associated with a document.
        """
        # Delete from MongoDB
        await document_service.delete_document(user_id, document_id)
        
        # Delete from Pinecone
        return pinecone_service.delete_embeddings(
            user_id=user_id,
            filter={"document_id": {"$eq": document_id}}
        )


# Singleton instance
ingestion_service = DocumentIngestionService()
