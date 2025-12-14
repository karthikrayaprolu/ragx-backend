from typing import List, Dict, Any, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for managing Pinecone vector database operations with user-specific namespaces."""
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.index_name = settings.PINECONE_INDEX_NAME
    
    def _ensure_initialized(self):
        """Lazy initialization of Pinecone client and index."""
        if self.pc is None:
            from pinecone import Pinecone, ServerlessSpec
            logger.info("Initializing Pinecone service...")
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self._ensure_index_exists()
            self.index = self.pc.Index(self.index_name)
            logger.info("Pinecone service initialized.")

    def _ensure_index_exists(self):
        """Create the index if it doesn't exist."""
        from pinecone import ServerlessSpec
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Index {self.index_name} created successfully")
        else:
            logger.info(f"Index {self.index_name} already exists")
    
    def _get_user_namespace(self, user_id: str) -> str:
        """Generate a namespace for user-specific embeddings."""
        return f"user_{user_id}"
    
    def upsert_embeddings(
        self,
        user_id: str,
        vectors: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Upsert embeddings to user-specific namespace.
        
        Args:
            user_id: The user's unique identifier
            vectors: List of dicts with 'id', 'values', and 'metadata'
            batch_size: Number of vectors to upsert per batch
            
        Returns:
            Number of vectors upserted
        """
        self._ensure_initialized()
        namespace = self._get_user_namespace(user_id)
        total_upserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
            logger.info(f"Upserted batch {i // batch_size + 1}, total: {total_upserted}")
        
        return total_upserted
    
    def query_embeddings(
        self,
        user_id: str,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query embeddings from user-specific namespace.
        
        Args:
            user_id: The user's unique identifier
            query_vector: The query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching documents with scores
        """
        self._ensure_initialized()
        namespace = self._get_user_namespace(user_id)
        
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else None
            }
            for match in results.matches
        ]
    
    def delete_embeddings(
        self,
        user_id: str,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        filter: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete embeddings from user-specific namespace.
        
        Args:
            user_id: The user's unique identifier
            ids: Specific vector IDs to delete
            delete_all: If True, delete all vectors in the namespace
            filter: Metadata filter for deletion
            
        Returns:
            True if deletion was successful
        """
        self._ensure_initialized()
        namespace = self._get_user_namespace(user_id)
        
        try:
            if delete_all:
                self.index.delete(delete_all=True, namespace=namespace)
            elif ids:
                self.index.delete(ids=ids, namespace=namespace)
            elif filter:
                self.index.delete(filter=filter, namespace=namespace)
            
            logger.info(f"Deleted embeddings from namespace: {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False
    
    def get_namespace_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's namespace."""
        self._ensure_initialized()
        namespace = self._get_user_namespace(user_id)
        stats = self.index.describe_index_stats()
        
        namespace_stats = stats.namespaces.get(namespace, {})
        return {
            "namespace": namespace,
            "vector_count": getattr(namespace_stats, 'vector_count', 0),
            "total_index_vectors": stats.total_vector_count
        }
    
    def delete_user_namespace(self, user_id: str) -> bool:
        """Delete all vectors in a user's namespace."""
        # delete_embeddings calls _ensure_initialized, so we are good
        return self.delete_embeddings(user_id, delete_all=True)


# Singleton instance
pinecone_service = PineconeService()
