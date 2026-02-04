"""
Quick Fix Script for RAG Knowledge Issue
Run this to diagnose and fix common issues
"""
import asyncio
import logging
from app.core.config import settings
from app.services.vector_db import pinecone_service
from app.rag.embeddings import embedding_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_dimension_mismatch():
    """Check if there's a dimension mismatch between embeddings and Pinecone index"""
    logger.info("=== Checking Embedding Dimensions ===")
    
    # Check current embedding dimension
    test_text = "test embedding"
    embedding = embedding_service.generate_embedding(test_text)
    current_dim = len(embedding)
    logger.info(f"Current embedding dimension: {current_dim}")
    logger.info(f"Config setting: {settings.EMBEDDING_DIMENSION}")
    
    # Check Pinecone index dimension
    try:
        pinecone_service._ensure_initialized()
        index_description = pinecone_service.pc.describe_index(settings.PINECONE_INDEX_NAME)
        index_dim = index_description.dimension
        logger.info(f"Pinecone index dimension: {index_dim}")
        
        if current_dim != index_dim:
            logger.error(f"❌ DIMENSION MISMATCH!")
            logger.error(f"   Embedding: {current_dim} dimensions")
            logger.error(f"   Pinecone:  {index_dim} dimensions")
            logger.error(f"")
            logger.error(f"   SOLUTION OPTIONS:")
            logger.error(f"   1. Create new index with {current_dim} dimensions")
            logger.error(f"   2. Delete and recreate index: {settings.PINECONE_INDEX_NAME}")
            logger.error(f"   3. Have users re-upload documents")
            return False
        else:
            logger.info(f"✅ Dimensions match!")
            return True
            
    except Exception as e:
        logger.error(f"Error checking Pinecone: {e}")
        return False


async def check_user_vectors(user_id: str):
    """Check if a user has vectors in their namespace"""
    logger.info(f"\n=== Checking Vectors for User: {user_id} ===")
    
    try:
        stats = pinecone_service.get_namespace_stats(user_id)
        logger.info(f"Namespace: {stats['namespace']}")
        logger.info(f"Vector count: {stats['vector_count']}")
        
        if stats['vector_count'] == 0:
            logger.warning(f"❌ No vectors found for user {user_id}")
            logger.warning(f"   User needs to upload documents")
            return False
        else:
            logger.info(f"✅ Found {stats['vector_count']} vectors")
            
            # Try a test query
            test_embedding = embedding_service.generate_embedding("test query")
            results = pinecone_service.query_embeddings(
                user_id=user_id,
                query_vector=test_embedding,
                top_k=3,
                include_metadata=True
            )
            
            logger.info(f"Test query returned {len(results)} results")
            if results:
                logger.info(f"Top result score: {results[0]['score']:.4f}")
                logger.info(f"Sample metadata: {results[0]['metadata']}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error checking vectors: {e}")
        return False


async def main():
    """Main diagnostic function"""
    logger.info("=" * 60)
    logger.info("RAG KNOWLEDGE DIAGNOSTIC TOOL")
    logger.info("=" * 60)
    
    # Check dimension compatibility
    dim_ok = await check_dimension_mismatch()
    
    if not dim_ok:
        logger.error("\n⚠️  CRITICAL: Fix dimension mismatch first!")
        logger.error("    See PINECONE_MIGRATION.md for instructions")
        return
    
    # Check specific user (replace with actual user ID)
    test_user_id = input("\nEnter user ID to check (or press Enter to skip): ").strip()
    
    if test_user_id:
        await check_user_vectors(test_user_id)
    
    logger.info("\n" + "=" * 60)
    logger.info("Diagnostic complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
