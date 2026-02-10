from typing import List
from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using HuggingFace sentence-transformers.
    
    Uses all-MiniLM-L6-v2 model (384 dimensions) - free and fast.
    """
    
    def __init__(self):
        self.dimension = settings.EMBEDDING_DIMENSION  # 384 for all-MiniLM-L6-v2
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize HuggingFace sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading HuggingFace model: {settings.HUGGINGFACE_MODEL}")
            
            # Use token if provided
            if settings.HUGGINGFACE_TOKEN:
                os.environ['HUGGINGFACE_TOKEN'] = settings.HUGGINGFACE_TOKEN
                self.model = SentenceTransformer(settings.HUGGINGFACE_MODEL, use_auth_token=settings.HUGGINGFACE_TOKEN)
            else:
                self.model = SentenceTransformer(settings.HUGGINGFACE_MODEL)
            
            logger.info(f"HuggingFace model loaded successfully (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise RuntimeError(f"Cannot initialize embedding service: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats (384 dimensions)
        """
        text = text.replace("\n", " ").strip()
        
        if not text:
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch (much faster).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (each 384 dimensions)
        """
        # Clean texts
        cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
        cleaned_texts = [text if text else " " for text in cleaned_texts]
        
        if not cleaned_texts:
            return []
        
        try:
            # Batch encoding is much faster
            embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {e}")
            raise


# Singleton instance
embedding_service = EmbeddingService()
