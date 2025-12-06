from sentence_transformers import SentenceTransformer
from typing import List
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.
    
    Since OpenRouter doesn't provide embedding endpoints, we use a local
    sentence-transformers model which is fast and free.
    """
    
    def __init__(self):
        # Using all-MiniLM-L6-v2 - fast, good quality, 384 dimensions
        # Or use all-mpnet-base-v2 for better quality (768 dimensions)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Matches the model output
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        text = text.replace("\n", " ").strip()
        
        if not text:
            raise ValueError("Cannot generate embedding for empty text")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Clean texts
        cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
        cleaned_texts = [text if text else " " for text in cleaned_texts]
        
        if not cleaned_texts:
            return []
        
        embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


# Singleton instance
embedding_service = EmbeddingService()
