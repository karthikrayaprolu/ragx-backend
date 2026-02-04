from typing import List
from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Uses Google's Gemini API for embeddings (low memory, no local models).
    Falls back to simple hashing for development/testing if API key not available.
    """
    
    def __init__(self):
        self.dimension = 768  # Gemini embedding dimension
        self.use_gemini = bool(settings.GEMINI_API_KEY)
        
        if not self.use_gemini:
            logger.warning("GEMINI_API_KEY not set - using fallback embedding (not recommended for production)")
            self.dimension = 384
    
    def _generate_gemini_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini API (low memory)"""
        try:
            import google.generativeai as genai
            
            if not hasattr(self, '_genai_configured'):
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._genai_configured = True
            
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            # Fallback to simple method
            return self._generate_fallback_embedding(text)
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Simple fallback embedding for development (not for production)"""
        import hashlib
        import struct
        
        # Create deterministic "embedding" from text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to floats
        embedding = []
        for i in range(0, min(len(hash_bytes), 384 // 8), 4):
            value = struct.unpack('f', hash_bytes[i:i+4])[0]
            embedding.append(value)
        
        # Pad to 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)
        
        return embedding[:384]
    
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
        
        if self.use_gemini:
            return self._generate_gemini_embedding(text)
        else:
            return self._generate_fallback_embedding(text)
    
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
        
        # Generate embeddings one by one (Gemini API doesn't support batch)
        embeddings = []
        for text in cleaned_texts:
            emb = self.generate_embedding(text)
            embeddings.append(emb)
        
        return embeddings


# Singleton instance
embedding_service = EmbeddingService()
