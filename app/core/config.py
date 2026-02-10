from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "ragster-index"
    
    # MongoDB Configuration
    MONGO_URI: str
    MONGO_DB_NAME: str = "ragx"
    
    # Gemini Configuration
    GEMINI_API_KEY: Optional[str] = None
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    
    # Groq Configuration (Free alternative to OpenAI)
    GROQ_API_KEY: Optional[str] = None
    
    # HuggingFace Configuration
    HUGGINGFACE_TOKEN: Optional[str] = None
    HUGGINGFACE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Model Configuration
    LLM_PROVIDER: str = "groq"  # Options: "openai" or "groq"
    LLM_MODEL: str = "llama-3.1-8b-instant"  # Groq model (free and fast)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embedding model
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 produces 384 dimensions
    LLM_TEMPERATURE: float = 0.7
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Ragster"
    
    # Frontend Configuration
    FRONTEND_URL: str = "https://rag-x.vercel.app"  # Override with production URL in env
    
    # Stripe Configuration
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    
    # Testing Configuration
    TEST_MODE: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra env variables


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
