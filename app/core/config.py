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
    
    # OpenRouter Configuration (Free OpenAI-compatible API)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Model Configuration
    LLM_MODEL: str = "openai/gpt-oss-20b:free"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local sentence-transformers model
    EMBEDDING_DIMENSION: int = 384  # Matches all-MiniLM-L6-v2 output
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
