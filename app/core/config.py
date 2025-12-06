from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "ragster-index"
    
    # MongoDB Configuration
    MONGO_URI: str = "mongodb+srv://ragx:ragx@ragx.4jl5sux.mongodb.net/?retryWrites=true&w=majority"
    MONGO_DB_NAME: str = "ragx"
    
    # OpenRouter Configuration (Free OpenAI-compatible API)
    OPENROUTER_API_KEY: str
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
