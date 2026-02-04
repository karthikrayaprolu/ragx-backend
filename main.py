from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import os
import sys

# Create FastAPI app
logger.info(f"Starting application on port {os.environ.get('PORT', 'unknown')}...")
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="RAG-powered document Q&A API with user-specific embeddings",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

from app.db.mongo import mongodb

@app.on_event("startup")
async def startup_db_client():
    await mongodb.connect_to_database()

@app.on_event("shutdown")
async def shutdown_db_client():
    await mongodb.close_database_connection()

# Configure CORS - Allow all origins for development/production
# Note: In production, you should restrict this to specific domains
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://rag-x.vercel.app",
    "https://ragx.appwrite.network",
    "https://ragx-backend.onrender.com",
]

# Add environment-based CORS origin if provided
if frontend_url := os.environ.get("FRONTEND_URL"):
    allowed_origins.append(frontend_url)

# For production, allow all Vercel preview deployments
if os.environ.get("ENVIRONMENT") == "production":
    # Use regex to allow all Vercel deployments
    from fastapi.middleware.cors import CORSMiddleware
    import re
    
    class CustomCORSMiddleware(CORSMiddleware):
        def is_allowed_origin(self, origin: str) -> bool:
            if super().is_allowed_origin(origin):
                return True
            # Allow all vercel.app subdomains
            if re.match(r"https://.*\.vercel\.app$", origin):
                return True
            return False
    
    app.add_middleware(
        CustomCORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info(f"Health check called. PORT: {os.environ.get('PORT')}")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Render sets PORT environment variable, default to 10000
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
