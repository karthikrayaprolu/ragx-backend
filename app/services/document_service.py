from app.db.mongo import get_database
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.collection_name = "documents"

    async def get_collection(self):
        db = await get_database()
        return db[self.collection_name]

    async def create_document(
        self, 
        user_id: str, 
        document_id: str, 
        filename: str, 
        file_type: str, 
        metadata: Dict[str, Any] = None
    ):
        """Register a new document in MongoDB."""
        collection = await self.get_collection()
        doc = {
            "document_id": document_id,
            "user_id": user_id,
            "filename": filename,
            "file_type": file_type,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "status": "processed"
        }
        await collection.insert_one(doc)
        logger.info(f"Registered document {document_id} for user {user_id}")
        return doc

    async def get_user_documents(self, user_id: str):
        """Get all documents for a user."""
        collection = await self.get_collection()
        cursor = collection.find({"user_id": user_id}).sort("created_at", -1)
        documents = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            documents.append(doc)
        return documents

    async def get_document_count(self, user_id: str) -> int:
        """Count total documents for a user."""
        collection = await self.get_collection()
        return await collection.count_documents({"user_id": user_id})

    async def delete_document(self, user_id: str, document_id: str):
        """Delete a document registry."""
        collection = await self.get_collection()
        await collection.delete_one({"document_id": document_id, "user_id": user_id})

    async def delete_all_documents(self, user_id: str):
        """Delete all documents for a user."""
        collection = await self.get_collection()
        await collection.delete_many({"user_id": user_id})

document_service = DocumentService()
