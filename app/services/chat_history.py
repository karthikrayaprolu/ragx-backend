from app.db.mongo import get_database
from app.schemas.chat_history import ChatSession, Message
from bson import ObjectId
from datetime import datetime
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)

class ChatHistoryService:
    def __init__(self):
        self.collection_name = "chat_sessions"

    async def get_collection(self):
        db = await get_database()
        return db[self.collection_name]

    async def create_session(self, user_id: str, title: str = "New Chat") -> ChatSession:
        collection = await self.get_collection()
        session = {
            "user_id": user_id,
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "messages": []
        }
        result = await collection.insert_one(session)
        session["_id"] = str(result.inserted_id)
        logger.info(f"Created chat session {session['_id']} for user {user_id}")
        return ChatSession(**session)

    async def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        collection = await self.get_collection()
        cursor = collection.find({"user_id": user_id}).sort("updated_at", -1)
        sessions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            sessions.append(ChatSession(**doc))
        return sessions

    async def get_session(self, session_id: str, user_id: str) -> Optional[ChatSession]:
        collection = await self.get_collection()
        try:
            doc = await collection.find_one({"_id": ObjectId(session_id), "user_id": user_id})
            if doc:
                doc["_id"] = str(doc["_id"])
                logger.info(f"Retrieved session {session_id} with {len(doc.get('messages', []))} messages")
                return ChatSession(**doc)
            logger.warning(f"Session {session_id} not found for user {user_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None

    async def add_message(self, session_id: str, user_id: str, message: Message):
        collection = await self.get_collection()
        try:
            result = await collection.update_one(
                {"_id": ObjectId(session_id), "user_id": user_id},
                {
                    "$push": {"messages": message.dict()},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            if result.modified_count == 0:
                logger.warning(f"Failed to add message to session {session_id}: Session not found or user mismatch")
            else:
                logger.info(f"Added message to session {session_id}")
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")

    async def delete_session(self, session_id: str, user_id: str):
        collection = await self.get_collection()
        await collection.delete_one({"_id": ObjectId(session_id), "user_id": user_id})

    async def update_session_title(self, session_id: str, user_id: str, title: str):
        collection = await self.get_collection()
        await collection.update_one(
            {"_id": ObjectId(session_id), "user_id": user_id},
            {"$set": {"title": title, "updated_at": datetime.utcnow()}}
        )

    async def get_total_queries(self, user_id: str) -> int:
        """Count total number of user messages across all sessions."""
        collection = await self.get_collection()
        
        # Aggregation to sum the length of 'messages' array for user's sessions
        # Or simply count total sessions if that's what we want. 
        # Requirement says "Total Queries", usually means individual messages.
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$project": {"message_count": {"$size": "$messages"}}},
            {"$group": {"_id": None, "total": {"$sum": "$message_count"}}}
        ]
        
        cursor = collection.aggregate(pipeline)
        result = await cursor.to_list(length=1)
        
        if result:
            return result[0]["total"]
        return 0

chat_history_service = ChatHistoryService()
