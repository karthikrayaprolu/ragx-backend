from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[List[dict]] = None

class ChatSession(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = []

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}

class CreateChatSession(BaseModel):
    title: Optional[str] = "New Chat"

class UpdateChatSession(BaseModel):
    title: Optional[str] = None
