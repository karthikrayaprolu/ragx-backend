from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., min_length=1, description="The user's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    session_id: Optional[str] = Field(default=None, description="Chat session ID to save history")


class TokenUsage(BaseModel):
    """Token usage information."""
    prompt: int
    completion: int
    total: int


class SourceDocument(BaseModel):
    """Source document reference."""
    filename: Optional[str]
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    """Response model for chat queries."""
    answer: str
    sources: List[SourceDocument]
    tokens_used: TokenUsage


class ChatStreamChunk(BaseModel):
    """Streaming response chunk."""
    content: str
    done: bool = False


class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ConversationRequest(BaseModel):
    """Request model for conversation with history."""
    messages: List[ConversationMessage]
    top_k: int = Field(default=5, ge=1, le=20)
    filter: Optional[Dict[str, Any]] = None
