from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.schemas.chat_history import ChatSession, CreateChatSession, UpdateChatSession
from app.services.chat_history import chat_history_service
from app.api.v1.auth import get_current_user_id

router = APIRouter(prefix="/history", tags=["Chat History"])

@router.get("/", response_model=List[ChatSession], response_model_by_alias=True)
async def get_chat_history(user_id: str = Depends(get_current_user_id)):
    """Get all chat sessions for the current user."""
    return await chat_history_service.get_user_sessions(user_id)

@router.post("/", response_model=ChatSession, response_model_by_alias=True)
async def create_chat_session(
    session_data: CreateChatSession,
    user_id: str = Depends(get_current_user_id)
):
    """Create a new chat session."""
    return await chat_history_service.create_session(user_id, session_data.title)

@router.get("/{session_id}", response_model=ChatSession, response_model_by_alias=True)
async def get_chat_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """Get a specific chat session."""
    session = await chat_history_service.get_session(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session

@router.delete("/{session_id}")
async def delete_chat_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """Delete a chat session."""
    await chat_history_service.delete_session(session_id, user_id)
    return {"status": "success"}

@router.patch("/{session_id}")
async def update_chat_session(
    session_id: str,
    session_data: UpdateChatSession,
    user_id: str = Depends(get_current_user_id)
):
    """Update a chat session title."""
    if session_data.title:
        await chat_history_service.update_session_title(session_id, user_id, session_data.title)
    return {"status": "success"}
