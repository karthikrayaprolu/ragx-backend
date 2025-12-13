from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Optional
from app.rag.chain import rag_chain
from app.schemas.chat import ChatRequest, ChatResponse
from app.api.v1.auth import get_current_user_id
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


from app.services.chat_history import chat_history_service
from app.schemas.chat_history import Message

@router.post("/query", response_model=ChatResponse)
async def query_documents(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Query the user's documents and get an AI-generated response.
    
    The query is embedded, matched against stored documents, 
    and relevant context is used to generate a response.
    """
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session = await chat_history_service.create_session(
                user_id=user_id,
                title=request.query[:50] + ("..." if len(request.query) > 50 else "")
            )
            session_id = session.id
        
        result = await rag_chain.query(
            user_id=user_id,
            query=request.query,
            top_k=request.top_k,
            filter=request.filter,
            system_prompt=request.system_prompt,
            session_id=session_id
        )
        
        # History is now handled automatically by the RAGChain via MongoDB integration
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            tokens_used=result["tokens_used"]
        )
    
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@router.post("/stream")
async def stream_query(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Stream a response from the user's documents.
    
    Returns a streaming response with chunks of the AI-generated answer.
    """
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session = await chat_history_service.create_session(
                user_id=user_id,
                title=request.query[:50] + ("..." if len(request.query) > 50 else "")
            )
            session_id = session.id
        
        async def generate():
            full_answer = ""
            async for chunk in rag_chain.query_stream(
                user_id=user_id,
                query=request.query,
                top_k=request.top_k,
                filter=request.filter,
                system_prompt=request.system_prompt,
                session_id=session_id
            ):
                full_answer += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            
            # History is now handled automatically by the RAGChain via MongoDB integration
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except Exception as e:
        logger.error(f"Error streaming response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stream response: {str(e)}")


@router.get("/health")
async def health_check():
    """Check if the chat service is healthy."""
    return {"status": "healthy", "service": "chat"}
