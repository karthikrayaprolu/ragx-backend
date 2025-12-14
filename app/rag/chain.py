from langchain_mongodb import MongoDBChatMessageHistory
from typing import List, Dict, Any, Optional, Generator
from app.core.config import settings
from app.rag.embeddings import embedding_service
from app.services.vector_db import pinecone_service
from app.schemas.chat_history import Message
import logging
import asyncio

logger = logging.getLogger(__name__)


def get_session_history(session_id: str):
    """
    Factory function to get MongoDB chat history for a session.
    """
    return MongoDBChatMessageHistory(
        connection_string=settings.MONGO_URI,
        session_id=session_id,
        database_name=settings.MONGO_DB_NAME,
        collection_name="chat_history"
    )


class RAGChain:
    """RAG chain for querying user-specific documents ensuring chat history is maintained in MongoDB."""
    
    def __init__(self):
        self.llm = None
        self.prompt = None
        self.chain = None
        self.chain_with_history = None

    def _ensure_initialized(self):
        """Lazy initialization of RAG chain components."""
        if self.chain_with_history is None:
            logger.info("Initializing RAGChain components...")
            # Import here to avoid heavy startup cost
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.runnables.history import RunnableWithMessageHistory
            
            # Initialize Chat Model
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL
            )
            
            # Define Prompt Template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ])
            
            # Create Basic Chain
            self.chain = self.prompt | self.llm
            
            # Wrap with History Support
            self.chain_with_history = RunnableWithMessageHistory(
                self.chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )
            logger.info("RAGChain initialized.")
    
    def _retrieve_context(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context from user's documents."""
        # Generate query embedding
        query_embedding = embedding_service.generate_embedding(query)
        
        # Query Pinecone
        results = pinecone_service.query_embeddings(
            user_id=user_id,
            query_vector=query_embedding,
            top_k=top_k,
            filter=filter,
            include_metadata=True
        )
        
        return results
    
    def _build_context_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            text = metadata.get("text", "")
            source = metadata.get("filename") or metadata.get("source", "Unknown")
            score = result.get("score", 0)
            
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    async def query(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query user's documents and generate a response with history context.
        """
        self._ensure_initialized()
        
        # Retrieve relevant context
        results = self._retrieve_context(user_id, query, top_k, filter)
        context = self._build_context_prompt(results)
        
        # Default system prompt
        if not system_prompt:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Answer based ONLY on the provided context. If the context doesn't contain relevant information, say so clearly."""
        
        # Determine invocation
        input_data = {
            "question": query,
            "context": context,
            "system_prompt": system_prompt
        }
        
        if session_id:
            # Use chain with history
            response = self.chain_with_history.invoke(
                input_data,
                config={"configurable": {"session_id": session_id}}
            )
        else:
            # Fallback to stateless chain if no session_id
            response = self.chain.invoke({
                **input_data,
                "history": [] # Empty history
            })
            
        answer = response.content
        
        # Extract usage if available
        usage = response.response_metadata.get("token_usage", {})
        
        sources = [
            {
                "filename": r.get("metadata", {}).get("filename") or r.get("metadata", {}).get("source"),
                "score": r.get("score"),
                "text_preview": r.get("metadata", {}).get("text", "")[:200]
            }
            for r in results
        ]
        
        # Save messages to chat_sessions collection
        if session_id:
            from app.services.chat_history import chat_history_service
            from datetime import datetime
            
            # Add user message
            user_message = Message(
                role="user",
                content=query,
                timestamp=datetime.utcnow()
            )
            await chat_history_service.add_message(session_id, user_id, user_message)
            
            # Add assistant message
            assistant_message = Message(
                role="assistant",
                content=answer,
                timestamp=datetime.utcnow(),
                sources=sources
            )
            await chat_history_service.add_message(session_id, user_id, assistant_message)
        
        return {
            "answer": answer,
            "sources": sources,
            "tokens_used": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0)
            }
        }
    
    async def query_stream(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Stream query response with history context.
        Returns tuple of (generator, results) for saving messages after streaming.
        """
        self._ensure_initialized()
        
        # Retrieve relevant context
        results = self._retrieve_context(user_id, query, top_k, filter)
        context = self._build_context_prompt(results)
        
        if not system_prompt:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Answer based ONLY on the provided context. If the context doesn't contain relevant information, say so clearly."""
        
        input_data = {
            "question": query,
            "context": context,
            "system_prompt": system_prompt
        }
        
        if session_id:
            stream = self.chain_with_history.stream(
                input_data,
                config={"configurable": {"session_id": session_id}}
            )
        else:
            stream = self.chain.stream({
                **input_data,
                "history": []
            })
        
        # Collect chunks and yield them
        full_response = ""
        for chunk in stream:
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        # Save messages to chat_sessions collection after streaming
        if session_id:
            from app.services.chat_history import chat_history_service
            from datetime import datetime
            
            sources = [
                {
                    "filename": r.get("metadata", {}).get("filename") or r.get("metadata", {}).get("source"),
                    "score": r.get("score"),
                    "text_preview": r.get("metadata", {}).get("text", "")[:200]
                }
                for r in results
            ]
            
            # Add user message
            user_message = Message(
                role="user",
                content=query,
                timestamp=datetime.utcnow()
            )
            await chat_history_service.add_message(session_id, user_id, user_message)
            
            # Add assistant message
            assistant_message = Message(
                role="assistant",
                content=full_response,
                timestamp=datetime.utcnow(),
                sources=sources
            )
            await chat_history_service.add_message(session_id, user_id, assistant_message)


# Singleton instance
rag_chain = RAGChain()
