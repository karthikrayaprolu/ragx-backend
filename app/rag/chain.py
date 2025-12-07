from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb import MongoDBChatMessageHistory
from typing import List, Dict, Any, Optional, Generator
from app.core.config import settings
from app.rag.embeddings import embedding_service
from app.services.vector_db import pinecone_service
import logging

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
    
    def query(
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
        
        return {
            "answer": answer,
            "sources": [
                {
                    "filename": r.get("metadata", {}).get("filename") or r.get("metadata", {}).get("source"),
                    "score": r.get("score"),
                    "text_preview": r.get("metadata", {}).get("text", "")[:200]
                }
                for r in results
            ],
            "tokens_used": {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0)
            }
        }
    
    def query_stream(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream query response with history context.
        """
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
        
        for chunk in stream:
            if chunk.content:
                yield chunk.content


# Singleton instance
rag_chain = RAGChain()
