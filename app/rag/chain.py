from openai import OpenAI
from typing import List, Dict, Any, Optional, Generator
from app.core.config import settings
from app.rag.embeddings import embedding_service
from app.services.vector_db import pinecone_service
import logging

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain for querying user-specific documents and generating responses using OpenRouter."""
    
    def __init__(self):
        # Use OpenRouter API (OpenAI-compatible)
        self.client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY
        )
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
    
    def _retrieve_context(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from user's documents.
        
        Args:
            user_id: The user's unique identifier
            query: The user's query
            top_k: Number of relevant chunks to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of relevant document chunks with metadata
        """
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
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query user's documents and generate a response.
        
        Args:
            user_id: The user's unique identifier
            query: The user's question
            top_k: Number of relevant chunks to retrieve
            filter: Optional metadata filter
            system_prompt: Optional custom system prompt
            
        Returns:
            Response with answer and source documents
        """
        # Retrieve relevant context
        results = self._retrieve_context(user_id, query, top_k, filter)
        context = self._build_context_prompt(results)
        
        # Default system prompt
        if not system_prompt:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            
Rules:
- Answer based ONLY on the provided context
- If the context doesn't contain relevant information, say so clearly
- Cite your sources when possible
- Be concise but thorough
- If you're unsure, express uncertainty"""
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        answer = response.choices[0].message.content
        
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
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        }
    
    def query_stream(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream query response for real-time output.
        
        Args:
            user_id: The user's unique identifier
            query: The user's question
            top_k: Number of relevant chunks to retrieve
            filter: Optional metadata filter
            system_prompt: Optional custom system prompt
            
        Yields:
            Response chunks as they're generated
        """
        # Retrieve relevant context
        results = self._retrieve_context(user_id, query, top_k, filter)
        context = self._build_context_prompt(results)
        
        if not system_prompt:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Answer based ONLY on the provided context. If the context doesn't contain relevant information, say so clearly."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # Stream response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Singleton instance
rag_chain = RAGChain()
