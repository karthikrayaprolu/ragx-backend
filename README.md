RAGx Backend API (Ragster)

The powerful FastAPI-based engine driving the RAGx platform.

This backend handles Multi-Tenant Retrieval-Augmented Generation (RAG), secure document ingestion, vector database management via Pinecone, and seamless integration with the Next.js frontend using Firebase Authentication.

Key Features

FastAPI Architecture: High-performance, asynchronous REST API built with Python.

Secure Multi-Tenancy:

Uses Firebase Admin SDK to verify ID tokens from the frontend.

Implements Pinecone Namespaces to strictly isolate vector data per user (User A cannot query User B's data).

Advanced RAG Pipeline:

Ingestion: Parses PDFs and splits text into semantic chunks using LangChain.

Embedding: Generates vectors using OpenAI (text-embedding-3-small).

Retrieval: Context-aware querying with strict relevance thresholds.

Universal File Parsing: Supports PDF, CSV, TXT, and JSON ingestion.

Hybrid Database Strategy:

Pinecone for vector storage (embeddings).

PostgreSQL/SQL for user metadata and document processing status.

Tech Stack
Component	Technology	Description
Framework	FastAPI	Modern Python web framework
Vector DB	Pinecone	Serverless vector storage with Namespace support
LLM & Embeddings	OpenAI	GPT-4o / text-embedding-3-small
Orchestration	LangChain	RAG chains and document loaders
Auth	Firebase Admin	Server-side token verification
Database	SQLAlchemy	ORM for metadata storage
Validation	Pydantic	Strict data validation and settings management
Project Structure

Designed for scalability using the Router-Service-Controller pattern.

ragster-backend/
├── app/
│   ├── api/v1/         # REST Endpoints (Routes)
│   ├── core/           # Config & Security (Auth Middleware)
│   ├── db/             # Database Models & Session
│   ├── rag/            # AI Logic (Chains, Embeddings, Ingestion)
│   ├── schemas/        # Pydantic Data Models
│   ├── services/       # Business Logic (File handling, User ops)
│   └── utils/          # Helper functions
├── main.py             # App Entry Point
└── requirements.txt    # Dependencies