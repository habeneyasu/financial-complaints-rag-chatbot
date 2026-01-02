"""
RAG (Retrieval-Augmented Generation) modules for the Financial Complaints Chatbot.

This package contains modules for text chunking, embedding generation, and vector store management.
"""

from src.rag.chunker import TextChunker
from src.rag.embedder import EmbeddingGenerator
from src.rag.vectorstore import VectorStore

__all__ = ['TextChunker', 'EmbeddingGenerator', 'VectorStore']

