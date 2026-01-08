"""
RAG (Retrieval-Augmented Generation) modules for the Financial Complaints Chatbot.

This package contains modules for text chunking, embedding generation, vector store management,
retrieval, prompt engineering, generation, and evaluation.
"""

from src.rag.chunker import TextChunker
from src.rag.embedder import EmbeddingGenerator
from src.rag.vectorstore import VectorStore
from src.rag.retriever import RAGRetriever
from src.rag.prompts import PromptTemplate
from src.rag.generator import RAGGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.evaluator import RAGEvaluator

__all__ = [
    'TextChunker',
    'EmbeddingGenerator',
    'VectorStore',
    'RAGRetriever',
    'PromptTemplate',
    'RAGGenerator',
    'RAGPipeline',
    'RAGEvaluator'
]

