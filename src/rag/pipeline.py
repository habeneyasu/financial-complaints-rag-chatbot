"""
Complete RAG pipeline module.

This module provides the complete Retrieval-Augmented Generation pipeline
that combines retrieval, prompt engineering, and generation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from src.rag.retriever import RAGRetriever
from src.rag.prompts import PromptTemplate
from src.rag.generator import RAGGenerator
from src.rag.vectorstore import VectorStore
from src.rag.embedder import EmbeddingGenerator
from src.config import config
from src.utils.logger import get_logger


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    This class provides an end-to-end RAG pipeline that:
    1. Takes a user question
    2. Retrieves relevant context from the vector store
    3. Formats a prompt with the context
    4. Generates an answer using an LLM
    
    Attributes:
        retriever: RAGRetriever instance
        generator: RAGGenerator instance
        prompt_template: PromptTemplate instance
        logger: Logger instance
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        top_k: int = 5,
        max_context_length: int = 2000
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            vector_store: Initialized VectorStore instance
            embedding_model: Embedding model name (defaults to config)
            llm_model: LLM model name (default: "google/flan-t5-base")
            prompt_template: Prompt template (defaults to PromptTemplate.get_default())
            top_k: Number of documents to retrieve (default: 5)
            max_context_length: Maximum context length in characters (default: 2000)
        """
        self.logger = get_logger(__name__)
        
        # Initialize components
        embedding_model = embedding_model or config.embedding.model_name
        embedder = EmbeddingGenerator(
            model_name=embedding_model,
            device=config.embedding.device
        )
        
        self.retriever = RAGRetriever(
            vector_store=vector_store,
            embedder=embedder
        )
        
        self.generator = RAGGenerator(
            model_name=llm_model or "google/flan-t5-base",
            device=config.embedding.device
        )
        
        self.prompt_template = prompt_template or PromptTemplate.get_default()
        self.top_k = top_k
        self.max_context_length = max_context_length
        
        self.logger.info("RAG Pipeline initialized")
        self.logger.info(f"  Embedding model: {embedding_model}")
        self.logger.info(f"  LLM model: {llm_model or 'google/flan-t5-base'}")
        self.logger.info(f"  Top-k: {top_k}")
    
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        This method:
        1. Retrieves relevant documents
        2. Formats context and prompt
        3. Generates answer using LLM
        4. Returns answer with sources
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve (overrides default)
            metadata_filter: Optional metadata filter for retrieval
            return_sources: Whether to include retrieved sources in response
            
        Returns:
            Dictionary containing:
            - answer: Generated answer
            - sources: List of retrieved documents (if return_sources=True)
            - context: Formatted context string
            - question: Original question
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        self.logger.info(f"Processing question: '{question[:50]}...'")
        
        # Step 1: Retrieve relevant documents
        k = top_k or self.top_k
        retrieved_docs = self.retriever.retrieve(
            question=question,
            k=k,
            metadata_filter=metadata_filter
        )
        
        if not retrieved_docs:
            self.logger.warning("No documents retrieved")
            return {
                "answer": "I don't have enough information in the provided context to answer this question.",
                "sources": [],
                "context": "",
                "question": question
            }
        
        # Step 2: Format context
        context = self.retriever.format_context(
            retrieved_docs=retrieved_docs,
            max_length=self.max_context_length
        )
        
        # Step 3: Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # Step 4: Generate answer
        try:
            answer = self.generator.generate_with_retries(prompt)
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            answer = "I encountered an error while generating an answer. Please try again."
        
        # Step 5: Prepare response
        response = {
            "answer": answer.strip(),
            "question": question,
            "context": context
        }
        
        if return_sources:
            # Format sources for display
            sources = []
            for doc in retrieved_docs[:3]:  # Top 3 sources
                source_info = {
                    "rank": doc['rank'],
                    "text": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                    "metadata": {
                        "product": doc['metadata'].get('product', 'Unknown'),
                        "issue": doc['metadata'].get('issue', 'Unknown'),
                        "company": doc['metadata'].get('company', 'Unknown')
                    },
                    "distance": doc.get('distance')
                }
                sources.append(source_info)
            response["sources"] = sources
        
        self.logger.info("✅ Answer generated successfully")
        
        return response
    
    @classmethod
    def from_vector_store_path(
        cls,
        vector_store_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> 'RAGPipeline':
        """
        Create RAG pipeline from vector store path.
        
        Args:
            vector_store_path: Path to vector store directory
            collection_name: Collection name (defaults to config)
            **kwargs: Additional arguments for RAGPipeline.__init__
            
        Returns:
            Initialized RAGPipeline instance
        """
        if vector_store_path is None:
            vector_store_path = config.data.vector_store_dir
        
        if collection_name is None:
            collection_name = config.vectorstore.collection_name
        
        vector_store = VectorStore(
            persist_directory=str(vector_store_path),
            collection_name=collection_name
        )
        
        return cls(vector_store=vector_store, **kwargs)

