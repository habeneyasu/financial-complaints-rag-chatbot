"""
Retriever module for RAG pipeline.

This module provides functionality for retrieving relevant documents from the vector store
based on user questions using semantic similarity search.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.rag.vectorstore import VectorStore
from src.rag.embedder import EmbeddingGenerator
from src.config import config
from src.exceptions import DataProcessingError
from src.utils.logger import get_logger


class RAGRetriever:
    """
    A class for retrieving relevant documents from the vector store.
    
    This class handles question embedding and similarity search to retrieve
    the most relevant text chunks for RAG-based question answering.
    
    Attributes:
        vector_store: VectorStore instance for querying
        embedder: EmbeddingGenerator instance for question embedding
        logger: Logger instance
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[EmbeddingGenerator] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the RAG Retriever.
        
        Args:
            vector_store: VectorStore instance (must be initialized with vector store)
            embedder: Optional EmbeddingGenerator instance (creates new if not provided)
            embedding_model: Model name for embedding (defaults to config)
        """
        self.vector_store = vector_store
        self.logger = get_logger(__name__)
        
        # Initialize embedder if not provided
        if embedder is None:
            embedding_model = embedding_model or config.embedding.model_name
            self.embedder = EmbeddingGenerator(
                model_name=embedding_model,
                device=config.embedding.device
            )
        else:
            self.embedder = embedder
        
        self.logger.info("RAG Retriever initialized")
    
    def retrieve(
        self,
        question: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a given question.
        
        This method:
        1. Embeds the question using the same model from Task 2 (all-MiniLM-L6-v2)
        2. Performs similarity search against the vector store
        3. Returns the top-k most relevant text chunks
        
        Args:
            question: User's question (string)
            k: Number of top results to retrieve (default: 5)
            metadata_filter: Optional metadata filter (e.g., {"Product": "Credit card"})
            
        Returns:
            List of dictionaries, each containing:
            - text: The retrieved text chunk
            - metadata: Associated metadata (product, issue, company, etc.)
            - distance: Similarity distance (lower is more similar)
            - id: Document ID
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        self.logger.info(f"Retrieving top-{k} documents for question: '{question[:50]}...'")
        
        try:
            # Step 1: Embed the question using the same model from Task 2
            question_embedding = self.embedder.encode(
                texts=question,
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=config.embedding.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Convert to list format for ChromaDB
            if isinstance(question_embedding, np.ndarray):
                if question_embedding.ndim > 1:
                    question_embedding = question_embedding[0].tolist()
                else:
                    question_embedding = question_embedding.tolist()
            
            self.logger.debug(f"Question embedded. Dimension: {len(question_embedding)}")
            
            # Step 2: Query vector store with embeddings (not query_text to ensure we use our model)
            results = self.vector_store.query(
                query_embeddings=[question_embedding],
                n_results=k,
                where=metadata_filter
            )
            
            # Step 3: Format results
            retrieved_docs = []
            for i in range(len(results['ids'])):
                doc = {
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] else {},
                    'distance': results['distances'][i] if results['distances'] else None,
                    'rank': i + 1
                }
                retrieved_docs.append(doc)
            
            self.logger.info(f"✅ Retrieved {len(retrieved_docs)} documents")
            
            # Log retrieved sources
            for i, doc in enumerate(retrieved_docs[:3], 1):
                product = doc['metadata'].get('product', 'Unknown')
                self.logger.debug(f"  {i}. Product: {product}, Distance: {doc['distance']:.4f}")
            
            return retrieved_docs
            
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg, details={"question": question, "k": k})
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """
        Format retrieved documents into a context string for the prompt.
        
        Args:
            retrieved_docs: List of retrieved documents from retrieve()
            max_length: Maximum length of context string (default: 2000 chars)
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Format each document
            text = doc['text']
            metadata = doc.get('metadata', {})
            
            # Add metadata info if available
            product = metadata.get('product', '')
            issue = metadata.get('issue', '')
            
            doc_text = f"[Document {i}]"
            if product:
                doc_text += f" Product: {product}"
            if issue:
                doc_text += f", Issue: {issue}"
            doc_text += f"\n{text}\n"
            
            # Check if adding this would exceed max length
            if current_length + len(doc_text) > max_length:
                # Truncate the last document if needed
                remaining = max_length - current_length - len(f"[Document {i}]...\n")
                if remaining > 100:  # Only add if we have meaningful space
                    doc_text = f"[Document {i}]"
                    if product:
                        doc_text += f" Product: {product}"
                    doc_text += f"\n{text[:remaining]}...\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n".join(context_parts)

