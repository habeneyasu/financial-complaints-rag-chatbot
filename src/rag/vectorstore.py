"""
Vector store module for storing and retrieving embeddings using ChromaDB.

This module provides functionality for creating, populating, and querying a vector store
with embeddings and associated metadata for traceability.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import warnings
import os

# Suppress ChromaDB telemetry warnings
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    # Suppress telemetry warnings
    warnings.filterwarnings('ignore', message='.*telemetry.*')
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from src.config import config
from src.exceptions import DataProcessingError
from src.utils.logger import get_logger


class VectorStore:
    """
    A class for managing vector embeddings in ChromaDB.
    
    This class provides methods for creating, populating, and querying a vector store
    with embeddings and metadata for traceability back to source complaints.
    
    Attributes:
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist the vector store
        client: ChromaDB client instance
        collection: ChromaDB collection instance
        logger: Logger instance
    """
    
    def __init__(
        self,
        collection_name: str = "financial_complaints",
        persist_directory: Optional[Union[str, Path]] = None,
        reset: bool = False
    ):
        """
        Initialize the VectorStore.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store (default: config)
            reset: Whether to reset/delete existing collection (default: False)
            
        Raises:
            DataProcessingError: If ChromaDB is not available
        """
        if not CHROMADB_AVAILABLE:
            raise DataProcessingError(
                "ChromaDB is not available. Install it with: pip install chromadb",
                details={"collection_name": collection_name}
            )
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory else config.data.vector_store_dir
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Delete existing collection if reset is requested
            if reset:
                try:
                    self.client.delete_collection(name=collection_name)
                    self.logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    pass  # Collection might not exist
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Financial complaints RAG vector store"}
            )
            
            self.logger.info(
                f"✅ VectorStore initialized: {collection_name} "
                f"(persist_dir: {self.persist_directory})"
            )
            
        except Exception as e:
            error_msg = f"Error initializing VectorStore: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(
                error_msg,
                details={"collection_name": collection_name, "persist_directory": str(self.persist_directory)}
            )
    
    def add_embeddings(
        self,
        df: pd.DataFrame,
        embedding_column: str = 'embedding',
        id_column: str = 'chunk_id',
        text_column: str = 'chunk_text',
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> int:
        """
        Add embeddings to the vector store with metadata.
        
        Args:
            df: DataFrame containing embeddings and metadata
            embedding_column: Name of the column containing embeddings
            id_column: Name of the column to use as document IDs
            text_column: Name of the column containing text chunks
            metadata_columns: List of columns to include as metadata (default: all except embedding/text/id)
            batch_size: Batch size for adding documents
            
        Returns:
            Number of documents added
        """
        if embedding_column not in df.columns:
            raise DataProcessingError(
                f"Embedding column '{embedding_column}' not found",
                details={"available_columns": list(df.columns)}
            )
        
        if id_column not in df.columns:
            raise DataProcessingError(
                f"ID column '{id_column}' not found",
                details={"available_columns": list(df.columns)}
            )
        
        if text_column not in df.columns:
            raise DataProcessingError(
                f"Text column '{text_column}' not found",
                details={"available_columns": list(df.columns)}
            )
        
        # Determine metadata columns
        if metadata_columns is None:
            # Include all columns except embedding, text, and id columns
            exclude_cols = {embedding_column, text_column, id_column}
            metadata_columns = [col for col in df.columns if col not in exclude_cols]
        
        self.logger.info(
            f"Adding {len(df):,} embeddings to vector store '{self.collection_name}'"
        )
        self.logger.info(f"Metadata columns: {metadata_columns}")
        
        # Prepare data for ChromaDB
        ids = df[id_column].astype(str).tolist()
        texts = df[text_column].astype(str).tolist()
        
        # Convert embeddings to list format if needed
        embeddings = df[embedding_column].tolist()
        if isinstance(embeddings[0], np.ndarray):
            embeddings = [emb.tolist() for emb in embeddings]
        
        # Prepare metadata
        metadatas = []
        for _, row in df.iterrows():
            metadata = {}
            for col in metadata_columns:
                value = row[col]
                # ChromaDB metadata must be strings, numbers, or booleans
                if pd.isna(value):
                    metadata[col] = None
                elif isinstance(value, (str, int, float, bool)):
                    metadata[col] = value
                else:
                    metadata[col] = str(value)
            metadatas.append(metadata)
        
        # Add in batches
        total_added = 0
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(df), batch_size), desc="Adding to vector store")
        except ImportError:
            iterator = range(0, len(df), batch_size)
        
        for i in iterator:
            batch_end = min(i + batch_size, len(df))
            batch_ids = ids[i:batch_end]
            batch_texts = texts[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            total_added += len(batch_ids)
        
        self.logger.info(f"✅ Added {total_added:,} documents to vector store")
        return total_added
    
    def query(
        self,
        query_text: Optional[str] = None,
        query_embeddings: Optional[Union[List[float], np.ndarray]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text (will be embedded if query_embeddings not provided)
            query_embeddings: Pre-computed query embeddings
            n_results: Number of results to return
            where: Metadata filter (e.g., {"Product": "Credit card"})
            where_document: Document content filter
            
        Returns:
            Dictionary with query results including ids, distances, documents, and metadata
        """
        if query_text is None and query_embeddings is None:
            raise ValueError("Either query_text or query_embeddings must be provided")
        
        # Convert query_embeddings to list if numpy array
        if query_embeddings is not None:
            if isinstance(query_embeddings, np.ndarray):
                query_embeddings = query_embeddings.tolist()
            if isinstance(query_embeddings[0], np.ndarray):
                query_embeddings = [emb.tolist() for emb in query_embeddings]
        
        try:
            results = self.collection.query(
                query_texts=[query_text] if query_text else None,
                query_embeddings=query_embeddings if query_embeddings else None,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            # Format results for easier use
            formatted_results = {
                "ids": results["ids"][0] if results["ids"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else []
            }
            
            self.logger.debug(f"Query returned {len(formatted_results['ids'])} results")
            return formatted_results
            
        except Exception as e:
            error_msg = f"Error querying vector store: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg)
    
    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve documents by their IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        try:
            results = self.collection.get(ids=ids)
            return results
        except Exception as e:
            error_msg = f"Error retrieving documents by IDs: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory)
        }
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            error_msg = f"Error deleting collection: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg)
    
    def reset_collection(self) -> None:
        """Reset the collection (delete and recreate)."""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Financial complaints RAG vector store"}
        )
        self.logger.info(f"Reset collection: {self.collection_name}")

