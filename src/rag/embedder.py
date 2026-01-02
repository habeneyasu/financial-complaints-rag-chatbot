"""
Embedding generation module for creating vector representations of text chunks.

This module provides functionality for generating embeddings using sentence-transformers
models, specifically optimized for semantic search and retrieval tasks.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from src.config import config
from src.exceptions import DataProcessingError
from src.utils.logger import get_logger


class EmbeddingGenerator:
    """
    A class for generating embeddings from text using sentence-transformers models.
    
    This class provides methods for generating embeddings from text chunks,
    with support for batch processing and progress tracking.
    
    Attributes:
        model_name: Name of the sentence-transformers model
        model: Loaded SentenceTransformer model
        device: Device to run the model on ('cpu' or 'cuda')
        logger: Logger instance
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
            cache_folder: Optional folder to cache the model
            
        Raises:
            DataProcessingError: If sentence-transformers is not available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise DataProcessingError(
                "sentence-transformers is not available. Install it with: pip install sentence-transformers",
                details={"model_name": model_name}
            )
        
        self.model_name = model_name
        self.logger = get_logger(__name__)
        
        # Determine device
        if device is None:
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Load the model
        try:
            self.logger.info(f"Loading embedding model: {model_name}")
            self.logger.info(f"Device: {self.device}")
            
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=cache_folder
            )
            
            # Log model info
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(
                f"✅ Model loaded successfully. Embedding dimension: {self.embedding_dimension}"
            )
            
        except Exception as e:
            error_msg = f"Error loading embedding model: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(
                error_msg,
                details={"model_name": model_name, "device": self.device}
            )
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings (L2 normalization)
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [str(text) if text is not None else "" for text in texts]
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=convert_to_numpy
            )
            
            self.logger.debug(
                f"Generated embeddings for {len(valid_texts)} texts. "
                f"Shape: {embeddings.shape}"
            )
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(
                error_msg,
                details={"num_texts": len(valid_texts), "batch_size": batch_size}
            )
    
    def generate_embeddings_for_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'chunk_text',
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True
    ) -> pd.DataFrame:
        """
        Generate embeddings for all texts in a DataFrame and add as a new column.
        
        Args:
            df: DataFrame containing text chunks
            text_column: Name of the column containing text to embed
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            DataFrame with added 'embedding' column containing numpy arrays
        """
        if text_column not in df.columns:
            raise DataProcessingError(
                f"Text column '{text_column}' not found in DataFrame",
                details={"available_columns": list(df.columns)}
            )
        
        self.logger.info(
            f"Generating embeddings for {len(df):,} texts from column '{text_column}'"
        )
        self.logger.info(f"Model: {self.model_name}, Batch size: {batch_size}")
        
        # Get texts
        texts = df[text_column].astype(str).tolist()
        
        # Generate embeddings
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings
        )
        
        # Add embeddings to DataFrame
        result_df = df.copy()
        result_df['embedding'] = [emb for emb in embeddings]
        
        self.logger.info(
            f"✅ Generated {len(embeddings):,} embeddings "
            f"(dimension: {self.embedding_dimension})"
        )
        
        return result_df
    
    def save_embeddings(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = 'parquet'
    ) -> Path:
        """
        Save DataFrame with embeddings to disk.
        
        Note: Parquet format is recommended for preserving numpy arrays.
        
        Args:
            df: DataFrame containing embeddings
            output_path: Path to save the file
            format: File format ('parquet', 'pickle', or 'npy' for embeddings only)
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'parquet':
                # For parquet, we need to convert embeddings to list format
                df_save = df.copy()
                if 'embedding' in df_save.columns:
                    df_save['embedding'] = df_save['embedding'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                df_save.to_parquet(output_path, index=False, engine='pyarrow')
                
            elif format == 'pickle':
                df.to_pickle(output_path)
                
            elif format == 'npy':
                # Save only embeddings as numpy array
                if 'embedding' not in df.columns:
                    raise DataProcessingError("No 'embedding' column found in DataFrame")
                embeddings = np.array(df['embedding'].tolist())
                np.save(output_path, embeddings)
                
            else:
                raise DataProcessingError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved embeddings to {output_path} ({format})")
            return output_path
            
        except Exception as e:
            error_msg = f"Error saving embeddings: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(
                error_msg,
                details={"path": str(output_path), "format": format}
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', None)
        }

