"""
Text chunking module for splitting long narratives into appropriately sized chunks.

This module provides functionality for chunking text documents using LangChain's
RecursiveCharacterTextSplitter or custom chunking strategies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None

from src.config import config
from src.exceptions import DataProcessingError
from src.utils.logger import get_logger


class TextChunker:
    """
    A class for chunking text documents into smaller pieces for embedding.
    
    This class provides methods for chunking complaint narratives using
    LangChain's RecursiveCharacterTextSplitter or custom chunking strategies.
    
    Attributes:
        chunk_size: Target size of chunks (in characters or tokens)
        chunk_overlap: Overlap between chunks
        splitter: Text splitter instance
        logger: Logger instance
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: Optional[callable] = None,
        use_langchain: bool = True
    ):
        """
        Initialize the TextChunker.
        
        Args:
            chunk_size: Target size of chunks (default: 500 characters)
            chunk_overlap: Overlap between chunks (default: 50 characters)
            separators: List of separators to use for splitting (default: None, uses LangChain defaults)
            length_function: Function to measure text length (default: len)
            use_langchain: Whether to use LangChain's RecursiveCharacterTextSplitter (default: True)
            
        Raises:
            DataProcessingError: If LangChain is requested but not available
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_langchain = use_langchain
        self.logger = get_logger(__name__)
        
        if use_langchain:
            if not LANGCHAIN_AVAILABLE:
                raise DataProcessingError(
                    "LangChain is not available. Install it with: pip install langchain",
                    details={"use_langchain": use_langchain}
                )
            
            # Default separators for RecursiveCharacterTextSplitter
            if separators is None:
                separators = [
                    "\n\n",  # Paragraphs
                    "\n",   # Lines
                    ". ",   # Sentences
                    "! ",   # Exclamations
                    "? ",   # Questions
                    "; ",   # Semicolons
                    ", ",   # Commas
                    " ",    # Words
                    ""      # Characters
                ]
            
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=length_function or len
            )
            self.logger.info(
                f"Initialized TextChunker with LangChain RecursiveCharacterTextSplitter "
                f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
            )
        else:
            self.splitter = None
            self.length_function = length_function or len
            self.logger.info(
                f"Initialized TextChunker with custom chunking "
                f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
            )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk a single text document.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or pd.isna(text) or text.strip() == "":
            return []
        
        if self.use_langchain and self.splitter:
            try:
                chunks = self.splitter.split_text(text)
                return chunks
            except Exception as e:
                self.logger.error(f"Error chunking text with LangChain: {str(e)}")
                raise DataProcessingError(
                    f"Error chunking text: {str(e)}",
                    details={"text_length": len(text), "chunk_size": self.chunk_size}
                )
        else:
            # Custom chunking implementation
            return self._custom_chunk_text(text)
    
    def _custom_chunk_text(self, text: str) -> List[str]:
        """
        Custom chunking implementation when LangChain is not used.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        text_length = self.length_function(text)
        
        if text_length <= self.chunk_size:
            return [text]
        
        # Start from the beginning
        start = 0
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            if end >= text_length:
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to break at a sentence boundary
            # Look for sentence endings before the end position
            sentence_endings = ['. ', '! ', '? ', '\n\n', '\n']
            best_break = end
            
            for ending in sentence_endings:
                # Search backwards from end position
                break_pos = text.rfind(ending, start, end)
                if break_pos != -1:
                    best_break = break_pos + len(ending)
                    break
            
            # If no sentence boundary found, try word boundary
            if best_break == end:
                word_break = text.rfind(' ', start, end)
                if word_break != -1:
                    best_break = word_break + 1
            
            chunks.append(text[start:best_break].strip())
            
            # Move start position with overlap
            start = best_break - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'Consumer complaint narrative',
        metadata_columns: Optional[List[str]] = None,
        progress_bar: bool = True
    ) -> pd.DataFrame:
        """
        Chunk all narratives in a DataFrame and create a new DataFrame with chunks.
        
        Args:
            df: DataFrame containing narratives
            text_column: Name of the column containing text to chunk
            metadata_columns: List of columns to include as metadata (default: all except text_column)
            progress_bar: Whether to show progress bar (default: True)
            
        Returns:
            DataFrame with chunks, each row is a chunk with metadata
        """
        if text_column not in df.columns:
            raise DataProcessingError(
                f"Text column '{text_column}' not found in DataFrame",
                details={"available_columns": list(df.columns)}
            )
        
        if metadata_columns is None:
            # Include all columns except the text column as metadata
            metadata_columns = [col for col in df.columns if col != text_column]
        
        self.logger.info(f"Chunking {len(df)} narratives from column '{text_column}'")
        self.logger.info(f"Chunk parameters: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        chunk_data = []
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            iterator = tqdm(df.iterrows(), total=len(df), desc="Chunking") if progress_bar else df.iterrows()
        except ImportError:
            iterator = df.iterrows()
            if progress_bar:
                self.logger.warning("tqdm not available, progress bar disabled")
        
        for idx, row in iterator:
            text = row[text_column]
            
            if pd.isna(text) or str(text).strip() == "":
                continue
            
            chunks = self.chunk_text(str(text))
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_row = {
                    'chunk_id': f"{idx}_{chunk_idx}",
                    'original_index': idx,
                    'chunk_index': chunk_idx,
                    'chunk_text': chunk,
                    'chunk_length': len(chunk),
                    'num_chunks': len(chunks)
                }
                
                # Add metadata columns
                for col in metadata_columns:
                    chunk_row[col] = row[col]
                
                chunk_data.append(chunk_row)
        
        chunk_df = pd.DataFrame(chunk_data)
        
        self.logger.info(
            f"Created {len(chunk_df)} chunks from {len(df)} narratives "
            f"(avg {len(chunk_df)/len(df):.2f} chunks per narrative)"
        )
        
        return chunk_df
    
    def analyze_chunking(
        self,
        df: pd.DataFrame,
        text_column: str = 'Consumer complaint narrative'
    ) -> Dict[str, Any]:
        """
        Analyze chunking statistics for a dataset.
        
        Args:
            df: DataFrame containing narratives
            text_column: Name of the column containing text
            
        Returns:
            Dictionary with chunking statistics
        """
        if text_column not in df.columns:
            raise DataProcessingError(f"Text column '{text_column}' not found")
        
        chunk_lengths = []
        chunks_per_doc = []
        
        for text in df[text_column].dropna():
            if str(text).strip() == "":
                continue
            
            chunks = self.chunk_text(str(text))
            chunks_per_doc.append(len(chunks))
            
            for chunk in chunks:
                chunk_lengths.append(len(chunk))
        
        if not chunk_lengths:
            return {"error": "No valid text found for chunking analysis"}
        
        chunk_lengths = np.array(chunk_lengths)
        chunks_per_doc = np.array(chunks_per_doc)
        
        return {
            "total_documents": len(df[text_column].dropna()),
            "total_chunks": len(chunk_lengths),
            "avg_chunks_per_doc": float(chunks_per_doc.mean()),
            "median_chunks_per_doc": float(np.median(chunks_per_doc)),
            "max_chunks_per_doc": int(chunks_per_doc.max()),
            "chunk_length_stats": {
                "mean": float(chunk_lengths.mean()),
                "median": float(np.median(chunk_lengths)),
                "std": float(chunk_lengths.std()),
                "min": int(chunk_lengths.min()),
                "max": int(chunk_lengths.max()),
                "q25": float(np.percentile(chunk_lengths, 25)),
                "q75": float(np.percentile(chunk_lengths, 75)),
                "q90": float(np.percentile(chunk_lengths, 90)),
                "q95": float(np.percentile(chunk_lengths, 95))
            },
            "chunks_within_target": {
                "within_10_percent": int(((chunk_lengths >= self.chunk_size * 0.9) & 
                                          (chunk_lengths <= self.chunk_size * 1.1)).sum()),
                "within_20_percent": int(((chunk_lengths >= self.chunk_size * 0.8) & 
                                          (chunk_lengths <= self.chunk_size * 1.2)).sum()),
                "percentage_within_10": float(((chunk_lengths >= self.chunk_size * 0.9) & 
                                              (chunk_lengths <= self.chunk_size * 1.1)).sum() / len(chunk_lengths) * 100),
                "percentage_within_20": float(((chunk_lengths >= self.chunk_size * 0.8) & 
                                              (chunk_lengths <= self.chunk_size * 1.2)).sum() / len(chunk_lengths) * 100)
            }
        }

