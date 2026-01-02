"""
Data preprocessing module for CFPB complaint dataset.

This module provides a comprehensive data preprocessing class that handles
cleaning, transformation, and preparation of complaint data for the RAG pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

from src.config import config
from src.exceptions import DataProcessingError, DataValidationError
from src.utils.logger import get_logger


class DataPreprocessor:
    """
    A class for preprocessing CFPB complaint data.
    
    This class handles data cleaning, transformation, and preparation
    following industry best practices for data preprocessing.
    
    Attributes:
        df: DataFrame to preprocess
        logger: Logger instance
        narrative_col: Name of the narrative column
    """
    
    def __init__(self, df: pd.DataFrame, narrative_col: str = "Consumer complaint narrative"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            df: DataFrame to preprocess
            narrative_col: Name of the narrative column
        """
        if df is None or df.empty:
            raise DataValidationError("DataFrame is None or empty")
        
        self.df = df.copy()
        self.logger = get_logger(__name__)
        self.narrative_col = narrative_col
        self._original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataPreprocessor':
        """
        Remove duplicate rows from the dataset.
        
        Args:
            subset: Optional list of columns to consider for duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Self for method chaining
        """
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_count - len(self.df)
        
        if removed > 0:
            self.logger.info(f"Removed {removed:,} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")
        
        return self
    
    def clean_narratives(
        self,
        remove_whitespace: bool = True,
        remove_extra_spaces: bool = True,
        min_length: int = 10
    ) -> 'DataPreprocessor':
        """
        Clean narrative text data.
        
        Args:
            remove_whitespace: Remove leading/trailing whitespace
            remove_extra_spaces: Remove multiple consecutive spaces
            min_length: Minimum character length for valid narratives
            
        Returns:
            Self for method chaining
        """
        if self.narrative_col not in self.df.columns:
            self.logger.warning(f"Column '{self.narrative_col}' not found. Skipping narrative cleaning.")
            return self
        
        initial_non_null = self.df[self.narrative_col].notna().sum()
        
        # Convert to string and handle NaN
        self.df[self.narrative_col] = self.df[self.narrative_col].astype(str)
        
        # Replace 'nan' strings with actual NaN
        self.df[self.narrative_col] = self.df[self.narrative_col].replace(['nan', 'None', ''], np.nan)
        
        if remove_whitespace:
            self.df[self.narrative_col] = self.df[self.narrative_col].str.strip()
        
        if remove_extra_spaces:
            self.df[self.narrative_col] = self.df[self.narrative_col].str.replace(r'\s+', ' ', regex=True)
        
        # Remove narratives that are too short
        if min_length > 0:
            mask = (self.df[self.narrative_col].str.len() < min_length) & (self.df[self.narrative_col].notna())
            self.df.loc[mask, self.narrative_col] = np.nan
            removed = mask.sum()
            if removed > 0:
                self.logger.info(f"Marked {removed:,} narratives as null (length < {min_length} chars)")
        
        final_non_null = self.df[self.narrative_col].notna().sum()
        self.logger.info(f"Narrative cleaning complete. Non-null narratives: {initial_non_null:,} -> {final_non_null:,}")
        
        return self
    
    def handle_missing_values(
        self,
        strategy: str = 'drop',
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None
    ) -> 'DataPreprocessor':
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: Strategy to use ('drop', 'fill', 'keep')
            columns: Optional list of columns to process (default: all)
            fill_value: Value to fill with if strategy is 'fill'
            
        Returns:
            Self for method chaining
        """
        columns = columns or self.df.columns.tolist()
        
        if strategy == 'drop':
            initial_count = len(self.df)
            # Only drop rows where all specified columns are null
            self.df = self.df.dropna(subset=columns, how='all')
            removed = initial_count - len(self.df)
            if removed > 0:
                self.logger.info(f"Dropped {removed:,} rows with all missing values in specified columns")
        
        elif strategy == 'fill':
            if fill_value is None:
                raise DataProcessingError("fill_value must be provided when strategy is 'fill'")
            
            for col in columns:
                if col in self.df.columns:
                    null_count = self.df[col].isnull().sum()
                    if null_count > 0:
                        self.df[col] = self.df[col].fillna(fill_value)
                        self.logger.info(f"Filled {null_count:,} missing values in '{col}' with {fill_value}")
        
        elif strategy == 'keep':
            self.logger.info("Keeping missing values as-is")
        
        else:
            raise DataProcessingError(f"Unknown strategy: {strategy}")
        
        return self
    
    def standardize_dates(self, date_columns: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Standardize date columns to datetime format.
        
        Args:
            date_columns: Optional list of date column names
            
        Returns:
            Self for method chaining
        """
        if date_columns is None:
            # Auto-detect date columns
            date_columns = [col for col in self.df.columns if 'date' in col.lower()]
        
        for col in date_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    converted = self.df[col].notna().sum()
                    self.logger.info(f"Converted '{col}' to datetime. {converted:,} valid dates")
                except Exception as e:
                    self.logger.warning(f"Could not convert '{col}' to datetime: {e}")
        
        return self
    
    def add_derived_features(self) -> 'DataPreprocessor':
        """
        Add derived features useful for analysis and RAG pipeline.
        
        Returns:
            Self for method chaining
        """
        # Add word count if narrative column exists
        if self.narrative_col in self.df.columns:
            def word_count(text):
                if pd.isna(text) or text == '':
                    return 0
                return len(str(text).split())
            
            self.df['narrative_word_count'] = self.df[self.narrative_col].apply(word_count)
            self.logger.info("Added 'narrative_word_count' feature")
        
        # Add character count
        if self.narrative_col in self.df.columns:
            def char_count(text):
                if pd.isna(text) or text == '':
                    return 0
                return len(str(text))
            
            self.df['narrative_char_count'] = self.df[self.narrative_col].apply(char_count)
            self.logger.info("Added 'narrative_char_count' feature")
        
        # Add has_narrative flag
        if self.narrative_col in self.df.columns:
            self.df['has_narrative'] = self.df[self.narrative_col].notna() & (self.df[self.narrative_col] != '')
            self.logger.info("Added 'has_narrative' feature")
        
        return self
    
    def filter_by_conditions(
        self,
        conditions: Dict[str, Any],
        keep: bool = True
    ) -> 'DataPreprocessor':
        """
        Filter dataset based on conditions.
        
        Args:
            conditions: Dictionary of column: value pairs for filtering
            keep: If True, keep rows matching conditions; if False, drop them
            
        Returns:
            Self for method chaining
        """
        initial_count = len(self.df)
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        for col, value in conditions.items():
            if col in self.df.columns:
                if isinstance(value, (list, tuple)):
                    col_mask = self.df[col].isin(value)
                else:
                    col_mask = self.df[col] == value
                
                if keep:
                    mask = mask & col_mask
                else:
                    mask = mask & ~col_mask
        
        self.df = self.df[mask].reset_index(drop=True)
        removed = initial_count - len(self.df)
        
        if removed > 0:
            action = "Kept" if keep else "Removed"
            self.logger.info(f"{action} {removed:,} rows based on conditions")
        
        return self
    
    def select_columns(self, columns: List[str]) -> 'DataPreprocessor':
        """
        Select specific columns from the dataset.
        
        Args:
            columns: List of column names to keep
            
        Returns:
            Self for method chaining
        """
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            self.logger.warning(f"Columns not found: {missing_cols}")
        
        available_cols = [col for col in columns if col in self.df.columns]
        self.df = self.df[available_cols]
        
        self.logger.info(f"Selected {len(available_cols)} columns")
        return self
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of preprocessing operations.
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            "original_shape": self._original_shape,
            "current_shape": self.df.shape,
            "rows_removed": self._original_shape[0] - self.df.shape[0],
            "columns_removed": self._original_shape[1] - self.df.shape[1],
            "current_columns": list(self.df.columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def save_processed_data(self, file_path: Optional[Path] = None, format: str = 'parquet') -> Path:
        """
        Save processed data to file.
        
        Args:
            file_path: Optional path to save file (defaults to config)
            format: File format ('parquet', 'csv', 'pickle')
            
        Returns:
            Path to saved file
        """
        if file_path is None:
            file_path = config.data.processed_data_path
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'parquet':
                self.df.to_parquet(file_path, index=False, engine='pyarrow')
            elif format == 'csv':
                self.df.to_csv(file_path, index=False)
            elif format == 'pickle':
                self.df.to_pickle(file_path)
            else:
                raise DataProcessingError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved processed data to {file_path} ({format})")
            return file_path
            
        except Exception as e:
            error_msg = f"Error saving processed data: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg, details={"path": str(file_path), "format": format})
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.
        
        Returns:
            Processed DataFrame
        """
        return self.df.copy()
    
    def reset(self) -> 'DataPreprocessor':
        """
        Reset to original data (not implemented - would need to store original).
        
        Returns:
            Self for method chaining
        """
        self.logger.warning("Reset not implemented. Create new DataPreprocessor instance.")
        return self

