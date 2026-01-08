"""
Data loader module for CFPB complaint dataset.

This module provides a robust, reusable class for loading the CFPB complaint
dataset with comprehensive error handling and validation.
"""

import pandas as pd
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
import logging

from src.config import config
from src.exceptions import DataLoadError, FileNotFoundError
from src.utils.logger import get_logger


class DataLoader:
    """
    A class for loading CFPB complaint dataset with error handling and validation.
    
    This class follows the Single Responsibility Principle by focusing solely
    on data loading operations. It provides methods for downloading, validating,
    and loading the dataset.
    
    Attributes:
        data_path: Path to the data file
        logger: Logger instance for this class
    """
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Optional path to data file (defaults to config)
            chunk_size: Optional chunk size for reading large files
        """
        self.data_path = data_path or config.data.raw_data_path
        self.chunk_size = chunk_size or config.data.chunk_size
        self.logger = get_logger(__name__)
        self._data: Optional[pd.DataFrame] = None
    
    def download_data(self, url: Optional[str] = None, force: bool = False) -> Path:
        """
        Download the CFPB complaint dataset from the official source.
        
        Args:
            url: Optional URL to download from (defaults to config)
            force: If True, re-download even if file exists
            
        Returns:
            Path to the downloaded file
            
        Raises:
            DataLoadError: If download fails
        """
        url = url or config.data.cfpb_data_url
        if not url:
            raise DataLoadError("No download URL provided")
        
        # Check if file already exists
        if self.data_path.exists() and not force:
            self.logger.info(f"Data file already exists at {self.data_path}")
            return self.data_path
        
        try:
            self.logger.info(f"Downloading data from {url}")
            self.logger.info(f"Target location: {self.data_path}")
            
            # Ensure directory exists
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            urlretrieve(url, self.data_path)
            self.logger.info("Download completed successfully")
            
            return self.data_path
            
        except HTTPError as e:
            error_msg = f"HTTP error {e.code} while downloading: {e.reason}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg, details={"url": url, "http_code": e.code})
            
        except URLError as e:
            error_msg = f"URL error while downloading: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg, details={"url": url})
            
        except Exception as e:
            error_msg = f"Unexpected error during download: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg, details={"url": url, "error_type": type(e).__name__})
    
    def _extract_zip_if_needed(self, file_path: Path) -> Path:
        """
        Extract CSV from ZIP file if needed.
        
        Args:
            file_path: Path to the ZIP file
            
        Returns:
            Path to the extracted CSV file
        """
        if file_path.suffix == ".zip":
            self.logger.info(f"Extracting ZIP file: {file_path}")
            extract_dir = file_path.parent / file_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find CSV file in extracted directory
                csv_files = list(extract_dir.glob("*.csv"))
                if csv_files:
                    csv_path = csv_files[0]
                    self.logger.info(f"Extracted CSV to: {csv_path}")
                    return csv_path
                else:
                    raise DataLoadError(
                        f"No CSV file found in ZIP archive: {file_path}",
                        details={"extract_dir": str(extract_dir)}
                    )
            except zipfile.BadZipFile as e:
                raise DataLoadError(
                    f"Invalid ZIP file: {file_path}",
                    details={"error": str(e)}
                )
        
        return file_path
    
    def load_data(
        self,
        file_path: Optional[Path] = None,
        sample_size: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load the complaint dataset from file.
        
        Args:
            file_path: Optional path to data file
            sample_size: Optional number of rows to sample (for testing)
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            DataFrame containing the complaint data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            DataLoadError: If loading fails
        """
        file_path = file_path or self.data_path
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}",
                details={"path": str(file_path)}
            )
        
        try:
            self.logger.info(f"Loading data from: {file_path}")
            
            # Extract ZIP if needed
            actual_path = self._extract_zip_if_needed(file_path)
            
            # Determine if we should read in chunks
            file_size = actual_path.stat().st_size
            use_chunks = file_size > 100 * 1024 * 1024  # > 100MB
            
            if use_chunks:
                self.logger.info("Large file detected, reading in chunks")
                chunks = []
                for chunk in pd.read_csv(actual_path, chunksize=self.chunk_size, **kwargs):
                    chunks.append(chunk)
                    if sample_size and len(pd.concat(chunks, ignore_index=True)) >= sample_size:
                        break
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(actual_path, **kwargs)
            
            # Apply sampling if requested
            if sample_size and len(df) > sample_size:
                self.logger.info(f"Sampling {sample_size} rows from {len(df)} total rows")
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            self.logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            self._data = df
            
            return df
            
        except pd.errors.EmptyDataError as e:
            error_msg = f"Data file is empty: {file_path}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg, details={"path": str(file_path)})
            
        except pd.errors.ParserError as e:
            error_msg = f"Error parsing CSV file: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg, details={"path": str(file_path), "error": str(e)})
            
        except Exception as e:
            error_msg = f"Unexpected error loading data: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg, details={"path": str(file_path), "error_type": type(e).__name__})
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the loaded dataset structure and content.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
            
        Raises:
            DataValidationError: If validation fails
        """
        self.logger.info("Validating dataset structure")
        
        validation_results = {
            "is_valid": True,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict()
        }
        
        # Check for required columns (common CFPB complaint columns)
        expected_columns = [
            "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
            "Consumer complaint narrative", "Company public response",
            "Company", "State", "ZIP code", "Tags", "Consumer consent provided?",
            "Submitted via", "Date sent to company", "Company response to consumer",
            "Timely response?", "Consumer disputed?", "Complaint ID"
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Missing expected columns: {missing_columns}")
            validation_results["missing_columns"] = missing_columns
        
        # Check for empty dataset
        if len(df) == 0:
            validation_results["is_valid"] = False
            self.logger.error("Dataset is empty")
        
        # Log validation summary
        self.logger.info(f"Validation complete. Valid: {validation_results['is_valid']}")
        self.logger.info(f"Rows: {validation_results['row_count']}, "
                        f"Columns: {validation_results['column_count']}")
        
        return validation_results
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the loaded data."""
        return self._data
    
    def load_data_optimized(
        self,
        essential_columns: Optional[List[str]] = None,
        file_path: Optional[Path] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data with optimization for large files (loads only essential columns).
        
        This method is designed to prevent kernel crashes when loading very large files
        by loading only essential columns and using optimized data types.
        
        Args:
            essential_columns: List of column names to load (if None, loads all columns)
            file_path: Optional path to data file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            DataFrame containing the complaint data
        """
        file_path = file_path or self.data_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Default essential columns for Task 1 and Task 2
        if essential_columns is None:
            essential_columns = [
                'Date received',
                'Product',
                'Sub-product',
                'Issue',
                'Sub-issue',
                'Consumer complaint narrative',
                'Company',
                'State',
                'ZIP code',
                'Complaint ID'
            ]
        
        try:
            self.logger.info(f"Loading optimized data from: {file_path}")
            
            # Extract ZIP if needed
            actual_path = self._extract_zip_if_needed(file_path)
            
            # Check available columns
            sample_df = pd.read_csv(actual_path, nrows=100)
            available_columns = [col for col in essential_columns if col in sample_df.columns]
            missing_columns = [col for col in essential_columns if col not in sample_df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}")
            
            self.logger.info(f"Loading {len(available_columns)} essential columns")
            
            # Optimized dtype mapping
            optimized_dtypes = {
                'Complaint ID': 'str',
                'ZIP code': 'str',
                'State': 'category',
                'Product': 'category',
                'Issue': 'category'
            }
            
            # Filter dtypes to only include columns that exist
            dtype_dict = {k: v for k, v in optimized_dtypes.items() if k in available_columns}
            
            # Determine if we should read in chunks
            file_size = actual_path.stat().st_size
            use_chunks = file_size > 100 * 1024 * 1024  # > 100MB
            
            if use_chunks:
                self.logger.info("Large file detected, reading in chunks with essential columns")
                self.logger.info(f"Chunk size: {self.chunk_size:,} rows")
                
                chunks = []
                total_rows = 0
                chunk_count = 0
                
                # Use smaller chunks for very large files to prevent memory issues
                effective_chunk_size = min(self.chunk_size, 50000)  # Max 50k rows per chunk
                
                try:
                    from tqdm import tqdm
                    iterator = tqdm(
                        pd.read_csv(
                            actual_path,
                            chunksize=effective_chunk_size,
                            usecols=available_columns,
                            low_memory=False,
                            dtype=dtype_dict,
                            **kwargs
                        ),
                        desc="Loading chunks"
                    )
                except ImportError:
                    iterator = pd.read_csv(
                        actual_path,
                        chunksize=effective_chunk_size,
                        usecols=available_columns,
                        low_memory=False,
                        dtype=dtype_dict,
                        **kwargs
                    )
                
                for chunk in iterator:
                    # Optimize chunk immediately to reduce memory
                    for col in chunk.columns:
                        if chunk[col].dtype == 'object' and col not in ['Consumer complaint narrative', 'Date received']:
                            unique_ratio = chunk[col].nunique() / len(chunk)
                            if unique_ratio < 0.5:
                                try:
                                    chunk[col] = chunk[col].astype('category')
                                except:
                                    pass
                    
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    chunk_count += 1
                    
                    # Log progress every 10 chunks
                    if chunk_count % 10 == 0:
                        self.logger.info(f"Processed {chunk_count} chunks: {total_rows:,} rows loaded...")
                    
                    # Safety check: if we're using too much memory, warn user
                    if chunk_count % 50 == 0:
                        current_memory = sum(c.memory_usage(deep=True).sum() for c in chunks) / 1024**2
                        if current_memory > 2000:  # More than 2GB
                            self.logger.warning(f"High memory usage: {current_memory:.2f} MB. Consider using a sample.")
                
                # Concatenate chunks in smaller batches to avoid memory spikes
                self.logger.info(f"Combining {len(chunks)} chunks...")
                if len(chunks) > 20:
                    # For many chunks, combine in batches
                    batch_size = 20
                    combined_chunks = []
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        combined = pd.concat(batch, ignore_index=True)
                        combined_chunks.append(combined)
                        self.logger.debug(f"Combined batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                    df = pd.concat(combined_chunks, ignore_index=True)
                else:
                    df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(
                    actual_path,
                    usecols=available_columns,
                    low_memory=False,
                    dtype=dtype_dict,
                    **kwargs
                )
            
            # Final memory optimization (only if not already done in chunks)
            if not use_chunks:
                for col in df.columns:
                    if df[col].dtype == 'object' and col not in ['Consumer complaint narrative', 'Date received']:
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio < 0.5:  # Less than 50% unique values
                            try:
                                df[col] = df[col].astype('category')
                            except:
                                pass
            
            self.logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns (optimized)")
            self._data = df
            
            return df
            
        except Exception as e:
            error_msg = f"Error loading optimized data: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg, details={"path": str(file_path), "error_type": type(e).__name__})
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the loaded dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if self._data is None:
            raise DataLoadError("No data loaded. Call load_data() first.")
        
        return {
            "shape": self._data.shape,
            "columns": list(self._data.columns),
            "dtypes": self._data.dtypes.to_dict(),
            "memory_usage_mb": self._data.memory_usage(deep=True).sum() / 1024**2,
            "null_counts": self._data.isnull().sum().to_dict()
        }

