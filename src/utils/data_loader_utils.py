"""
Data Loading Utilities for Notebooks

This module provides reusable, memory-efficient data loading functions
that can be used across all notebooks. It implements industry best practices:
- Lazy loading
- Caching (Parquet format)
- Memory optimization
- Smart fallback strategies
- Progress indicators

Usage:
    from src.utils.data_loader_utils import load_complaints_data
    
    df = load_complaints_data()  # Automatically handles caching and optimization
"""

import sys
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np
import warnings
import os

# Optional: psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global cache for loaded data (shared across notebook cells)
_LOADED_DATA_CACHE = {}


def _get_available_memory() -> float:
    """
    Get available system memory in MB.
    
    Returns:
        Available memory in MB
    """
    if not PSUTIL_AVAILABLE:
        # Fallback: assume 8GB if psutil not available
        return 8000
    
    try:
        # Get available system memory
        system_mem = psutil.virtual_memory()
        return system_mem.available / (1024**2)  # Convert to MB
    except Exception:
        # Fallback: assume 8GB on error
        return 8000


def load_complaints_data(
    use_cache: bool = True,
    use_sample: Optional[int] = None,
    force_reload: bool = False,
    project_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load complaints data with intelligent caching and memory optimization.
    
    This function implements a multi-tier loading strategy:
    1. Check if data is already in memory (from previous notebook cell)
    2. Check for cached Parquet file (fastest)
    3. Load from CSV with optimization (only if needed)
    
    Args:
        use_cache: Whether to use cached data if available (default: True)
        use_sample: If provided, load only this many rows (for testing)
        force_reload: Force reload even if data is cached (default: False)
        project_root: Project root directory (auto-detected if None)
        
    Returns:
        DataFrame with complaints data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        MemoryError: If system doesn't have enough memory
        
    Example:
        >>> # Basic usage - uses cache if available
        >>> df = load_complaints_data()
        
        >>> # Load sample for testing
        >>> df = load_complaints_data(use_sample=100000)
        
        >>> # Force reload from source
        >>> df = load_complaints_data(force_reload=True)
    """
    global _LOADED_DATA_CACHE
    
    # Auto-detect project root if not provided
    if project_root is None:
        # Try to detect from current working directory
        cwd = Path.cwd()
        if cwd.name == 'notebooks':
            project_root = cwd.parent
        else:
            project_root = cwd
    
    cache_key = f"complaints_data_{use_sample}"
    
    # Tier 1: Check memory cache (fastest - instant)
    if use_cache and not force_reload and cache_key in _LOADED_DATA_CACHE:
        logger.info("Using data from memory cache")
        return _LOADED_DATA_CACHE[cache_key]
    
    # Tier 2: Check for cached Parquet file (very fast - seconds)
    parquet_path = project_root / "data" / "raw" / "complaints_optimized.parquet"
    
    if use_cache and not force_reload and parquet_path.exists():
        try:
            logger.info(f"Loading from cached Parquet file: {parquet_path.name}")
            
            # Check file size first
            parquet_size_mb = parquet_path.stat().st_size / (1024**2)
            logger.info(f"Parquet file size: {parquet_size_mb:.2f} MB")
            
            if use_sample:
                # For sampling, use efficient chunked reading for large files
                if parquet_size_mb > 500:  # If > 500MB, use chunked reading to avoid memory issues
                    logger.info(f"Large Parquet file detected ({parquet_size_mb:.1f} MB). Using efficient sampling...")
                    import pyarrow.parquet as pq
                    
                    # Get file metadata without loading
                    parquet_file = pq.ParquetFile(parquet_path)
                    total_rows = parquet_file.metadata.num_rows
                    
                    # Calculate how many rows we need to read to get a good sample
                    # Read more rows than needed, then sample (more efficient than loading full file)
                    rows_to_read = min(total_rows, use_sample * 3)  # Read 3x to ensure good sample
                    
                    # Read only a subset of row groups to minimize memory
                    num_row_groups = parquet_file.num_row_groups
                    row_groups_to_read = max(1, int((rows_to_read / total_rows) * num_row_groups))
                    row_groups_to_read = min(row_groups_to_read, num_row_groups)
                    
                    # Read selected row groups
                    row_group_indices = list(range(row_groups_to_read))
                    df = parquet_file.read_row_groups(row_group_indices).to_pandas()
                    
                    # Sample from the loaded subset
                    if len(df) > use_sample:
                        df = df.sample(n=min(use_sample, len(df)), random_state=42).reset_index(drop=True)
                    
                    logger.info(f"Sampled {len(df):,} rows from {row_groups_to_read} row groups (total: {num_row_groups})")
                else:
                    # For smaller files, load and sample normally
                    df = pd.read_parquet(parquet_path)
                    if len(df) > use_sample:
                        df = df.sample(n=use_sample, random_state=42).reset_index(drop=True)
                        logger.info(f"Sampled {use_sample:,} rows from cached file")
            else:
                # For full dataset, check memory before loading
                available_memory_mb = _get_available_memory()
                logger.info(f"Available memory: {available_memory_mb:.0f} MB")
                
                # Estimate memory usage (Parquet files expand when loaded)
                estimated_memory_mb = parquet_size_mb * 1.5  # Parquet compresses, so expands when loaded
                
                # Use 90% threshold (less conservative) - only block if truly insufficient
                if estimated_memory_mb > available_memory_mb * 0.9:  # If > 90% of available memory
                    error_msg = (
                        f"Insufficient memory to load full dataset.\n"
                        f"  Parquet file: {parquet_size_mb:.0f} MB\n"
                        f"  Estimated memory needed: {estimated_memory_mb:.0f} MB\n"
                        f"  Available memory: {available_memory_mb:.0f} MB\n\n"
                        f"SOLUTION: Use a sample instead:\n"
                        f"  df = load_complaints_data(use_sample=1000000)"
                    )
                    logger.error(error_msg)
                    raise MemoryError(error_msg)
                elif estimated_memory_mb > available_memory_mb * 0.8:  # Warn if > 80%
                    logger.warning(
                        f"Loading large dataset may use most available memory.\n"
                        f"  Estimated: {estimated_memory_mb:.0f} MB / Available: {available_memory_mb:.0f} MB\n"
                        f"  Consider using a sample: load_complaints_data(use_sample=1000000)"
                    )
                
                # Load with memory monitoring
                try:
                    df = pd.read_parquet(parquet_path)
                except MemoryError as e:
                    error_msg = (
                        f"Memory error loading Parquet file.\n"
                        f"Try: load_complaints_data(use_sample=100000)"
                    )
                    logger.error(error_msg)
                    raise MemoryError(error_msg) from e
            
            # Cache in memory (only if reasonable size)
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            if memory_usage < 5000:  # Only cache if < 5GB
                _LOADED_DATA_CACHE[cache_key] = df
            else:
                logger.info("Dataset too large for memory cache, skipping cache")
            
            logger.info(f"✅ Loaded from cache: {df.shape[0]:,} rows × {df.shape[1]} columns")
            logger.info(f"   Memory usage: {memory_usage:.2f} MB")
            
            return df
            
        except MemoryError as e:
            logger.error(f"Memory error loading Parquet: {e}")
            raise MemoryError(
                f"Not enough memory to load Parquet file. "
                f"Try: load_complaints_data(use_sample=100000)"
            ) from e
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}. Falling back to CSV loading.")
    
    # Tier 3: Load from CSV with optimization (slower - minutes)
    csv_path = project_root / "data" / "raw" / "complaints.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Please ensure the complaints.csv file is in the data/raw directory."
        )
    
    logger.info("Loading from CSV with optimization...")
    
    # Initialize DataLoader
    loader = DataLoader(data_path=csv_path, chunk_size=100000)
    
    try:
        if use_sample:
            # Use regular load_data with sampling (more memory efficient)
            logger.info(f"Loading sample of {use_sample:,} rows...")
            df = loader.load_data(sample_size=use_sample)
        else:
            # Use optimized loading (only essential columns)
            logger.info("Loading full dataset with optimization (essential columns only)...")
            df = loader.load_data_optimized()
        
        # Cache in memory
        _LOADED_DATA_CACHE[cache_key] = df
        
        # Try to save as Parquet for future use (if not sampling)
        if not use_sample and not parquet_path.exists():
            try:
                logger.info("Saving optimized version to Parquet for faster future loads...")
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(parquet_path, index=False, compression='snappy')
                logger.info(f"✅ Cached to: {parquet_path.name}")
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")
        
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"✅ Successfully loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"   Memory usage: {memory_usage:.2f} MB")
        
        return df
        
    except MemoryError as e:
        error_msg = (
            f"Memory error while loading data: {e}\n\n"
            f"SOLUTIONS:\n"
            f"1. Use a sample: load_complaints_data(use_sample=100000)\n"
            f"2. Restart kernel to free memory\n"
            f"3. Use a machine with more RAM\n"
            f"4. Load from cached Parquet if available"
        )
        logger.error(error_msg)
        raise MemoryError(error_msg) from e
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def clear_data_cache():
    """
    Clear the in-memory data cache.
    
    Useful when you need to free up memory or force a reload.
    
    Example:
        >>> clear_data_cache()
        >>> df = load_complaints_data()  # Will reload from source
    """
    global _LOADED_DATA_CACHE
    _LOADED_DATA_CACHE.clear()
    logger.info("Data cache cleared")


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about the loaded dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    return {
        "shape": df.shape,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum()
    }


# Convenience function for notebooks
def quick_load(use_sample: Optional[int] = None) -> pd.DataFrame:
    """
    Quick load function for notebook use.
    
    This is a convenience wrapper around load_complaints_data() with
    sensible defaults for notebook environments.
    
    Args:
        use_sample: Optional sample size (default: None = full dataset)
        
    Returns:
        DataFrame with complaints data
        
    Example:
        >>> df = quick_load()  # Load full dataset
        >>> df = quick_load(100000)  # Load 100k sample
    """
    return load_complaints_data(use_cache=True, use_sample=use_sample)

