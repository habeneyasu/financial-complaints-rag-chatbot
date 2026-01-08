"""
Optimized data loading strategies for large CSV files.

This script provides best practices for loading large datasets without crashing the kernel.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.config import config


def load_data_optimized(
    data_path: Path,
    use_only_essential_columns: bool = True,
    sample_size: Optional[int] = None,
    chunk_size: int = 100000,
    convert_to_parquet: bool = True
) -> pd.DataFrame:
    """
    Load large CSV file with memory optimization.
    
    Args:
        data_path: Path to CSV file
        use_only_essential_columns: If True, only load essential columns
        sample_size: Optional number of rows to sample (for testing)
        chunk_size: Number of rows to process at a time
        convert_to_parquet: If True, convert to parquet for faster future loads
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from: {data_path}")
    print(f"File size: {data_path.stat().st_size / (1024**3):.2f} GB")
    
    # Essential columns for Task 1 (EDA and preprocessing)
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
    
    # Strategy 1: Load only essential columns
    if use_only_essential_columns:
        print("\n📋 Strategy: Loading only essential columns to reduce memory usage")
        print(f"Essential columns: {len(essential_columns)}")
        
        # First, read a small sample to get column names
        sample_df = pd.read_csv(data_path, nrows=1000)
        available_columns = [col for col in essential_columns if col in sample_df.columns]
        missing_columns = [col for col in essential_columns if col not in sample_df.columns]
        
        if missing_columns:
            print(f"⚠️ Warning: Missing columns: {missing_columns}")
        
        print(f"Loading {len(available_columns)} columns: {available_columns}")
        
        # Load in chunks with only essential columns
        chunks = []
        total_rows = 0
        
        print("\n📊 Loading data in chunks...")
        for i, chunk in enumerate(pd.read_csv(
            data_path,
            chunksize=chunk_size,
            usecols=available_columns,
            low_memory=False,
            dtype={
                'Complaint ID': 'str',  # Prevent ID from being converted to float
                'ZIP code': 'str',
                'State': 'category',
                'Product': 'category',
                'Issue': 'category'
            }
        )):
            chunks.append(chunk)
            total_rows += len(chunk)
            print(f"  Processed chunk {i+1}: {total_rows:,} rows loaded", end='\r')
            
            # Stop early if sampling
            if sample_size and total_rows >= sample_size:
                break
        
        print(f"\n✅ Loaded {total_rows:,} rows")
        df = pd.concat(chunks, ignore_index=True)
        
        # Apply sampling if requested
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size:,} rows...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    else:
        # Strategy 2: Load all columns but in chunks
        print("\n📋 Strategy: Loading all columns in chunks")
        chunks = []
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(
            data_path,
            chunksize=chunk_size,
            low_memory=False
        )):
            chunks.append(chunk)
            total_rows += len(chunk)
            print(f"  Processed chunk {i+1}: {total_rows:,} rows loaded", end='\r')
            
            if sample_size and total_rows >= sample_size:
                break
        
        print(f"\n✅ Loaded {total_rows:,} rows")
        df = pd.concat(chunks, ignore_index=True)
        
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Optimize data types to reduce memory
    print("\n🔧 Optimizing data types...")
    memory_before = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory before optimization: {memory_before:.2f} MB")
    
    # Convert object columns to category where appropriate
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column has reasonable number of unique values for category
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                try:
                    df[col] = df[col].astype('category')
                except:
                    pass
    
    memory_after = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory after optimization: {memory_after:.2f} MB")
    print(f"Memory saved: {memory_before - memory_after:.2f} MB ({((memory_before - memory_after) / memory_before * 100):.1f}%)")
    
    # Convert to parquet for faster future loads
    if convert_to_parquet:
        parquet_path = data_path.parent / f"{data_path.stem}_optimized.parquet"
        print(f"\n💾 Saving optimized version to: {parquet_path}")
        df.to_parquet(parquet_path, index=False, compression='snappy')
        print(f"✅ Saved! File size: {parquet_path.stat().st_size / (1024**2):.2f} MB")
        print(f"   Use this file for faster loading in the future!")
    
    return df


def load_with_progress_bar(data_path: Path, **kwargs) -> pd.DataFrame:
    """
    Load data with progress bar using tqdm.
    """
    try:
        from tqdm import tqdm
        
        # Get total number of rows (approximate)
        total_rows = sum(1 for _ in open(data_path, 'r')) - 1  # -1 for header
        
        chunks = []
        chunk_size = kwargs.get('chunk_size', 100000)
        
        with tqdm(total=total_rows, desc="Loading data", unit="rows") as pbar:
            for chunk in pd.read_csv(data_path, chunksize=chunk_size, **kwargs):
                chunks.append(chunk)
                pbar.update(len(chunk))
        
        return pd.concat(chunks, ignore_index=True)
    except ImportError:
        print("tqdm not available, loading without progress bar...")
        return load_data_optimized(data_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    data_path = Path("data/raw/complaints.csv")
    
    if data_path.exists():
        print("=" * 80)
        print("OPTIMIZED DATA LOADING")
        print("=" * 80)
        
        # Option 1: Load only essential columns (RECOMMENDED)
        print("\n" + "=" * 80)
        print("OPTION 1: Load only essential columns (RECOMMENDED)")
        print("=" * 80)
        df = load_data_optimized(
            data_path,
            use_only_essential_columns=True,
            chunk_size=100000,
            convert_to_parquet=True
        )
        
        print(f"\n✅ Successfully loaded {len(df):,} rows × {len(df.columns)} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    else:
        print(f"Data file not found: {data_path}")

