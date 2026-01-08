"""
Script to populate the vector store from embeddings file.

This script loads pre-computed embeddings and adds them to the ChromaDB vector store.
For large files (>100k rows), it processes data in chunks to avoid memory issues.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.vectorstore import VectorStore
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def populate_vectorstore(
    embeddings_path: Path = None,
    reset: bool = False,
    chunk_size: int = 50000
):
    """
    Populate vector store from embeddings file.
    
    Args:
        embeddings_path: Path to embeddings parquet file
        reset: Whether to reset the vector store before adding
        chunk_size: Number of rows to process at a time for large files
    """
    # Default embeddings path - prioritize processed files (12k sample) over raw (full dataset)
    if embeddings_path is None:
        possible_paths = [
            Path("data/processed/chunked_complaints_with_embeddings.parquet"),  # Expected: 12k sample
            Path("notebooks/data/processed/chunked_complaints_with_embeddings.parquet"),  # Alternative location
            Path("data/raw/complaint_embeddings.parquet"),  # Fallback: full dataset (2.3GB)
        ]
        
        found_paths = [p for p in possible_paths if p.exists()]
        
        if not found_paths:
            raise FileNotFoundError(
                f"Embeddings file not found. Tried: {[str(p) for p in possible_paths]}\n"
                f"Please ensure you have run the embedding generation notebook (07_embedding_generation.ipynb) "
                f"for your 12,000 record sample."
            )
        
        # Prefer processed files (12k sample) over raw (full dataset)
        embeddings_path = found_paths[0]
        
        # Warn if using the large raw file
        if embeddings_path.name == "complaint_embeddings.parquet":
            file_size_mb = embeddings_path.stat().st_size / (1024**2)
            logger.warning(
                f"⚠️  Using large raw embeddings file ({file_size_mb:.1f} MB). "
                f"This appears to be the full dataset, not the 12k sample.\n"
                f"   Consider generating embeddings for your 12k sample instead.\n"
                f"   Expected file: data/processed/chunked_complaints_with_embeddings.parquet"
            )
    
    logger.info(f"Loading embeddings from: {embeddings_path}")
    
    # Check file size and determine processing strategy
    try:
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(embeddings_path)
        num_rows = parquet_file.metadata.num_rows
        file_size_mb = embeddings_path.stat().st_size / (1024**2)
        logger.info(f"File contains {num_rows:,} rows ({file_size_mb:.1f} MB)")
        
        # Warn if file seems too large for 12k sample
        if num_rows > 100000 or file_size_mb > 500:
            logger.warning(
                f"⚠️  File appears to contain more than the expected 12k sample.\n"
                f"   Expected: ~12,000-15,000 records\n"
                f"   Found: {num_rows:,} records\n"
                f"   This may take a long time to process."
            )
        
        # Read first 100 rows to get structure
        df_sample = pd.read_parquet(embeddings_path, nrows=100)
        
        # Verify required columns
        required_cols = ['embedding', 'chunk_id', 'chunk_text']
        missing_cols = [col for col in required_cols if col not in df_sample.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get metadata columns
        metadata_candidates = [
            'Product', 'Issue', 'Company', 'State', 'Date received', 
            'Complaint ID', 'original_index', 'chunk_index', 'num_chunks', 'chunk_length'
        ]
        available_metadata = [col for col in metadata_candidates if col in df_sample.columns]
        logger.info(f"Available metadata columns: {available_metadata}")
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            collection_name=config.vectorstore.collection_name,
            persist_directory=str(config.data.vector_store_dir),
            reset=reset
        )
        
        info = vector_store.get_collection_info()
        logger.info(f"Current document count: {info['document_count']:,}")
        
        if info['document_count'] > 0 and not reset:
            logger.warning(f"Vector store already has {info['document_count']:,} documents.")
            logger.warning("Set reset=True to recreate from scratch.")
            response = input("Continue adding to existing store? (y/n): ")
            if response.lower() != 'y':
                logger.info("Aborted.")
                return
        
        # Process file in chunks if large
        if num_rows > 100000:
            logger.info(f"Large file detected ({num_rows:,} rows). Processing in chunks of {chunk_size:,}...")
            total_added = 0
            
            for batch_num, batch_df in enumerate(pd.read_parquet(embeddings_path, chunksize=chunk_size)):
                logger.info(f"Processing batch {batch_num + 1} ({len(batch_df):,} rows)...")
                
                num_added = vector_store.add_embeddings(
                    df=batch_df,
                    embedding_column='embedding',
                    id_column='chunk_id',
                    text_column='chunk_text',
                    metadata_columns=available_metadata,
                    batch_size=config.vectorstore.batch_size
                )
                total_added += num_added
                logger.info(f"  Added {num_added:,} documents (total: {total_added:,})")
        else:
            # Small file - load all at once
            logger.info("Loading full file into memory...")
            df = pd.read_parquet(embeddings_path)
            logger.info(f"✅ Loaded dataset: {df.shape[0]:,} chunks × {df.shape[1]} columns")
            
            logger.info("Adding embeddings to vector store...")
            total_added = vector_store.add_embeddings(
                df=df,
                embedding_column='embedding',
                id_column='chunk_id',
                text_column='chunk_text',
                metadata_columns=available_metadata,
                batch_size=config.vectorstore.batch_size
            )
        
        logger.info(f"✅ Successfully added {total_added:,} documents to vector store")
        
        # Verify
        info = vector_store.get_collection_info()
        logger.info(f"Total documents in collection: {info['document_count']:,}")
        
        # Test query
        logger.info("\nTesting query...")
        results = vector_store.query(query_text="credit card", n_results=3)
        logger.info(f"Test query returned {len(results['ids'])} results")
        if results['ids']:
            logger.info(f"First result: {results['documents'][0][:100]}...")
        
        return total_added
        
    except Exception as e:
        logger.error(f"Error processing embeddings file: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate vector store from embeddings")
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=None,
        help="Path to embeddings parquet file"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before adding (deletes existing data)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of rows to process at a time for large files (default: 50000)"
    )
    
    args = parser.parse_args()
    
    try:
        populate_vectorstore(
            embeddings_path=args.embeddings_path,
            reset=args.reset,
            chunk_size=args.chunk_size
        )
        print("\n✅ Vector store populated successfully!")
        print("You can now run the app: python app.py")
    except Exception as e:
        logger.error(f"Error populating vector store: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
