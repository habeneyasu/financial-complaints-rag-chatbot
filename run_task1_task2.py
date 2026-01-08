#!/usr/bin/env python3
"""
Task 1 & 2: Complete EDA, Data Preprocessing, and Vector Store Setup Script

This script demonstrates all the key functions required for Task 1 & 2:
1. EDA (Exploratory Data Analysis)
2. Filtering/cleaning (product filtering, narrative removal)
3. Stratified sampling (10k-15k with proportional representation)
4. Text chunking
5. Embedding generation
6. Vector store creation
7. Saving filtered_complaints.csv

Usage:
    python run_task1_task2.py [--data-path PATH] [--sample-size N] [--skip-vector-store]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.eda.analyzer import EDAAnalyzer
from src.rag.chunker import TextChunker
from src.rag.embedder import EmbeddingGenerator
from src.rag.vectorstore import VectorStore
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_eda(df, output_dir: Path) -> Dict[str, Any]:
    """
    Step 1: Perform Exploratory Data Analysis
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save EDA results
        
    Returns:
        Dictionary with EDA results
    """
    logger.info("=" * 80)
    logger.info("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("=" * 80)
    
    analyzer = EDAAnalyzer(df)
    
    # Analyze product distribution
    logger.info("\n📊 Analyzing product distribution...")
    product_dist = analyzer.analyze_product_distribution()
    logger.info(f"Products found: {list(product_dist['counts'].keys())}")
    logger.info(f"Total records: {product_dist['total']:,}")
    
    # Analyze narrative length
    logger.info("\n📏 Analyzing narrative length...")
    narrative_length = analyzer.analyze_narrative_length('Consumer complaint narrative')
    logger.info(f"Mean length: {narrative_length['mean']:.1f} characters")
    logger.info(f"Median length: {narrative_length['median']:.1f} characters")
    
    # Check for missing narratives
    logger.info("\n🔍 Checking for missing narratives...")
    missing_count = df['Consumer complaint narrative'].isna().sum()
    empty_count = (df['Consumer complaint narrative'].astype(str).str.strip() == '').sum()
    logger.info(f"Missing narratives: {missing_count:,}")
    logger.info(f"Empty narratives: {empty_count:,}")
    
    # Generate visualizations
    logger.info("\n📈 Generating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        analyzer.plot_product_distribution(save_path=output_dir / "product_distribution.png")
        logger.info("✅ Saved: product_distribution.png")
    except Exception as e:
        logger.warning(f"Could not save product distribution plot: {e}")
    
    try:
        analyzer.plot_narrative_length_distribution(
            narrative_col='Consumer complaint narrative',
            save_path=output_dir / "narrative_length_distribution.png"
        )
        logger.info("✅ Saved: narrative_length_distribution.png")
    except Exception as e:
        logger.warning(f"Could not save narrative length plot: {e}")
    
    return {
        'product_distribution': product_dist,
        'narrative_length': narrative_length,
        'missing_narratives': missing_count,
        'empty_narratives': empty_count
    }


def run_filtering_and_cleaning(df, output_dir: Path) -> Dict[str, Any]:
    """
    Step 2: Filter by products and remove empty narratives
    
    Args:
        df: DataFrame to filter
        output_dir: Directory to save filtered dataset
        
    Returns:
        Dictionary with filtered DataFrame and statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: FILTERING AND CLEANING")
    logger.info("=" * 80)
    
    # Target products as specified in assignment
    target_products = [
        "Credit card",
        "Personal loan",
        "Savings account",
        "Money transfers"
    ]
    
    logger.info(f"\n🎯 Filtering to target products: {target_products}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(df, narrative_col='Consumer complaint narrative')
    
    # Apply Task 1 filtering (product filtering + narrative removal + EDA)
    logger.info("\n🔧 Applying Task 1 filtering workflow...")
    task1_results = preprocessor.apply_task1_filtering(
        target_products=target_products,
        perform_eda=True
    )
    
    df_filtered = task1_results['filtered_dataframe']
    filtering_stats = task1_results['filtering_stats']
    
    logger.info(f"\n✅ Filtering complete:")
    logger.info(f"   Initial records: {filtering_stats['initial_count']:,}")
    logger.info(f"   After product filter: {filtering_stats['after_product_filter']:,}")
    logger.info(f"   After narrative filter: {filtering_stats['after_narrative_filter']:,}")
    logger.info(f"   Retention rate: {filtering_stats['retention_rate']:.2f}%")
    
    # Save filtered dataset as task1_filtered_complaints.csv
    logger.info("\n💾 Saving filtered dataset...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = preprocessor.save_filtered_dataset(
        output_dir=output_dir,
        filename="task1_filtered_complaints",
        save_csv=True,
        save_parquet=True
    )
    
    logger.info(f"✅ Saved filtered dataset:")
    logger.info(f"   CSV: {saved_files.get('csv')}")
    logger.info(f"   Parquet: {saved_files.get('parquet')}")
    
    return {
        'filtered_dataframe': df_filtered,
        'filtering_stats': filtering_stats,
        'saved_files': saved_files,
        'eda_results': task1_results.get('eda_results')
    }


def run_stratified_sampling(df, sample_size: int, output_dir: Path) -> Dict[str, Any]:
    """
    Step 3: Create stratified sample with proportional representation
    
    Args:
        df: Filtered DataFrame
        sample_size: Number of samples (10k-15k range)
        output_dir: Directory to save stratified sample
        
    Returns:
        Dictionary with sampled DataFrame and statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: STRATIFIED SAMPLING")
    logger.info("=" * 80)
    
    logger.info(f"\n📊 Creating stratified sample of {sample_size:,} records...")
    logger.info("   Stratifying by: Product")
    logger.info("   Ensuring proportional representation across products")
    
    preprocessor = DataPreprocessor(df, narrative_col='Consumer complaint narrative')
    
    # Create and save stratified sample
    sample_results = preprocessor.create_and_save_stratified_sample(
        n_samples=sample_size,
        stratify_col='Product',
        random_state=42,
        output_path=output_dir / "stratified_sample.parquet"
    )
    
    df_sampled = sample_results['sampled_dataframe']
    sampling_stats = sample_results['sampling_stats']
    
    logger.info(f"\n✅ Stratified sampling complete:")
    logger.info(f"   Original size: {len(df):,}")
    logger.info(f"   Sample size: {len(df_sampled):,}")
    
    # Show proportional representation
    logger.info("\n📈 Proportional representation:")
    for product, stats in sampling_stats['proportion_comparison'].items():
        orig_pct = stats['original_pct']
        sample_pct = stats['sample_pct']
        diff = abs(orig_pct - sample_pct)
        logger.info(f"   {product}:")
        logger.info(f"      Original: {orig_pct:.2f}% | Sample: {sample_pct:.2f}% | Diff: {diff:.2f}%")
    
    logger.info(f"\n💾 Saved stratified sample: {sample_results['saved_path']}")
    
    return {
        'sampled_dataframe': df_sampled,
        'sampling_stats': sampling_stats,
        'saved_path': sample_results['saved_path']
    }


def run_chunking(df, output_dir: Path) -> Dict[str, Any]:
    """
    Step 4: Chunk text narratives
    
    Args:
        df: DataFrame with narratives to chunk
        output_dir: Directory to save chunked data
        
    Returns:
        Dictionary with chunked DataFrame and statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: TEXT CHUNKING")
    logger.info("=" * 80)
    
    chunk_size = 500
    chunk_overlap = 75
    
    logger.info(f"\n✂️  Chunking narratives...")
    logger.info(f"   Chunk size: {chunk_size} characters")
    logger.info(f"   Chunk overlap: {chunk_overlap} characters")
    
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_langchain=True
    )
    
    # Chunk the dataframe
    chunked_df = chunker.chunk_dataframe(
        df,
        text_column='Consumer complaint narrative',
        metadata_columns=['Product', 'Issue', 'Company', 'State', 'Date received', 'Complaint ID'],
        progress_bar=True
    )
    
    logger.info(f"\n✅ Chunking complete:")
    logger.info(f"   Original records: {len(df):,}")
    logger.info(f"   Total chunks: {len(chunked_df):,}")
    logger.info(f"   Average chunks per record: {len(chunked_df) / len(df):.2f}")
    
    # Save chunked data
    output_dir.mkdir(parents=True, exist_ok=True)
    chunked_path = output_dir / "chunked_complaints.parquet"
    chunked_df.to_parquet(chunked_path, index=False)
    logger.info(f"\n💾 Saved chunked data: {chunked_path}")
    
    return {
        'chunked_dataframe': chunked_df,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'saved_path': chunked_path
    }


def run_embedding(df, output_dir: Path) -> Dict[str, Any]:
    """
    Step 5: Generate embeddings
    
    Args:
        df: DataFrame with chunked text
        output_dir: Directory to save embeddings
        
    Returns:
        Dictionary with embeddings and statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: EMBEDDING GENERATION")
    logger.info("=" * 80)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = 32
    
    logger.info(f"\n🔢 Generating embeddings...")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Texts to embed: {len(df):,}")
    
    embedder = EmbeddingGenerator(
        model_name=model_name,
        device=None  # Auto-detect
    )
    
    # Generate embeddings
    texts = df['chunk_text'].tolist()
    embeddings = embedder.generate_embeddings(
        texts=texts,
        batch_size=batch_size,
        show_progress=True
    )
    
    logger.info(f"\n✅ Embeddings generated:")
    logger.info(f"   Total embeddings: {len(embeddings):,}")
    logger.info(f"   Embedding dimension: {len(embeddings[0])}")
    
    # Add embeddings to dataframe
    df_with_embeddings = df.copy()
    df_with_embeddings['embedding'] = [emb.tolist() for emb in embeddings]
    
    # Save embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "chunked_complaints_with_embeddings.parquet"
    df_with_embeddings.to_parquet(embedding_path, index=False)
    logger.info(f"\n💾 Saved embeddings: {embedding_path}")
    
    return {
        'embeddings': embeddings,
        'dataframe_with_embeddings': df_with_embeddings,
        'model_name': model_name,
        'embedding_dimension': len(embeddings[0]),
        'saved_path': embedding_path
    }


def run_vector_store(df, embeddings, vector_store_path: Path) -> Dict[str, Any]:
    """
    Step 6: Create vector store
    
    Args:
        df: DataFrame with chunked text and metadata
        embeddings: List of embedding arrays
        vector_store_path: Path to save vector store
        
    Returns:
        Dictionary with vector store information
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: VECTOR STORE CREATION")
    logger.info("=" * 80)
    
    collection_name = "financial_complaints"
    
    logger.info(f"\n💾 Creating vector store...")
    logger.info(f"   Collection: {collection_name}")
    logger.info(f"   Documents: {len(df):,}")
    logger.info(f"   Path: {vector_store_path}")
    
    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=str(vector_store_path),
        reset=True  # Start fresh
    )
    
    # Prepare metadata
    metadata = []
    for idx, row in df.iterrows():
        metadata.append({
            'product': str(row.get('Product', '')),
            'issue': str(row.get('Issue', '')),
            'company': str(row.get('Company', '')),
            'state': str(row.get('State', '')),
            'date_received': str(row.get('Date received', '')),
            'complaint_id': str(row.get('Complaint ID', '')),
            'chunk_id': str(row.get('chunk_id', f'chunk_{idx}')),
            'chunk_length': int(row.get('chunk_length', 0))
        })
    
    # Add documents to vector store
    texts = df['chunk_text'].tolist()
    ids = [f"chunk_{i}" for i in range(len(df))]
    
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )
    
    # Persist vector store
    vector_store.persist()
    
    logger.info(f"\n✅ Vector store created successfully!")
    logger.info(f"   Collection: {collection_name}")
    logger.info(f"   Documents indexed: {len(df):,}")
    logger.info(f"   Location: {vector_store_path}")
    
    return {
        'vector_store': vector_store,
        'collection_name': collection_name,
        'document_count': len(df),
        'path': str(vector_store_path)
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run Task 1 & 2: EDA, Preprocessing, and Vector Store Setup"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to complaints CSV file (default: config.data.raw_data_path)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=12500,
        help='Stratified sample size (default: 12500, range: 10000-15000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data (default: data/processed)'
    )
    parser.add_argument(
        '--vector-store-path',
        type=str,
        default='vector_store',
        help='Path to vector store directory (default: vector_store)'
    )
    parser.add_argument(
        '--skip-vector-store',
        action='store_true',
        help='Skip vector store creation (faster for testing)'
    )
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip EDA analysis (faster for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate sample size
    if not (10000 <= args.sample_size <= 15000):
        logger.warning(
            f"Sample size {args.sample_size} is outside recommended range (10k-15k)"
        )
    
    # Setup paths
    output_dir = Path(args.output_dir)
    vector_store_path = Path(args.vector_store_path)
    
    logger.info("=" * 80)
    logger.info("TASK 1 & 2: EDA, DATA PREPROCESSING, AND VECTOR STORE SETUP")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"   Data path: {args.data_path or config.data.raw_data_path}")
    logger.info(f"   Sample size: {args.sample_size:,}")
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Vector store path: {vector_store_path}")
    logger.info(f"   Skip EDA: {args.skip_eda}")
    logger.info(f"   Skip vector store: {args.skip_vector_store}")
    
    try:
        # Step 0: Load data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 0: LOADING DATA")
        logger.info("=" * 80)
        
        data_path = Path(args.data_path) if args.data_path else config.data.raw_data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"\n📥 Loading data from: {data_path}")
        loader = DataLoader(data_path=data_path)
        df = loader.load_data()
        logger.info(f"✅ Loaded {len(df):,} records")
        
        # Step 1: EDA
        eda_results = None
        if not args.skip_eda:
            eda_results = run_eda(df, output_dir / "eda_results")
        
        # Step 2: Filtering and cleaning
        filtering_results = run_filtering_and_cleaning(df, output_dir)
        df_filtered = filtering_results['filtered_dataframe']
        
        # Step 3: Stratified sampling
        sampling_results = run_stratified_sampling(df_filtered, args.sample_size, output_dir)
        df_sampled = sampling_results['sampled_dataframe']
        
        # Step 4: Chunking
        chunking_results = run_chunking(df_sampled, output_dir)
        df_chunked = chunking_results['chunked_dataframe']
        
        # Step 5: Embedding
        embedding_results = run_embedding(df_chunked, output_dir)
        embeddings = embedding_results['embeddings']
        
        # Step 6: Vector store
        if not args.skip_vector_store:
            vector_store_results = run_vector_store(
                df_chunked,
                embeddings,
                vector_store_path
            )
        else:
            logger.info("\n⏭️  Skipping vector store creation (--skip-vector-store)")
            vector_store_results = None
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ TASK 1 & 2 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nSummary:")
        logger.info(f"   Initial records: {len(df):,}")
        logger.info(f"   After filtering: {len(df_filtered):,}")
        logger.info(f"   Stratified sample: {len(df_sampled):,}")
        logger.info(f"   Text chunks: {len(df_chunked):,}")
        logger.info(f"   Embeddings: {len(embeddings):,}")
        if vector_store_results:
            logger.info(f"   Vector store documents: {vector_store_results['document_count']:,}")
        
        logger.info(f"\n📁 Output files:")
        logger.info(f"   Filtered CSV: {filtering_results['saved_files'].get('csv')}")
        logger.info(f"   Stratified sample: {sampling_results['saved_path']}")
        logger.info(f"   Chunked data: {chunking_results['saved_path']}")
        logger.info(f"   Embeddings: {embedding_results['saved_path']}")
        if vector_store_results:
            logger.info(f"   Vector store: {vector_store_results['path']}")
        
        logger.info("\n✅ All steps completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

