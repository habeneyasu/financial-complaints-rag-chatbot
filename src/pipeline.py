"""
Complete preprocessing and embedding pipeline for Financial Complaints RAG Chatbot.

This module provides an executable pipeline that integrates:
1. Data loading
2. Task 1 filtering (products + narratives)
3. Stratified sampling (10k-15k with proportional representation)
4. Text chunking
5. Embedding generation
6. Vector store creation

All steps are configurable and reproducible.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.rag.chunker import TextChunker
from src.rag.embedder import EmbeddingGenerator
from src.rag.vectorstore import VectorStore
from src.config import config
from src.utils.logger import get_logger


logger = get_logger(__name__)


def run_complete_pipeline(
    data_path: Optional[Path] = None,
    target_products: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    stratify_column: Optional[str] = None,
    random_state: Optional[int] = None,
    create_stratified_sample: bool = True,
    perform_eda: bool = True,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_model: Optional[str] = None,
    vector_store_path: Optional[Path] = None,
    save_intermediate: bool = True
) -> Dict[str, Any]:
    """
    Run the complete preprocessing and embedding pipeline.
    
    This is executable code that can be run from command line or imported to
    reliably reproduce the vector store with configurable parameters.
    
    Args:
        data_path: Path to raw data file (defaults to config)
        target_products: Products to filter to (defaults to Task 1 products)
        sample_size: Stratified sample size (defaults to config.sampling.sample_size)
        stratify_column: Column to stratify on (defaults to config.sampling.stratify_column)
        random_state: Random seed (defaults to config.sampling.random_state)
        create_stratified_sample: Whether to create stratified sample (default: True)
        perform_eda: Whether to perform EDA (default: True)
        chunk_size: Text chunk size (defaults to config.chunking.chunk_size)
        chunk_overlap: Chunk overlap (defaults to config.chunking.chunk_overlap)
        embedding_model: Embedding model name (defaults to config.embedding.model_name)
        vector_store_path: Path to save vector store (defaults to config.data.vector_store_dir)
        save_intermediate: Whether to save intermediate files (default: True)
        
    Returns:
        Dictionary with all pipeline results and paths
    """
    logger.info("=" * 100)
    logger.info("FINANCIAL COMPLAINTS RAG PIPELINE")
    logger.info("=" * 100)
    logger.info("This pipeline will:")
    logger.info("  1. Load and filter data (Task 1)")
    logger.info("  2. Create stratified sample (10k-15k, proportional representation)")
    logger.info("  3. Chunk text narratives")
    logger.info("  4. Generate embeddings")
    logger.info("  5. Create vector store")
    logger.info("=" * 100)
    
    pipeline_results = {}
    
    # Step 1: Load data
    logger.info("\n📥 Step 1: Loading Data")
    if data_path is None:
        data_path = config.data.raw_data_path
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    loader = DataLoader(data_path=data_path, chunk_size=config.data.chunk_size)
    df = loader.load_data()
    logger.info(f"✅ Loaded {len(df):,} rows from {data_path}")
    pipeline_results['data_loaded'] = len(df)
    
    # Step 2: Preprocessing pipeline (filtering + sampling)
    logger.info("\n🔧 Step 2: Preprocessing Pipeline")
    preprocessor = DataPreprocessor(df, narrative_col='Consumer complaint narrative')
    
    if target_products is None:
        target_products = [
            "Credit card",
            "Personal loan",
            "Savings account",
            "Money transfers"
        ]
    
    preprocessing_results = preprocessor.apply_complete_pipeline(
        target_products=target_products,
        create_stratified_sample=create_stratified_sample,
        n_samples=sample_size,
        perform_eda=perform_eda,
        save_filtered=save_intermediate,
        save_stratified=save_intermediate
    )
    
    pipeline_results['preprocessing'] = preprocessing_results
    df_final = preprocessing_results['final_dataframe']
    logger.info(f"✅ Preprocessing complete: {len(df_final):,} rows ready for embedding")
    
    # Step 3: Text chunking
    logger.info("\n✂️  Step 3: Text Chunking")
    chunk_size = chunk_size or config.chunking.chunk_size
    chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
    
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_langchain=config.chunking.use_langchain
    )
    
    chunked_df = chunker.chunk_dataframe(
        df_final,
        text_column='Consumer complaint narrative',
        metadata_columns=['Product', 'Issue', 'Company', 'State', 'Date received', 'Complaint ID'],
        progress_bar=True
    )
    
    logger.info(f"✅ Chunking complete: {len(chunked_df):,} chunks created")
    pipeline_results['chunking'] = {
        'total_chunks': len(chunked_df),
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    }
    
    # Step 4: Embedding generation
    logger.info("\n🔢 Step 4: Generating Embeddings")
    embedding_model = embedding_model or config.embedding.model_name
    
    embedder = EmbeddingGenerator(
        model_name=embedding_model,
        batch_size=config.embedding.batch_size,
        normalize_embeddings=config.embedding.normalize_embeddings,
        device=config.embedding.device
    )
    
    embeddings = embedder.generate_embeddings(
        texts=chunked_df['chunk_text'].tolist(),
        show_progress=True
    )
    
    logger.info(f"✅ Embeddings generated: {len(embeddings):,} vectors, "
                f"dimension: {len(embeddings[0])}")
    pipeline_results['embeddings'] = {
        'count': len(embeddings),
        'dimension': len(embeddings[0]),
        'model': embedding_model
    }
    
    # Step 5: Vector store creation
    logger.info("\n💾 Step 5: Creating Vector Store")
    if vector_store_path is None:
        vector_store_path = config.data.vector_store_dir
    
    vector_store = VectorStore(
        persist_directory=str(vector_store_path),
        collection_name=config.vectorstore.collection_name
    )
    
    # Prepare metadata
    metadata = []
    for _, row in chunked_df.iterrows():
        metadata.append({
            'chunk_id': row.get('chunk_id', ''),
            'product': row.get('Product', ''),
            'issue': row.get('Issue', ''),
            'company': row.get('Company', ''),
            'state': row.get('State', ''),
            'date_received': str(row.get('Date received', '')),
            'complaint_id': str(row.get('Complaint ID', '')),
            'chunk_length': row.get('chunk_length', 0)
        })
    
    vector_store.add_documents(
        texts=chunked_df['chunk_text'].tolist(),
        embeddings=embeddings,
        metadatas=metadata,
        ids=[f"chunk_{i}" for i in range(len(chunked_df))]
    )
    
    vector_store.persist()
    
    logger.info(f"✅ Vector store created: {vector_store_path}")
    logger.info(f"   Collection: {config.vectorstore.collection_name}")
    logger.info(f"   Documents: {len(chunked_df):,}")
    
    pipeline_results['vector_store'] = {
        'path': str(vector_store_path),
        'collection': config.vectorstore.collection_name,
        'documents': len(chunked_df)
    }
    
    # Final summary
    logger.info("\n" + "=" * 100)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 100)
    logger.info(f"Initial data: {pipeline_results['data_loaded']:,} rows")
    logger.info(f"After filtering: {preprocessing_results['pipeline_summary']['after_filtering']:,} rows")
    logger.info(f"Final sample: {preprocessing_results['pipeline_summary']['final_size']:,} rows")
    logger.info(f"Chunks created: {pipeline_results['chunking']['total_chunks']:,}")
    logger.info(f"Embeddings: {pipeline_results['embeddings']['count']:,} vectors")
    logger.info(f"Vector store: {vector_store_path}")
    logger.info("=" * 100)
    
    return pipeline_results


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete preprocessing and embedding pipeline"
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        default=None,
        help='Path to raw data file (default: config.data.raw_data_path)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help=f'Sample size for stratified sampling (default: {config.sampling.sample_size})'
    )
    parser.add_argument(
        '--stratify-column',
        type=str,
        default=None,
        help=f'Column to stratify on (default: {config.sampling.stratify_column})'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=None,
        help=f'Random seed (default: {config.sampling.random_state})'
    )
    parser.add_argument(
        '--no-sampling',
        action='store_true',
        help='Skip stratified sampling'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help=f'Text chunk size (default: {config.chunking.chunk_size})'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=None,
        help=f'Chunk overlap (default: {config.chunking.chunk_overlap})'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=None,
        help=f'Embedding model (default: {config.embedding.model_name})'
    )
    parser.add_argument(
        '--vector-store-path',
        type=Path,
        default=None,
        help='Path to save vector store (default: config.data.vector_store_dir)'
    )
    parser.add_argument(
        '--no-intermediate',
        action='store_true',
        help='Skip saving intermediate files'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_complete_pipeline(
            data_path=args.data_path,
            sample_size=args.sample_size,
            stratify_column=args.stratify_column,
            random_state=args.random_state,
            create_stratified_sample=not args.no_sampling,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            vector_store_path=args.vector_store_path,
            save_intermediate=not args.no_intermediate
        )
        
        logger.info("\n✅ Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

