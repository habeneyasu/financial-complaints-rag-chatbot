"""
Configuration management module for the Financial Complaints RAG Chatbot.

This module provides centralized configuration management using environment variables
and default values, following the 12-factor app methodology.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class SamplingConfig:
    """Configuration for stratified sampling."""
    
    # Sample size configuration (target range: 10,000-15,000)
    sample_size: int = int(os.getenv("SAMPLE_SIZE", "12500"))
    sample_size_min: int = 10000
    sample_size_max: int = 15000
    
    # Stratification configuration
    stratify_column: str = os.getenv("STRATIFY_COLUMN", "Product")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    min_samples_per_stratum: int = int(os.getenv("MIN_SAMPLES_PER_STRATUM", "1"))
    
    # Output configuration
    stratified_sample_filename: str = os.getenv(
        "STRATIFIED_SAMPLE_FILENAME",
        "stratified_sample.parquet"
    )
    
    def __post_init__(self) -> None:
        """Validate sampling configuration."""
        if not (self.sample_size_min <= self.sample_size <= self.sample_size_max):
            raise ValueError(
                f"Sample size {self.sample_size} must be between "
                f"{self.sample_size_min} and {self.sample_size_max}"
            )
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        if self.min_samples_per_stratum < 1:
            raise ValueError("min_samples_per_stratum must be at least 1")
    
    @property
    def stratified_sample_path(self) -> Path:
        """Get the full path to stratified sample file."""
        # This will be set by DataConfig
        return Path("data/processed") / self.stratified_sample_filename


@dataclass
class DataConfig:
    """Configuration for data paths and settings."""
    
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    vector_store_dir: Path = Path("vector_store")
    
    # CFPB dataset configuration
    cfpb_data_url: Optional[str] = os.getenv(
        "CFPB_DATA_URL",
        "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
    )
    cfpb_data_filename: str = "complaints.csv"
    processed_filename: str = "processed_complaints.parquet"
    
    # Data processing settings
    chunk_size: int = 10000  # For reading large files in chunks
    sample_size: Optional[int] = None  # For testing with smaller dataset (deprecated, use SamplingConfig)
    
    def __post_init__(self) -> None:
        """Ensure directories exist after initialization."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def raw_data_path(self) -> Path:
        """Get the full path to raw data file."""
        return self.raw_data_dir / self.cfpb_data_filename
    
    @property
    def processed_data_path(self) -> Path:
        """Get the full path to processed data file."""
        return self.processed_data_dir / self.processed_filename
    
    @property
    def stratified_sample_path(self) -> Path:
        """Get the full path to stratified sample file."""
        return self.processed_data_dir / "stratified_sample.parquet"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[Path] = Path("logs/app.log")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    def __post_init__(self) -> None:
        """Ensure log directory exists."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    use_langchain: bool = os.getenv("USE_LANGCHAIN_CHUNKING", "true").lower() == "true"
    
    def __post_init__(self) -> None:
        """Validate chunking configuration."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    model_name: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    normalize_embeddings: bool = os.getenv(
        "NORMALIZE_EMBEDDINGS", "true"
    ).lower() == "true"
    device: Optional[str] = os.getenv("EMBEDDING_DEVICE", None)
    
    def __post_init__(self) -> None:
        """Validate embedding configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    
    collection_name: str = os.getenv("VECTOR_STORE_COLLECTION", "financial_complaints")
    batch_size: int = int(os.getenv("VECTOR_STORE_BATCH_SIZE", "100"))
    
    def __post_init__(self) -> None:
        """Validate vector store configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class AppConfig:
    """Main application configuration."""
    
    data: DataConfig = None
    logging: LoggingConfig = None
    chunking: ChunkingConfig = None
    embedding: EmbeddingConfig = None
    vectorstore: VectorStoreConfig = None
    sampling: SamplingConfig = None
    
    def __post_init__(self) -> None:
        """Initialize sub-configurations if not provided."""
        if self.data is None:
            self.data = DataConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.vectorstore is None:
            self.vectorstore = VectorStoreConfig()
        if self.sampling is None:
            self.sampling = SamplingConfig()


# Global configuration instance
config = AppConfig()

