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
    sample_size: Optional[int] = None  # For testing with smaller dataset
    
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
class AppConfig:
    """Main application configuration."""
    
    data: DataConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self) -> None:
        """Initialize sub-configurations if not provided."""
        if self.data is None:
            self.data = DataConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


# Global configuration instance
config = AppConfig()

