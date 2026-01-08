"""Utility modules for the Financial Complaints RAG Chatbot."""

from .logger import setup_logger, get_logger

# Import data loading utilities (optional - only if available)
try:
    from .data_loader_utils import (
        load_complaints_data,
        quick_load,
        clear_data_cache,
        get_data_info
    )
    __all__ = [
        "setup_logger",
        "get_logger",
        "load_complaints_data",
        "quick_load",
        "clear_data_cache",
        "get_data_info"
    ]
except ImportError:
    # If data_loader_utils is not available, just export logger functions
    __all__ = ["setup_logger", "get_logger"]
