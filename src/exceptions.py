"""
Custom exception classes for the Financial Complaints RAG Chatbot.

This module defines custom exceptions to provide more specific error handling
and better error messages for different failure scenarios.
"""


class FinancialComplaintsError(Exception):
    """Base exception class for all custom exceptions in this project."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataLoadError(FinancialComplaintsError):
    """Raised when data loading fails."""
    pass


class DataValidationError(FinancialComplaintsError):
    """Raised when data validation fails."""
    pass


class DataProcessingError(FinancialComplaintsError):
    """Raised when data processing fails."""
    pass


class FileNotFoundError(FinancialComplaintsError):
    """Raised when a required file is not found."""
    pass


class ConfigurationError(FinancialComplaintsError):
    """Raised when there's a configuration error."""
    pass

