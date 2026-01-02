"""
Data preprocessing module for CFPB complaint dataset.

This module provides a comprehensive data preprocessing class that handles
cleaning, transformation, and preparation of complaint data for the RAG pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

from src.config import config
from src.exceptions import DataProcessingError, DataValidationError
from src.utils.logger import get_logger
from src.eda.analyzer import EDAAnalyzer


class DataPreprocessor:
    """
    A class for preprocessing CFPB complaint data.
    
    This class handles data cleaning, transformation, and preparation
    following industry best practices for data preprocessing.
    
    Attributes:
        df: DataFrame to preprocess
        logger: Logger instance
        narrative_col: Name of the narrative column
    """
    
    def __init__(self, df: pd.DataFrame, narrative_col: str = "Consumer complaint narrative"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            df: DataFrame to preprocess
            narrative_col: Name of the narrative column
        """
        if df is None or df.empty:
            raise DataValidationError("DataFrame is None or empty")
        
        self.df = df.copy()
        self.logger = get_logger(__name__)
        self.narrative_col = narrative_col
        self._original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataPreprocessor':
        """
        Remove duplicate rows from the dataset.
        
        Args:
            subset: Optional list of columns to consider for duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Self for method chaining
        """
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_count - len(self.df)
        
        if removed > 0:
            self.logger.info(f"Removed {removed:,} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")
        
        return self
    
    def clean_narratives(
        self,
        remove_whitespace: bool = True,
        remove_extra_spaces: bool = True,
        min_length: int = 10
    ) -> 'DataPreprocessor':
        """
        Clean narrative text data.
        
        Args:
            remove_whitespace: Remove leading/trailing whitespace
            remove_extra_spaces: Remove multiple consecutive spaces
            min_length: Minimum character length for valid narratives
            
        Returns:
            Self for method chaining
        """
        if self.narrative_col not in self.df.columns:
            self.logger.warning(f"Column '{self.narrative_col}' not found. Skipping narrative cleaning.")
            return self
        
        initial_non_null = self.df[self.narrative_col].notna().sum()
        
        # Convert to string and handle NaN
        self.df[self.narrative_col] = self.df[self.narrative_col].astype(str)
        
        # Replace 'nan' strings with actual NaN
        self.df[self.narrative_col] = self.df[self.narrative_col].replace(['nan', 'None', ''], np.nan)
        
        if remove_whitespace:
            self.df[self.narrative_col] = self.df[self.narrative_col].str.strip()
        
        if remove_extra_spaces:
            self.df[self.narrative_col] = self.df[self.narrative_col].str.replace(r'\s+', ' ', regex=True)
        
        # Remove narratives that are too short
        if min_length > 0:
            mask = (self.df[self.narrative_col].str.len() < min_length) & (self.df[self.narrative_col].notna())
            self.df.loc[mask, self.narrative_col] = np.nan
            removed = mask.sum()
            if removed > 0:
                self.logger.info(f"Marked {removed:,} narratives as null (length < {min_length} chars)")
        
        final_non_null = self.df[self.narrative_col].notna().sum()
        self.logger.info(f"Narrative cleaning complete. Non-null narratives: {initial_non_null:,} -> {final_non_null:,}")
        
        return self
    
    def clean_text_narratives(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        remove_boilerplate: bool = True,
        normalize_whitespace: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_phone_numbers: bool = False,
        preserve_punctuation: bool = False
    ) -> 'DataPreprocessor':
        """
        Clean text narratives for improved embedding quality.
        
        This method performs comprehensive text cleaning including:
        - Lowercasing
        - Removing special characters
        - Removing boilerplate text
        - Text normalization
        
        Args:
            lowercase: Convert text to lowercase
            remove_special_chars: Remove special characters (keeps alphanumeric and basic punctuation)
            remove_boilerplate: Remove common boilerplate phrases
            normalize_whitespace: Normalize whitespace (multiple spaces to single space)
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses from text
            remove_phone_numbers: Remove phone numbers from text
            preserve_punctuation: If True, keeps basic punctuation; if False, removes all punctuation
            
        Returns:
            Self for method chaining
        """
        if self.narrative_col not in self.df.columns:
            self.logger.warning(f"Column '{self.narrative_col}' not found. Skipping text cleaning.")
            return self
        
        initial_count = self.df[self.narrative_col].notna().sum()
        self.logger.info(f"Starting text cleaning for {initial_count:,} narratives")
        
        # Work on a copy to avoid SettingWithCopyWarning
        narratives = self.df[self.narrative_col].copy()
        
        # Convert to string (handles NaN)
        narratives = narratives.astype(str)
        
        # Replace 'nan' strings with actual NaN
        narratives = narratives.replace(['nan', 'None', 'NaN'], np.nan)
        
        # Only process non-null values
        mask = narratives.notna()
        
        if lowercase:
            narratives.loc[mask] = narratives.loc[mask].str.lower()
            self.logger.info("Applied lowercasing")
        
        if remove_urls:
            # Remove URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            narratives.loc[mask] = narratives.loc[mask].str.replace(url_pattern, '', regex=True)
            self.logger.info("Removed URLs")
        
        if remove_emails:
            # Remove email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            narratives.loc[mask] = narratives.loc[mask].str.replace(email_pattern, '', regex=True)
            self.logger.info("Removed email addresses")
        
        if remove_phone_numbers:
            # Remove phone numbers (various formats)
            phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            narratives.loc[mask] = narratives.loc[mask].str.replace(phone_pattern, '', regex=True)
            self.logger.info("Removed phone numbers")
        
        if remove_boilerplate:
            # Common boilerplate phrases to remove
            boilerplate_patterns = [
                r'i am writing to file a complaint',
                r'i am writing to complain',
                r'this is a complaint',
                r'i would like to file a complaint',
                r'i want to file a complaint',
                r'please note',
                r'please be advised',
                r'kindly note',
                r'for your information',
                r'fyi',
            ]
            
            for pattern in boilerplate_patterns:
                narratives.loc[mask] = narratives.loc[mask].str.replace(
                    pattern, '', case=False, regex=True
                )
            
            self.logger.info("Removed boilerplate phrases")
        
        if remove_special_chars:
            if preserve_punctuation:
                # Keep alphanumeric, spaces, and basic punctuation
                narratives.loc[mask] = narratives.loc[mask].str.replace(
                    r'[^a-z0-9\s.,!?;:\'\"-]', '', regex=True
                )
            else:
                # Keep only alphanumeric and spaces
                narratives.loc[mask] = narratives.loc[mask].str.replace(
                    r'[^a-z0-9\s]', '', regex=True
                )
            self.logger.info(f"Removed special characters (preserve_punctuation={preserve_punctuation})")
        
        if normalize_whitespace:
            # Replace multiple spaces/newlines/tabs with single space
            narratives.loc[mask] = narratives.loc[mask].str.replace(r'\s+', ' ', regex=True)
            # Strip leading/trailing whitespace
            narratives.loc[mask] = narratives.loc[mask].str.strip()
            self.logger.info("Normalized whitespace")
        
        # Update the dataframe
        self.df[self.narrative_col] = narratives
        
        # Remove narratives that became empty after cleaning
        empty_after_cleaning = (self.df[self.narrative_col].isna() | 
                               (self.df[self.narrative_col].str.strip() == '')).sum()
        
        if empty_after_cleaning > 0:
            self.logger.warning(f"{empty_after_cleaning:,} narratives became empty after cleaning")
            # Optionally set to NaN
            self.df.loc[self.df[self.narrative_col].str.strip() == '', self.narrative_col] = np.nan
        
        final_count = self.df[self.narrative_col].notna().sum()
        self.logger.info(f"Text cleaning complete. Non-null narratives: {initial_count:,} -> {final_count:,}")
        
        return self
    
    def handle_missing_values(
        self,
        strategy: str = 'drop',
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None
    ) -> 'DataPreprocessor':
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: Strategy to use ('drop', 'fill', 'keep')
            columns: Optional list of columns to process (default: all)
            fill_value: Value to fill with if strategy is 'fill'
            
        Returns:
            Self for method chaining
        """
        columns = columns or self.df.columns.tolist()
        
        if strategy == 'drop':
            initial_count = len(self.df)
            # Only drop rows where all specified columns are null
            self.df = self.df.dropna(subset=columns, how='all')
            removed = initial_count - len(self.df)
            if removed > 0:
                self.logger.info(f"Dropped {removed:,} rows with all missing values in specified columns")
        
        elif strategy == 'fill':
            if fill_value is None:
                raise DataProcessingError("fill_value must be provided when strategy is 'fill'")
            
            for col in columns:
                if col in self.df.columns:
                    null_count = self.df[col].isnull().sum()
                    if null_count > 0:
                        self.df[col] = self.df[col].fillna(fill_value)
                        self.logger.info(f"Filled {null_count:,} missing values in '{col}' with {fill_value}")
        
        elif strategy == 'keep':
            self.logger.info("Keeping missing values as-is")
        
        else:
            raise DataProcessingError(f"Unknown strategy: {strategy}")
        
        return self
    
    def standardize_dates(self, date_columns: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Standardize date columns to datetime format.
        
        Args:
            date_columns: Optional list of date column names
            
        Returns:
            Self for method chaining
        """
        if date_columns is None:
            # Auto-detect date columns
            date_columns = [col for col in self.df.columns if 'date' in col.lower()]
        
        for col in date_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    converted = self.df[col].notna().sum()
                    self.logger.info(f"Converted '{col}' to datetime. {converted:,} valid dates")
                except Exception as e:
                    self.logger.warning(f"Could not convert '{col}' to datetime: {e}")
        
        return self
    
    def add_derived_features(self) -> 'DataPreprocessor':
        """
        Add derived features useful for analysis and RAG pipeline.
        
        Returns:
            Self for method chaining
        """
        # Add word count if narrative column exists
        if self.narrative_col in self.df.columns:
            def word_count(text):
                if pd.isna(text) or text == '':
                    return 0
                return len(str(text).split())
            
            self.df['narrative_word_count'] = self.df[self.narrative_col].apply(word_count)
            self.logger.info("Added 'narrative_word_count' feature")
        
        # Add character count
        if self.narrative_col in self.df.columns:
            def char_count(text):
                if pd.isna(text) or text == '':
                    return 0
                return len(str(text))
            
            self.df['narrative_char_count'] = self.df[self.narrative_col].apply(char_count)
            self.logger.info("Added 'narrative_char_count' feature")
        
        # Add has_narrative flag
        if self.narrative_col in self.df.columns:
            self.df['has_narrative'] = self.df[self.narrative_col].notna() & (self.df[self.narrative_col] != '')
            self.logger.info("Added 'has_narrative' feature")
        
        return self
    
    def filter_by_products(
        self,
        products: List[str],
        product_col: str = 'Product'
    ) -> 'DataPreprocessor':
        """
        Filter dataset to include only specified products.
        
        This method explicitly filters the dataset to the assignment's specified product set.
        It validates that the products exist in the dataset and logs the filtering results.
        
        Args:
            products: List of product names to include
            product_col: Name of the product column (default: 'Product')
            
        Returns:
            Self for method chaining
            
        Raises:
            DataValidationError: If product column doesn't exist
        """
        if product_col not in self.df.columns:
            raise DataValidationError(f"Product column '{product_col}' not found in dataset")
        
        initial_count = len(self.df)
        
        # Get available products in dataset
        available_products = self.df[product_col].unique()
        
        # Validate and filter to only products that exist
        valid_products = [p for p in products if p in available_products]
        missing_products = [p for p in products if p not in available_products]
        
        if missing_products:
            self.logger.warning(
                f"The following products were not found in dataset: {missing_products}"
            )
        
        if not valid_products:
            raise DataValidationError(
                f"None of the specified products were found in the dataset. "
                f"Available products: {sorted(available_products)[:10]}..."
            )
        
        # Filter to valid products
        self.df = self.df[self.df[product_col].isin(valid_products)].reset_index(drop=True)
        removed = initial_count - len(self.df)
        
        self.logger.info(
            f"Filtered to {len(valid_products)} products: {valid_products}. "
            f"Kept {len(self.df):,} rows (removed {removed:,} rows, "
            f"{removed/initial_count*100:.2f}%)"
        )
        
        # Log product distribution after filtering
        if len(self.df) > 0:
            product_counts = self.df[product_col].value_counts()
            self.logger.info("Product distribution after filtering:")
            for product, count in product_counts.items():
                self.logger.info(f"  {product}: {count:,} ({count/len(self.df)*100:.2f}%)")
        
        return self
    
    def remove_empty_narratives(
        self,
        narrative_col: Optional[str] = None
    ) -> 'DataPreprocessor':
        """
        Remove records with empty or missing Consumer complaint narratives.
        
        This method explicitly removes records where the narrative column is:
        - NaN/null
        - Empty string
        - Whitespace-only string
        
        This ensures the dataset only contains records with valid narrative text,
        as required for the RAG pipeline.
        
        Args:
            narrative_col: Name of the narrative column. If None, uses self.narrative_col
            
        Returns:
            Self for method chaining
            
        Raises:
            DataValidationError: If narrative column doesn't exist
        """
        col = narrative_col or self.narrative_col
        
        if col not in self.df.columns:
            raise DataValidationError(f"Narrative column '{col}' not found in dataset")
        
        initial_count = len(self.df)
        
        # Identify empty narratives (NaN, empty string, or whitespace-only)
        has_narrative = (
            self.df[col].notna() & 
            (self.df[col].astype(str).str.strip() != '') &
            (self.df[col].astype(str) != 'nan') &
            (self.df[col].astype(str) != 'None')
        )
        
        empty_count = (~has_narrative).sum()
        
        # Filter to keep only records with narratives
        self.df = self.df[has_narrative].reset_index(drop=True)
        removed = initial_count - len(self.df)
        
        self.logger.info(
            f"Removed {removed:,} records with empty narratives "
            f"({removed/initial_count*100:.2f}%). "
            f"Kept {len(self.df):,} records with valid narratives."
        )
        
        if removed != empty_count:
            self.logger.warning(
                f"Expected to remove {empty_count} empty narratives, "
                f"but removed {removed} records. This may indicate data inconsistencies."
            )
        
        return self
    
    def filter_by_conditions(
        self,
        conditions: Dict[str, Any],
        keep: bool = True
    ) -> 'DataPreprocessor':
        """
        Filter dataset based on conditions.
        
        Args:
            conditions: Dictionary of column: value pairs for filtering
            keep: If True, keep rows matching conditions; if False, drop them
            
        Returns:
            Self for method chaining
        """
        initial_count = len(self.df)
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        for col, value in conditions.items():
            if col in self.df.columns:
                if isinstance(value, (list, tuple)):
                    col_mask = self.df[col].isin(value)
                else:
                    col_mask = self.df[col] == value
                
                if keep:
                    mask = mask & col_mask
                else:
                    mask = mask & ~col_mask
        
        self.df = self.df[mask].reset_index(drop=True)
        removed = initial_count - len(self.df)
        
        if removed > 0:
            action = "Kept" if keep else "Removed"
            self.logger.info(f"{action} {removed:,} rows based on conditions")
        
        return self
    
    def select_columns(self, columns: List[str]) -> 'DataPreprocessor':
        """
        Select specific columns from the dataset.
        
        Args:
            columns: List of column names to keep
            
        Returns:
            Self for method chaining
        """
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            self.logger.warning(f"Columns not found: {missing_cols}")
        
        available_cols = [col for col in columns if col in self.df.columns]
        self.df = self.df[available_cols]
        
        self.logger.info(f"Selected {len(available_cols)} columns")
        return self
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of preprocessing operations.
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            "original_shape": self._original_shape,
            "current_shape": self.df.shape,
            "rows_removed": self._original_shape[0] - self.df.shape[0],
            "columns_removed": self._original_shape[1] - self.df.shape[1],
            "current_columns": list(self.df.columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def perform_eda_analysis(
        self,
        product_col: str = 'Product',
        narrative_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform EDA analysis integrated with preprocessing pipeline.
        
        This method performs the EDA workflow (distribution, length, missingness)
        that is tightly coupled with the preprocessing pipeline. It ensures that
        EDA is performed on the current state of the preprocessed data.
        
        Args:
            product_col: Name of the product column (default: 'Product')
            narrative_col: Name of the narrative column (uses self.narrative_col if None)
            
        Returns:
            Dictionary containing EDA results:
            - product_distribution: Product distribution analysis
            - narrative_length: Narrative length statistics
            - narrative_presence: Narrative presence analysis
            - summary_statistics: Overall dataset summary
        """
        narrative_col = narrative_col or self.narrative_col
        
        self.logger.info("Performing EDA analysis on preprocessed dataset...")
        
        # Initialize EDA analyzer with current dataframe
        analyzer = EDAAnalyzer(self.df)
        
        eda_results = {}
        
        # 1. Product distribution analysis
        try:
            if product_col in self.df.columns:
                eda_results['product_distribution'] = analyzer.analyze_product_distribution(
                    product_col=product_col
                )
                self.logger.info(
                    f"Product distribution: {eda_results['product_distribution']['total_products']} "
                    f"unique products"
                )
            else:
                self.logger.warning(f"Product column '{product_col}' not found. Skipping product distribution.")
                eda_results['product_distribution'] = None
        except Exception as e:
            self.logger.error(f"Error analyzing product distribution: {e}")
            eda_results['product_distribution'] = None
        
        # 2. Narrative length analysis
        try:
            if narrative_col in self.df.columns:
                eda_results['narrative_length'] = analyzer.analyze_narrative_length(
                    narrative_col=narrative_col
                )
                if 'word_count_stats' in eda_results['narrative_length']:
                    mean_words = eda_results['narrative_length']['word_count_stats']['mean']
                    self.logger.info(f"Narrative length: mean {mean_words:.1f} words")
            else:
                self.logger.warning(f"Narrative column '{narrative_col}' not found. Skipping narrative length.")
                eda_results['narrative_length'] = None
        except Exception as e:
            self.logger.error(f"Error analyzing narrative length: {e}")
            eda_results['narrative_length'] = None
        
        # 3. Narrative presence analysis
        try:
            if narrative_col in self.df.columns:
                eda_results['narrative_presence'] = analyzer.analyze_narrative_presence(
                    narrative_col=narrative_col
                )
                with_narrative = eda_results['narrative_presence']['with_narrative']
                self.logger.info(
                    f"Narrative presence: {with_narrative['count']:,} "
                    f"({with_narrative['percentage']:.1f}%) with narratives"
                )
            else:
                self.logger.warning(f"Narrative column '{narrative_col}' not found. Skipping narrative presence.")
                eda_results['narrative_presence'] = None
        except Exception as e:
            self.logger.error(f"Error analyzing narrative presence: {e}")
            eda_results['narrative_presence'] = None
        
        # 4. Summary statistics
        try:
            eda_results['summary_statistics'] = analyzer.get_summary_statistics()
            self.logger.info(
                f"Summary: {eda_results['summary_statistics']['shape'][0]:,} rows, "
                f"{eda_results['summary_statistics']['shape'][1]} columns"
            )
        except Exception as e:
            self.logger.error(f"Error getting summary statistics: {e}")
            eda_results['summary_statistics'] = None
        
        self.logger.info("EDA analysis complete")
        return eda_results
    
    def apply_task1_filtering(
        self,
        target_products: List[str],
        product_col: str = 'Product',
        narrative_col: Optional[str] = None,
        perform_eda: bool = True
    ) -> Dict[str, Any]:
        """
        Apply complete Task 1 filtering workflow.
        
        This method performs the complete Task 1 filtering workflow:
        1. Filters to specified product set
        2. Removes records without narratives
        3. Optionally performs EDA analysis
        
        This ensures reproducibility of the exact Task 1 dataset.
        
        Args:
            target_products: List of product names to include
            product_col: Name of the product column (default: 'Product')
            narrative_col: Name of the narrative column (uses self.narrative_col if None)
            perform_eda: Whether to perform EDA analysis after filtering (default: True)
            
        Returns:
            Dictionary containing:
            - filtered_dataframe: The filtered DataFrame
            - preprocessing_summary: Summary of preprocessing operations
            - eda_results: EDA analysis results (if perform_eda=True)
            - filtering_stats: Statistics about filtering operations
        """
        narrative_col = narrative_col or self.narrative_col
        
        initial_count = len(self.df)
        self.logger.info(
            f"Starting Task 1 filtering workflow on {initial_count:,} records. "
            f"Target products: {target_products}"
        )
        
        # Step 1: Filter by products
        self.filter_by_products(products=target_products, product_col=product_col)
        after_product_filter = len(self.df)
        
        # Step 2: Remove empty narratives
        self.remove_empty_narratives(narrative_col=narrative_col)
        after_narrative_filter = len(self.df)
        
        # Collect filtering statistics
        filtering_stats = {
            "initial_count": initial_count,
            "after_product_filter": after_product_filter,
            "after_narrative_filter": after_narrative_filter,
            "removed_by_product": initial_count - after_product_filter,
            "removed_by_narrative": after_product_filter - after_narrative_filter,
            "total_removed": initial_count - after_narrative_filter,
            "retention_rate": (after_narrative_filter / initial_count * 100) if initial_count > 0 else 0
        }
        
        # Step 3: Perform EDA if requested
        eda_results = None
        if perform_eda:
            eda_results = self.perform_eda_analysis(
                product_col=product_col,
                narrative_col=narrative_col
            )
        
        # Get preprocessing summary
        preprocessing_summary = self.get_preprocessing_summary()
        
        self.logger.info(
            f"Task 1 filtering complete: {after_narrative_filter:,} records "
            f"({filtering_stats['retention_rate']:.2f}% retention)"
        )
        
        return {
            "filtered_dataframe": self.df.copy(),
            "preprocessing_summary": preprocessing_summary,
            "eda_results": eda_results,
            "filtering_stats": filtering_stats
        }
    
    def create_and_save_stratified_sample(
        self,
        n_samples: Optional[int] = None,
        stratify_col: Optional[str] = None,
        random_state: Optional[int] = None,
        output_path: Optional[Path] = None,
        min_samples_per_stratum: int = 1
    ) -> Dict[str, Any]:
        """
        Create and save stratified sample with proportional representation across products.
        
        This method creates a stratified sample ensuring proportional representation
        across all product categories and saves it with clear naming for reproducibility.
        This is executable code (not just a notebook reference) that can be integrated
        into the preprocessing/embedding pipeline.
        
        Args:
            n_samples: Total number of samples (defaults to config.sampling.sample_size)
            stratify_col: Column to stratify on (defaults to config.sampling.stratify_column)
            random_state: Random seed (defaults to config.sampling.random_state)
            output_path: Path to save sample (defaults to config.data.stratified_sample_path)
            min_samples_per_stratum: Minimum samples per category (default: 1)
            
        Returns:
            Dictionary containing:
            - sampled_dataframe: The stratified sample DataFrame
            - sampling_stats: Statistics about the sampling process
            - saved_path: Path where sample was saved
        """
        from src.config import config
        
        # Use configuration defaults if not provided
        n_samples = n_samples or config.sampling.sample_size
        stratify_col = stratify_col or config.sampling.stratify_column
        random_state = random_state if random_state is not None else config.sampling.random_state
        
        if output_path is None:
            output_path = config.data.stratified_sample_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"Creating stratified sample: {n_samples:,} samples, "
            f"stratify by '{stratify_col}', random_state={random_state}"
        )
        
        # Create stratified sample
        df_sampled = self.stratified_sample(
            n_samples=n_samples,
            stratify_col=stratify_col,
            random_state=random_state,
            min_samples_per_stratum=min_samples_per_stratum
        )
        
        # Calculate sampling statistics
        original_counts = self.df[stratify_col].value_counts()
        sample_counts = df_sampled[stratify_col].value_counts()
        original_proportions = (original_counts / len(self.df) * 100).round(2)
        sample_proportions = (sample_counts / len(df_sampled) * 100).round(2)
        
        # Calculate differences
        proportion_diffs = {}
        for product in original_proportions.index:
            orig_pct = original_proportions[product]
            sample_pct = sample_proportions.get(product, 0)
            proportion_diffs[product] = {
                'original_pct': float(orig_pct),
                'sample_pct': float(sample_pct),
                'difference_pct': float(sample_pct - orig_pct),
                'original_count': int(original_counts[product]),
                'sample_count': int(sample_counts.get(product, 0))
            }
        
        sampling_stats = {
            "original_size": len(self.df),
            "sample_size": len(df_sampled),
            "target_size": n_samples,
            "sampling_ratio": len(df_sampled) / len(self.df) * 100,
            "within_target_range": 10000 <= len(df_sampled) <= 15000,
            "stratify_column": stratify_col,
            "random_state": random_state,
            "product_proportions": proportion_diffs,
            "max_proportion_diff": max([abs(d['difference_pct']) for d in proportion_diffs.values()]),
            "mean_proportion_diff": sum([abs(d['difference_pct']) for d in proportion_diffs.values()]) / len(proportion_diffs)
        }
        
        # Save stratified sample
        self.logger.info(f"Saving stratified sample to: {output_path}")
        df_sampled.to_parquet(output_path, index=False, engine='pyarrow')
        file_size_mb = output_path.stat().st_size / (1024**2)
        
        self.logger.info(
            f"✅ Stratified sample saved: {len(df_sampled):,} rows, "
            f"{file_size_mb:.2f} MB, max proportion diff: {sampling_stats['max_proportion_diff']:.2f}%"
        )
        
        return {
            "sampled_dataframe": df_sampled,
            "sampling_stats": sampling_stats,
            "saved_path": output_path
        }
    
    def apply_complete_pipeline(
        self,
        target_products: List[str],
        create_stratified_sample: bool = True,
        n_samples: Optional[int] = None,
        perform_eda: bool = True,
        save_filtered: bool = True,
        save_stratified: bool = True
    ) -> Dict[str, Any]:
        """
        Apply complete preprocessing pipeline: filtering -> sampling -> ready for embedding.
        
        This method integrates the complete workflow:
        1. Task 1 filtering (products + narratives)
        2. Optional stratified sampling (10k-15k with proportional representation)
        3. Saves outputs with clear naming for reproducibility
        
        This is executable code that can be run from scripts or notebooks to reliably
        reproduce the vector store.
        
        Args:
            target_products: List of products to filter to
            create_stratified_sample: Whether to create stratified sample (default: True)
            n_samples: Sample size (defaults to config)
            perform_eda: Whether to perform EDA analysis (default: True)
            save_filtered: Whether to save filtered dataset (default: True)
            save_stratified: Whether to save stratified sample (default: True)
            
        Returns:
            Dictionary with all pipeline results and statistics
        """
        from src.config import config
        
        self.logger.info("=" * 100)
        self.logger.info("STARTING COMPLETE PREPROCESSING PIPELINE")
        self.logger.info("=" * 100)
        
        pipeline_results = {}
        
        # Step 1: Task 1 filtering
        self.logger.info("\n📋 Step 1: Task 1 Filtering (Products + Narratives)")
        task1_results = self.apply_task1_filtering(
            target_products=target_products,
            perform_eda=perform_eda
        )
        df_filtered = task1_results['filtered_dataframe']
        pipeline_results['task1'] = task1_results
        
        # Update preprocessor to use filtered data
        self.df = df_filtered
        
        # Save filtered dataset if requested
        if save_filtered:
            self.logger.info("\n💾 Saving filtered dataset...")
            saved_files = self.save_filtered_dataset(
                filename="task1_filtered_complaints",
                save_csv=True,
                save_parquet=True
            )
            pipeline_results['filtered_files'] = saved_files
        
        # Step 2: Stratified sampling (if requested)
        if create_stratified_sample:
            self.logger.info("\n📊 Step 2: Creating Stratified Sample")
            sample_results = self.create_and_save_stratified_sample(
                n_samples=n_samples,
                output_path=config.data.stratified_sample_path if save_stratified else None
            )
            pipeline_results['stratified_sample'] = sample_results
            df_final = sample_results['sampled_dataframe']
        else:
            self.logger.info("\n⏭️  Skipping stratified sampling")
            df_final = df_filtered
            pipeline_results['stratified_sample'] = None
        
        pipeline_results['final_dataframe'] = df_final
        pipeline_results['pipeline_summary'] = {
            "initial_size": task1_results['filtering_stats']['initial_count'],
            "after_filtering": len(df_filtered),
            "final_size": len(df_final),
            "sampling_applied": create_stratified_sample,
            "ready_for_embedding": True
        }
        
        self.logger.info("\n" + "=" * 100)
        self.logger.info("✅ COMPLETE PIPELINE FINISHED")
        self.logger.info("=" * 100)
        self.logger.info(f"Final dataset size: {len(df_final):,} rows")
        self.logger.info(f"Ready for: chunking → embedding → vector store creation")
        
        return pipeline_results
    
    def save_processed_data(self, file_path: Optional[Path] = None, format: str = 'parquet') -> Path:
        """
        Save processed data to file.
        
        Args:
            file_path: Optional path to save file (defaults to config)
            format: File format ('parquet', 'csv', 'pickle')
            
        Returns:
            Path to saved file
        """
        if file_path is None:
            file_path = config.data.processed_data_path
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'parquet':
                self.df.to_parquet(file_path, index=False, engine='pyarrow')
            elif format == 'csv':
                self.df.to_csv(file_path, index=False)
            elif format == 'pickle':
                self.df.to_pickle(file_path)
            else:
                raise DataProcessingError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved processed data to {file_path} ({format})")
            return file_path
            
        except Exception as e:
            error_msg = f"Error saving processed data: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg, details={"path": str(file_path), "format": format})
    
    def save_filtered_dataset(
        self,
        output_dir: Optional[Path] = None,
        filename: str = "task1_filtered_complaints",
        save_csv: bool = True,
        save_parquet: bool = True
    ) -> Dict[str, Path]:
        """
        Save filtered dataset with clear naming convention for Task 1.
        
        This method saves the filtered dataset with a clearly named CSV file
        as requested for Task 1 deliverables. The filename clearly indicates
        this is the Task 1 filtered dataset.
        
        Args:
            output_dir: Directory to save files (defaults to config processed_data_dir)
            filename: Base filename (without extension)
            save_csv: Whether to save CSV format (default: True)
            save_parquet: Whether to save Parquet format (default: True)
            
        Returns:
            Dictionary with format as key and file path as value
            
        Raises:
            DataProcessingError: If saving fails
        """
        if output_dir is None:
            output_dir = config.data.processed_data_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            if save_csv:
                csv_path = output_dir / f"{filename}.csv"
                self.df.to_csv(csv_path, index=False)
                csv_size_mb = csv_path.stat().st_size / (1024**2)
                self.logger.info(
                    f"Saved filtered CSV: {csv_path} "
                    f"({csv_size_mb:.2f} MB, {len(self.df):,} rows)"
                )
                saved_files['csv'] = csv_path
            
            if save_parquet:
                parquet_path = output_dir / f"{filename}.parquet"
                self.df.to_parquet(parquet_path, index=False, engine='pyarrow')
                parquet_size_mb = parquet_path.stat().st_size / (1024**2)
                self.logger.info(
                    f"Saved filtered Parquet: {parquet_path} "
                    f"({parquet_size_mb:.2f} MB, {len(self.df):,} rows)"
                )
                saved_files['parquet'] = parquet_path
            
            if not saved_files:
                raise DataProcessingError("No files were saved. Both save_csv and save_parquet are False.")
            
            return saved_files
            
        except Exception as e:
            error_msg = f"Error saving filtered dataset: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg, details={"output_dir": str(output_dir), "filename": filename})
    
    def stratified_sample(
        self,
        n_samples: int,
        stratify_col: str = 'Product',
        random_state: Optional[int] = 42,
        min_samples_per_stratum: int = 1
    ) -> pd.DataFrame:
        """
        Create a stratified random sample from the dataset.
        
        This method ensures proportional representation across all categories
        in the specified stratification column. If a category has fewer samples
        than required proportionally, all available samples are taken.
        
        Args:
            n_samples: Total number of samples to draw (target range: 10,000-15,000)
            stratify_col: Column name to stratify on (default: 'Product')
            random_state: Random seed for reproducibility
            min_samples_per_stratum: Minimum samples per category (default: 1)
            
        Returns:
            Stratified sample DataFrame
            
        Raises:
            DataValidationError: If stratification column doesn't exist or has insufficient data
        """
        if stratify_col not in self.df.columns:
            raise DataValidationError(
                f"Stratification column '{stratify_col}' not found in dataset"
            )
        
        if n_samples <= 0:
            raise DataValidationError(f"Sample size must be positive, got {n_samples}")
        
        if n_samples > len(self.df):
            self.logger.warning(
                f"Requested sample size ({n_samples:,}) exceeds dataset size ({len(self.df):,}). "
                f"Returning full dataset."
            )
            return self.df.copy()
        
        # Get value counts for stratification column
        value_counts = self.df[stratify_col].value_counts()
        total_rows = len(self.df)
        
        self.logger.info(f"Creating stratified sample of {n_samples:,} from {total_rows:,} rows")
        self.logger.info(f"Stratifying by: {stratify_col}")
        self.logger.info(f"Unique categories: {value_counts.shape[0]}")
        
        # Calculate proportional allocation
        proportions = value_counts / total_rows
        allocations = (proportions * n_samples).astype(int)
        
        # Ensure minimum samples per stratum
        allocations = allocations.clip(lower=min_samples_per_stratum)
        
        # Adjust for rounding errors - distribute remaining samples
        allocated_total = allocations.sum()
        remaining = n_samples - allocated_total
        
        if remaining > 0:
            # Distribute remaining samples to largest categories
            sorted_indices = allocations.sort_values(ascending=False).index
            for idx in sorted_indices[:remaining]:
                allocations[idx] += 1
        elif remaining < 0:
            # If we over-allocated, reduce from smallest categories
            sorted_indices = allocations.sort_values(ascending=True).index
            for idx in sorted_indices:
                if allocations[idx] > min_samples_per_stratum and remaining < 0:
                    reduction = min(abs(remaining), allocations[idx] - min_samples_per_stratum)
                    allocations[idx] -= reduction
                    remaining += reduction
                    if remaining >= 0:
                        break
        
        # Sample from each stratum
        sampled_dfs = []
        sampling_summary = {}
        
        for category, n_sample in allocations.items():
            category_df = self.df[self.df[stratify_col] == category]
            available = len(category_df)
            n_sample = min(n_sample, available)  # Don't sample more than available
            
            if n_sample > 0:
                sampled = category_df.sample(n=n_sample, random_state=random_state)
                sampled_dfs.append(sampled)
                
                sampling_summary[category] = {
                    'requested': int(allocations[category]),
                    'sampled': len(sampled),
                    'available': available,
                    'proportion': f"{(available / total_rows * 100):.2f}%"
                }
        
        # Combine all sampled dataframes
        if sampled_dfs:
            result_df = pd.concat(sampled_dfs, ignore_index=True)
            # Shuffle the final result
            result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            actual_sample_size = len(result_df)
            self.logger.info(f"✅ Stratified sample created: {actual_sample_size:,} rows")
            
            # Log sampling summary
            self.logger.info("\nSampling Summary:")
            self.logger.info("=" * 80)
            for category, info in sampling_summary.items():
                self.logger.info(
                    f"{category}: {info['sampled']:,} sampled "
                    f"(requested: {info['requested']:,}, available: {info['available']:,}, "
                    f"proportion: {info['proportion']})"
                )
            self.logger.info("=" * 80)
            
            return result_df
        else:
            raise DataValidationError("No samples could be drawn from any category")
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.
        
        Returns:
            Processed DataFrame
        """
        return self.df.copy()
    
    def reset(self) -> 'DataPreprocessor':
        """
        Reset to original data (not implemented - would need to store original).
        
        Returns:
            Self for method chaining
        """
        self.logger.warning("Reset not implemented. Create new DataPreprocessor instance.")
        return self

