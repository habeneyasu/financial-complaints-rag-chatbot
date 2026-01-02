"""
Exploratory Data Analysis (EDA) analyzer module.

This module provides reusable EDA functionality that can be used
in notebooks or scripts for consistent analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger


class EDAAnalyzer:
    """
    A class for performing exploratory data analysis.
    
    This class provides reusable methods for common EDA tasks,
    ensuring consistency across different analysis notebooks.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA Analyzer.
        
        Args:
            df: DataFrame to analyze
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")
        
        self.df = df
        self.logger = get_logger(__name__)
    
    def analyze_product_distribution(self, product_col: str = 'Product') -> Dict[str, Any]:
        """
        Analyze distribution of complaints across products.
        
        Args:
            product_col: Name of the product column
            
        Returns:
            Dictionary with analysis results
        """
        if product_col not in self.df.columns:
            raise ValueError(f"Column '{product_col}' not found")
        
        product_counts = self.df[product_col].value_counts()
        product_percentages = (self.df[product_col].value_counts(normalize=True) * 100).round(2)
        
        return {
            "total_products": self.df[product_col].nunique(),
            "product_counts": product_counts.to_dict(),
            "product_percentages": product_percentages.to_dict(),
            "most_common": {
                "product": product_counts.index[0],
                "count": int(product_counts.iloc[0]),
                "percentage": float(product_percentages.iloc[0])
            },
            "least_common": {
                "product": product_counts.index[-1],
                "count": int(product_counts.iloc[-1]),
                "percentage": float(product_percentages.iloc[-1])
            },
            "median_count": float(product_counts.median())
        }
    
    def analyze_narrative_length(
        self,
        narrative_col: str = 'Consumer complaint narrative'
    ) -> Dict[str, Any]:
        """
        Analyze narrative length distribution.
        
        Args:
            narrative_col: Name of the narrative column
            
        Returns:
            Dictionary with analysis results
        """
        if narrative_col not in self.df.columns:
            raise ValueError(f"Column '{narrative_col}' not found")
        
        def word_count(text):
            if pd.isna(text) or text == '':
                return 0
            return len(str(text).split())
        
        # Calculate word counts
        word_counts = self.df[narrative_col].apply(word_count)
        df_with_narratives = self.df[self.df[narrative_col].notna() & (self.df[narrative_col] != '')]
        
        if len(df_with_narratives) == 0:
            return {"error": "No narratives found"}
        
        narrative_word_counts = df_with_narratives[narrative_col].apply(word_count)
        
        return {
            "total_rows": len(self.df),
            "rows_with_narratives": len(df_with_narratives),
            "rows_without_narratives": len(self.df) - len(df_with_narratives),
            "word_count_stats": {
                "mean": float(narrative_word_counts.mean()),
                "median": float(narrative_word_counts.median()),
                "std": float(narrative_word_counts.std()),
                "min": int(narrative_word_counts.min()),
                "max": int(narrative_word_counts.max()),
                "q25": float(narrative_word_counts.quantile(0.25)),
                "q75": float(narrative_word_counts.quantile(0.75)),
                "q90": float(narrative_word_counts.quantile(0.90)),
                "q95": float(narrative_word_counts.quantile(0.95)),
                "q99": float(narrative_word_counts.quantile(0.99))
            },
            "very_short_count": int((narrative_word_counts < 10).sum()),
            "very_long_threshold": float(narrative_word_counts.quantile(0.95)),
            "very_long_count": int((narrative_word_counts > narrative_word_counts.quantile(0.95)).sum())
        }
    
    def analyze_narrative_presence(
        self,
        narrative_col: str = 'Consumer complaint narrative'
    ) -> Dict[str, Any]:
        """
        Analyze presence of narratives in complaints.
        
        Args:
            narrative_col: Name of the narrative column
            
        Returns:
            Dictionary with analysis results
        """
        if narrative_col not in self.df.columns:
            raise ValueError(f"Column '{narrative_col}' not found")
        
        has_narrative = self.df[narrative_col].notna() & (self.df[narrative_col] != '')
        
        with_count = int(has_narrative.sum())
        without_count = int((~has_narrative).sum())
        total = len(self.df)
        
        return {
            "total_complaints": total,
            "with_narrative": {
                "count": with_count,
                "percentage": float((with_count / total * 100).round(2))
            },
            "without_narrative": {
                "count": without_count,
                "percentage": float((without_count / total * 100).round(2))
            }
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get overall summary statistics of the dataset.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / 1024**2),
            "null_counts": self.df.isnull().sum().to_dict(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum())
        }

