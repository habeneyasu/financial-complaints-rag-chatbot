"""
Common setup utilities for Jupyter notebooks.
This module provides reusable setup code to avoid duplication across notebooks.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings


def setup_notebook(
    project_root: Path = None,
    suppress_warnings: bool = True,
    set_pandas_options: bool = True,
    print_info: bool = True
) -> Path:
    """
    Common setup for Jupyter notebooks.
    
    Args:
        project_root: Path to project root. If None, auto-detects.
        suppress_warnings: Whether to suppress warnings.
        set_pandas_options: Whether to set pandas display options.
        print_info: Whether to print setup information.
    
    Returns:
        Path to project root.
    """
    # Auto-detect project root if not provided
    if project_root is None:
        cwd = Path.cwd()
        if cwd.name == 'notebooks':
            project_root = cwd.parent
        else:
            project_root = cwd
    
    # Add src to path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Suppress warnings
    if suppress_warnings:
        warnings.filterwarnings('ignore')
    
    # Set pandas display options
    if set_pandas_options:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 100)
    
    # Print setup information
    if print_info:
        print(f"Project root: {project_root}")
        print(f"Python version: {sys.version}")
        print(f"Pandas version: {pd.__version__}")
    
    return project_root


def setup_visualization():
    """
    Setup for notebooks that use visualization (matplotlib, seaborn).
    Call this after setup_notebook() if you need plotting.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        
        print(f"Matplotlib version: {plt.matplotlib.__version__}")
        print(f"Seaborn version: {sns.__version__}")
        
        return plt, sns
    except ImportError as e:
        print(f"Warning: Visualization libraries not available: {e}")
        return None, None

