# Financial Complaints RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for answering questions about financial complaints using vector embeddings and LLMs.

## Project Structure

```
├── .vscode/
│   └── settings.json              # VS Code configuration
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline for unit tests
├── data/
│   ├── raw/                       # Raw CFPB complaint data
│   └── processed/                 # Processed/cleaned data
├── vector_store/                  # Persisted FAISS/ChromaDB index
├── notebooks/
│   ├── 01_load_dataset.ipynb      # Task 1: Load CFPB dataset
│   ├── 02_eda_analysis.ipynb      # Task 1: Exploratory Data Analysis
│   └── README.md                  # Notebook documentation
├── src/
│   ├── data/                      # Data processing modules
│   │   ├── loader.py              # DataLoader class
│   │   └── preprocessor.py        # DataPreprocessor class
│   ├── eda/                       # EDA utilities
│   │   └── analyzer.py              # EDAAnalyzer class
│   ├── config.py                  # Configuration management
│   ├── exceptions.py              # Custom exceptions
│   └── utils/                     # Utility modules
│       └── logger.py              # Logging utilities
├── tests/                         # Unit tests
├── app.py                         # Gradio/Streamlit interface
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## Tasks Overview

### Task 1: Exploratory Data Analysis and Data Preprocessing ✅
- **Notebooks**: `01_load_dataset.ipynb`, `02_eda_analysis.ipynb`
- **Modules**: `src/data/loader.py`, `src/data/preprocessor.py`, `src/eda/analyzer.py`
- **Objectives**:
  - Load CFPB complaint dataset
  - Perform EDA (product distribution, narrative analysis)
  - Clean and preprocess data for RAG pipeline

### Task 2: Vector Store Creation (Upcoming)
- Create embeddings for complaint narratives
- Build and persist vector store

### Task 3: RAG Pipeline Implementation (Upcoming)
- Implement retrieval component
- Integrate with LLM
- Build complete RAG pipeline

### Task 4: Application Development (Upcoming)
- Build Gradio/Streamlit interface
- Deploy and evaluate

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Notebooks

Start Jupyter:
```bash
jupyter notebook
```

Then open:
- `notebooks/01_load_dataset.ipynb` - Load the CFPB dataset
- `notebooks/02_eda_analysis.ipynb` - Perform EDA

### Using the Code Modules

```python
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.eda.analyzer import EDAAnalyzer

# Load data
loader = DataLoader(data_path="data/raw/complaints.csv")
df = loader.load_data()

# Preprocess
preprocessor = DataPreprocessor(df)
df_clean = preprocessor.clean_narratives().remove_duplicates().get_data()

# Analyze
analyzer = EDAAnalyzer(df_clean)
product_dist = analyzer.analyze_product_distribution()
```

## Development

Run tests:
```bash
pytest tests/ -v
```

## Code Organization Principles

- **Modularity**: Each class has a single responsibility
- **Reusability**: Classes can be used in notebooks or scripts
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Logging**: All operations are logged for debugging
- **Type Hints**: Full type annotations for better code clarity
- **Documentation**: Clear docstrings following Google style

## License

[Add your license here]
