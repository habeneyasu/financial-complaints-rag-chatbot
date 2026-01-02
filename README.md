# Financial Complaints RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot system for answering questions about financial complaints using the Consumer Financial Protection Bureau (CFPB) dataset. This project leverages vector embeddings, semantic search, and Large Language Models (LLMs) to provide accurate, context-aware responses to user queries about financial complaint data.

## 🎯 Features

- **Intelligent Question Answering**: Query financial complaint data using natural language
- **Semantic Search**: Leverage vector embeddings for contextually relevant retrieval
- **Data Preprocessing Pipeline**: Automated cleaning and normalization of complaint narratives
- **Exploratory Data Analysis**: Comprehensive EDA tools for understanding complaint patterns
- **Modular Architecture**: Clean, maintainable codebase following SOLID principles
- **Production-Ready**: Comprehensive error handling, logging, and configuration management
- **Interactive UI**: Gradio-based web interface for easy interaction

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │
│  ┌───────────┐  │
│  │ Retrieval │──┼──► Vector Store (ChromaDB)
│  └───────────┘  │
│  ┌───────────┐  │
│  │ Generation│──┼──► LLM (Transformers)
│  └───────────┘  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Response       │
└─────────────────┘
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **LangChain**: RAG pipeline orchestration
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Embedding generation
- **Transformers**: LLM integration
- **PyTorch**: Deep learning framework

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **NLTK**: Natural language processing utilities

### Visualization & UI
- **Matplotlib & Seaborn**: Data visualization
- **Gradio**: Interactive web interface

### Development Tools
- **Jupyter**: Interactive development and analysis
- **Pytest**: Unit testing framework

## 📋 Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **8GB+ RAM** (recommended for processing large datasets)
- **CUDA-capable GPU** (optional, for faster model inference)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd financial-complaints-rag-chatbot
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLTK Data (if needed)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## ⚙️ Configuration

The project uses environment variables for configuration. Create a `.env` file in the root directory:

```env
# Data Configuration
CFPB_DATA_URL=https://files.consumerfinance.gov/ccdb/complaints.csv.zip

# Logging Configuration
LOG_LEVEL=INFO

# Optional: Sample size for testing
SAMPLE_SIZE=10000
```

Configuration is managed through `src/config.py` using the 12-factor app methodology.

## 📖 Usage

### Running the Application

```bash
python app.py
```

This will start the Gradio interface, typically accessible at `http://localhost:7860`.

### Using the Code Modules

#### Data Loading and Preprocessing

```python
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

# Load data
loader = DataLoader(data_path="data/raw/complaints.csv")
df = loader.load_data()

# Preprocess data
preprocessor = DataPreprocessor(df)
df_clean = preprocessor.clean_narratives().remove_duplicates().get_data()

# Save processed data
preprocessor.save_processed_data("data/processed/complaints_clean.parquet")
```

#### Exploratory Data Analysis

```python
from src.eda.analyzer import EDAAnalyzer

# Initialize analyzer
analyzer = EDAAnalyzer(df_clean)

# Analyze product distribution
product_dist = analyzer.analyze_product_distribution()
print(product_dist)

# Generate visualizations
analyzer.plot_product_distribution()
analyzer.plot_narrative_length_distribution()
```

### Running Jupyter Notebooks

The project includes sequential notebooks for data exploration and processing:

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

**Notebook Sequence:**
1. `01_load_dataset.ipynb` - Load and validate CFPB dataset
2. `02_eda_analysis.ipynb` - Exploratory data analysis
3. `03_filter_dataset.ipynb` - Filter and subset data
4. `04_clean_text_narratives.ipynb` - Text cleaning for embeddings
5. `05_stratified_sampling.ipynb` - Create stratified sample (10,000-15,000 complaints) with proportional product representation
6. `06_text_chunking_experiment.ipynb` - Text chunking strategy experimentation and optimization
7. `07_embedding_vectorstore.ipynb` - Embedding generation and vector store indexing (Task 2)

## 📋 Tasks Overview

### Task 1: Exploratory Data Analysis and Data Preprocessing ✅
- **Notebooks**: `01_load_dataset.ipynb`, `02_eda_analysis.ipynb`, `03_filter_dataset.ipynb`, `04_clean_text_narratives.ipynb`
- **Modules**: `src/data/loader.py`, `src/data/preprocessor.py`, `src/eda/analyzer.py`
- **Objectives**:
  - Load CFPB complaint dataset
  - Perform exploratory data analysis (product distribution, narrative analysis)
  - Filter and subset data for processing
  - Clean and preprocess complaint narratives
  - Normalize text for embedding generation

### Task 2: Text Chunking, Embedding, and Vector Store Indexing 🚧
- **Objective**: Convert the cleaned text narratives into a format suitable for efficient semantic search
- **Prerequisites**: 
  - Stratified sample of 10,000-15,000 complaints (notebook `05_stratified_sampling.ipynb`)
- **Key Components**:
  - **Text Chunking**: Split long narratives into appropriately sized chunks for embedding
  - **Embedding Generation**: Create vector embeddings using sentence transformers
  - **Vector Store Indexing**: Build and persist vector store (ChromaDB) for semantic search
  - **Metadata Management**: Store relevant complaint metadata alongside embeddings
- **Expected Deliverables**:
  - Chunked text data with optimal chunk sizes
  - Generated embeddings for all complaint narratives
  - Persisted vector store ready for retrieval
  - Indexing pipeline for efficient querying

### Task 3: RAG Pipeline Implementation (Upcoming)
- **Objective**: Build the complete Retrieval-Augmented Generation pipeline
- **Key Components**:
  - Implement retrieval component for semantic search
  - Integrate with Large Language Model (LLM)
  - Build query processing and response generation
  - Implement context management and prompt engineering

### Task 4: Application Development (Upcoming)
- **Objective**: Create user-friendly interface and deploy the system
- **Key Components**:
  - Build Gradio/Streamlit web interface
  - Implement user interaction flows
  - Add performance monitoring and evaluation
  - Deploy and optimize for production

## 📁 Project Structure

```
financial-complaints-rag-chatbot/
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline configuration
├── data/
│   ├── raw/                       # Raw CFPB complaint data
│   │   └── complaints.csv         # Source dataset
│   └── processed/                 # Processed/cleaned data
│       └── processed_complaints.parquet
├── vector_store/                  # Persisted vector embeddings (ChromaDB)
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_load_dataset.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_filter_dataset.ipynb
│   ├── 04_clean_text_narratives.ipynb
│   ├── 05_stratified_sampling.ipynb      # Stratified sampling (10K-15K samples)
│   ├── 06_text_chunking_experiment.ipynb # Text chunking experimentation
│   ├── 07_embedding_vectorstore.ipynb    # Task 2: Embedding & vector store
│   └── README.md
├── src/                           # Source code
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py              # DataLoader class
│   │   └── preprocessor.py        # DataPreprocessor class
│   ├── eda/                       # EDA utilities
│   │   ├── __init__.py
│   │   └── analyzer.py            # EDAAnalyzer class
│   ├── rag/                       # RAG pipeline modules
│   │   ├── __init__.py
│   │   └── chunker.py             # TextChunker class
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py
│   │   └── logger.py              # Logging utilities
│   ├── config.py                  # Configuration management
│   ├── exceptions.py              # Custom exception classes
│   └── __init__.py
├── tests/                         # Unit tests
│   └── __init__.py
├── app.py                         # Main application entry point
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── .env.example                   # Example environment variables
└── README.md                      # This file
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_loader.py -v
```

## 🔧 Development

### Code Organization Principles

- **Modularity**: Each class has a single responsibility (SRP)
- **Reusability**: Classes can be used in notebooks or scripts
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Logging**: All operations are logged for debugging and monitoring
- **Type Hints**: Full type annotations for better code clarity
- **Documentation**: Clear docstrings following Google style guide

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test**
   ```bash
   pytest tests/ -v
   ```

3. **Commit changes**
   ```bash
   git commit -m "feat: add new feature"
   ```

4. **Push and create pull request**

### Code Style

The project follows PEP 8 style guidelines. Consider using:
- **Black** for code formatting
- **flake8** or **ruff** for linting
- **mypy** for type checking

## 📊 Project Status

### ✅ Completed
- [x] Data loading and validation pipeline
- [x] Exploratory data analysis tools
- [x] Data preprocessing and cleaning
- [x] Text normalization for embeddings
- [x] Stratified sampling (10,000-15,000 complaints with proportional product representation)
- [x] Modular codebase architecture
- [x] Configuration management
- [x] Logging infrastructure

### 🚧 In Progress
- [x] Text chunking strategy implementation and experimentation
- [ ] Embedding generation pipeline
- [ ] Vector store creation and persistence (ChromaDB)
- [ ] Vector indexing and metadata management

### 📅 Planned
- [ ] RAG pipeline implementation (Task 3)
- [ ] Retrieval component development
- [ ] LLM integration and prompt engineering
- [ ] Application development with Gradio interface (Task 4)
- [ ] Performance optimization and evaluation
- [ ] Deployment configuration
- [ ] API endpoint development
- [ ] Model fine-tuning capabilities

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Consumer Financial Protection Bureau (CFPB)** for providing the complaint dataset
- **LangChain** community for RAG framework
- **Hugging Face** for transformer models and tools
- **ChromaDB** team for vector database solution

## 📚 References

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

## 📧 Contact

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**Note**: This project is part of a training portfolio and is intended for educational and demonstration purposes.
