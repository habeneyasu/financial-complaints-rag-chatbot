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

### Running the Complete Pipeline

Execute the complete preprocessing and embedding pipeline:

```bash
# Run with default configuration
python -m src.pipeline

# Run with custom parameters
python -m src.pipeline \
    --sample-size 12500 \
    --stratify-column Product \
    --random-state 42 \
    --chunk-size 500 \
    --chunk-overlap 75 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Skip stratified sampling (use full filtered dataset)
python -m src.pipeline --no-sampling
```

The pipeline integrates:
1. **Data loading** from raw CSV
2. **Task 1 filtering** (products + narratives)
3. **Stratified sampling** (10k-15k with proportional representation)
4. **Text chunking**
5. **Embedding generation**
6. **Vector store creation**

All steps are configurable and reproducible.

### Using the Code Modules

#### Data Loading and Preprocessing

```python
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

# Load data
loader = DataLoader(data_path="data/raw/complaints.csv")
df = loader.load_data()

# Task 1: Apply complete filtering workflow (explicit product filtering + narrative removal + EDA)
preprocessor = DataPreprocessor(df, narrative_col='Consumer complaint narrative')
task1_results = preprocessor.apply_task1_filtering(
    target_products=["Credit card", "Personal loan", "Savings account", "Money transfers"],
    perform_eda=True
)
df_filtered = task1_results['filtered_dataframe']

# Save filtered dataset with clear naming
preprocessor.save_filtered_dataset(
    filename="task1_filtered_complaints",
    save_csv=True,
    save_parquet=True
)
```

#### Complete Pipeline with Stratified Sampling

```python
from src.pipeline import run_complete_pipeline

# Run complete pipeline: filtering -> stratified sampling -> chunking -> embedding -> vector store
results = run_complete_pipeline(
    sample_size=12500,  # Stratified sample size (10k-15k range)
    stratify_column='Product',  # Column for proportional representation
    random_state=42,  # For reproducibility
    create_stratified_sample=True,
    chunk_size=500,
    chunk_overlap=75
)

# Or use the preprocessor directly
from src.data.preprocessor import DataPreprocessor
preprocessor = DataPreprocessor(df, narrative_col='Consumer complaint narrative')

# Complete pipeline with stratified sampling
pipeline_results = preprocessor.apply_complete_pipeline(
    target_products=["Credit card", "Personal loan", "Savings account", "Money transfers"],
    create_stratified_sample=True,
    n_samples=12500,
    perform_eda=True
)

# Create and save stratified sample separately
sample_results = preprocessor.create_and_save_stratified_sample(
    n_samples=12500,
    stratify_col='Product',
    random_state=42
)
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

#### RAG Pipeline Usage

```python
from src.rag.pipeline import RAGPipeline
from src.rag.evaluator import RAGEvaluator

# Initialize RAG pipeline from vector store
rag_pipeline = RAGPipeline.from_vector_store_path(
    vector_store_path="vector_store",
    collection_name="financial_complaints",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="google/flan-t5-base",
    top_k=5
)

# Answer a question
result = rag_pipeline.answer("What are the most common issues with credit cards?")
print(result['answer'])
print(result['sources'])  # Top retrieved sources

# Run evaluation
evaluator = RAGEvaluator(rag_pipeline)
questions = [{"question": "What are common credit card issues?"}]
results = evaluator.evaluate_questions(questions)
eval_table = evaluator.create_evaluation_table(results)
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
3. `03_filter_dataset.ipynb` - **Task 1 filtering** (explicit product filtering, narrative removal, integrated EDA) → outputs `task1_filtered_complaints.csv`
4. `04_clean_text_narratives.ipynb` - Text cleaning for embeddings
5. `05_stratified_sampling.ipynb` - Create stratified sample (10,000-15,000 complaints) with proportional product representation
6. `06_text_chunking_experiment.ipynb` - Text chunking strategy experimentation and optimization
7. `07_embedding_generation.ipynb` - Embedding generation using sentence-transformers
8. `08_vectorstore_creation.ipynb` - Vector store creation and indexing (Task 2)
9. `09_rag_evaluation.ipynb` - **Task 3 RAG pipeline evaluation** (retrieval + generation + qualitative assessment)

## 📋 Tasks Overview

### Task 1: Exploratory Data Analysis and Data Preprocessing ✅
- **Notebooks**: `01_load_dataset.ipynb`, `02_eda_analysis.ipynb`, `03_filter_dataset.ipynb`, `04_clean_text_narratives.ipynb`
- **Modules**: `src/data/loader.py`, `src/data/preprocessor.py`, `src/eda/analyzer.py`
- **Objectives**:
  - Load CFPB complaint dataset
  - Perform exploratory data analysis (product distribution, narrative analysis)
  - **Explicitly filter** to assignment's specified product set (Credit card, Personal loan, Savings account, Money transfers)
  - **Explicitly remove** records without Consumer complaint narratives
  - **Integrated EDA workflow** (distribution, length, missingness) tightly coupled with preprocessing
  - Clean and preprocess complaint narratives
  - Normalize text for embedding generation
- **Deliverables**:
  - ✅ Filtered dataset saved as `task1_filtered_complaints.csv` (clearly named)
  - ✅ EDA analysis integrated with preprocessing pipeline
  - ✅ Fully reproducible workflow documented in code

### Task 2: Text Chunking, Embedding, and Vector Store Indexing ✅
- **Objective**: Convert the cleaned text narratives into a format suitable for efficient semantic search
- **Prerequisites**: 
  - Stratified sample of 10,000-15,000 complaints with proportional product representation
- **Key Components**:
  - **Stratified Sampling**: Executable code (not just notebook) ensuring 10k-15k samples with proportional representation across products
  - **Text Chunking**: Split long narratives into appropriately sized chunks for embedding (chunk_size=500, overlap=75)
  - **Embedding Generation**: Create vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - **Vector Store Indexing**: Build and persist vector store (ChromaDB) for semantic search
  - **Metadata Management**: Store relevant complaint metadata alongside embeddings for traceability
- **Notebooks**: `05_stratified_sampling.ipynb`, `06_text_chunking_experiment.ipynb`, `07_embedding_generation.ipynb`, `08_vectorstore_creation.ipynb`
- **Modules**: `src/data/preprocessor.py`, `src/pipeline.py`, `src/rag/chunker.py`, `src/rag/embedder.py`, `src/rag/vectorstore.py`
- **Executable Pipeline**: `src/pipeline.py` provides complete workflow as executable code
- **Configuration**: All parameters configurable via `config.py` and environment variables
- **Deliverables**:
  - ✅ Stratified sampling as executable code with configurable sample size, product column, and output locations
  - ✅ Chunked text data with optimal chunk sizes (500 chars, 15% overlap)
  - ✅ Generated embeddings for all complaint chunks (384-dimensional vectors)
  - ✅ Persisted ChromaDB vector store ready for retrieval
  - ✅ Metadata storage enabling full traceability to source complaints
  - ✅ Fully reproducible pipeline for reliable vector store creation

### Task 3: RAG Pipeline Implementation and Evaluation ✅
- **Objective**: Build the complete Retrieval-Augmented Generation pipeline and evaluate its effectiveness
- **Key Components**:
  - **Retriever**: Embeds questions using all-MiniLM-L6-v2 and performs similarity search (top-k=5)
  - **Prompt Engineering**: Robust template instructing LLM to act as financial analyst, use only context
  - **Generator**: LLM-based answer generation using Hugging Face transformers (flan-t5-base)
  - **Evaluation**: Qualitative assessment with 10 representative questions and quality scoring
- **Modules**: `src/rag/retriever.py`, `src/rag/prompts.py`, `src/rag/generator.py`, `src/rag/pipeline.py`, `src/rag/evaluator.py`
- **Notebook**: `09_rag_evaluation.ipynb` - Complete evaluation workflow
- **Deliverables**:
  - ✅ RAG pipeline with retriever and generator
  - ✅ Prompt template for context-based answering
  - ✅ Evaluation framework with 10 representative questions
  - ✅ Evaluation table with quality scores and analysis
  - ✅ Markdown evaluation report documenting results

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
│   ├── 07_embedding_generation.ipynb     # Embedding generation
│   ├── 08_vectorstore_creation.ipynb     # Task 2: Vector store creation
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
│   │   ├── chunker.py             # TextChunker class
│   │   ├── embedder.py            # EmbeddingGenerator class
│   │   └── vectorstore.py         # VectorStore class (ChromaDB)
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
- [x] Exploratory data analysis tools (integrated with preprocessing)
- [x] Data preprocessing and cleaning
- [x] **Explicit product filtering** to assignment's specified product set
- [x] **Explicit narrative removal** (records without narratives filtered in code)
- [x] **Clear CSV naming** for Task 1 deliverables (`task1_filtered_complaints.csv`)
- [x] **Reproducible Task 1 workflow** with integrated EDA
- [x] **Stratified sampling as executable code** (10k-15k with proportional representation)
- [x] **Configurable sampling** (sample size, product column, output locations)
- [x] **Complete pipeline integration** (filtering → sampling → embedding → vectorstore)
- [x] **Executable pipeline script** for reproducible vector store creation
- [x] Text normalization for embeddings
- [x] Modular codebase architecture
- [x] Configuration management
- [x] Logging infrastructure

### 🚧 In Progress
- [x] Text chunking strategy implementation and experimentation
- [x] Embedding generation pipeline (sentence-transformers/all-MiniLM-L6-v2)
- [x] Vector store creation and persistence (ChromaDB)
- [x] Vector indexing and metadata management

### ✅ Completed (Task 3)
- [x] RAG pipeline implementation with retriever and generator
- [x] Retrieval component with question embedding and similarity search
- [x] LLM integration using Hugging Face transformers
- [x] Prompt engineering with robust templates
- [x] Evaluation framework with qualitative assessment
- [x] Evaluation report generation

### 📅 Planned
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
