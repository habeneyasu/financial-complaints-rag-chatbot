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
│   ├── raw/                       # Raw data files
│   └── processed/                 # Processed/cleaned data
├── vector_store/                  # Persisted FAISS/ChromaDB index
├── notebooks/
│   ├── __init__.py
│   └── README.md                  # Notebook documentation
├── src/
│   ├── __init__.py                # Source code package
├── tests/
│   ├── __init__.py                # Unit tests
├── app.py                         # Gradio/Streamlit interface
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

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

Run the application:
```bash
python app.py
```

## Development

Run tests:
```bash
pytest tests/ -v
```

## License

[Add your license here]

