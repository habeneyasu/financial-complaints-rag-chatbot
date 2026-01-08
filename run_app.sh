#!/bin/bash
# Launcher script for the Financial Complaints RAG Chatbot

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Error: Virtual environment not found at venv/"
    echo "Please create a virtual environment first: python -m venv venv"
    exit 1
fi

# Check if sentence-transformers is available
python -c "from sentence_transformers import SentenceTransformer" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: sentence-transformers is not available in the virtual environment"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the app
echo "🚀 Starting Financial Complaints RAG Chatbot..."
python app.py

