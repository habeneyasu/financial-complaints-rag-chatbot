"""
Financial Complaints RAG Chatbot - Main Application

This module provides the Gradio interface for the RAG chatbot.
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Verify sentence-transformers is available before importing RAG components
import os
venv_python = Path(__file__).parent / 'venv' / 'bin' / 'python'

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"\n❌ ERROR: sentence-transformers is NOT available in the current Python environment")
    print(f"   Python executable: {sys.executable}")
    print(f"   Virtual env: {os.environ.get('VIRTUAL_ENV', 'Not detected')}")
    print(f"\n💡 Solution:")
    print(f"   1. Activate the virtual environment: source venv/bin/activate")
    print(f"   2. Or use the venv Python directly: {venv_python} app.py")
    print(f"   3. Or use the launcher script: ./run_app.sh")
    print(f"\n   If sentence-transformers is missing, install it: pip install sentence-transformers")
    sys.exit(1)

from src.rag.pipeline import RAGPipeline
from src.config import config
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Global variable to store RAG pipeline
rag_pipeline: Optional[RAGPipeline] = None


def initialize_rag_pipeline(
    vector_store_path: Optional[str] = None,
    collection_name: Optional[str] = None
) -> RAGPipeline:
    """
    Initialize the RAG pipeline.
    
    Args:
        vector_store_path: Path to vector store directory
        collection_name: Name of the collection
        
    Returns:
        Initialized RAGPipeline instance
    """
    global rag_pipeline
    
    try:
        logger.info("Initializing RAG pipeline...")
        
        if vector_store_path is None:
            vector_store_path = str(config.data.vector_store_dir)
        
        if collection_name is None:
            collection_name = config.vectorstore.collection_name
        
        rag_pipeline = RAGPipeline.from_vector_store_path(
            vector_store_path=Path(vector_store_path),
            collection_name=collection_name,
            top_k=5
        )
        
        logger.info("✅ RAG pipeline initialized successfully")
        return rag_pipeline
        
    except Exception as e:
        error_msg = f"Error initializing RAG pipeline: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def format_sources(sources: list) -> str:
    """
    Format source documents for display.
    
    Args:
        sources: List of source documents
        
    Returns:
        Formatted markdown string with source information
    """
    if not sources:
        return "**No sources retrieved.**"
    
    formatted_sources = []
    formatted_sources.append("### 📚 Retrieved Sources\n")
    
    for i, source in enumerate(sources, 1):
        rank = source.get('rank', i)
        text = source.get('text', 'N/A')
        metadata = source.get('metadata', {})
        distance = source.get('distance')
        
        product = metadata.get('product', 'Unknown')
        issue = metadata.get('issue', 'Unknown')
        company = metadata.get('company', 'Unknown')
        
        # Calculate relevance score (1 - distance, where distance is typically 0-2)
        relevance_score = None
        if distance is not None:
            relevance_score = max(0, min(1, 1 - distance))  # Normalize to 0-1
        
        source_text = f"#### Source {rank}\n\n"
        source_text += f"**Product:** {product}  \n"
        source_text += f"**Issue:** {issue}  \n"
        source_text += f"**Company:** {company}  \n"
        if relevance_score is not None:
            source_text += f"**Relevance:** {relevance_score:.1%}  \n"
        source_text += f"\n**Text Excerpt:**\n> {text}\n\n"
        source_text += "---\n\n"
        
        formatted_sources.append(source_text)
    
    return "\n".join(formatted_sources)


def ask_question_streaming(question: str):
    """
    Process a user question with streaming response.
    
    This generator function yields the answer token-by-token for better UX.
    
    Args:
        question: User's question
        
    Yields:
        Tuple of (partial_answer, formatted_sources) as answer is generated
    """
    global rag_pipeline
    
    if not question or not question.strip():
        yield "Please enter a question.", "**No sources available.**"
        return
    
    if rag_pipeline is None:
        try:
            initialize_rag_pipeline()
        except Exception as e:
            error_msg = f"Failed to initialize RAG pipeline: {str(e)}"
            logger.error(error_msg)
            yield error_msg, "**No sources available.**"
            return
    
    try:
        logger.info(f"Processing question: {question[:100]}...")
        
        # Get answer from RAG pipeline
        result = rag_pipeline.answer(
            question=question,
            return_sources=True
        )
        
        answer = result.get('answer', 'No answer generated.')
        sources = result.get('sources', [])
        
        # Format sources for display
        formatted_sources = format_sources(sources)
        
        # Simulate streaming by yielding answer word-by-word
        # This provides better UX even if the model doesn't support true streaming
        words = answer.split()
        partial_answer = ""
        
        for word in words:
            partial_answer += word + " "
            yield partial_answer.strip(), formatted_sources
        
        # Final yield to ensure complete answer is shown
        yield answer, formatted_sources
        
        logger.info("✅ Question processed successfully")
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        logger.error(error_msg)
        yield f"An error occurred: {error_msg}", "**No sources available.**"


def ask_question(question: str) -> Tuple[str, str]:
    """
    Process a user question and return the answer with sources (non-streaming version).
    
    Args:
        question: User's question
        
    Returns:
        Tuple of (answer, formatted_sources)
    """
    global rag_pipeline
    
    if not question or not question.strip():
        return "Please enter a question.", "**No sources available.**"
    
    if rag_pipeline is None:
        try:
            initialize_rag_pipeline()
        except Exception as e:
            error_msg = f"Failed to initialize RAG pipeline: {str(e)}"
            logger.error(error_msg)
            return error_msg, "**No sources available.**"
    
    try:
        logger.info(f"Processing question: {question[:100]}...")
        
        # Get answer from RAG pipeline
        result = rag_pipeline.answer(
            question=question,
            return_sources=True
        )
        
        answer = result.get('answer', 'No answer generated.')
        sources = result.get('sources', [])
        
        # Format sources for display
        formatted_sources = format_sources(sources)
        
        logger.info("✅ Question processed successfully")
        
        return answer, formatted_sources
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        logger.error(error_msg)
        return f"An error occurred: {error_msg}", "**No sources available.**"


def clear_conversation() -> Tuple[str, str, str]:
    """
    Clear the conversation interface.
    
    Returns:
        Tuple of empty strings (question, answer, sources)
    """
    return "", "", "**Sources will appear here after you ask a question.**\n\nThe sources show the complaint excerpts that were used to generate the answer above."


def create_interface():
    """
    Create and launch the Gradio interface.
    """
    # Initialize RAG pipeline
    try:
        initialize_rag_pipeline()
        status_message = "✅ RAG pipeline initialized successfully!"
    except Exception as e:
        status_message = f"⚠️ Warning: Could not initialize RAG pipeline: {str(e)}\nThe interface will attempt to initialize when you ask a question."
        logger.warning(status_message)
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
    }
    .answer-box {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        min-height: 200px;
    }
    """
    
    # Create Gradio interface
    with gr.Blocks(css=custom_css, title="Financial Complaints RAG Chatbot") as demo:
        # Header
        gr.Markdown(
            """
            # 🏦 Financial Complaints RAG Chatbot
            
            Ask questions about financial complaints and get answers based on retrieved complaint data.
            """,
            elem_classes=["main-header"]
        )
        
        # Status message
        gr.Markdown(f"**Status:** {status_message}")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Question input
                question_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="e.g., What are the most common issues with credit cards?",
                    lines=3,
                    interactive=True
                )
                
                # Buttons
                with gr.Row():
                    submit_btn = gr.Button("Ask", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary", size="lg")
                
                # Answer display
                answer_output = gr.Textbox(
                    label="🤖 AI-Generated Answer",
                    lines=12,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="The answer will appear here as it's generated...",
                    elem_classes=["answer-box"]
                )
            
            with gr.Column(scale=1):
                # Sources display
                sources_output = gr.Markdown(
                    label="📚 Source Documents",
                    value="**Sources will appear here after you ask a question.**\n\n"
                          "The sources show the complaint excerpts that were used to generate the answer above. "
                          "This helps verify the accuracy and traceability of the response.",
                    elem_classes=["source-box"]
                )
        
        # Examples
        gr.Examples(
            examples=[
                "What are the most common issues with credit cards?",
                "What problems do customers face with personal loans?",
                "Tell me about money transfer complaints.",
                "What are common complaints about savings accounts?",
                "How do customers describe issues with their credit reports?"
            ],
            inputs=question_input
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### ℹ️ About This Chatbot
            
            This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer questions 
            based on financial complaint data. 
            
            **How it works:**
            1. Your question is converted to a search query
            2. Relevant complaint excerpts are retrieved from the vector database
            3. An AI model generates an answer based on the retrieved context
            4. Source documents are displayed for verification and transparency
            
            **Features:**
            - ✅ Real-time streaming responses
            - ✅ Source citation for every answer
            - ✅ Based on real financial complaint data
            - ✅ Transparent and verifiable responses
            """
        )
        
        # Event handlers with streaming support
        # Use streaming for better UX (answer appears token-by-token)
        submit_btn.click(
            fn=ask_question_streaming,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        question_input.submit(
            fn=ask_question_streaming,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            inputs=None,
            outputs=[question_input, answer_output, sources_output]
        )
    
    return demo


def main():
    """
    Main application entry point.
    """
    logger.info("Starting Financial Complaints RAG Chatbot application...")
    
    # Create and launch interface
    demo = create_interface()
    
    # Launch with sharing disabled by default (can be enabled for public access)
    demo.launch(
        server_name="0.0.0.0",  # Allow access from network
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public link
        show_error=True
    )


if __name__ == "__main__":
    main()
