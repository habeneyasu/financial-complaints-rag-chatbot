"""
Prompt engineering module for RAG pipeline.

This module provides robust prompt templates for guiding the LLM to generate
accurate, context-based answers about financial complaints.
"""

from typing import List, Dict, Any


class PromptTemplate:
    """
    A class for managing prompt templates for RAG-based question answering.
    
    This class provides robust prompt templates that instruct the LLM to:
    - Act as a helpful financial analyst assistant
    - Use only the provided context
    - Answer questions based on that context
    - State when information is insufficient
    """
    
    DEFAULT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""
    
    DETAILED_TEMPLATE = """You are an expert financial analyst assistant for CrediTrust, a financial services company. Your role is to help answer questions about customer complaints by analyzing retrieved complaint excerpts.

Instructions:
1. Carefully read the provided context (retrieved complaint excerpts)
2. Use ONLY the information from the context to answer the question
3. If the context contains relevant information, provide a clear, concise answer
4. If the context does NOT contain enough information to answer the question, explicitly state: "I don't have enough information in the provided context to answer this question."
5. Do not make up or infer information that is not in the context
6. If multiple complaints are relevant, you can reference patterns or common issues

Context (Retrieved Complaint Excerpts):
{context}

Question: {question}

Answer:"""
    
    CONCISE_TEMPLATE = """You are a financial analyst assistant. Answer the question using ONLY the provided context. If the context doesn't contain the answer, say "I don't have enough information."

Context: {context}

Question: {question}

Answer:"""
    
    def __init__(self, template: str = None):
        """
        Initialize the prompt template.
        
        Args:
            template: Custom template string (uses DEFAULT_TEMPLATE if not provided)
        """
        self.template = template or self.DEFAULT_TEMPLATE
    
    def format(
        self,
        context: str,
        question: str,
        **kwargs
    ) -> str:
        """
        Format the prompt template with context and question.
        
        Args:
            context: Formatted context string from retrieved documents
            question: User's question
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        return self.template.format(
            context=context,
            question=question,
            **kwargs
        )
    
    @classmethod
    def get_default(cls) -> 'PromptTemplate':
        """Get default prompt template."""
        return cls(cls.DEFAULT_TEMPLATE)
    
    @classmethod
    def get_detailed(cls) -> 'PromptTemplate':
        """Get detailed prompt template with more instructions."""
        return cls(cls.DETAILED_TEMPLATE)
    
    @classmethod
    def get_concise(cls) -> 'PromptTemplate':
        """Get concise prompt template."""
        return cls(cls.CONCISE_TEMPLATE)

