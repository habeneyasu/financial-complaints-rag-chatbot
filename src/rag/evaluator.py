"""
Evaluation module for RAG pipeline.

This module provides functionality for qualitative evaluation of the RAG system
by testing it with representative questions and analyzing the results.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

from src.rag.pipeline import RAGPipeline
from src.utils.logger import get_logger


class RAGEvaluator:
    """
    A class for evaluating the RAG pipeline with qualitative assessment.
    
    This class provides methods to:
    - Run evaluation on a set of questions
    - Generate evaluation tables
    - Analyze results and provide insights
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the RAG Evaluator.
        
        Args:
            rag_pipeline: Initialized RAGPipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.logger = get_logger(__name__)
        self.evaluation_results = []
    
    def evaluate_question(
        self,
        question: str,
        expected_answer: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline on a single question.
        
        Args:
            question: Question to evaluate
            expected_answer: Optional expected answer (for reference)
            metadata_filter: Optional metadata filter for retrieval
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating question: '{question[:50]}...'")
        
        # Get answer from RAG pipeline
        result = self.rag_pipeline.answer(
            question=question,
            return_sources=True,
            metadata_filter=metadata_filter
        )
        
        # Format evaluation result
        evaluation = {
            "question": question,
            "generated_answer": result["answer"],
            "retrieved_sources": result.get("sources", [])[:2],  # Top 2 sources
            "num_sources": len(result.get("sources", [])),
            "expected_answer": expected_answer,
            "context_used": result.get("context", "")[:500]  # First 500 chars
        }
        
        return evaluation
    
    def evaluate_questions(
        self,
        questions: List[Dict[str, Any]],
        auto_score: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the RAG pipeline on multiple questions.
        
        Args:
            questions: List of question dictionaries, each with:
                - question: The question string
                - expected_answer: Optional expected answer
                - metadata_filter: Optional metadata filter
            auto_score: Whether to attempt automatic scoring (default: False, manual)
            
        Returns:
            List of evaluation results
        """
        self.logger.info(f"Evaluating {len(questions)} questions")
        
        results = []
        for i, q_dict in enumerate(questions, 1):
            self.logger.info(f"\n[{i}/{len(questions)}] Processing question...")
            
            evaluation = self.evaluate_question(
                question=q_dict["question"],
                expected_answer=q_dict.get("expected_answer"),
                metadata_filter=q_dict.get("metadata_filter")
            )
            
            # Add manual scoring placeholders
            evaluation["quality_score"] = None  # To be filled manually
            evaluation["comments"] = ""  # To be filled manually
            
            results.append(evaluation)
        
        self.evaluation_results = results
        return results
    
    def create_evaluation_table(self, results: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Create an evaluation table from results.
        
        Args:
            results: Evaluation results (uses self.evaluation_results if not provided)
            
        Returns:
            DataFrame with evaluation table
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            raise ValueError("No evaluation results available")
        
        # Prepare table data
        table_data = []
        for i, result in enumerate(results, 1):
            # Format sources for display
            sources_text = ""
            if result.get("retrieved_sources"):
                for j, source in enumerate(result["retrieved_sources"][:2], 1):
                    product = source.get("metadata", {}).get("product", "Unknown")
                    text_preview = source.get("text", "")[:100]
                    sources_text += f"Source {j} ({product}): {text_preview}...\n"
            
            table_data.append({
                "Question": result["question"],
                "Generated Answer": result["generated_answer"][:300] + "..." if len(result["generated_answer"]) > 300 else result["generated_answer"],
                "Retrieved Sources": sources_text.strip(),
                "Quality Score": result.get("quality_score", "N/A"),
                "Comments/Analysis": result.get("comments", "")
            })
        
        return pd.DataFrame(table_data)
    
    def generate_markdown_report(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a markdown evaluation report.
        
        Args:
            results: Evaluation results (uses self.evaluation_results if not provided)
            output_path: Optional path to save report
            
        Returns:
            Markdown report string
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            raise ValueError("No evaluation results available")
        
        # Generate markdown
        markdown = "# RAG Pipeline Evaluation Report\n\n"
        markdown += "## Overview\n\n"
        markdown += f"This report evaluates the RAG pipeline on {len(results)} representative questions.\n\n"
        
        markdown += "## Evaluation Table\n\n"
        markdown += "| Question | Generated Answer | Retrieved Sources | Quality Score | Comments/Analysis |\n"
        markdown += "|----------|------------------|------------------|---------------|-------------------|\n"
        
        for result in results:
            question = result["question"].replace("|", "\\|")
            answer = result["generated_answer"][:200].replace("|", "\\|")
            
            # Format sources
            sources_text = ""
            if result.get("retrieved_sources"):
                for j, source in enumerate(result["retrieved_sources"][:2], 1):
                    product = source.get("metadata", {}).get("product", "Unknown")
                    sources_text += f"**Source {j}** ({product}): {source.get('text', '')[:80]}...<br>"
            
            quality_score = result.get("quality_score", "N/A")
            comments = result.get("comments", "").replace("|", "\\|")
            
            markdown += f"| {question} | {answer}... | {sources_text} | {quality_score} | {comments} |\n"
        
        markdown += "\n## Analysis\n\n"
        markdown += "### What Worked Well\n\n"
        markdown += "- [To be filled based on evaluation]\n\n"
        markdown += "### Areas for Improvement\n\n"
        markdown += "- [To be filled based on evaluation]\n\n"
        markdown += "### Recommendations\n\n"
        markdown += "- [To be filled based on evaluation]\n\n"
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown)
            self.logger.info(f"Report saved to: {output_path}")
        
        return markdown

