"""
Generator module for RAG pipeline.

This module provides functionality for generating answers using Large Language Models
based on retrieved context and user questions.
"""

from typing import Optional, Dict, Any
import warnings

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    from transformers import AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None

try:
    from langchain.llms import HuggingFacePipeline
    from langchain import PromptTemplate as LangChainPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    HuggingFacePipeline = None
    LangChainPromptTemplate = None

from src.exceptions import DataProcessingError
from src.utils.logger import get_logger


class RAGGenerator:
    """
    A class for generating answers using LLMs in the RAG pipeline.
    
    This class combines the prompt, user question, and retrieved chunks,
    then sends them to an LLM to generate a response.
    
    Attributes:
        model_name: Name of the LLM model
        model: Loaded model or pipeline
        tokenizer: Tokenizer for the model
        logger: Logger instance
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        use_pipeline: bool = True
    ):
        """
        Initialize the RAG Generator.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run on ('cpu', 'cuda', or None for auto)
            max_length: Maximum length of generated text
            temperature: Sampling temperature (lower = more deterministic)
            use_pipeline: Whether to use Hugging Face pipeline (simpler)
            
        Raises:
            DataProcessingError: If transformers is not available
        """
        if not TRANSFORMERS_AVAILABLE:
            raise DataProcessingError(
                "transformers is not available. Install it with: pip install transformers",
                details={"model_name": model_name}
            )
        
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.use_pipeline = use_pipeline
        self.logger = get_logger(__name__)
        
        # Determine device
        if device is None:
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Load model
        try:
            self.logger.info(f"Loading LLM: {model_name}")
            self.logger.info(f"Device: {self.device}")
            
            if use_pipeline:
                # Use Hugging Face pipeline (simpler, works with many models)
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=model_name,
                    device=0 if self.device == 'cuda' else -1,
                    max_length=max_length,
                    temperature=temperature
                )
                self.model = None
                self.tokenizer = None
            else:
                # Load model and tokenizer separately (more control)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Try to determine if it's a seq2seq or causal model
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                except:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
                self.model.to(self.device)
                self.pipeline = None
            
            self.logger.info("✅ LLM loaded successfully")
            
        except Exception as e:
            error_msg = f"Error loading LLM: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(
                error_msg,
                details={"model_name": model_name, "device": self.device}
            )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True
    ) -> str:
        """
        Generate an answer from a formatted prompt.
        
        Args:
            prompt: Formatted prompt string (context + question)
            max_new_tokens: Maximum number of tokens to generate (overrides max_length)
            temperature: Sampling temperature (overrides default)
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            Generated answer string
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            if self.use_pipeline and self.pipeline:
                # Use pipeline
                result = self.pipeline(
                    prompt,
                    max_length=max_new_tokens or self.max_length,
                    temperature=temperature or self.temperature,
                    do_sample=do_sample,
                    num_return_sequences=1
                )
                
                # Extract generated text
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    # Remove the prompt from the beginning if it's included
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    return generated_text
                else:
                    return str(result)
            
            else:
                # Use model directly
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens or (self.max_length - inputs['input_ids'].shape[1]),
                        temperature=temperature or self.temperature,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove prompt if included
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
                
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg, details={"prompt_length": len(prompt)})
    
    def generate_with_retries(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Generate answer with retry logic for robustness.
        
        Args:
            prompt: Formatted prompt string
            max_retries: Maximum number of retry attempts
            **kwargs: Additional arguments for generate()
            
        Returns:
            Generated answer string
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    raise DataProcessingError(
                        f"Failed to generate answer after {max_retries} attempts: {str(last_error)}"
                    )
        
        # Should not reach here, but just in case
        raise DataProcessingError("Failed to generate answer")

