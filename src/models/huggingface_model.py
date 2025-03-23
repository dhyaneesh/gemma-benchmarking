import torch
from typing import List, Dict, Any, Optional, Union
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class HuggingFaceModel(BaseModel):
    """Wrapper for HuggingFace transformer models."""
    
    def __init__(
        self, 
        model_name: str, 
        cache_dir: str = None, 
        device: str = None,
        torch_dtype: torch.dtype = None,
        low_memory: bool = False,
        **kwargs
    ):
        """
        Initialize a HuggingFace model.
        
        Args:
            model_name: Name or path of the model
            cache_dir: Directory to store downloaded models
            device: Device to put the model on ('cpu', 'cuda', 'cuda:0', etc.)
            torch_dtype: Data type for model weights (float16, bfloat16, etc.)
            low_memory: Whether to use optimization for low memory environments
            **kwargs: Additional model-specific arguments
        """
        super().__init__(model_name, cache_dir, device)
        
        # Set torch dtype for model loading
        if torch_dtype is None:
            if torch.cuda.is_available():
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
            
        self.low_memory = low_memory
        self.kwargs = kwargs
        
    def load(self):
        """Load the model and tokenizer from HuggingFace."""
        logger.info(f"Loading model {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True
        )
        
        # Set tokenizer padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device if self.low_memory else None,
        }
        
        # Update with additional kwargs
        model_kwargs.update(self.kwargs)
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if not self.low_memory:
                self.model.to(self.device)
                
            logger.info(f"Successfully loaded {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            return False
    
    def tokenize(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize the input text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load() first.")
        
        # Handle both single strings and lists of strings
        is_single_text = isinstance(text, str)
        if is_single_text:
            text = [text]
            
        # Tokenize with padding to the max length in batch
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        return encoded
    
    def prepare_inputs(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model inference."""
        return self.tokenize(prompts)
    
    def generate(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 256, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return per prompt
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated text responses
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded. Call load() first.")
        
        # Prepare inputs
        inputs = self.prepare_inputs(prompts)
        
        # Set up generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate outputs
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode the generated token IDs to text
            decoded_outputs = self.tokenizer.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )
            
            # Process the outputs based on num_return_sequences
            results = []
            if num_return_sequences == 1:
                # Simple case: one output per prompt
                for i, output in enumerate(decoded_outputs):
                    prompt_len = len(self.tokenizer.decode(inputs.input_ids[i], skip_special_tokens=True))
                    results.append(output[prompt_len:].strip())
            else:
                # Multiple outputs per prompt
                for i in range(0, len(decoded_outputs), num_return_sequences):
                    prompt_outputs = decoded_outputs[i:i+num_return_sequences]
                    prompt_len = len(self.tokenizer.decode(
                        inputs.input_ids[i//num_return_sequences], 
                        skip_special_tokens=True
                    ))
                    # Extract only the newly generated text for each output
                    prompt_results = [output[prompt_len:].strip() for output in prompt_outputs]
                    results.extend(prompt_results)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return ["Error: " + str(e)] * len(prompts)
