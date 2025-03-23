from src.models.huggingface_model import HuggingFaceModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GemmaModel(HuggingFaceModel):
    """Wrapper specifically for Gemma models."""
    
    def __init__(
        self, 
        model_name: str, 
        cache_dir: str = None, 
        device: str = None,
        **kwargs
    ):
        """
        Initialize a Gemma model.
        
        Args:
            model_name: Model size (e.g., "gemma-2b", "gemma-7b")
            cache_dir: Directory to store downloaded models
            device: Device to put the model on
            **kwargs: Additional model-specific arguments
        """
        # Map model name to HF model ID
        model_map = {
            "gemma-2b": "google/gemma-2b",
            "gemma-2b-it": "google/gemma-2b-it",
            "gemma-7b": "google/gemma-7b",
            "gemma-7b-it": "google/gemma-7b-it"
        }
        
        if model_name in model_map:
            hf_model_name = model_map[model_name]
        else:
            # If not in map, assume it's a full HF model ID
            hf_model_name = model_name
            
        super().__init__(hf_model_name, cache_dir, device, **kwargs)
        self.original_model_name = model_name
        
    def format_prompt(self, prompt: str) -> str:
        """Format the prompt according to Gemma's expected format."""
        return f"{prompt}"
