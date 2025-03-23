from src.models.huggingface_model import HuggingFaceModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MistralModel(HuggingFaceModel):
    """Wrapper specifically for Mistral models."""
    
    def __init__(
        self, 
        model_name: str, 
        cache_dir: str = None, 
        device: str = None,
        **kwargs
    ):
        """
        Initialize a Mistral model.
        
        Args:
            model_name: Model size (e.g., "mistral-7b")
            cache_dir: Directory to store downloaded models
            device: Device to put the model on
            **kwargs: Additional model-specific arguments
        """
        # Map model name to HF model ID
        model_map = {
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1"
        }
        
        if model_name in model_map:
            hf_model_name = model_map[model_name]
        else:
            # If not in map, assume it's a full HF model ID
            hf_model_name = model_name
            
        super().__init__(hf_model_name, cache_dir, device, **kwargs)
        self.original_model_name = model_name
        
    def format_prompt(self, prompt: str) -> str:
        """Format the prompt according to Mistral's expected format."""
        return f"{prompt}"
