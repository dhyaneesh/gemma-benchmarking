from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Any, Optional, Union
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base abstract class for all model implementations."""
    
    def __init__(self, model_name: str, cache_dir: str = None, device: str = None, **kwargs):
        """
        Initialize the base model.
        
        Args:
            model_name: Name or path of the model
            cache_dir: Directory to store downloaded models
            device: Device to put the model on ('cpu', 'cuda', 'cuda:0', etc.)
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing model {model_name} on {self.device}")
        
        # These will be populated by child classes
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for the given prompts."""
        pass
    
    def prepare_inputs(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model inference."""
        pass
    
    def tokenize(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize the input text."""
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get the current memory usage of the model."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        memory_stats = {}
        if hasattr(torch.cuda, 'memory_allocated') and torch.cuda.is_available():
            memory_stats['allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_stats['reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
        
        return memory_stats
    
    def unload(self):
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model {self.model_name} unloaded from memory")
