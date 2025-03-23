# src/models/model_factory.py
import logging
from typing import Dict, Any, Optional
import os

from src.models.gemma_model import GemmaModel
from src.models.mistral_model import MistralModel
from src.models.huggingface_model import HuggingFaceModel
from src.utils.huggingface_auth import setup_huggingface_auth, is_authenticated

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating model instances based on model type."""
    
    @staticmethod
    def create_model(model_config: Dict[str, Any]):
        """
        Create and return a model instance based on configuration.
        
        Args:
            model_config: Dictionary containing model configuration
                Required keys:
                - 'name': Model name/identifier
                - 'type': Model type (e.g., 'gemma', 'mistral', 'huggingface')
                
                Optional keys:
                - 'cache_dir': Directory to cache model files
                - 'device': Device to load model on
                - 'hf_token': Hugging Face token for authentication
                - 'hf_token_path': Path to a file containing the Hugging Face token
                - Any other model-specific parameters
                
        Returns:
            An instance of the appropriate model class
        """
        model_name = model_config.get('name')
        model_type = model_config.get('type', '').lower()
        
        if not model_name:
            raise ValueError("Model name must be specified in configuration")
        
        # Extract common parameters
        cache_dir = model_config.get('cache_dir')
        device = model_config.get('device')
        
        # Handle Hugging Face authentication if needed
        # Models like Gemma require authentication
        if 'gemma' in model_type or model_config.get('requires_auth', False):
            hf_token = model_config.get('hf_token')
            hf_token_path = model_config.get('hf_token_path')
            
            if not is_authenticated():
                auth_success = setup_huggingface_auth(token=hf_token, token_path=hf_token_path)
                if not auth_success:
                    logger.warning(f"Failed to authenticate with Hugging Face. Model {model_name} may not load correctly.")
        
        # Remove known keys to get model-specific parameters
        model_params = model_config.copy()
        for key in ['name', 'type', 'cache_dir', 'device', 'hf_token', 'hf_token_path', 'requires_auth']:
            if key in model_params:
                del model_params[key]
        
        # Create the appropriate model based on type
        if 'gemma' in model_type:
            logger.info(f"Creating Gemma model: {model_name}")
            return GemmaModel(model_name, cache_dir, device, **model_params)
        
        elif 'mistral' in model_type:
            logger.info(f"Creating Mistral model: {model_name}")
            return MistralModel(model_name, cache_dir, device, **model_params)
        
        else:
            # Default to generic HuggingFace model
            logger.info(f"Creating generic HuggingFace model: {model_name}")
            return HuggingFaceModel(model_name, cache_dir, device, **model_params)