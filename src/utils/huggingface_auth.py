import os
import logging
from huggingface_hub import login, HfFolder
from typing import Optional

logger = logging.getLogger(__name__)

def setup_huggingface_auth(token: Optional[str] = None, token_path: Optional[str] = None) -> bool:
    """
    Set up Hugging Face authentication.
    
    Args:
        token: Hugging Face API token. If None, will try to get from env var or token path.
        token_path: Path to a file containing the Hugging Face token. If None, uses default ~/.huggingface/token.
        
    Returns:
        bool: True if authentication was successful, False otherwise.
    """
    try:
        # Try to get token from parameters
        hf_token = token
        
        # If not provided, try environment variable
        if hf_token is None:
            hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            if hf_token:
                logger.info("Using Hugging Face token from environment variable")
        
        # If still not found, try token path
        if hf_token is None and token_path:
            try:
                with open(token_path, 'r') as f:
                    hf_token = f.read().strip()
                logger.info(f"Using Hugging Face token from file: {token_path}")
            except Exception as e:
                logger.warning(f"Failed to read token from {token_path}: {str(e)}")
        
        # If still not found, check if already logged in
        if hf_token is None:
            if HfFolder.get_token() is not None:
                logger.info("Already logged in to Hugging Face")
                return True
            else:
                logger.warning("No Hugging Face token provided and not already logged in")
                return False
        
        # Login with the token
        login(token=hf_token, add_to_git_credential=False)
        logger.info("Successfully authenticated with Hugging Face")
        return True
        
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {str(e)}")
        return False


def is_authenticated() -> bool:
    """
    Check if already authenticated with Hugging Face.
    
    Returns:
        bool: True if authenticated, False otherwise.
    """
    return HfFolder.get_token() is not None
