# scripts/model_example.py
import sys
import os
import argparse
import logging
import json
from pathlib import Path

# Add the project root to path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import ModelFactory
from src.utils.huggingface_auth import setup_huggingface_auth, is_authenticated

def setup_logging(log_level="INFO"):
    """Set up basic logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Ensure logs directory exists
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(logs_dir, "model_test.log"))
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Test model loading and inference")
    parser.add_argument("--config", type=str, default=os.path.join(project_root, "configs", "default.json"),
                        help="Path to configuration file")
    parser.add_argument("--model_name", type=str, help="Override model name from config")
    parser.add_argument("--model_type", type=str, help="Override model type from config")
    parser.add_argument("--prompt", type=str, default="Explain what makes language models effective for NLP tasks.",
                        help="Prompt to test with the model")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--hf_token", type=str, help="Hugging Face API token")
    parser.add_argument("--hf_token_path", type=str, help="Path to file containing Hugging Face API token")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return 1
    
    # Handle Hugging Face authentication
    hf_token = args.hf_token
    hf_token_path = args.hf_token_path or config.get("huggingface", {}).get("token_path")
    
    if config.get("huggingface", {}).get("use_auth", False):
        if not is_authenticated():
            logger.info("Setting up Hugging Face authentication")
            auth_success = setup_huggingface_auth(token=hf_token, token_path=hf_token_path)
            if not auth_success:
                logger.warning("Failed to authenticate with Hugging Face. Some models may not load correctly.")
        else:
            logger.info("Already authenticated with Hugging Face")
    
    # Determine which model to use
    if args.model_name and args.model_type:
        # Use command line arguments
        model_key = args.model_name
        model_type = args.model_type
        model_config = {
            "name": model_key,
            "type": model_type,
            "cache_dir": config.get("cache_dir", "model_cache"),
            "hf_token": hf_token,
            "hf_token_path": hf_token_path
        }
    else:
        # Use first model from config
        model_family = next(iter(config["models"]))
        model_key = next(iter(config["models"][model_family]))
        model_config = config["models"][model_family][model_key].copy()
        model_config["cache_dir"] = config.get("cache_dir", "model_cache")
        model_config["hf_token"] = hf_token
        model_config["hf_token_path"] = hf_token_path
    
    # Create and test the model
    try:
        logger.info(f"Creating model: {model_config['name']} of type {model_config['type']}")
        model = ModelFactory.create_model(model_config)
        
        logger.info("Loading model...")
        success = model.load()
        if not success:
            logger.error("Failed to load model")
            return 1
        
        # Get memory usage
        memory_stats = model.get_memory_usage()
        logger.info(f"Model memory usage: {memory_stats}")
        
        # Run inference with the test prompt
        logger.info(f"Running inference with prompt: '{args.prompt}'")
        results = model.generate(
            [args.prompt], 
            max_new_tokens=config.get("inference", {}).get("max_new_tokens", 100), 
            temperature=config.get("inference", {}).get("temperature", 0.7)
        )
        
        logger.info("Generation results:")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}:\n{result}")
        
        # Unload the model
        logger.info("Unloading model...")
        model.unload()
        logger.info("Done")
        
    except Exception as e:
        logger.error(f"Error during model testing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())