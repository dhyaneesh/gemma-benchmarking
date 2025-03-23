import argparse
import json
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logger
from src.benchmarks.runner import BenchmarkRunner

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gemma Models Benchmarking Suite")
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(Path("configs") / "default.json"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Directory to save results (overrides config)"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+",
        help="Models to benchmark (overrides config)"
    )
    parser.add_argument(
        "--benchmarks", 
        type=str, 
        nargs="+",
        help="Benchmarks to run (overrides config)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        config_path = Path(config_path)  # Convert string path to Path object
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Remove huggingface token configuration if it exists
        if "huggingface" in config:
            del config["huggingface"]
            
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def main():
    """Main entry point for benchmarking suite."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.models:
        # Convert flat list to the structured format in config
        config["models"] = {"custom": {model: {"name": model} for model in args.models}}
    if args.benchmarks:
        config["benchmarks"] = {benchmark: {} for benchmark in args.benchmarks}
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("gemma_benchmark", level=log_level)
    logger.info("Starting Gemma benchmarking suite")
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize and run benchmarks
    runner = BenchmarkRunner(config)
    runner.run_benchmarks()
    
    logger.info("Benchmarking complete")

if __name__ == "__main__":
    main()