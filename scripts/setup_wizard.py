import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import rich
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/setup.log")
        ]
    )

def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    console.print("\n[bold]Checking prerequisites...[/bold]")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        console.print("[red]❌ Python 3.8 or higher is required[/red]")
        return False
        
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            console.print("[green]✓ CUDA is available[/green]")
        else:
            console.print("[yellow]⚠ CUDA is not available. GPU acceleration will not be available.[/yellow]")
    except ImportError:
        console.print("[yellow]⚠ PyTorch not installed. GPU acceleration will not be available.[/yellow]")
    
    # Check HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        console.print("[yellow]⚠ HuggingFace token not found in environment variables[/yellow]")
    
    return True

def setup_environment() -> bool:
    """Set up the Python environment."""
    console.print("\n[bold]Setting up environment...[/bold]")
    
    # Check if conda is available
    conda_available = os.system("conda --version") == 0
    
    if conda_available:
        console.print("Conda is available. Would you like to use conda for environment setup?")
        use_conda = Confirm.ask("Use conda?", default=True)
        
        if use_conda:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating conda environment...", total=None)
                result = os.system("conda env create -f environment.yml")
                progress.update(task, completed=True)
                
            if result != 0:
                console.print("[red]❌ Failed to create conda environment[/red]")
                return False
                
            console.print("[green]✓ Conda environment created successfully[/green]")
            return True
    
    # Fallback to venv
    console.print("Using Python venv for environment setup...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating virtual environment...", total=None)
        result = os.system("python -m venv venv")
        progress.update(task, completed=True)
    
    if result != 0:
        console.print("[red]❌ Failed to create virtual environment[/red]")
        return False
    
    # Install requirements
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Installing dependencies...", total=None)
        result = os.system("pip install -r requirements.txt")
        progress.update(task, completed=True)
    
    if result != 0:
        console.print("[red]❌ Failed to install dependencies[/red]")
        return False
    
    console.print("[green]✓ Environment setup completed successfully[/green]")
    return True

def configure_models() -> Dict:
    """Configure model settings."""
    console.print("\n[bold]Configuring models...[/bold]")
    
    models_config = {
        "models": {
            "gemma": {},
            "mistral": {}
        }
    }
    
    # Configure Gemma models
    if Confirm.ask("Would you like to benchmark Gemma models?", default=True):
        console.print("\nAvailable Gemma models:")
        console.print("1. gemma-2b")
        console.print("2. gemma-7b")
        
        model_choice = Prompt.ask(
            "Select models to benchmark (comma-separated numbers)",
            default="1"
        )
        
        for choice in model_choice.split(","):
            choice = choice.strip()
            if choice == "1":
                models_config["models"]["gemma"]["gemma-2b"] = {
                    "type": "gemma",
                    "name": "google/gemma-2b",
                    "torch_dtype": "float16",
                    "low_memory": False,
                    "requires_auth": True
                }
            elif choice == "2":
                models_config["models"]["gemma"]["gemma-7b"] = {
                    "type": "gemma",
                    "name": "google/gemma-7b",
                    "torch_dtype": "float16",
                    "low_memory": True,
                    "requires_auth": True
                }
    
    # Configure Mistral models
    if Confirm.ask("Would you like to benchmark Mistral models?", default=True):
        models_config["models"]["mistral"]["mistral-7b"] = {
            "type": "mistral",
            "name": "mistralai/Mistral-7B-v0.1",
            "torch_dtype": "float16",
            "low_memory": True,
            "requires_auth": False
        }
    
    return models_config

def configure_benchmarks() -> Dict:
    """Configure benchmark settings."""
    console.print("\n[bold]Configuring benchmarks...[/bold]")
    
    benchmarks_config = {
        "benchmarks": {}
    }
    
    # Configure MMLU
    if Confirm.ask("Would you like to run the MMLU benchmark?", default=True):
        num_shots = Prompt.ask(
            "Number of few-shot examples per subject",
            default="5"
        )
        benchmarks_config["benchmarks"]["mmlu"] = {
            "subjects": ["all"],
            "num_shots": int(num_shots)
        }
    
    # Configure GSM8K
    if Confirm.ask("Would you like to run the GSM8K benchmark?", default=True):
        num_shots = Prompt.ask(
            "Number of few-shot examples",
            default="3"
        )
        benchmarks_config["benchmarks"]["gsm8k"] = {
            "num_shots": int(num_shots),
            "split": "test"
        }
    
    return benchmarks_config

def save_config(config: Dict, output_dir: str = "configs"):
    """Save the configuration to a file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / "custom.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    console.print(f"\n[green]✓ Configuration saved to {config_file}[/green]")
    return config_file

def main():
    """Main entry point for the setup wizard."""
    parser = argparse.ArgumentParser(description="Setup wizard for LLM Benchmarking Suite")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs",
        help="Directory to save configuration"
    )
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Display welcome message
        console.print(Panel.fit(
            "[bold blue]Welcome to the LLM Benchmarking Suite Setup Wizard[/bold blue]\n"
            "This wizard will help you set up and configure the benchmarking suite.",
            title="Setup Wizard"
        ))
        
        # Check prerequisites
        if not check_prerequisites():
            console.print("\n[red]❌ Prerequisites check failed. Please fix the issues and try again.[/red]")
            return
        
        # Set up environment
        if not setup_environment():
            console.print("\n[red]❌ Environment setup failed. Please fix the issues and try again.[/red]")
            return
        
        # Configure models
        models_config = configure_models()
        
        # Configure benchmarks
        benchmarks_config = configure_benchmarks()
        
        # Combine configurations
        config = {
            "output_dir": "results",
            "cache_dir": "model_cache",
            "log_level": "INFO",
            "seed": 42,
            "huggingface": {
                "token_path": ".hf_token",
                "use_auth": True
            },
            **models_config,
            **benchmarks_config,
            "inference": {
                "batch_size": 4,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            },
            "report_format": ["json", "md"]
        }
        
        # Save configuration
        config_file = save_config(config, args.output_dir)
        
        # Display next steps
        console.print("\n[bold green]Setup completed successfully![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Activate your environment:")
        console.print("   - If using conda: [code]conda activate gemma-benchmark[/code]")
        console.print("   - If using venv: [code]source venv/bin/activate[/code] (Unix) or [code]venv\\Scripts\\activate[/code] (Windows)")
        console.print("\n2. Run the benchmarks:")
        console.print(f"   [code]python src/main.py --config {config_file}[/code]")
        console.print("\n3. Generate reports:")
        console.print("   [code]python scripts/generate_report.py[/code]")
        
    except Exception as e:
        logger.error(f"Setup wizard failed: {e}")
        console.print(f"\n[red]❌ Setup wizard failed: {e}[/red]")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 