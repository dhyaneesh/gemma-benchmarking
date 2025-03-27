import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

from .mmlu import MMLUBenchmark
from .gsm8k import GSM8KBenchmark
from src.models.model_factory import ModelFactory

class BenchmarkRunner:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.model_factory = ModelFactory()
        
    def _initialize_model(self, model_config: Dict) -> Optional[object]:
        """Initialize a model from configuration."""
        try:
            return self.model_factory.create_model(
                model_type=model_config["type"],
                model_name=model_config["name"],
                torch_dtype=model_config["torch_dtype"],
                low_memory=model_config.get("low_memory", False),
                requires_auth=model_config.get("requires_auth", True)
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize model {model_config['name']}: {e}")
            return None
            
    def run_mmlu_benchmark(self, model: object, model_config: Dict) -> Optional[Dict]:
        """Run MMLU benchmark for a specific model."""
        try:
            mmlu_config = self.config["benchmarks"]["mmlu"]
            benchmark = MMLUBenchmark(mmlu_config)
            results = benchmark.run(model)
            return results
        except Exception as e:
            self.logger.error(f"Failed to run MMLU benchmark: {e}")
            return None
            
    def run_gsm8k_benchmark(self, model: object, model_config: Dict) -> Optional[Dict]:
        """Run GSM8K benchmark for a specific model."""
        try:
            gsm8k_config = self.config["benchmarks"]["gsm8k"]
            benchmark = GSM8KBenchmark(gsm8k_config)
            results = benchmark.run(model)
            return results
        except Exception as e:
            self.logger.error(f"Failed to run GSM8K benchmark: {e}")
            return None
    
    def save_results(self, model_name: str, benchmark_name: str, results: Dict):
        """Save benchmark results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{model_name}_{benchmark_name}_{timestamp}.json"
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def run_benchmarks(self):
        """Run all configured benchmarks for all models."""
        for model_category, models in self.config["models"].items():
            for model_name, model_config in models.items():
                self.logger.info(f"Running benchmarks for {model_name}")
                
                # Initialize model
                model = self._initialize_model(model_config)
                if model is None:
                    continue
                
                try:
                    # Run MMLU benchmark if configured
                    if "mmlu" in self.config["benchmarks"]:
                        self.logger.info(f"Running MMLU benchmark for {model_name}")
                        results = self.run_mmlu_benchmark(model, model_config)
                        if results:
                            self.save_results(model_name, "mmlu", results)
                    
                    # Run GSM8K benchmark if configured
                    if "gsm8k" in self.config["benchmarks"]:
                        self.logger.info(f"Running GSM8K benchmark for {model_name}")
                        results = self.run_gsm8k_benchmark(model, model_config)
                        if results:
                            self.save_results(model_name, "gsm8k", results)
                            
                except Exception as e:
                    self.logger.error(f"Error running benchmarks for {model_name}: {e}")
                finally:
                    # Clean up model resources
                    if hasattr(model, "cleanup"):
                        model.cleanup() 