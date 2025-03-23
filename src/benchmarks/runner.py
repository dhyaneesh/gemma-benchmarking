import json
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

from .mmlu import MMLUBenchmark

class BenchmarkRunner:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def run_mmlu_benchmark(self, model_config: Dict) -> Dict:
        """Run MMLU benchmark for a specific model."""
        mmlu_config = self.config["benchmarks"]["mmlu"]
        
        benchmark = MMLUBenchmark(
            model_name=model_config["name"],
            model_type=model_config["type"],
            token_path=self.config["huggingface"]["token_path"],
            torch_dtype=model_config["torch_dtype"],
            low_memory=model_config.get("low_memory", False),
            requires_auth=model_config.get("requires_auth", True)
        )
        
        results = benchmark.run_benchmark(
            subjects=mmlu_config["subjects"],
            num_shots=mmlu_config["num_shots"]
        )
        
        return results
    
    def save_results(self, model_name: str, benchmark_name: str, results: Dict):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{model_name}_{benchmark_name}_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def run_benchmarks(self):
        """Run all configured benchmarks for all models."""
        for model_category, models in self.config["models"].items():
            for model_name, model_config in models.items():
                self.logger.info(f"Running benchmarks for {model_name}")
                
                if "mmlu" in self.config["benchmarks"]:
                    self.logger.info(f"Running MMLU benchmark for {model_name}")
                    results = self.run_mmlu_benchmark(model_config)
                    self.save_results(model_name, "mmlu", results)
                
                # Add other benchmark types here as needed 