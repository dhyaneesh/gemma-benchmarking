from abc import ABC, abstractmethod
from typing import Dict

from src.models.base_model import BaseModel

class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, config: Dict):
        """Initialize the benchmark with configuration."""
        self.config = config
        
    @abstractmethod
    def run(self, model: BaseModel) -> Dict:
        """Run the benchmark on the given model.
        
        Args:
            model: The model to evaluate
            
        Returns:
            Dict containing benchmark results
        """
        pass
        
    def save_results(self, results: Dict, output_dir: str) -> None:
        """Save benchmark results to disk.
        
        Args:
            results: The benchmark results to save
            output_dir: Directory to save results to
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results in JSON format
        results_file = output_path / f"{self.__class__.__name__.lower()}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
    def load_results(self, results_file: str) -> Dict:
        """Load benchmark results from disk.
        
        Args:
            results_file: Path to the results file
            
        Returns:
            Dict containing the loaded results
        """
        import json
        
        with open(results_file, "r") as f:
            return json.load(f) 