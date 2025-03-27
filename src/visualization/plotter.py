import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class BenchmarkPlotter:
    """Class for visualizing benchmark results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_results(self, results_file: str) -> Dict:
        """Load results from a JSON file."""
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results from {results_file}: {e}")
            return {}
            
    def plot_mmlu_results(self, results_files: List[str], output_name: Optional[str] = None):
        """Plot MMLU benchmark results."""
        results = []
        for file in results_files:
            data = self.load_results(file)
            if not data:
                continue
                
            model_name = Path(file).stem.split("_")[0]
            for subject, score in data.items():
                results.append({
                    "Model": model_name,
                    "Subject": subject,
                    "Score": score
                })
                
        if not results:
            logger.warning("No valid MMLU results to plot")
            return
            
        df = pd.DataFrame(results)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Create bar plot
        sns.barplot(data=df, x="Subject", y="Score", hue="Model")
        plt.xticks(rotation=45, ha="right")
        plt.title("MMLU Benchmark Results by Subject")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        
        # Save plot
        if output_name is None:
            output_name = f"mmlu_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        plt.savefig(self.output_dir / f"{output_name}.png")
        plt.close()
        
    def plot_gsm8k_results(self, results_files: List[str], output_name: Optional[str] = None):
        """Plot GSM8K benchmark results."""
        results = []
        for file in results_files:
            data = self.load_results(file)
            if not data:
                continue
                
            model_name = Path(file).stem.split("_")[0]
            results.append({
                "Model": model_name,
                "Accuracy": data["accuracy"],
                "Total Examples": data["total_examples"],
                "Correct": data["total_correct"]
            })
            
        if not results:
            logger.warning("No valid GSM8K results to plot")
            return
            
        df = pd.DataFrame(results)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot accuracy
        sns.barplot(data=df, x="Model", y="Accuracy", ax=ax1)
        ax1.set_title("GSM8K Benchmark Results")
        ax1.set_ylabel("Accuracy")
        ax1.tick_params(axis="x", rotation=45)
        
        # Plot total examples and correct answers
        df_melted = df.melt(id_vars=["Model"], value_vars=["Total Examples", "Correct"])
        sns.barplot(data=df_melted, x="Model", y="value", hue="variable", ax=ax2)
        ax2.set_title("GSM8K Examples Processed")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if output_name is None:
            output_name = f"gsm8k_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        plt.savefig(self.output_dir / f"{output_name}.png")
        plt.close()
        
    def generate_summary_report(self, results_dir: str, output_name: Optional[str] = None):
        """Generate a summary report of all benchmark results."""
        results_dir = Path(results_dir)
        if not results_dir.exists():
            logger.error(f"Results directory {results_dir} does not exist")
            return
            
        # Collect all results
        mmlu_files = list(results_dir.glob("*_mmlu_*.json"))
        gsm8k_files = list(results_dir.glob("*_gsm8k_*.json"))
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {}
        }
        
        # Process MMLU results
        if mmlu_files:
            mmlu_results = {}
            for file in mmlu_files:
                data = self.load_results(str(file))
                if data:
                    model_name = file.stem.split("_")[0]
                    mmlu_results[model_name] = data
            summary["benchmarks"]["mmlu"] = mmlu_results
            
        # Process GSM8K results
        if gsm8k_files:
            gsm8k_results = {}
            for file in gsm8k_files:
                data = self.load_results(str(file))
                if data:
                    model_name = file.stem.split("_")[0]
                    gsm8k_results[model_name] = {
                        "accuracy": data["accuracy"],
                        "total_examples": data["total_examples"],
                        "total_correct": data["total_correct"]
                    }
            summary["benchmarks"]["gsm8k"] = gsm8k_results
            
        # Save summary
        if output_name is None:
            output_name = f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(self.output_dir / f"{output_name}.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        # Generate plots
        if mmlu_files:
            self.plot_mmlu_results([str(f) for f in mmlu_files], f"{output_name}_mmlu")
        if gsm8k_files:
            self.plot_gsm8k_results([str(f) for f in gsm8k_files], f"{output_name}_gsm8k") 