import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from src.benchmarks.base_benchmark import BaseBenchmark
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class GSM8KBenchmark(BaseBenchmark):
    """GSM8K (Grade School Math 8K) benchmark implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.dataset = None
        self.num_shots = config.get("num_shots", 3)
        self.split = config.get("split", "test")
        
    def load_dataset(self) -> None:
        """Load the GSM8K dataset."""
        try:
            logger.info("Loading GSM8K dataset...")
            self.dataset = load_dataset("gsm8k", split=self.split)
            logger.info(f"Loaded {len(self.dataset)} examples from GSM8K {self.split} split")
        except Exception as e:
            logger.error(f"Error loading GSM8K dataset: {e}")
            raise
            
    def format_prompt(self, question: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """Format the prompt with few-shot examples if provided."""
        prompt = "Let's solve this math problem step by step.\n\n"
        
        if few_shot_examples:
            for example in few_shot_examples:
                prompt += f"Question: {example['question']}\n"
                prompt += f"Let's solve this step by step:\n{example['answer']}\n\n"
                
        prompt += f"Question: {question}\n"
        prompt += "Let's solve this step by step:\n"
        return prompt
        
    def extract_answer(self, response: str) -> float:
        """Extract the final numerical answer from the model's response."""
        try:
            # Look for the last number in the response
            numbers = [float(s) for s in response.split() if s.replace('.', '').replace('-', '').isdigit()]
            if numbers:
                return numbers[-1]
            return None
        except Exception as e:
            logger.warning(f"Error extracting answer from response: {e}")
            return None
            
    def evaluate_example(self, model: BaseModel, example: Dict, few_shot_examples: List[Dict]) -> Dict:
        """Evaluate a single example."""
        prompt = self.format_prompt(example["question"], few_shot_examples)
        
        try:
            response = model.generate(prompt)
            predicted_answer = self.extract_answer(response)
            correct_answer = float(example["answer"].split()[-1])
            
            is_correct = abs(predicted_answer - correct_answer) < 1e-6 if predicted_answer is not None else False
            
            return {
                "question": example["question"],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "response": response,
                "is_correct": is_correct
            }
        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            return {
                "question": example["question"],
                "error": str(e)
            }
            
    def run(self, model: BaseModel) -> Dict:
        """Run the GSM8K benchmark on the given model."""
        if self.dataset is None:
            self.load_dataset()
            
        results = []
        total_correct = 0
        total_examples = 0
        
        # Get few-shot examples
        few_shot_examples = self.dataset.select(range(self.num_shots))
        
        # Evaluate each example
        for example in tqdm(self.dataset.select(range(self.num_shots, len(self.dataset))), desc="Evaluating GSM8K"):
            result = self.evaluate_example(model, example, few_shot_examples)
            results.append(result)
            
            if "is_correct" in result:
                total_examples += 1
                if result["is_correct"]:
                    total_correct += 1
                    
        accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        return {
            "accuracy": accuracy,
            "total_examples": total_examples,
            "total_correct": total_correct,
            "detailed_results": results
        } 