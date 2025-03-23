import os
import json
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path

class MMLUBenchmark:
    def __init__(
        self,
        model_name: str,
        model_type: str,
        token_path: str,
        torch_dtype: str = "float16",
        low_memory: bool = False,
        requires_auth: bool = True
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.torch_dtype = getattr(torch, torch_dtype)
        self.low_memory = low_memory
        self.requires_auth = requires_auth
        
        # Load HuggingFace token
        if requires_auth:
            with open(token_path, "r") as f:
                self.hf_token = f.read().strip()
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
        
        # Initialize model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # Load MMLU dataset
        self.dataset = load_dataset("cais/mmlu", "all")
        
    def _load_model(self):
        """Load the model and tokenizer with appropriate settings."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=self.low_memory,
            trust_remote_code=True
        ).to(self.device)
        
    def _format_prompt(self, question: str, choices: List[str], num_shots: int = 5) -> str:
        """Format the prompt with few-shot examples."""
        # TODO: Implement proper few-shot examples
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer:"
        return prompt
    
    def _get_answer(self, prompt: str) -> str:
        """Get model's answer for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        answer = self.tokenizer.decode(outputs[0][-1], skip_special_tokens=True)
        return answer.strip()
    
    def evaluate_subject(self, subject: str, num_shots: int = 5) -> Dict:
        """Evaluate the model on a specific MMLU subject."""
        subject_data = self.dataset[subject]
        correct = 0
        total = 0
        
        for item in tqdm(subject_data, desc=f"Evaluating {subject}"):
            prompt = self._format_prompt(
                item["question"],
                item["choices"],
                num_shots
            )
            answer = self._get_answer(prompt)
            
            # Convert answer to index (A=0, B=1, etc.)
            try:
                predicted_idx = ord(answer.upper()) - ord('A')
                if 0 <= predicted_idx < len(item["choices"]):
                    if predicted_idx == item["answer"]:
                        correct += 1
                    total += 1
            except:
                continue
        
        accuracy = correct / total if total > 0 else 0
        return {
            "subject": subject,
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def run_benchmark(self, subjects: List[str] = None, num_shots: int = 5) -> Dict:
        """Run the MMLU benchmark on specified subjects."""
        if subjects is None or "all" in subjects:
            subjects = self.dataset.keys()
        
        results = []
        for subject in subjects:
            if subject in self.dataset:
                result = self.evaluate_subject(subject, num_shots)
                results.append(result)
        
        # Calculate average accuracy
        avg_accuracy = np.mean([r["accuracy"] for r in results])
        
        return {
            "average_accuracy": avg_accuracy,
            "subject_results": results
        } 