import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load environment variables from .env file
load_dotenv()

# Import detection libraries
from llm_guard.input_scanners.prompt_injection import PromptInjection as LLMGuardScanner


@dataclass
class BenchmarkResult:
    library_name: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_detection_time: float

class PromptInjectionBenchmark:
    def __init__(self):
        # Initialize detection libraries
        self.llm_guard = LLMGuardScanner()
        
        # Load dataset
        self.dataset = load_dataset("deepset/prompt-injections", split="train")
        
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> BenchmarkResult:
        y_true = [r['actual'] for r in results]
        y_pred = [r['predicted'] for r in results]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics using sklearn
        accuracy = accuracy_score(y_true, y_pred) if len(results) > 0 else 0
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0) 
        f1 = f1_score(y_true, y_pred, zero_division=0)
        avg_time = sum(r['detection_time'] for r in results) / len(results) if len(results) > 0 else 0
        
        return BenchmarkResult(
            library_name=results[0]['library'],
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_detection_time=avg_time
        )

    def run_llm_guard(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            _, results_valid, _ = self.llm_guard.scan(prompt)
            is_injection = not results_valid
        except Exception as e:
            print(f"LLM Guard error: {str(e)}")
            is_injection = False
        detection_time = time.time() - start_time
        
        return {
            "library": "LLM Guard",
            "predicted": is_injection,
            "detection_time": detection_time
        }

    def run_benchmark(self) -> Dict[str, BenchmarkResult]:
        results = {
            "LLM Guard": []
        }
        
        for example in tqdm(self.dataset, desc="Running benchmark"):
            prompt = example['text']
            is_injection = example['label'] == 1
            
            # Run detector
            llm_guard_result = self.run_llm_guard(prompt)
            llm_guard_result['actual'] = is_injection
            results["LLM Guard"].append(llm_guard_result)
        
        # Calculate metrics for each library
        benchmark_results = {}
        for library_name, library_results in results.items():
            benchmark_results[library_name] = self.calculate_metrics(library_results)
        
        return benchmark_results

    def save_results(self, results: Dict[str, BenchmarkResult], output_file: str):
        """Save benchmark results to a JSON file"""
        output = {
            "benchmark_date": datetime.now().isoformat(),
            "results": {
                name: {
                    "true_positives": result.true_positives,
                    "false_positives": result.false_positives,
                    "true_negatives": result.true_negatives,
                    "false_negatives": result.false_negatives,
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "avg_detection_time": result.avg_detection_time
                }
                for name, result in results.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

def main():
    benchmark = PromptInjectionBenchmark()
    print("Starting benchmark...")
    results = benchmark.run_benchmark()
    
    # Print results
    print("\nBenchmark Results:")
    print("-" * 80)
    for library_name, result in results.items():
        print(f"\n{library_name}:")
        print(f"Accuracy: {result.accuracy:.4f}")
        print(f"Precision: {result.precision:.4f}")
        print(f"Recall: {result.recall:.4f}")
        print(f"F1 Score: {result.f1_score:.4f}")
        print(f"Average Detection Time: {result.avg_detection_time:.4f} seconds")
    
    # Save results to file
    benchmark.save_results(results, "benchmark_results.json")
    print("\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main()
