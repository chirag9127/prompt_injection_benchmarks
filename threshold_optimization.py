import numpy as np
from typing import Tuple, List
import time
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from llm_guard.input_scanners.prompt_injection import PromptInjection as LLMGuardScanner
from pytector import PromptInjectionDetector

class ThresholdOptimizer:
    def __init__(self):
        # Initialize detectors
        self.llm_guard = LLMGuardScanner()
        self.pytector = PromptInjectionDetector(model_name_or_url="deberta")
        
        # Load dataset
        self.dataset = load_dataset("deepset/prompt-injections", split="train")
        
        # Define threshold ranges to test
        self.thresholds = np.linspace(0.0, 1.0, 30)  # Test 30 thresholds from 0.1 to 0.9
    
    def get_llm_guard_score(self, prompt: str) -> float:
        """Get raw score from LLM Guard (higher means more likely to be injection)"""
        try:
            _, _, risk_score = self.llm_guard.scan(prompt)
            return risk_score
        except Exception as e:
            print(f"LLM Guard error: {str(e)}")
            return 0.0

    def get_pytector_score(self, prompt: str) -> float:
        """Get probability score from Pytector"""
        try:
            _, probability = self.pytector.detect_injection(prompt)
            return probability
        except Exception as e:
            print(f"Pytector error: {str(e)}")
            return 0.0

    def evaluate_threshold(self, y_true: List[int], scores: List[float], threshold: float) -> Tuple[float, float, float]:
        """Calculate metrics for a given threshold"""
        y_pred = [1 if score >= threshold else 0 for score in scores]
        return (
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0)
        )

    def plot_metrics(self, thresholds: List[float], precisions: List[float], 
                    recalls: List[float], f1_scores: List[float], title: str):
        """Plot metrics across different thresholds"""
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', marker='o')
        plt.plot(thresholds, recalls, label='Recall', marker='o')
        plt.plot(thresholds, f1_scores, label='F1 Score', marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'{title} Metrics vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{title.lower().replace(" ", "_")}_threshold_metrics.png')
        plt.close()

    def plot_precision_recall_curve(self, y_true: List[int], scores: List[float], title: str):
        """Plot precision-recall curve and calculate AUC-PR"""
        # Calculate precision-recall curve
        precisions, recalls, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recalls, precisions)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, label=f'{title} (AUC = {pr_auc:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{title} Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        # Add the baseline (random classifier)
        baseline = sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='r', linestyle='--', 
                   label=f'Baseline (No Skill) = {baseline:.3f}')
        
        plt.legend()
        plt.savefig(f'{title.lower().replace(" ", "_")}_pr_curve.png')
        plt.close()
        
        return pr_auc

    def optimize(self):
        # Collect ground truth and scores
        y_true = []
        llm_guard_scores = []
        pytector_scores = []
        
        print("Collecting scores from both detectors...")
        for example in tqdm(self.dataset):
            prompt = example['text']
            y_true.append(example['label'])
            
            llm_guard_scores.append(self.get_llm_guard_score(prompt))
            pytector_scores.append(self.get_pytector_score(prompt))

        # Plot precision-recall curves
        print("\nGenerating precision-recall curves...")
        llm_guard_auc = self.plot_precision_recall_curve(y_true, llm_guard_scores, 'LLM Guard')
        pytector_auc = self.plot_precision_recall_curve(y_true, pytector_scores, 'Pytector')
        
        print(f"\nAUC-PR Scores:")
        print(f"LLM Guard: {llm_guard_auc:.3f}")
        print(f"Pytector: {pytector_auc:.3f}")

        # Evaluate metrics across thresholds
        results = {
            'LLM Guard': {'precisions': [], 'recalls': [], 'f1_scores': []},
            'Pytector': {'precisions': [], 'recalls': [], 'f1_scores': []}
        }

        print("\nEvaluating thresholds...")
        for threshold in tqdm(self.thresholds):
            # LLM Guard
            p, r, f1 = self.evaluate_threshold(y_true, llm_guard_scores, threshold)
            results['LLM Guard']['precisions'].append(p)
            results['LLM Guard']['recalls'].append(r)
            results['LLM Guard']['f1_scores'].append(f1)
            
            # Pytector
            p, r, f1 = self.evaluate_threshold(y_true, pytector_scores, threshold)
            results['Pytector']['precisions'].append(p)
            results['Pytector']['recalls'].append(r)
            results['Pytector']['f1_scores'].append(f1)

        # Find optimal thresholds
        for detector in ['LLM Guard', 'Pytector']:
            best_threshold_idx = np.argmax(results[detector]['f1_scores'])
            best_threshold = self.thresholds[best_threshold_idx]
            
            print(f"\n{detector} Optimal Results:")
            print(f"Best Threshold: {best_threshold:.3f}")
            print(f"Best F1 Score: {results[detector]['f1_scores'][best_threshold_idx]:.3f}")
            print(f"Precision: {results[detector]['precisions'][best_threshold_idx]:.3f}")
            print(f"Recall: {results[detector]['recalls'][best_threshold_idx]:.3f}")
            
            # Plot metrics
            self.plot_metrics(
                self.thresholds,
                results[detector]['precisions'],
                results[detector]['recalls'],
                results[detector]['f1_scores'],
                detector
            )

def main():
    optimizer = ThresholdOptimizer()
    optimizer.optimize()

if __name__ == "__main__":
    main() 