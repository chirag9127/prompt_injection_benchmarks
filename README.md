# Prompt Injection Detection Benchmark

This project provides a benchmarking framework for evaluating prompt injection detection libraries. Currently, it benchmarks [LLM Guard](https://github.com/laiyer-ai/llm-guard) and [Pytector](https://github.com/MaxMLang/pytector)'s prompt injection detection capabilities against the [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) dataset. Pytector is a Python package that uses state-of-the-art transformer models to detect prompt injection attempts in text inputs.

## Features

- Evaluates detection accuracy, precision, recall, and F1 score
- Measures average detection time

## Initial Results

Initial benchmark results:

LLM Guard:
| Metric | Value |
|--------|-------|
| Accuracy | 79.30% |
| Precision | 95.92% |
| Recall | 46.31% |
| F1 Score | 62.46% |
| Avg Detection Time | 0.093s |

Pytector:
| Metric | Value |
|--------|-------|
| Accuracy | 79.85% |
| Precision | 95.15% |
| Recall | 48.28% |
| F1 Score | 64.05% |
| Avg Detection Time | 0.155s |

These results suggest that LLM Guard has high precision but lower recall. The detection speed is quite fast at under 0.1 seconds per prompt.

## Installation

1. Clone the repository
2. Create a virtual environment (recommended)
3. Install dependencies
4. Run `python main.py`


## Threshold Optimization

The `threshold_optimization.py` script performs analysis to find optimal detection thresholds for both LLM Guard and Pytector. Key functionality includes:

- Tests 30 different threshold values from 0.0 to 1.0
- Evaluates precision, recall and F1 score at each threshold
- Generates visualization plots:
  - Precision-Recall curves with AUC scores
  - Metrics vs threshold plots showing precision, recall and F1 trade-offs
- Automatically determines optimal threshold values that maximize F1 score
- Outputs detailed metrics at the optimal threshold points

The analysis helps balance the trade-off between false positives and false negatives by finding threshold values that optimize overall detection performance. The precision-recall curves provide insight into detector performance across the full range of possible thresholds.

Based on this analysis, we selected a threshold of 0.034 for both detectors as it provided the best balance of precision and recall on the benchmark dataset.

