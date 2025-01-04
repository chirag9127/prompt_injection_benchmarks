# Prompt Injection Detection Benchmark

This project provides a benchmarking framework for evaluating prompt injection detection libraries. Currently, it benchmarks [LLM Guard](https://github.com/laiyer-ai/llm-guard)'s prompt injection detection capabilities against the [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) dataset.

## Features

- Evaluates detection accuracy, precision, recall, and F1 score
- Measures average detection time

## Initial Results

Initial benchmark results for LLM Guard:

| Metric | Value |
|--------|-------|
| Accuracy | 77.29% |
| Precision | 96.47% |
| Recall | 40.39% |
| F1 Score | 56.94% |
| Avg Detection Time | 0.099s |

These results suggest that LLM Guard has high precision but lower recall. The detection speed is quite fast at under 0.1 seconds per prompt.

## Installation

1. Clone the repository
2. Create a virtual environment (recommended)
3. Install dependencies
4. Run `python main.py`