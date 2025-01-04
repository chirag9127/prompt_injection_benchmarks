# Prompt Injection Detection Benchmark

This project provides a benchmarking framework for evaluating prompt injection detection libraries. Currently, it benchmarks [LLM Guard](https://github.com/laiyer-ai/llm-guard)'s prompt injection detection capabilities against the [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) dataset.

## Features

- Evaluates detection accuracy, precision, recall, and F1 score
- Measures average detection time

## Installation

1. Clone the repository
2. Create a virtual environment (recommended)
3. Install dependencies
4. Run `python main.py`