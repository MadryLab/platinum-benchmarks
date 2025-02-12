#!/bin/bash

python src/run_benchmark.py \
    --model-list \
        meta-llama/Meta-Llama-3.1-405B-Instruct \
        claude-3-5-sonnet \
        o1-mini \
        gpt-4o-2024-11-20 \
        gpt-4o-2024-08-06 \
        gemini-1.5-pro \
        mistral-large \
        meta-llama/Meta-Llama-3.1-70B-Instruct \
        gpt-4o-mini \
        gemini-1.5-flash \
        mistral-small \
        o1-preview-2024-09-12 \
        meta-llama/Llama-3.3-70B-Instruct \
        grok-2-1212 \
        claude-3-5-sonnet-20241022 \
        claude-3-5-haiku \
        gemini-2.0-flash \
        gemini-2.0-flash-thinking \
        deepseek/deepseek-chat \
        Qwen/Qwen2.5-72B-Instruct \
        o1-2024-12-17-med \
        o1-2024-12-17-high \
        deepseek-r1 \
        o3-mini-2025-01-31-high \
        gemini-2.0-pro-02-05 \
        Qwen2.5-Max \
    --output-file ./outputs/live_results.csv \
    --save-errors
