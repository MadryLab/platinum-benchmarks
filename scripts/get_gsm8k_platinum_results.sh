#!/bin/bash

python src/run_gsm8k_platinum.py \
    --model-list \
        meta-llama/Meta-Llama-3.1-405B-Instruct \
        gpt-4o-2024-11-20 \
        o1-2024-12-17-high \
        o1-2024-12-17-med \
        claude-3-5-sonnet-20241022 \
        gemini-2.0-flash-thinking-01-21 \
        o3-mini-2025-01-31-high \
        gpt-4o-mini \
        gemini-2.0-pro-02-05 \
        deepseek-r1 \
        claude-3-7-sonnet-20250219 \
        claude-3-7-sonnet-20250219-thinking \
        gpt-4.5-preview-2025-02-27 \
    --output-file ./outputs/results_gsm8k_platinum.csv \
    --save-errors
