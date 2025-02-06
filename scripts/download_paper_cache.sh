#!/bin/bash

# This script downloads the cache files with the LLM inferences used to generate the results in the paper.
# The script assumes that you have installed the Hugging Face CLI (pip install huggingface_hub).
# Usage: bash scripts/download_paper_cache.sh

DATASET_REPO="madrylab/platinum-bench-paper-cache"
LOCAL_DIR="./cache"

if find "$LOCAL_DIR" -maxdepth 1 -type f -name "*.pkl" | grep -q .; then
    echo "Error: The cache directory '$LOCAL_DIR' is not empty."
    echo "       To avoid overwriting existing files, please move or delete the files in this directory."
    exit 1
fi

# Download the dataset from Hugging Face, including only pkl files
huggingface-cli download "$DATASET_REPO" \
    --repo-type dataset \
    --include "*.pkl" \
    --local-dir "$LOCAL_DIR"