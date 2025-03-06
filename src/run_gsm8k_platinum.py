"""Evaluate models on GSM8K-Platinum.

This code was adapted from `run_benchmark.py`

Usage:
python src/run_gsm8k_platinum.py --model-list gpt-4o-mini
"""

import datasets
from dotenv import load_dotenv
import pandas as pd
import os
import json
import argparse

from utils import get_parse_fn, get_prompt, run_predictions, run_predictions_parallel

template_gsm8k = """Solve the following math word problem.

{question}

Think step-by-step. Then, provide the final answer as a single integer in the format "Answer: XXX" with no extra formatting."""

def run_benchmark(model_list, output_file, parallelism=1, save_errors=False):
    load_dotenv()

    benchmark_path = "madrylab/gsm8k-platinum"

    dataset_name = 'gsm8k_full'
    parsing_strategy = 'math'

    gsm8k = datasets.load_dataset(benchmark_path, "main", split='test')
    print('Loaded dataset with', len(gsm8k), 'examples')

    platinum_dataset = gsm8k.map(lambda q: {
        'platinum_target': q['answer'].split('\n#### ')[-1].replace(',', ''),
        'platinum_prompt': template_gsm8k.format(question=q['question']),
        'platinum_prompt_no_cot': template_gsm8k.format(question=q['question']).replace('Think step-by-step. ', ''),
    })

    parse_fn = get_parse_fn(parsing_strategy)
    
    errors = {}
    for model_name in model_list:
        errors[model_name] = []

        if parallelism > 1:
            outputs = run_predictions_parallel(platinum_dataset, dataset_name, model_name, load_only=False, num_threads=parallelism)
        else:
            outputs = run_predictions(platinum_dataset, dataset_name, model_name, load_only=False)

        empty_count = 0
        for example, output in zip(platinum_dataset, outputs):
            platinum_target = example['platinum_target']
            prompt = get_prompt(example, model_name)

            if output is None:
                empty_count += 1

            try:
                prediction = parse_fn(output)
                correct = float(platinum_target) == float(prediction)
            except:
                prediction = 'parsing error'
                correct = False

            if not correct:
                errors[model_name].append({
                    'prompt': prompt,
                    'platinum_target': platinum_target,
                    'prediction': prediction,
                    'explanation': output,
                })
        
        if empty_count > 0:
            print(f"WARN: Model {model_name} had {empty_count} empty outputs for dataset {dataset_name}, perhaps due to API errors.")
            

        if save_errors:
            errors_dir = './outputs/errors'
            os.makedirs(errors_dir, exist_ok=True)
            with open(os.path.join(errors_dir, f'errors_{dataset_name}.json'), 'w') as f:
                json.dump(errors, f, indent=2)

    df = pd.DataFrame([{
        'model': model_name,
        'error_count': len(errors[model_name]),
        } for model_name in model_list])
    print(df.to_string(index=False))

    # Save the results to a file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate models on Platinum Benchmarks')

    parser.add_argument('--model-list', type=str, nargs="+", default=None, help='A space-separated list of models to be evaluated')
    parser.add_argument('--output-file', type=str, default='./outputs/results_gsm8k_platinum.csv', help='Output file name to save the results')
    parser.add_argument('--parallel', type=int, default=1, help='Number of threads to use for parallel prediction. If more than 1, will use parallelism')
    parser.add_argument('--save-errors', action='store_true', help='Save errors for each dataset to the directory ./outputs/errors')

    args = parser.parse_args()

    run_benchmark(args.model_list, args.output_file, parallelism=args.parallel, save_errors=args.save_errors)