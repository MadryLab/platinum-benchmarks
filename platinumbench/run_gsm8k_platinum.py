"""Evaluate models on GSM8K-Platinum.

This code was adapted from `run_benchmark.py`

Usage:
python platinumbench/run_gsm8k_platinum.py --model-list gpt-4o-mini
"""

import datasets
from dotenv import load_dotenv
import pandas as pd
import os
import json
import argparse
import torch
from types import SimpleNamespace


from .utils import get_parse_fn, get_prompt, run_predictions, run_predictions_parallel, fix_seed, clear_device_cache

try:
    import wandb
except ImportError:
    wandb = None

template_gsm8k = """Solve the following math word problem.

{question}

Think step-by-step. Then, provide the final answer as a single integer in the format "Answer: XXX" with no extra formatting."""

def run_gsm8k_benchmark(model_list, output_file, parallelism=1, errors_dir=None, wandb=None,seed=42, args=SimpleNamespace()):
    
    """Runs the benchmark for the specified models and saves the results to a CSV file.
    Args:
        model_list: List of model names or dict of {model_name: tuple(model torch.nn.Module, tokenizer)} to evaluate.
        output_file: Path to the output CSV file.
        parallelism: Number of threads to use for parallel prediction.
        errors_dir: error dir 
        use_paper_version: Whether to use the version of the benchmark used in the paper.
        use_unfiltered_version: Whether to use the unfiltered benchmark, including rejected examples.
        dataset_names: ["singleop", "singleq", "multiarith",
                        "svamp", "gsm8k", "mmlu_math",
                        "bbh_logical_deduction_three_objects", "bbh_object_counting",
                        "bbh_navigate","tab_fact","hotpotqa",
                        "squad","drop","winograd_wsc"]
        args: Additional arguments for model inference in form of dict(as in argparse). Examples include: 
              temperature: Temperature for the model default is 0.5., 
              reasoning_model:Indicate if the model is in reasoning mode., etc.
        
        return: pandas datasets with error count and accuracies for each model and datasets
    """

    if not getattr(args, "seed", False):
        args.seed=seed
    
    fix_seed(seed=args.seed)

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
    
    if wandb:
        wandb.log({f"#_{dataset_name}": len(platinum_dataset)})
        artifacts={}
        for model_name in model_list:
            uniquename=(model_name + str(args.seed)).replace("/", "--") if str(args.seed) not in model_name else model_name.replace("/", "--")
            artifacts[model_name] = wandb.Artifact(f"{uniquename}", type="inference")
        
    errors = {}
    for model_name in model_list:
        if isinstance(model_list[model_name], tuple):
                if isinstance(model_list[model_name][0],torch.nn.Module):
                    args.model=model_list[model_name][0]
                    args.tokenizer=model_list[model_name][1]
                elif isinstance(model_list[model_name][1],torch.nn.Module):
                    args.model=model_list[model_name][1]
                    args.tokenizer=model_list[model_name][0]
                else:
                    raise NotImplementedError
                
        print(model_name)
        errors[model_name] = []

        if parallelism > 1:
            outputs = run_predictions_parallel(platinum_dataset, dataset_name, model_name, load_only=False, num_threads=parallelism, args=args)
        else:
            outputs = run_predictions(platinum_dataset, dataset_name, model_name, load_only=False, args=args)
        
        if isinstance(model_list, dict):
            args.model.to("cpu")
        
        empty_count = 0
        for example, output in zip(platinum_dataset, outputs):
            platinum_target = example['platinum_target']
            prompt = get_prompt(example, model_name, args = args)

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
            
        
        if errors_dir:
            model_errors_dir = errors_dir + f'/{model_name}/'
            os.makedirs(model_errors_dir, exist_ok=True)

            with open(os.path.join(model_errors_dir, f'errors_{dataset_name}.json'), 'w') as f:
                json.dump(errors, f, indent=2)
            
        if wandb:
            with artifacts[model_name].new_file(f"errors_{dataset_name}.json", mode="w") as f:
                json.dump(errors, f, indent=2)
    if wandb:
        if len(model_list)>1:
            wandb.log({f"platinumbench/{dataset_name}": errors})
            # raise "I don't want to deal with this , go implement yourself"
        wandb.log({f"platinumbench/errors.{dataset_name}": errors[model_name]})
    
    df = pd.DataFrame([{
        'model': model_name,
        'error_count': len(errors[model_name]),
        } for model_name in model_list])
    
    print(df.to_string(index=False))

    # Save the results to a file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    return df
