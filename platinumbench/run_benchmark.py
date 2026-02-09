"""Evaluate models on Platinum Benchmarks

Usage:
python platinumbench/run_benchmark.py --model-list gpt-4o-mini
"""

import datasets
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import os
import json

import torch
from types import SimpleNamespace

try:
    import wandb
except ImportError:
    wandb = None


from .utils import get_parse_fn, check_prediction, get_prompt, run_predictions, run_predictions_parallel, fix_seed, clear_device_cache

DATASET_NAMES = [
        "singleop",
        "singleq",
        "multiarith",
        "svamp",
        "gsm8k",
        "mmlu_math",
        "bbh_logical_deduction_three_objects",
        "bbh_object_counting",
        "bbh_navigate",
        "tab_fact",
        "hotpotqa",
        "squad",
        "drop",
        "winograd_wsc",
    ]

def run_benchmark(model_list, output_file=None, parallelism=1, errors_dir=None, use_paper_version=False, use_unfiltered_version=False, dataset_names=DATASET_NAMES, seed=42, wandb=None, args=SimpleNamespace()):
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

    assert isinstance(model_list, dict) or isinstance(model_list, list), "model_list should be a list of model names or dict of models and tokenizer ."
    
    if not getattr(args, "seed", False):
        args.seed=seed
    
    fix_seed(seed=args.seed)

    
    load_dotenv()
    
    print("args:", args)


    if use_paper_version:
        benchmark_path = "madrylab/platinum-bench-paper-version"
    else:
        benchmark_path = "madrylab/platinum-bench"

    acc_count_dict = {'model': model_list} if isinstance(model_list, list) else {'model': model_list.keys()}
    error_count_dict = {'model': model_list} if isinstance(model_list, list) else {'model': model_list.keys()}
    if wandb:
        artifacts={}
        for model_name in model_list:
            uniquename=(model_name + str(args.seed)).replace("/", "--") if str(args.seed) not in model_name else model_name.replace("/", "--")
            artifacts[model_name] = wandb.Artifact(f"{uniquename}", type="inference")
        
    for dataset_name in dataset_names:
        
        if "gsm8k_full" in dataset_name:
            continue
        
        print(f"Running predictions for {dataset_name}")

        platinum_dataset = datasets.load_dataset(benchmark_path, dataset_name, split='test')
        if not use_unfiltered_version:
            platinum_dataset = platinum_dataset.filter(lambda x: x['cleaning_status'] != 'rejected')

        parsing_strategy = platinum_dataset[0]['platinum_parsing_strategy']
        parse_fn = get_parse_fn(parsing_strategy)
        
        errors = {}
        if wandb:
            wandb.log({f"#_{dataset_name}": len(platinum_dataset)})
            


        for model_name in model_list: #TODO FLIP with dataset loop

            if isinstance(model_list, dict) and isinstance(model_list[model_name], tuple):
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
                # clear_device_cache(garbage_collection=True)

            
            correct_answers = 0
            incorrect_answers = 0
            empty_count = 0
            for example, output in zip(platinum_dataset, outputs):
                platinum_target = example['platinum_target']
                prompt = get_prompt(example, model_name, args = args)
                if output is None:
                    empty_count += 1
                try:
                    prediction = parse_fn(output)
                    correct = check_prediction(prediction, platinum_target, prompt, dataset_name)
                except:
                    prediction = 'parsing error'
                    correct = False
                if correct:
                    correct_answers += 1
                else:
                    incorrect_answers += 1

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
                    print(f"writing into errors_{dataset_name}.json.")
                    json.dump(errors, f, indent=2)
            print(f"accuracy of {model_name=} on {dataset_name=}",correct_answers/(correct_answers+incorrect_answers))
        error_count_dict[dataset_name] = [len(errors[model_name]) for model_name in model_list]
        acc_count_dict[dataset_name] = [(1-len(errors[model_name])/len(platinum_dataset))*100 for model_name in model_list]
        
        if wandb:
            if len(model_list)>1:
                df_dt = pd.DataFrame([{
                        'model': model_name,
                        'error_count': len(errors[model_name]),
                        } for model_name in model_list])
                print(df_dt)
                wandb.log({f"platinumbench/{dataset_name}": df_dt})
            else:
                wandb.log({f"platinumbench/errors.{dataset_name}": error_count_dict[dataset_name][0]})
                wandb.log({f"platinumbench/accuracy.{dataset_name}": acc_count_dict[dataset_name][0]})

    # print(error_count_dict)
    df = pd.DataFrame(error_count_dict)
    df['average'] = df.mean(numeric_only=True, axis=1)
    print(df)
    if wandb:
        for model_name in model_list:
            print(artifacts[model_name])
            wandb.log_artifact(artifacts[model_name])


    if output_file:
        # Save the results to a file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

    df_acc = pd.DataFrame(acc_count_dict)
    df_acc['average'] = df_acc.mean(numeric_only=True, axis=1)
    print(df_acc)

    if output_file:
        name, ext = output_file.rsplit(".", 1)
        new_filename = f"{name}_acc.{ext}"
        df_acc.to_csv(new_filename, index=False)
        print(f"Saved accuracy results to {new_filename}")

    return df, df_acc