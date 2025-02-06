"""Evaluate models on Platinum Benchmarks

Usage:
python src/run_vision_benchmark.py --model-list gpt-4o-mini --coco-path PATH
"""

import datasets
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import os
import json
import re
import argparse

from models import ModelInferenceEngine
from utils import get_llm_cache


def parse_fn_vqa(response):
    first_word = response.strip("\n").split("\n")[0].split(" ")[0].lower().strip().strip('.').strip(",")
    last_word = response.strip("\n").split("\n")[-1].split(" ")[-1].lower().strip().strip('.').strip(",")

    pattern_yes = r'\byes\b'
    pattern_no = r'\bno\b'

    yes_exists = re.search(pattern_yes, response, flags=re.IGNORECASE)
    no_exists = re.search(pattern_no, response, flags=re.IGNORECASE)
    
    if yes_exists and not no_exists:
        return 'yes'
    elif no_exists and not yes_exists:
        return 'no'
    if first_word in ['yes', 'no']:
        return first_word
    if last_word in ['yes', 'no']:
        return last_word
    else:
        return "Parsing error"
        

def run_vision_benchmark(model_list, output_file, coco_path):
    load_dotenv()

    dataset = datasets.load_dataset("madrylab/platinum-bench", 'vqa', split='test')

    response_cache = get_llm_cache('vqa')
    engine = ModelInferenceEngine(response_cache=response_cache)

    errors = []
    for model_name in model_list:
        error_count = 0
        
        for example in dataset:
            image_path = os.path.join(coco_path, example['image_path'])
            prompt = example['platinum_prompt']
                
            key, response, _ = engine.run_inference(prompt, model_name=model_name, image_path=image_path) 
            output = parse_fn_vqa(response)
    
            if output not in example['platinum_target']:
                error_count += 1        
            
    
        errors.append(error_count)
    

    combined = [[x, y] for x, y in zip(model_list, errors)]
    df = pd.DataFrame(combined, columns=['Model Names', 'Errors'])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate models on Platinum Benchmarks')

    parser.add_argument('--model-list', type=str, nargs="+", default=None, help='A space-separated list of models to be evaluated')
    parser.add_argument('--output-file', type=str, default='./outputs/results_vision.csv', help='Output file name to save the results')
    parser.add_argument('--coco-path', type=str, default=None, help='Path to directory where val2014 folder is locate.')

    args = parser.parse_args()

    run_vision_benchmark(args.model_list, args.output_file, args.coco_path)