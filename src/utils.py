import os
import pickle
import re
import time

from tqdm import tqdm

from models import ModelInferenceEngine, ModelEngineFactory
import openai

class LLMCache:
    def __init__(self, cache_file='llm_cache.pkl'):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def has(self, key):
        return key in self.cache

    def get(self, key):
        return self.cache[key]
    
    def set(self, key, response):
        self.cache[key] = response
        self.save_cache()

    def set_if_not_exists(self, prompt, response):
        if not self.has(prompt):
            self.set(prompt, response)


def get_llm_cache(dataset_name):
    return LLMCache(cache_file=f'cache/reliability_benchmark_cache_{dataset_name}.pkl')


def get_parse_fn(parsing_strategy):
    def parse_fn_math(output):
        """Used for singleop, singleeq, multiarith, gsm8k, and svamp"""
        return re.sub(r"\.0+$", "", (re.search(r'-?[0-9.]*[0-9]', output.replace('*','').replace('#','').lower().split('answer: ')[-1].replace(',', '')).group()))

    def parse_fn_multiple_choice(output):
        """Used for mmlu math and winograd schema challenge"""
        #return output.replace('*', '').lower().split("answer: ")[-1].replace(".", "").strip()[0:1].lower()

        x = output.replace('*', '').lower().split("answer: ")[-1].replace(".", "").strip()
        
        pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(pattern, x)
        if match:
            return match.group(1)[0:1]
        else:
            return x[0:1]
    
    def parse_bbh_multiple_choice(output):
        """Used for BBH multiple choice questions, where the answer is in the form (A)""" 
        result = output.replace('*', '').replace('#', '').lower().split('answer: ')[-1].replace('.', '').replace('\'', '').replace('\"', '').strip().lower()
        result = re.search(r'\([a-z]\)', result).group(0)
        return result

    def parse_fn_text(output):
        """Used by DROP and hotpotqa, where the answer is a string"""
        return (output.replace("#","").replace("*","").replace("\"", "").replace('\xa0', ' ')
                      .lower().split("answer: ")[-1].split('\n')[0].replace(",", "")
                      .replace(".","").split("}")[0].strip())
    
    def parse_fn_squad(output):
        """Like rext parsing, but explicitly handles the case when there is text after n/a"""
        output_clean = parse_fn_text(output)
        if output_clean.startswith('n/a '):
            return 'n/a'
        return output_clean
    
    def create_parse_fn(specific_parsing_fn):

        def parse_fn(output):
            # tex_pattern = r'\\boxed\{([^{}]+)\}|\\boxed\{\\text\{([^}]+)\}\}'
            tex_pattern = r'\\boxed\{(\\text\{)?([^\\{}]+)\}'

            # If answer is on the last line as expected, run as usual
            if "answer:" in output.lower().replace("*", ""):
                # If the answer is wrapped in latex (e.g., \boxed{...}), extract the content
                answer_section = output.lower().split("answer: ")[-1]
                if re.search(tex_pattern, answer_section):
                    match = re.search(tex_pattern, answer_section).group(2)
                    output = "Answer: " + match
            elif re.search(tex_pattern, output):
                # If the answer is not on the last line, try to recover by looking for a box
                output = "Answer: " + re.search(tex_pattern, output).group(2)
            else: 
                # Otherwise, just return the last line
                last_line = output.strip("\n").split("\n")[-1].lower()
                output = "Answer: " + last_line
            return specific_parsing_fn(output)
        
        return parse_fn

        
    if parsing_strategy == 'math':
        return create_parse_fn(parse_fn_math)
    elif parsing_strategy == 'multiple_choice':
        return create_parse_fn(parse_fn_multiple_choice)
    elif parsing_strategy == 'bbh_multiple_choice':
        return create_parse_fn(parse_bbh_multiple_choice)
    elif parsing_strategy == 'text':
        return create_parse_fn(parse_fn_text)
    elif parsing_strategy == 'squad':
        return create_parse_fn(parse_fn_squad)
    else:
        raise ValueError(f"Invalid parsing strategy: {parsing_strategy}")
    
def check_prediction(prediction, platinum_target, prompt, dataset_name):
    math_datasets = ['math_eval__multiarith', 'math_eval__singleop', 'math_eval__singleq', 'gsm8k', 'svamp',
                 'multiarith', 'singleop', 'singleq', 'bbh_object_counting']
    if dataset_name in math_datasets and prediction != 'Parsing error':
        correct = float(platinum_target[0]) == float(prediction)
    else:
        correct = prediction in platinum_target
    return correct


def get_prompt(example, model_name):
    #o1-preview refuses to answer sometimes unless we remove "Then" for some reason.
    if model_name.startswith('o1-preview') or model_name.startswith('o1-2024-12-17'):
        return example['platinum_prompt_no_cot'].replace('Then, provide', 'Provide')
    
    if model_name in ModelEngineFactory.reasoning_models:
        return example['platinum_prompt_no_cot']
    else:
        return example['platinum_prompt']

def process_single_example(example, model_name, dataset_name, inference_engine=None, force_refresh=False, load_only=False):
    """Runs a single prediction and returns the result."""

    # Initialize a separate cache and inference engine for each thread if needed
    if inference_engine is None:
        response_cache = get_llm_cache(dataset_name)
        inference_engine = ModelInferenceEngine(response_cache)

    prompt = get_prompt(example, model_name)
    
    try:
        return inference_engine.run_inference(
            prompt, 
            model_name=model_name,
            force_refresh=force_refresh,
            load_only=load_only
        )
    except openai.BadRequestError as e:
        print(f"Got bad request error for example with {model_name} on {dataset_name}")
        print(e)
        return None, '', False


def run_predictions(dataset, dataset_name, model_name, force_refresh=False, load_only=False):
    """Runs the model on the full dataset and caches the results."""
    response_cache = get_llm_cache(dataset_name)
    inference_engine = ModelInferenceEngine(response_cache)

    predictions = []
    for example in tqdm(dataset):
        key, response, set_cache = process_single_example(example, model_name, dataset_name, inference_engine=inference_engine,
                                                     force_refresh=force_refresh, load_only=load_only)
        predictions.append(response)
        if set_cache:
            response_cache.set(key, response)
    
    return predictions


def run_predictions_parallel(dataset, dataset_name, model_name, force_refresh=False, load_only=False, num_threads=16):
    """Runs the model on the full dataset and caches the results. Parallelized version."""
    import multiprocess as mp

    # Create a common, unchanging cache for all threads
    response_cache = get_llm_cache(dataset_name)

    def inference_fn(example):
        # Initialize a separate inference engine for each thread
        inference_engine = ModelInferenceEngine(response_cache)

        try:
            return process_single_example(example, model_name, dataset_name, inference_engine=inference_engine, force_refresh=force_refresh, load_only=load_only)
        except Exception as e:
            print(f"Error processing example: {example}")
            print(e)
            return None, None, False
    
    # Separately, create a mutable cache where we can store the results
    response_cache_mutable = get_llm_cache(dataset_name)
    
    results = []
    with mp.Pool(num_threads) as pool:
    # with mp.Pool(n) as pool:
        for result in tqdm(pool.imap(lambda example: inference_fn(example), 
                                     dataset, chunksize=2), total=len(dataset)):
            results.append(result)
    
            key, response, set_cache = result
            if set_cache:
                response_cache_mutable.set(key, response)

    predictions = [response for _, response, _ in results]
    return predictions
