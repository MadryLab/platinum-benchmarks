# Platinum Benchmarks

[**🏆 Leaderboard**](http://platinum-bench.csail.mit.edu/) &nbsp;|&nbsp; [**📖 Paper**](https://arxiv.org/abs/2502.03461) &nbsp;|&nbsp; [**🤗 Dataset**](https://huggingface.co/datasets/madrylab/platinum-bench)

This repository contains the evaluation code for "[Do Large Language Model Benchmarks Test Reliability?](https://arxiv.org/abs/2502.03461)."

We introduce **platinum benchmarks**, LLM benchmarks designed to test the reliability. Platinum benchmarks are carefully curated to minimize label errors and ambiguity, so that perfect performance is possible. It turns out, frontier language models still make mistakes on surprisingly simple tasks.


## Dataset

Our benchmark consists of fifteen platinum benchmarks adapted from existing datasets. We plan to occasionally update the benchmark as new issues with questions are found. Please refer to [the 🤗HuggingFace dataset page](https://huggingface.co/datasets/madrylab/platinum-bench) for more details.

## Evaluation

### Installation Quickstart

Clone and navigate into the repository:

```bash
git clone https://github.com/MadryLab/platinum-benchmarks.git
cd platinum-benchmarks
```

Set up the environment and install the required dependencies:

```bash
conda create -y -n platinum-bench python=3.10
conda activate platinum-bench
pip install -r requirements.txt
```

Create a `.env` file with API keys for the models you'd like to use, such as:

```
OPENAI_API_KEY={your-key-here}
ANTHROPIC_API_KEY={your-key-here}
```

You can start from the example file we provide in `.env_example`.

### Usage

The main evaluation script is `src/run_benchmark.py`. Here are the key options:

```bash
python src/run_benchmark.py --model-list MODEL_NAME [MODEL_NAME ...] \
                       [--output-file PATH] \
                       [--parallel N] \
                       [--save-errors] \
                       [--paper-version] \
                       [--unfiltered]
```

Arguments:
- `--model-list`: One or more model names to evaluate (required)
- `--output-file`: Path to save results CSV (default: './outputs/results.csv')
- `--parallel`: Number of threads for parallel evaluation (default: 1)
- `--save-errors`: Save errors for each dataset to './outputs/errors/'
- `--paper-version`: Use the benchmark version from the paper
- `--unfiltered`: Don't filter out bad questions (i.e., ambiguous or poorly worded questions that we exclude during out filtering process)

Example usage:
```bash
python run_benchmark.py --model-list gpt-4o-mini
```

Or, just use the script we provide to get results for all models we evaluate:

```bash
bash scripts/get_results.sh
```

### Reproduce Our Paper Results

Our code is designed to cache LLM outputs to avoid unnecessary API calls. The evaluation script will automatically use cached results when available and only make new API calls for uncached queries. If you'd like to reproduce our exact results:

1. Clone the repository and install dependencies as described above
2. Download our LLM inference cache to `./cache` (see below)
3. Run the paper evaluation script:

```bash
bash scripts/get_paper_results.sh
```

**LLM Inference Cache**

We provide a cache of LLM inferences that we used to generate our results. You can download the cache yourself from HuggingFace, or use the script below to automatically download the cache and place it in the correct directory.

```bash
bash scripts/download_paper_cache.sh
```

## Evaluating New Models

Models for which we already have an API implemented (e.g. OpenAI) can be added by simply adding a new entry in `ModelEngineFactory` in `src/models.py`. For example:

```python
...
elif model_name == "gpt-4o-2024-11-20":
  engine = OpenAIModel(api_name="gpt-4o-2024-11-20")
...
```

For other models, you can implement a new model class in `src/models.py` that inherits from `Model` and implements the `init_client` and `predict` methods. Then, add a new entry in `ModelEngineFactory` as above.

You should now be able to evaluate your new model as normal, e.g. by running:

```bash
python run_benchmark.py --model-list your-new-model
```

## Citation

Cite this dataset and the source datasets (see `sources.bib`).

```
@misc{vendrow2025largelanguagemodelbenchmarks,
      title={Do Large Language Model Benchmarks Test Reliability?}, 
      author={Joshua Vendrow and Edward Vendrow and Sara Beery and Aleksander Madry},
      year={2025},
      eprint={2502.03461},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.03461}, 
}
```

