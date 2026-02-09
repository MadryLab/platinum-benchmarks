import argparse
from .run_benchmark import run_benchmark
from .run_gsm8k_platinum import run_gsm8k_benchmark
try:
    import wandb
except ImportError:
    wandb = None

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


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on Platinum Benchmarks')
    parser.add_argument('--model-list', type=str, nargs="+", default=None, help='A space-separated list of models to be evaluated')
    parser.add_argument('--vllm', action='store_true', help='The model is served with vllm.')
    parser.add_argument('--port', type=int, default=8000, help='Port number for vllm server.')
    parser.add_argument('--host', type=str, default='localhost', help='Host for vllm server.')
    parser.add_argument('--api-key', type=str, default='token-abc123', help='API key for the model, if required.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for the model default is 0.5.')
    parser.add_argument('--reasoning-model', action='store_true', help='Indicate if the model is in reasoning mode.')
    parser.add_argument('--output-file', type=str, default='./outputs/results.csv', help='Output file name to save the results')
    parser.add_argument('--parallel', type=int, default=1, help='Number of threads to use for parallel prediction. If more than 1, will use parallelism')
    parser.add_argument('--errors-dir', type=str, default=None, help='Output directory where to save the errors')
    parser.add_argument('--paper-version', action='store_true', help='Use the version of the benchmark used in the paper, which is less recent. Use this flag if you want to reproduce the paper results.')
    parser.add_argument('--unfiltered', action='store_true', help='Use the unfiltered benchmark, which including examples that were rejected by the cleaning process.')
    parser.add_argument('--seed', type=int, default=42, help='Fixing random seed for reproducibility.')
    parser.add_argument("--dataset_names",
                        type=str,
                        nargs="+",                       # allow multiple values
                        choices=DATASET_NAMES+["gsm8k_full"],           # restrict to this list
                        default=DATASET_NAMES,           # default: all datasets
                        help="Datasets to run; default is all.",
                    )
    # Logging params
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log to wandb."
    )

    args = parser.parse_args()
    if len(args.model_list) > 1 and args.vllm:
        raise ValueError("vllm serving with multiple models is not supported yet.")
    print(f"Evaluating on {args.dataset_names=}")
    if args.log_wandb:
        assert wandb is not None, "wandb is not installed. Please install wandb `pip install wandb`."
        wandb.init(config=args)

    run_benchmark(args.model_list, args.output_file, parallelism=args.parallel, errors_dir=args.errors_dir, use_paper_version=args.paper_version, use_unfiltered_version=args.unfiltered,dataset_names=args.dataset_names, seed=args.seed, wandb=wandb if args.log_wandb else None, args=args)
    
    if "gsm8k_full" in args.dataset_names:
        run_gsm8k_benchmark(args.model_list, args.output_file, parallelism=args.parallel, errors_dir=args.errors_dir, seed=args.seed, wandb=wandb if args.log_wandb else None, args=args)

if __name__ == "__main__":
    main()