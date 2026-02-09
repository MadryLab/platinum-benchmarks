import subprocess
from pathlib import Path
import time
import argparse

# Function to serve model with vllm
def serve_model(model_name, models_path, port, dtype="bfloat16", api_key="token-abc123"):
    print(f"\n=== Serving model: {model_name} ===")
    # Example vllm serve command (adjust args as needed)
    serve_cmd = [
        "vllm", "serve",
        f"{models_path/model_name}",
        "--dtype", dtype,
        "--api-key", api_key,
        "--gpu-memory-utilization", "0.9",
        "--port", str(port)
    ]
    # Start the server
    server_proc = subprocess.Popen(serve_cmd)
    return server_proc

# Function to run benchmark
def run_benchmark(model_name, benchmark_script="platinumbench/run_benchmark.py", models_path=Path("models/"), port="8000", host="localhost",api_key="token-abc123",temperature=0.5):
    print(f"\n--- Running benchmark for: {model_name} ---")
    print(f"{model_name}")
    benchmark_cmd = [
        "python", benchmark_script,
        "--model-list", f"{models_path/model_name}",
        "--output-file", "outputs/"+f"{model_name}.csv",
        "--vllm",
        "--port", port,
        "--host", host,
        "--api-key", api_key,
        "--temperature", str(temperature),
        "--save-errors"
    ]
    subprocess.run(benchmark_cmd, check=True)

def main(args):
    # Path to your models folder
   
    models_path = Path(args.models_path)
    # Collect all model folders
    model_folders = [f.name for f in models_path.iterdir() if f.is_dir()]

    # Base path to your benchmark script
    benchmark_script = args.benchmark_script

    # Loop through models
    for model in model_folders:
        # Serve the model
        print(f"Starting to serve {model}...")
        server_proc = serve_model(model, models_path=models_path, port=args.port, api_key=args.api_key)
        
        # Give the server a few seconds to start
        time.sleep(60)
        print(f"Server for {model} should be up. Running benchmark...")
    
        try:
            # Run the benchmark
            run_benchmark(model, benchmark_script=benchmark_script, models_path=models_path, port=str(args.port), host=args.host, api_key=args.api_key,temperature=args.temperature)
        finally:
            print(f"Cleaning up for model {model}...")
            # Stop the server
            server_proc.terminate()
            server_proc.wait()
            print(f"Stopped serving {model}\n")
            time.sleep(10)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate batch of quantized models on Platinum Benchmarks')

    parser.add_argument('--models_path', type=str, default="models/", help='Path to the folder containing quantized models')
    parser.add_argument('--benchmark_script', type=str, default="platinumbench/run_benchmark.py", help='Path to the benchmark script')
    parser.add_argument('--output_csv', type=str, default="./outputs/live_results.csv", help='Output CSV file to save the results')
    parser.add_argument('--output-file', type=str, default='./outputs/results_vision.csv', help='Output file name to save the results')
    parser.add_argument('--port', type=int, default=8000, help='Port number for vllm server.')
    parser.add_argument('--host', type=str, default='localhost', help='Host for vllm server.')
    parser.add_argument('--api-key', type=str, default='token-abc123', help='API key for the model, if required.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for the model default is 0.5.')
    args = parser.parse_args()
    main(args)