import os
import subprocess
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
FASTA_PATH = BASE_DIR / "data" / "input.fasta"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_alphafold():
    command = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{BASE_DIR}:/app",
        "-w", "/app/task3_alphafold",  # locations of the script
        "--env", "TF_FORCE_UNIFIED_MEMORY=1",
        "--env", "XLA_PYTHON_CLIENT_MEM_FRACTION=4.0",
        "alphafold",  # Name of the Docker image
        "bash", "/app/alphafold/docker/run_alphafold.sh",
        f"--fasta_paths=/app/data/input.fasta",
        f"--output_dir=/app/output",
        f"--model_preset=monomer",
        f"--data_dir=/app/data",
        f"--db_preset=reduced_dbs",
        f"--use_gpu_relax=True"
    ]
    subprocess.run(command)

if __name__ == "__main__":
    run_alphafold()
