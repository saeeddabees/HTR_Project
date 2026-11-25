
import subprocess
import sys
import os
from pathlib import Path


# ==========================================
# Project Configuration
# ==========================================

# Automatically detect the project root
project_dir = Path(__file__).resolve().parent.parent

env = os.environ.copy()
env["PYTHONPATH"] = (
    str(project_dir)
    + os.pathsep
    + env.get("PYTHONPATH", "")
)


# ==========================================
# Hyperparameter Experiments
# ==========================================

experiments = [
    {
        "lr": "0.0005",
        "seed": 0,
        "name": "hp_attention_lr0005"
    },
    {
        "lr": "0.001",
        "seed": 0,
        "name": "hp_attention_lr001"
    },
    {
        "lr": "0.002",
        "seed": 0,
        "name": "hp_attention_lr002"
    },
    {
        "lr": "0.002",
        "seed": 1,
        "name": "hp_attention_lr002_seed1"
    },
    {
        "lr": "0.002",
        "seed": 2,
        "name": "hp_attention_lr002_seed2"
    }
]


# ==========================================
# Run Experiments
# ==========================================

total_experiments = len(experiments)

print("=" * 65)
print("HYPERPARAMETER SEARCH")
print("=" * 65)
print(f"Project directory: {project_dir}")
print(f"Total experiments: {total_experiments}")
print("=" * 65)


for index, exp in enumerate(experiments, start=1):

    print("\n" + "=" * 65)
    print(f"Experiment {index}/{total_experiments}")
    print(f"Name: {exp['name']}")
    print(f"Learning Rate: {exp['lr']}")
    print(f"Seed: {exp['seed']}")
    print("=" * 65, flush=True)

    command = [
        sys.executable,
        "-m", "src.train",
        "--model", "attention",
        "--epochs", "10",
        "--lr", exp["lr"],
        "--batch_size", "8",
        "--seed", str(exp["seed"]),
        "--name", exp["name"]
    ]

    subprocess.run(
        command,
        cwd=str(project_dir),
        env=env,
        check=True
    )

    print(f"\nCompleted successfully: {exp['name']}", flush=True)


# ==========================================
# Finished
# ==========================================

print("\n" + "=" * 65)
print("ALL HYPERPARAMETER EXPERIMENTS COMPLETED!")
print("=" * 65)
