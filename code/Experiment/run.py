"""
run.py — CLI entrypoint for the conjoint experiment pipeline.

Usage:
    # Run the full pipeline (experiment → preprocess)
    Open Terminal in ~/Thesis/code/modular$ 
    then enter: ~/myenv/bin/python ~/Thesis/code/modular/run.py configs/exp_01.yaml


    # Run only specific stages
    ~/myenv/bin/python ~/Thesis/code/modular/run.py configs/exp_01.yaml --stage experiment
    ~/myenv/bin/python ~/Thesis/code/modular/run.py configs/exp_01.yaml --stage preprocess

    
"""

import argparse
import sys
from pathlib import Path

from src.config import ExperimentConfig


def run_experiment_stage(cfg: ExperimentConfig):
    from src.experiment import run_experiment
    run_experiment(cfg)


def run_preprocess_stage(cfg: ExperimentConfig):
    from src.preprocessing import preprocess
    preprocess(cfg)


        
def main():
    parser = argparse.ArgumentParser(
        description="Conjoint experiment pipeline"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment YAML config (e.g. configs/exp_01.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=["experiment", "preprocess", "analysis", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Loaded config: {cfg.experiment_id}")

    stages = {
        "experiment": run_experiment_stage,
        "preprocess": run_preprocess_stage,

    }

    if args.stage == "all":
        for name, fn in stages.items():
            print(f"\n{'='*50}")
            print(f"  Stage: {name}")
            print(f"{'='*50}")
            fn(cfg)
    else:
        stages[args.stage](cfg)


if __name__ == "__main__":
    main()
