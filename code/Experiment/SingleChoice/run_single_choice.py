"""
run_single_choice.py - CLI entrypoint for the single-profile conjoint pipeline.
Usage:
cd to: jovyan@jupyter-joneskatrina:~/Thesis/code/modular$ 

~/myenv/bin/python ~/Thesis/code/modular/run_single_choice.py configs/L_Qwen_Software_Engineer.yaml
single_choice_Triage.yaml
"""

import argparse

from src.config import ExperimentConfig


def run_experiment_stage(cfg: ExperimentConfig):
    from src.single_choice_experiment import run_single_choice_experiment

    run_single_choice_experiment(cfg)


def run_preprocess_stage(cfg: ExperimentConfig):
    from src.single_choice_preprocessing import preprocess_single_choice

    preprocess_single_choice(cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Single-profile conjoint experiment pipeline"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment YAML config, e.g. configs/single_choice_teacher.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=["experiment", "preprocess", "all"],
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
            print(f"\n{'=' * 50}")
            print(f"  Stage: {name}")
            print(f"{'=' * 50}")
            fn(cfg)
    else:
        stages[args.stage](cfg)


if __name__ == "__main__":
    main()
