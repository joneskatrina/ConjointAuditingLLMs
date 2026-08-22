"""
src/experiment.py

Core experiment loop. Reads everything it needs from an ExperimentConfig —
no hardcoded attributes, prompts, or model names.

Produces a CSV in cfg.output_dir / raw / {experiment_id}.csv
"""

import re
import csv
import random
import traceback
from pathlib import Path

from src.config import ExperimentConfig
from src.model import load_model, get_choice_logprob


    
# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------

def _build_columns(attributes: dict) -> list[str]:
    attr_cols = (
        [f"a_{a}" for a in attributes]
        + [f"b_{a}" for a in attributes]
    )
    return ["iteration", "scenario", *attr_cols, "choice", "logprob_a", "logprob_b"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(rows: list, path: Path, columns: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(cfg: ExperimentConfig):
    """
    Run the full conjoint experiment defined by cfg.
    Saves raw results to: cfg.output_dir / raw / {experiment_id}.csv
    """
    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)
    
    tokenizer, model = load_model(cfg.model)

    columns     = _build_columns(cfg.attributes)
    output_path = cfg.output_dir / "raw" / f"{cfg.experiment_id}.csv"
    counts      = {"A": 0, "B": 0, "INVALID": 0}
    invalid_streak = 0
    results     = []

    print(f"\nExperiment : {cfg.experiment_id}")
    print(f"Model      : {cfg.model}")
    print(f"Iterations : {cfg.n_runs}")
    print(f"Output     : {output_path}\n")

    for i in range(1, cfg.n_runs + 1):

        # One scenario per iteration (extend this if you have multiple scenarios)
        scenario = random.choice(cfg.scenarios) if cfg.scenarios else {"id": 1}
        row = {"iteration": i, "scenario": scenario.get("id", 1)}

        try:
            profile_a = {attr: random.choice(levels)
                         for attr, levels in cfg.attributes.items()}
            profile_b = {attr: random.choice(levels)
                         for attr, levels in cfg.attributes.items()}

            for attr, val in profile_a.items():
                row[f"a_{attr}"] = val
            for attr, val in profile_b.items():
                row[f"b_{attr}"] = val

            profiles_text = cfg.render_profiles(profile_a, profile_b)
            user_text = profiles_text + "\n\n" + cfg.instruction_prompt
            result = get_choice_logprob(tokenizer, model, system_text=cfg.system_prompt, user_text=user_text)

            row["choice"]    = result["choice"]
            row["logprob_a"] = result["logprob_a"]
            row["logprob_b"] = result["logprob_b"]
            counts[result["choice"]] = counts.get(result["choice"], 0) + 1
            invalid_streak = invalid_streak + 1 if result["choice"] == "INVALID" else 0


        except Exception:
            traceback.print_exc()
            for col in columns:
                row.setdefault(col, "INVALID")
            row.update({"choice": "INVALID", "logprob_a": None, "logprob_b": None})
            counts["INVALID"] = counts.get("INVALID", 0) + 1
            invalid_streak += 1

        results.append(row)

        if invalid_streak >= 5:
            _save_checkpoint(results, output_path, columns)
            raise RuntimeError(
                "Stopping after 5 consecutive INVALID results. "
                "Check the API/model before spending more calls."
            )


        if i % cfg.checkpoint_every == 0:
            _save_checkpoint(results, output_path, columns)    
            #print(
            #    f"[{i}/{cfg.n_runs}]  "
            #    f"A={counts.get('A',0)}  "
            #    f"B={counts.get('B',0)}  "
            #    f"INVALID={counts.get('INVALID',0)}"
            #)

    _save_checkpoint(results, output_path, columns)

    valid = counts.get("A", 0) + counts.get("B", 0)
    print(f"\n=== Done: {cfg.experiment_id} ===")
    print(f"Valid   : {valid}  (A={counts.get('A',0)}, B={counts.get('B',0)})")
    print(f"Invalid : {counts.get('INVALID', 0)}")
    print(f"Saved   : {output_path.resolve()}")
