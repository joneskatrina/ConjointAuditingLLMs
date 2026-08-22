"""
src/preprocessing.py

Cleans the raw experiment CSV and reshapes it into long format
for conjoint analysis. Produces:
  cfg.output_dir / processed / {experiment_id}_long.csv

Usage:
    from src.config import ExperimentConfig
    from src.preprocessing import preprocess

    cfg = ExperimentConfig.from_yaml("configs/exp_01.yaml")
    df_long = preprocess(cfg)
"""

import pandas as pd
from pathlib import Path

from src.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with 'Not Specified' (intentional missing in conjoint design)."""
    return df.fillna("Not Specified")


# ---------------------------------------------------------------------------
# Long format
# ---------------------------------------------------------------------------

def to_long_format(df: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    """
    Reshape wide-format conjoint results into long format.
    Each iteration produces two rows (one per profile).
    """
    base_cols = ["iteration", "scenario"]

    def make_profile_df(profile: str) -> pd.DataFrame:
        letter = profile.upper()
        rename = {f"{profile}_{a}": a for a in attributes}
        df_p = df[base_cols + list(rename.keys()) + ["choice"]].copy()
        df_p.rename(columns=rename, inplace=True)
        df_p["profile"] = letter
        df_p["chosen"]  = (df["choice"] == letter).astype(int)
        return df_p[base_cols + ["profile"] + attributes + ["chosen"]]

    df_long = pd.concat(
        [make_profile_df("a"), make_profile_df("b")],
        ignore_index=True,
    )
    df_long.rename(columns={"iteration": "observation_id"}, inplace=True)
    return df_long


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess(cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Full preprocessing pipeline for one experiment config.
    Reads raw CSV → cleans → long format → saves → returns df_long.
    """
    raw_path = cfg.output_dir / "raw" / f"{cfg.experiment_id}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw results not found: {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows from {raw_path}")

    df = fill_missing(df)

    attributes = list(cfg.attributes.keys())
    df_long = to_long_format(df, attributes)

    out_path = cfg.output_dir / "processed" / f"{cfg.experiment_id}_long.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(out_path, index=False)
    print(f"\nLong-format data saved to: {out_path}")

    return df_long
