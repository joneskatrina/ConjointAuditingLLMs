"""
src/single_choice_preprocessing.py

Preprocessing for single-profile conjoint results. Reads the raw CSV
and writes a clean analysis-ready CSV with columns in the correct order:

    observation_id, scenario, <attributes...>, chosen

Handles both old raw CSVs (column: iteration, accepted) and new ones
(column: observation_id, chosen) transparently.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import ExperimentConfig


def _fill_missing(value: Any) -> Any:
    if value is None or value == "":
        return "Not Specified"
    return value


def _get_observation_id(row: dict[str, Any]) -> Any:
    """Support both old ('iteration') and new ('observation_id') column names."""
    if "observation_id" in row:
        return row["observation_id"]
    return row["iteration"]


def _normalize_chosen(row: dict[str, Any]) -> int | None:
    """Support both old ('accepted') and new ('chosen') column names."""
    # Try new name first, then fall back to old name
    raw = str(row.get("chosen", row.get("accepted", ""))).strip().lower()
    decision = str(row.get("decision", "")).strip().upper()

    if raw in {"1", "1.0", "yes", "true"} or decision == "ACCEPT":
        return 1
    if raw in {"0", "0.0", "no", "false"} or decision == "REJECT":
        return 0
    return None


def to_single_choice_long_format(
    rows: list[dict[str, Any]],
    attributes: list[str],
) -> tuple[list[dict[str, Any]], int]:
    """
    Convert raw single-choice rows to analysis-ready rows.

    Returns (processed_rows, skipped_count). Rows with invalid/missing
    decisions are skipped because analysis requires binary 0/1 outcomes.
    """
    processed = []
    skipped = 0

    for row in rows:
        chosen = _normalize_chosen(row)
        if chosen is None:
            skipped += 1
            continue

        out = {
            "observation_id": _get_observation_id(row),
            "scenario":       row["scenario"],
        }
        for attr in attributes:
            out[attr] = _fill_missing(row.get(f"profile_{attr}"))
        out["chosen"] = chosen  # always last
        processed.append(out)

    return processed, skipped


def preprocess_single_choice(cfg: "ExperimentConfig") -> list[dict[str, Any]]:
    raw_path = cfg.output_dir / "raw" / f"{cfg.experiment_id}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw results not found: {raw_path}")

    with open(raw_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {raw_path}")

    attributes = list(cfg.attributes.keys())
    processed, skipped = to_single_choice_long_format(rows, attributes)

    # Column order: observation_id, scenario, <attrs...>, chosen
    columns = ["observation_id", "scenario", *attributes, "chosen"]

    out_path = cfg.output_dir / "processed" / f"{cfg.experiment_id}_long.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(processed)

    print(f"\nProcessed data saved to: {out_path}")
    if skipped:
        print(f"Skipped {skipped} rows with invalid or missing decisions.")

    return processed