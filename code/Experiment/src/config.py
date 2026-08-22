"""
src/config.py

Loads a YAML experiment config and exposes it as a typed dataclass.
Also resolves the render_profiles and render_single_profile strings to the
actual functions.

Usage:
    from src.config import ExperimentConfig
    cfg = ExperimentConfig.from_yaml("configs/exp_01.yaml")

YAML keys for rendering:
    render_profiles:        render_profiles_markdown      # two-profile experiments
    render_single_profile:  render_single_profile_markdown  # single-profile experiments

Valid render_profiles values:
    render_profiles_markdown
    render_profiles_list
    render_profiles_flowtext

Valid render_single_profile values:
    render_single_profile_markdown
    render_single_profile_list
    render_single_profile_flowtext
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.profiles import (
    render_profiles_markdown,
    render_profiles_list,
    render_profiles_flowtext,
    render_single_profile_markdown,
    render_single_profile_list,
    render_single_profile_flowtext,
    set_field_labels_from_attributes,
)


# ---------------------------------------------------------------------------
# Registries: map YAML string names to the actual render functions
# ---------------------------------------------------------------------------

# Two-profile (classic conjoint) renderers
RENDER_REGISTRY: dict[str, Callable] = {
    "render_profiles_markdown": render_profiles_markdown,
    "render_profiles_list":     render_profiles_list,
    "render_profiles_flowtext": render_profiles_flowtext,
}

# Single-profile (vignette / accept-reject) renderers
SINGLE_RENDER_REGISTRY: dict[str, Callable] = {
    "render_single_profile_markdown":  render_single_profile_markdown,
    "render_single_profile_list":      render_single_profile_list,
    "render_single_profile_flowtext":  render_single_profile_flowtext,
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    experiment_id:      str
    job_title:          str
    model:              str
    system_prompt:      str
    instruction_prompt: str
    attributes:         dict[str, list[str]]
    render_profiles:    Callable          # used by experiment.py
    render_single_profile: Callable       # used by single_choice_experiment.py
    n_runs:             int

    output_dir:       Path
    checkpoint_every: int
    scenarios:        list | None = None
    random_seed:      int | None  = None

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # ---- two-profile renderer ----
        render_key = raw.get("render_profiles", "render_profiles_markdown")
        if render_key not in RENDER_REGISTRY:
            raise ValueError(
                f"Unknown render_profiles '{render_key}'. "
                f"Valid options: {list(RENDER_REGISTRY)}"
            )

        # ---- single-profile renderer ----
        single_key = raw.get("render_single_profile", "render_single_profile_markdown")
        if single_key not in SINGLE_RENDER_REGISTRY:
            raise ValueError(
                f"Unknown render_single_profile '{single_key}'. "
                f"Valid options: {list(SINGLE_RENDER_REGISTRY)}"
            )

        job_title = raw["job_title"]

        # ---- defaults ----
        output_dir       = Path(raw.get("output_dir", "outputs"))
        checkpoint_every = int(raw.get("checkpoint_every", 100))
        scenarios        = raw.get("scenarios", None)

        # Convert YAML attributes into FIELD_LABELS for rendering
        attributes = {k: [str(v) for v in vs] for k, vs in raw["attributes"].items()}
        set_field_labels_from_attributes(attributes)

        return cls(
            experiment_id         = raw["experiment_id"],
            job_title             = job_title,
            model                 = raw["model"],
            system_prompt         = raw["system_prompt"].format(job_title=job_title),
            instruction_prompt    = raw["instruction_prompt"],
            attributes            = attributes,
            render_profiles       = RENDER_REGISTRY[render_key],
            render_single_profile = SINGLE_RENDER_REGISTRY[single_key],
            n_runs                = int(raw["n_runs"]),
            random_seed           = raw.get("random_seed", None),
            output_dir            = output_dir,
            checkpoint_every      = checkpoint_every,
            scenarios             = scenarios,
        )