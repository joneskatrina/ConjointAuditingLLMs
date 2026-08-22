"""
src/single_choice_experiment.py

Single-profile conjoint experiment loop. This is an additive alternative to
src/experiment.py: each task shows one profile and asks whether it should be
accepted/selected.

Produces a CSV in cfg.output_dir / raw / {cfg.experiment_id}.csv
"""

from __future__ import annotations

import csv
import random
import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import profiles
from src.model import APIModel, _json_post, load_model

if TYPE_CHECKING:
    from src.config import ExperimentConfig


ACCEPT_LABEL = "ACCEPT"
REJECT_LABEL = "REJECT"
INVALID_LABEL = "INVALID"


def _build_columns(attributes: dict) -> list[str]:
    attr_cols = [f"profile_{a}" for a in attributes]
    return [
        "iteration",
        "scenario",
        *attr_cols,
        "decision",
        "accepted",
        "logprob_accept",
        "logprob_reject",
    ]


def _save_checkpoint(rows: list[dict[str, Any]], path: Path, columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_single_profile_markdown(profile: dict) -> str:
    fields = profiles.FIELD_LABELS.copy()
    random.shuffle(fields)

    header = "| Attribute | Candidate |"
    separator = "|-----------|-----------|"
    rows = [f"| {label} | {profile[key]} |" for label, key in fields]
    return "\n".join([header, separator] + rows)


def render_single_profile_list(profile: dict) -> str:
    fields = profiles.FIELD_LABELS.copy()
    random.shuffle(fields)
    rows = [f"- {label}: {profile[key]}" for label, key in fields]
    return "Candidate profile:\n" + "\n".join(rows)


def _extract_single_choice(text: str) -> str:
    text = (text or "").strip().upper()
    normalized = re.sub(r"[^A-Z]+", " ", text).strip()
    if normalized in {"Y", "YES", "ACCEPT", "SELECT", "HIRE", "ADMIT"}:
        return ACCEPT_LABEL
    if normalized in {"N", "NO", "REJECT", "DECLINE", "DO NOT HIRE", "DO NOT ADMIT"}:
        return REJECT_LABEL

    first_word = normalized.split(" ", 1)[0] if normalized else ""
    if first_word in {"Y", "YES", "ACCEPT", "SELECT", "HIRE", "ADMIT"}:
        return ACCEPT_LABEL
    if first_word in {"N", "NO", "REJECT", "DECLINE"}:
        return REJECT_LABEL
    return INVALID_LABEL


def _api_single_choice(model: APIModel, system_text: str, user_text: str) -> dict:
    prompt = (
        f"{user_text.strip()}\n\n"
        "Respond with exactly one character: Y for accept/select, or N for reject/decline. "
        "Do not explain."
    )

    if model.provider == "openai":
        payload = {
            "model": model.model_name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            "temperature": model.temperature,
            "max_completion_tokens": model.max_output_tokens,
            #"logprobs": True,
            #"top_logprobs": 5,
        }
        response = _json_post(
            "https://api.openai.com/v1/chat/completions",
            {
                "Authorization": f"Bearer {model.api_key}",
                "Content-Type": "application/json",
            },
            payload,
        )
        choice = response["choices"][0]
        text = choice["message"].get("content", "")
        logprob_accept = None
        logprob_reject = None
        content_logprobs = (choice.get("logprobs") or {}).get("content") or []
        if content_logprobs:
            top_logprobs = content_logprobs[0].get("top_logprobs") or []
            for item in top_logprobs:
                token = item.get("token", "").strip().upper()
                if token == "Y":
                    logprob_accept = item.get("logprob")
                elif token == "N":
                    logprob_reject = item.get("logprob")
        return {
            "decision": _extract_single_choice(text),
            "logprob_accept": logprob_accept,
            "logprob_reject": logprob_reject,
        }

    if model.provider in {"gemini", "google"}:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model.model_name}:generateContent?key={model.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": model.temperature,
                "maxOutputTokens": model.max_output_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        response = _json_post(url, {"Content-Type": "application/json"}, payload)
        parts = response["candidates"][0]["content"].get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
        return {
            "decision": _extract_single_choice(text),
            "logprob_accept": None,
            "logprob_reject": None,
        }

    if model.provider in {"anthropic", "claude"}:
        payload = {
            "model": model.model_name,
            "system": system_text,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": model.temperature,
            "max_tokens": model.max_output_tokens,
        }
        response = _json_post(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": model.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload,
        )
        text = "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        )
        return {
            "decision": _extract_single_choice(text),
            "logprob_accept": None,
            "logprob_reject": None,
        }

    raise ValueError(f"Unsupported API provider: {model.provider}")


def get_single_choice_logprob(tokenizer, model, system_text: str, user_text: str) -> dict:
    """
    Return the model's accept/reject decision for a single profile.

    For local HuggingFace models, the decision is based on next-token logprobs
    for Y versus N. For hosted APIs, logprobs are captured where the provider
    exposes them through this repo's simple API client.
    """
    if isinstance(model, APIModel):
        return _api_single_choice(model, system_text, user_text)

    import torch
    import torch.nn.functional as F

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(model.device)
    else:
        input_ids = encoded.to(model.device)

    suffix_ids = tokenizer.encode(
        "Answer: ",
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    input_ids = torch.cat([input_ids, suffix_ids], dim=-1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    next_token_logits = outputs.logits[0, -1, :]
    log_probs = F.log_softmax(next_token_logits, dim=-1)

    def get_ids(label: str) -> list[int]:
        ids = []
        for candidate in [label, f" {label}"]:
            encoded_candidate = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded_candidate) == 1:
                ids.append(encoded_candidate[0])
        return ids

    ids_accept = get_ids("Y")
    ids_reject = get_ids("N")
    if not ids_accept or not ids_reject:
        return {
            "decision": INVALID_LABEL,
            "logprob_accept": None,
            "logprob_reject": None,
        }

    lp_accept = log_probs[ids_accept].max().item()
    lp_reject = log_probs[ids_reject].max().item()

    return {
        "decision": ACCEPT_LABEL if lp_accept >= lp_reject else REJECT_LABEL,
        "logprob_accept": lp_accept,
        "logprob_reject": lp_reject,
    }


def _get_renderer(cfg: "ExperimentConfig"):
    name = getattr(cfg.render_profiles, "__name__", "")
    if name.endswith("list") or name.endswith("flowtext"):
        return render_single_profile_list
    return render_single_profile_markdown


def run_single_choice_experiment(cfg: "ExperimentConfig"):
    """
    Run a single-profile conjoint experiment defined by cfg.
    Saves raw results to: cfg.output_dir / raw / {cfg.experiment_id}.csv
    """
    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)

    tokenizer, model = load_model(cfg.model)

    columns = _build_columns(cfg.attributes)
    output_path = cfg.output_dir / "raw" / f"{cfg.experiment_id}.csv"
    counts = {ACCEPT_LABEL: 0, REJECT_LABEL: 0, INVALID_LABEL: 0}
    invalid_streak = 0
    results = []
    render_profile = _get_renderer(cfg)

    print(f"\nSingle-choice experiment : {cfg.experiment_id}")
    print(f"Model                    : {cfg.model}")
    print(f"Iterations               : {cfg.n_runs}")
    print(f"Output                   : {output_path}\n")

    for i in range(1, cfg.n_runs + 1):
        scenario = random.choice(cfg.scenarios) if cfg.scenarios else {"id": 1}
        row = {"iteration": i, "scenario": scenario.get("id", 1)}

        try:
            profile = {
                attr: random.choice(levels)
                for attr, levels in cfg.attributes.items()
            }

            for attr, val in profile.items():
                row[f"profile_{attr}"] = val

            profile_text = render_profile(profile)
            user_text = profile_text + "\n\n" + cfg.instruction_prompt
            result = get_single_choice_logprob(
                tokenizer,
                model,
                system_text=cfg.system_prompt,
                user_text=user_text,
            )

            decision = result["decision"]
            row["decision"] = decision
            row["accepted"] = (
                1 if decision == ACCEPT_LABEL
                else 0 if decision == REJECT_LABEL
                else None
            )
            row["logprob_accept"] = result["logprob_accept"]
            row["logprob_reject"] = result["logprob_reject"]
            counts[decision] = counts.get(decision, 0) + 1
            invalid_streak = invalid_streak + 1 if decision == INVALID_LABEL else 0

        except Exception:
            traceback.print_exc()
            for col in columns:
                row.setdefault(col, "INVALID")
            row.update(
                {
                    "decision": INVALID_LABEL,
                    "accepted": None,
                    "logprob_accept": None,
                    "logprob_reject": None,
                }
            )
            counts[INVALID_LABEL] = counts.get(INVALID_LABEL, 0) + 1
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

    _save_checkpoint(results, output_path, columns)

    valid = counts.get(ACCEPT_LABEL, 0) + counts.get(REJECT_LABEL, 0)
    print(f"\n=== Done: {cfg.experiment_id} ===")
    print(
        f"Valid   : {valid}  "
        f"(accept={counts.get(ACCEPT_LABEL, 0)}, reject={counts.get(REJECT_LABEL, 0)})"
    )
    print(f"Invalid : {counts.get(INVALID_LABEL, 0)}")
    print(f"Saved   : {output_path.resolve()}")