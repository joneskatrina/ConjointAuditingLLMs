"""
src/model.py

Loads a HuggingFace causal LM or a hosted API model and exposes one
get_choice_logprob() function. Model and tokenizer are loaded once and reused
across all iterations.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


API_PROVIDERS = {"openai", "anthropic", "claude", "gemini", "google"}


@dataclass
class APIModel:
    provider: str
    model_name: str
    max_output_tokens: int = 10
    temperature: float = 0.0

    @property
    def api_key(self) -> str:
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "google": "GEMINI_API_KEY",
        }
        env_var = env_vars[self.provider]
        value = os.getenv(env_var)
        if not value:
            raise RuntimeError(
                f"Missing {env_var}. Set it before running this hosted model."
            )
        return value


def _split_api_model(model_name: str) -> tuple[str, str] | None:
    if ":" not in model_name:
        return None
    provider, hosted_model = model_name.split(":", 1)
    provider = provider.strip().lower()
    hosted_model = hosted_model.strip()
    if provider not in API_PROVIDERS or not hosted_model:
        return None
    return provider, hosted_model


def load_model(model_name: str):
    """
    Load tokenizer + model onto CUDA in bfloat16, or configure an API model.
    Returns (tokenizer, model).

    Hosted API model names use provider prefixes:
      - openai:gpt-4.1-mini
      - gemini:gemini-3.1-flash-lite-preview
      - anthropic:claude-haiku-4-5-20251001
    """
    api_model = _split_api_model(model_name)
    if api_model:
        provider, hosted_model = api_model
        return None, APIModel(provider=provider, model_name=hosted_model)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        offload_buffers=True,
    )
    model.eval()
    return tokenizer, model


def _json_post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed ({exc.code}): {body}") from exc


def _extract_choice(text: str) -> str:
    text = (text or "").strip()
    if text.upper() in {"A", "B"}:
        return text.upper()
    match = re.search(r"\b([AB])\b", text.upper())
    return match.group(1) if match else "INVALID"


def _choice_from_api(model: APIModel, system_text: str, user_text: str) -> dict:
    prompt = (
        f"{user_text.strip()}\n\n"
        "Respond with exactly one character: A or B. Do not explain."
    )

    if model.provider == "openai":
        payload = {
            "model": model.model_name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            "temperature": model.temperature,
            "max_tokens": model.max_output_tokens,
            "logprobs": True,
            "top_logprobs": 5,
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
        logprob_a = None
        logprob_b = None
        content_logprobs = (choice.get("logprobs") or {}).get("content") or []
        if content_logprobs:
            top_logprobs = content_logprobs[0].get("top_logprobs") or []
            for item in top_logprobs:
                token = item.get("token", "").strip().upper()
                if token == "A":
                    logprob_a = item.get("logprob")
                elif token == "B":
                    logprob_b = item.get("logprob")
        return {
            "choice": _extract_choice(text),
            "logprob_a": logprob_a,
            "logprob_b": logprob_b,
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
        return {"choice": _extract_choice(text), "logprob_a": None, "logprob_b": None}

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
        return {"choice": _extract_choice(text), "logprob_a": None, "logprob_b": None}

    raise ValueError(f"Unsupported API provider: {model.provider}")


def get_choice_logprob(tokenizer, model, system_text: str, user_text: str) -> dict:
    """
    Run a single forward pass and return log-probs for 'A' and 'B'
    as the next token. Choice is whichever has the higher log-prob.

    Uses enable_thinking=False (Qwen3-specific) to suppress chain-of-thought,
    and pre-fills the assistant turn so 'A'/'B' is the natural next token.

    Returns:
        {
          "choice":    "A" or "B",
          "logprob_a": float,
          "logprob_b": float,
        }
    """
    if isinstance(model, APIModel):
        return _choice_from_api(model, system_text, user_text)

    import torch
    import torch.nn.functional as F

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    # Handle both raw tensor and BatchEncoding
    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(model.device)
    else:
        input_ids = encoded.to(model.device)

    # Pre-fill the assistant turn so the next token is naturally "A" or "B"
    suffix_ids = tokenizer.encode(
        "Answer: ", #"The better profile is profile / I would hire candidate",
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    input_ids = torch.cat([input_ids, suffix_ids], dim=-1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    next_token_logits = outputs.logits[0, -1, :]
    log_probs = F.log_softmax(next_token_logits, dim=-1)

    def get_ids(letter: str) -> list[int]:
        candidates = [letter, f" {letter}"]
        ids = []
        for c in candidates:
            enc = tokenizer.encode(c, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        return ids

    ids_a = get_ids("A")
    ids_b = get_ids("B")

    if not ids_a or not ids_b:
        return {"choice": "INVALID", "logprob_a": None, "logprob_b": None}

    lp_a = log_probs[ids_a].max().item()
    lp_b = log_probs[ids_b].max().item()
    
    
    # ── debug shows top 5 tokens with highest log prob ────────────────────────────────────────────────────────────────
    print(f"ids_a={ids_a}  →  {[tokenizer.decode([i]) for i in ids_a]}")
    print(f"ids_b={ids_b}  →  {[tokenizer.decode([i]) for i in ids_b]}")
    print(f"lp_a={lp_a:.4f},  lp_b={lp_b:.4f}")
    top5 = log_probs.topk(5)
    for score, idx in zip(top5.values, top5.indices):
        print(f"  {repr(tokenizer.decode([idx.item()]))}: {score.item():.4f}") # last time i run it it was " A", " B", " "" ", "**",  -> so     A or B spelled big
    # ─────────────────────────────────────────────────────────────────────────
    
    return {
        "choice":    "A" if lp_a >= lp_b else "B",
        "logprob_a": lp_a,
        "logprob_b": lp_b,
    }
