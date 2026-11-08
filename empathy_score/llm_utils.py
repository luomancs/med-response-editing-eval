"""Shared helpers for loading a Hugging Face chat model and running generation."""

import json
import os
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str = "auto", hf_token: Optional[str] = None):
    """Load a causal LM and its tokenizer.

    `device` is either "auto" (let accelerate place the model across available
    GPUs) or an explicit torch device string such as "cuda:0" or "cpu".
    The Hugging Face token is read from `hf_token` or the HF_TOKEN env var;
    it is only needed for gated models (e.g. Llama 3.1).
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {"torch_dtype": "auto", "token": token}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512,
             use_chat_template: bool = True) -> str:
    """Greedy-decode a completion for a single user prompt."""
    if use_chat_template and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_json(text: str, required_key: str) -> Optional[dict]:
    """Return the last flat JSON object in `text` that contains `required_key`.

    Reasoning blocks (<think>...</think>, emitted by e.g. Qwen3) are stripped
    first so that braces inside the reasoning are not picked up.
    """
    text = _THINK_RE.sub("", text)
    for candidate in reversed(_JSON_RE.findall(text)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and required_key in obj:
            return obj
    return None


def clean_text(text) -> str:
    """Collapse a free-text field onto a single line."""
    return " ".join(str(text).split())
