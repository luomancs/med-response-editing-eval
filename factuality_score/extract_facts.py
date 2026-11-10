#!/usr/bin/env python3
"""Decompose physician and edited responses into atomic facts.

Input : CSV with patient_question, physician_response, and an edited-response
        column (produced by empathy_score/generate_responses.py).
Output: the same CSV with two new columns holding double-slash-delimited fact lists.

Examples
--------
python extract_facts.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --data-path ../empathy_score/outputs/example/responses.csv \
    --edited-column refined_edit \
    --method factehr_medical \
    --output-path outputs/facts.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factuality_score.prompts import SYSTEM_MEDICAL_EXPERT, get_decomposition_prompt


def parse_factehr_facts(text: str) -> list[str]:
    """Parse // -delimited fact list (factehr* methods)."""
    if "exception happens" in text.lower():
        return []
    if "Atomic facts:" in text:
        text = text.split("Atomic facts:", 1)[1]
    parts = text.split("//") if "//" in text else text.split("\n")
    facts = []
    for p in parts:
        p = re.sub(r"^(?:\d+[.)]\s+|[-–•]\s+)", "", p.strip()).strip()
        if p and not p.startswith("Note:") and not p.startswith("Facts:"):
            facts.append(p)
    return facts


def parse_medscore_facts(text: str) -> list[str]:
    """Parse dash-delimited fact list (medscore_verifiable)."""
    if "exception happens" in text.lower() or "No verifiable claim" in text:
        return []
    if "Facts:" in text:
        text = text.split("Facts:", 1)[1]
    facts = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            fact = line[1:].strip()
            if fact and fact != "No verifiable claim":
                facts.append(fact)
        elif line and not line.lower().startswith(("facts:", "context:", "please")):
            if line != "No verifiable claim":
                facts.append(line)
    return facts


def extract(model, tokenizer, text: str, method: str, context: str = None,
            max_new_tokens: int = 512) -> list[str]:
    from empathy_score.llm_utils import generate

    prompt = get_decomposition_prompt(text, method=method, context=context)
    prompt = f"{SYSTEM_MEDICAL_EXPERT}\n\n{prompt}"
    raw = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    if method.startswith("medscore"):
        return parse_medscore_facts(raw)
    return parse_factehr_facts(raw)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--edited-column", required=True,
                   help="Column with the edited response (e.g. 'refined_edit').")
    p.add_argument("--question-column", default="patient_question")
    p.add_argument("--physician-column", default="physician_response")
    p.add_argument("--method", default="factehr_medical",
                   choices=["factehr", "factehr_medical", "factehr_medical_adv",
                            "medscore_verifiable"])
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from empathy_score.llm_utils import load_model

    torch.manual_seed(args.seed)

    data = pd.read_csv(args.data_path)
    needed = [args.question_column, args.physician_column, args.edited_column]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if args.max_rows:
        data = data.head(args.max_rows)
    data = data.dropna(subset=needed).reset_index(drop=True)
    print(f"Extracting facts from {len(data)} rows (method={args.method})")

    model, tokenizer = load_model(args.model_name, args.device)

    orig_facts_col, edit_facts_col = [], []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Extracting facts"):
        context = row[args.question_column]
        orig = extract(model, tokenizer, row[args.physician_column],
                       args.method, context, args.max_new_tokens)
        edit = extract(model, tokenizer, row[args.edited_column],
                       args.method, context, args.max_new_tokens)
        orig_facts_col.append(" // ".join(orig))
        edit_facts_col.append(" // ".join(edit))

    data["original_facts"] = orig_facts_col
    data["edited_facts"] = edit_facts_col

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_path, index=False)
    print(f"Saved {len(data)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
