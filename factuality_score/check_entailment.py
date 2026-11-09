#!/usr/bin/env python3
"""Check entailment for every (premise, fact) pair produced by extract_facts.py.

For each row the script runs two directions:
  Recall    — premise = edited response,   hypothesis = each original fact
  Precision — premise = original response,  hypothesis = each edited fact

Output: a JSON file with per-pair results and a summary CSV.

Example
-------
python check_entailment.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --data-path outputs/facts.csv \
    --edited-column refined_edit \
    --output-path outputs/entailment.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "empathy_score"))
from llm_utils import extract_json, generate, load_model

from prompts import SYSTEM_FACT_CHECKER, get_entailment_prompt


def check(model, tokenizer, premise: str, hypothesis: str,
          max_new_tokens: int = 64) -> bool:
    """Return True if *premise* entails *hypothesis*."""
    prompt = f"{SYSTEM_FACT_CHECKER}\n\n{get_entailment_prompt(premise, hypothesis)}"
    raw = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    parsed = extract_json(raw, "entailment_prediction")
    if parsed is not None:
        return parsed.get("entailment_prediction", 0) == 1
    return "1" in raw


def split_facts(cell: str) -> list[str]:
    """Split a // -delimited fact string back into a list."""
    if pd.isna(cell) or not cell.strip():
        return []
    return [f.strip() for f in cell.split("//") if f.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", required=True)
    p.add_argument("--data-path", required=True,
                   help="CSV produced by extract_facts.py (must have original_facts, edited_facts).")
    p.add_argument("--output-path", required=True, help="Per-pair JSON results.")
    p.add_argument("--summary-path", default=None, help="Summary CSV. Default: <output stem>_summary.csv.")
    p.add_argument("--edited-column", required=True,
                   help="Column with the edited response text.")
    p.add_argument("--question-column", default="patient_question")
    p.add_argument("--physician-column", default="physician_response")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    import torch; torch.manual_seed(args.seed)

    data = pd.read_csv(args.data_path)
    for col in ["original_facts", "edited_facts", args.physician_column, args.edited_column]:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")
    if args.max_rows:
        data = data.head(args.max_rows)
    data = data.reset_index(drop=True)
    print(f"Checking entailment for {len(data)} rows")

    model, tokenizer = load_model(args.model_name, args.device)

    results = []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Entailment"):
        orig_facts = split_facts(row["original_facts"])
        edit_facts = split_facts(row["edited_facts"])
        edited_text = str(row[args.edited_column])
        original_text = str(row[args.physician_column])

        # Recall: does edited text preserve original facts?
        preservation = []
        for fact in orig_facts:
            preserved = check(model, tokenizer, edited_text, fact, args.max_new_tokens)
            preservation.append({"fact": fact, "preserved": preserved})

        # Precision: are edited facts grounded in the original?
        additions = []
        for fact in edit_facts:
            grounded = check(model, tokenizer, original_text, fact, args.max_new_tokens)
            additions.append({"fact": fact, "is_new": not grounded})

        preserved_n = sum(1 for p in preservation if p["preserved"])
        new_n = sum(1 for a in additions if a["is_new"])
        recall = preserved_n / len(orig_facts) if orig_facts else 1.0
        grounded_n = len(edit_facts) - new_n
        precision = grounded_n / len(edit_facts) if edit_facts else 1.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        results.append({
            "pair_id": idx,
            "question": row.get(args.question_column, ""),
            "original_fact_count": len(orig_facts),
            "edited_fact_count": len(edit_facts),
            "preserved_facts": preserved_n,
            "new_facts": new_n,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "preservation_details": preservation,
            "addition_details": additions,
        })

    # Save per-pair results
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]
    f1s = [r["f1_score"] for r in results]

    total_preserved = sum(r["preserved_facts"] for r in results)
    total_orig = sum(r["original_fact_count"] for r in results)
    total_grounded = sum(r["edited_fact_count"] - r["new_facts"] for r in results)
    total_edited = sum(r["edited_fact_count"] for r in results)

    micro_recall = total_preserved / total_orig if total_orig else 1.0
    micro_precision = total_grounded / total_edited if total_edited else 1.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)

    summary = pd.DataFrame([{
        "n_pairs": len(results),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "micro_recall": round(micro_recall, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_f1": round(micro_f1, 4),
    }])

    summary_path = Path(args.summary_path) if args.summary_path else out.with_suffix(".csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nPer-pair results: {out}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
