#!/usr/bin/env python3
"""3EM-Ranker: pairwise empathy judge.

For every row, an LLM judge sees the patient question and two responses
(Response 1 = --response1-column, Response 2 = --response2-column) and
answers "Response 1", "Response 2" or "Equal" with a short reason.

Output: the input CSV plus columns
    judge_choice      normalized label: response_1 / response_2 / tie / invalid
    judge_choice_raw  the "choice" string returned by the judge
    judge_reason      the judge's reason
    judge_raw_output  full generated text
and a JSON summary (see compute_empathy_score.py for the metric).

Example
-------
python empathy_ranker.py \
    --judge-model Qwen/Qwen3-8B \
    --data-path outputs/pairs.csv \
    --response1-column physician_response \
    --response2-column refined_edit \
    --output-path outputs/judge_qwen3_physician_vs_refined.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from compute_empathy_score import normalize_choice, summarize
from llm_utils import clean_text, extract_json, generate, load_model
from prompts import (DEFAULT_FEWSHOT_FILE, JSON_SUFFIX_JUDGE, JUDGE_FEW_SHOT,
                     JUDGE_ZERO_SHOT, load_fewshot_block)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-model", required=True,
                   help="Judge model, e.g. Qwen/Qwen3-8B or meta-llama/Llama-3.1-8B-Instruct.")
    p.add_argument("--data-path", required=True, help="CSV containing both responses.")
    p.add_argument("--output-path", required=True, help="Per-row judgments CSV.")
    p.add_argument("--summary-path", default=None,
                   help="Summary JSON. Default: <output-path stem>_summary.json.")
    p.add_argument("--question-column", default="patient_question")
    p.add_argument("--response1-column", required=True)
    p.add_argument("--response2-column", required=True)
    p.add_argument("--few-shot", action="store_true", help="Use the few-shot judge prompt.")
    p.add_argument("--fewshot-file", default=str(DEFAULT_FEWSHOT_FILE),
                   help="JSON file with in-context examples for --few-shot.")
    p.add_argument("--exclude-column", default=None,
                   help="Optional column used to drop rows (with --exclude-value).")
    p.add_argument("--exclude-value", default=None, help="Rows where exclude-column == this value are dropped.")
    p.add_argument("--device", default="auto", help="'auto', 'cuda:0', 'cpu', ...")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-rows", type=int, default=None, help="Only judge the first N rows (debugging).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data = pd.read_csv(args.data_path)
    cols = [args.question_column, args.response1_column, args.response2_column]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {args.data_path}: {missing}")

    if args.exclude_column and args.exclude_value is not None:
        before = len(data)
        data = data[data[args.exclude_column].astype(str) != args.exclude_value]
        print(f"Excluded {before - len(data)} rows where {args.exclude_column} == {args.exclude_value!r}")

    before = len(data)
    data = data.dropna(subset=cols)
    if len(data) < before:
        print(f"Dropped {before - len(data)} rows with an empty question/response.")
    if args.max_rows:
        data = data.head(args.max_rows)
    data = data.reset_index(drop=True)
    print(f"Judging {len(data)} pairs: R1='{args.response1_column}' vs R2='{args.response2_column}'")

    if args.few_shot:
        examples = load_fewshot_block(args.fewshot_file)
        template = JUDGE_FEW_SHOT.replace("{examples}", examples)
    else:
        template = JUDGE_ZERO_SHOT

    model, tokenizer = load_model(args.judge_model, args.device)

    choices_raw, reasons, raw_outputs = [], [], []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Judging empathy"):
        prompt = template.format(
            pq=clean_text(row[args.question_column]),
            r1=clean_text(row[args.response1_column]),
            r2=clean_text(row[args.response2_column]),
        ) + JSON_SUFFIX_JUDGE
        raw = generate(model, tokenizer, prompt, args.max_new_tokens)
        parsed = extract_json(raw, "choice") or {}
        choices_raw.append(parsed.get("choice", ""))
        reasons.append(parsed.get("reason", ""))
        raw_outputs.append(raw)

    data["judge_choice_raw"] = choices_raw
    data["judge_choice"] = [normalize_choice(c) for c in choices_raw]
    data["judge_reason"] = reasons
    data["judge_raw_output"] = raw_outputs

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)

    summary = summarize(data["judge_choice"])
    summary.update({
        "judge_model": args.judge_model,
        "mode": "few-shot" if args.few_shot else "zero-shot",
        "data_path": args.data_path,
        "response_1": args.response1_column,
        "response_2": args.response2_column,
    })
    summary_path = Path(args.summary_path) if args.summary_path else out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Per-row judgments: {out}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
