#!/usr/bin/env python3
"""Generate LLM-edited (or directly generated) responses to patient questions.

Input : CSV with a patient-question column and (for edit prompts) a
        physician-response column.
Output: the same CSV with one extra column holding the LLM output.

Examples
--------
# Edit physician responses with the refined prompt
python generate_responses.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --data-path ../data/example_qa.csv \
    --prompt refined \
    --output-column refined_edit \
    --output-path outputs/llama31_refined.csv

# Direct generation baseline (physician response is not shown to the model)
python generate_responses.py --model-name Qwen/Qwen3-8B \
    --data-path ../data/example_qa.csv --prompt direct \
    --output-column direct_response --output-path outputs/qwen3_direct.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from llm_utils import clean_text, extract_json, generate, load_model
from prompts import GENERATION_PROMPTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", required=True, help="Hugging Face model id or local path.")
    p.add_argument("--data-path", required=True, help="Input CSV.")
    p.add_argument("--output-path", required=True, help="Output CSV.")
    p.add_argument("--prompt", choices=sorted(GENERATION_PROMPTS), default="simple",
                   help="Prompt template (see prompts.py). Default: simple.")
    p.add_argument("--output-column", default=None,
                   help="Name of the new column. Default: '<prompt>_response'.")
    p.add_argument("--question-column", default="patient_question")
    p.add_argument("--physician-column", default="physician_response")
    p.add_argument("--device", default="auto", help="'auto', 'cuda:0', 'cpu', ...")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-rows", type=int, default=None, help="Only process the first N rows (debugging).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-chat-template", action="store_true",
                   help="Feed the raw prompt without the model's chat template.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    template, json_key = GENERATION_PROMPTS[args.prompt]
    out_col = args.output_column or f"{args.prompt}_response"

    data = pd.read_csv(args.data_path)
    if args.max_rows:
        data = data.head(args.max_rows)
    needed = [args.question_column] + ([] if args.prompt == "direct" else [args.physician_column])
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {args.data_path}: {missing}")
    print(f"Loaded {len(data)} rows from {args.data_path}")

    model, tokenizer = load_model(args.model_name, args.device)
    suffix = '{\n    "%s": "your_%s_here"\n    }' % (json_key, json_key)

    outputs, raw_outputs, n_failed = [], [], 0
    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Generating ({args.prompt})"):
        prompt = template.format(
            pq=clean_text(row[args.question_column]),
            r1=clean_text(row.get(args.physician_column, "")),
        ) + suffix
        raw = generate(model, tokenizer, prompt, args.max_new_tokens,
                       use_chat_template=not args.no_chat_template)
        parsed = extract_json(raw, json_key)
        if parsed is None:
            n_failed += 1
        outputs.append(parsed[json_key] if parsed else None)
        raw_outputs.append(raw)

    data[out_col] = outputs
    data[f"{out_col}_raw"] = raw_outputs
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_path, index=False)
    print(f"Saved {len(data)} rows to {args.output_path} "
          f"({n_failed} rows could not be parsed; '{out_col}' is empty for those).")


if __name__ == "__main__":
    main()
