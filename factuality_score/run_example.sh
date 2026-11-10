#!/usr/bin/env bash
# End-to-end MedFactChecking Score example.
#
#   1. Extract atomic facts from physician and edited responses.
#   2. Check entailment in both directions (recall + precision).
#   3. Aggregate into the factuality score.
#
# This script assumes you have already generated edited responses with
# empathy_score/generate_responses.py (or run_example.sh), producing a
# CSV that has both physician_response and at least one edit column.
#
# Usage:
#   bash run_example.sh
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct bash run_example.sh  # quick smoke test
#
# Gated models need `huggingface-cli login` or HF_TOKEN in your environment.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EDIT_COL="${EDIT_COL:-refined_edit}"
DATA="${DATA:-../empathy_score/outputs/example/responses.csv}"
OUT="${OUT:-outputs/example}"
DEVICE="${DEVICE:-auto}"
METHOD="${METHOD:-factehr_medical}"
PYTHON="${PYTHON:-python}"
mkdir -p "$OUT"

echo "== Step 1: extract facts (method=$METHOD)"
"$PYTHON" extract_facts.py \
    --model-name "$MODEL" --device "$DEVICE" \
    --data-path "$DATA" \
    --edited-column "$EDIT_COL" \
    --method "$METHOD" \
    --output-path "$OUT/facts.csv"

echo "== Step 2: check entailment"
"$PYTHON" check_entailment.py \
    --model-name "$MODEL" --device "$DEVICE" \
    --data-path "$OUT/facts.csv" \
    --edited-column "$EDIT_COL" \
    --output-path "$OUT/entailment.json"

echo "== Step 3: MedFactChecking Score"
"$PYTHON" compute_factuality_score.py "$OUT/entailment.json"
