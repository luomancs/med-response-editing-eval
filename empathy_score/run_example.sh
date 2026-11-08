#!/usr/bin/env bash
# End-to-end Empathy Ranking Score example on the synthetic data in ../data.
#
#   1. An editor LLM rewrites each physician response (simple + refined prompts).
#   2. A judge LLM compares physician vs. edited responses (3EM-Ranker).
#   3. Scores are aggregated into A>>B / B>>A / tie percentages.
#
# Usage:
#   bash run_example.sh
#   EDITOR_MODEL=Qwen/Qwen2.5-0.5B-Instruct JUDGE_MODELS="Qwen/Qwen2.5-0.5B-Instruct" bash run_example.sh   # quick smoke test
#
# Gated models (Llama 3.1) need `huggingface-cli login` or HF_TOKEN in your environment.
set -euo pipefail
cd "$(dirname "$0")"

EDITOR_MODEL="${EDITOR_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
JUDGE_MODELS="${JUDGE_MODELS:-Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct}"
DATA="${DATA:-../data/example_qa.csv}"
OUT="${OUT:-outputs/example}"
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-python}"
mkdir -p "$OUT"

echo "== Step 1: generate edited responses with $EDITOR_MODEL"
"$PYTHON" generate_responses.py --model-name "$EDITOR_MODEL" --device "$DEVICE" \
    --data-path "$DATA" --prompt simple --output-column simple_edit \
    --output-path "$OUT/responses.csv"
# Chain on the previous output so all responses live in one CSV.
"$PYTHON" generate_responses.py --model-name "$EDITOR_MODEL" --device "$DEVICE" \
    --data-path "$OUT/responses.csv" --prompt refined --output-column refined_edit \
    --output-path "$OUT/responses.csv"

echo "== Step 2: judge empathy (3EM-Ranker)"
for JUDGE in $JUDGE_MODELS; do
    TAG=$(basename "$JUDGE")
    for EDIT in simple_edit refined_edit; do
        "$PYTHON" empathy_ranker.py --judge-model "$JUDGE" --device "$DEVICE" \
            --data-path "$OUT/responses.csv" \
            --response1-column physician_response --response2-column "$EDIT" \
            --output-path "$OUT/judge_${TAG}_physician_vs_${EDIT}.csv"
    done
done

echo "== Step 3: Empathy Ranking Score"
"$PYTHON" compute_empathy_score.py "$OUT"/judge_*.csv
