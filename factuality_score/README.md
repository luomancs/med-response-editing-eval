# MedFactChecking Score

The MedFactChecking Score measures how well an LLM-edited response preserves the
medical facts of the original physician response without adding new, ungrounded ones.
It is adapted from the [FactEHR](https://arxiv.org/abs/2412.12422) benchmark.

## How it works

```
Original response  ──decompose──▶  C  = {c₁, c₂, …}
Edited   response  ──decompose──▶  C' = {c₁', c₂', …}

Recall    = (1/|C|)  Σ  𝟙[edited ⊨ c]     (fact preservation)
Precision = (1/|C'|) Σ  𝟙[original ⊨ c']   (no hallucinations)
F1        = harmonic mean
```

Both macro (per-pair mean) and micro (corpus-level) averages are reported.

The input uses the same CSV format as [`../data/README.md`](../data/README.md), so it
can be run on the outputs of `../empathy_score/generate_responses.py`.

## Pipeline

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `extract_facts.py` | Decomposes both responses into atomic facts |
| 2 | `check_entailment.py` | Checks entailment in both directions |
| 3 | `compute_factuality_score.py` | Aggregates into the factuality table |

All scripts share `../empathy_score/llm_utils.py` for model loading and generation.

## Quick start

First, generate edited responses with the empathy editing pipeline:

```bash
cd ../empathy_score
bash run_example.sh          # produces outputs/example/responses.csv
```

Then run the factuality evaluation:

```bash
cd ../factuality_score
bash run_example.sh          # uses the responses from above
```

Or step by step:

```bash
# 1. Extract facts
python extract_facts.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --data-path ../empathy_score/outputs/example/responses.csv \
    --edited-column refined_edit \
    --method factehr_medical \
    --output-path outputs/facts.csv

# 2. Check entailment
python check_entailment.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --data-path outputs/facts.csv \
    --edited-column refined_edit \
    --output-path outputs/entailment.json

# 3. Score
python compute_factuality_score.py outputs/entailment.json
```

### Smoke test (CPU, tiny model)

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct bash run_example.sh
```

## Decomposition methods

| Method | Description |
|--------|-------------|
| `factehr` | All atomic facts |
| `factehr_medical` | Medical/clinical facts only (default) |
| `factehr_medical_adv` | Stricter medical-only filter with explicit include/exclude |
| `medscore_verifiable` | Verifiable medical claims (MedScore-style) |

Select with `--method` on `extract_facts.py`.

## Output format

`check_entailment.py` produces a JSON array. Each entry:

```json
{
  "pair_id": 0,
  "question": "...",
  "original_fact_count": 3,
  "edited_fact_count": 5,
  "preserved_facts": 3,
  "new_facts": 1,
  "recall": 1.0,
  "precision": 0.8,
  "f1_score": 0.8889,
  "preservation_details": [{"fact": "...", "preserved": true}, ...],
  "addition_details": [{"fact": "...", "is_new": false}, ...]
}
```

`compute_factuality_score.py` prints a summary table:

```
file         n_pairs  macro_recall  macro_precision  macro_f1  micro_recall  micro_precision  micro_f1
entailment.json    5        0.9200           0.8400    0.8711        0.9333           0.8571    0.8936
```

## Validation

Run the regression tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

These tests require `numpy`, `pandas`, and `tqdm`. They use controlled model
responses and do not download models or require a GPU. A real-model smoke test
is still needed to assess model output quality.

The entailment checker requires a JSON prediction with an integer value of `0`
or `1`. Invalid output raises an error instead of being included in the scores.
