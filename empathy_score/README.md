# Empathy Ranking Score (3EM-Ranker)

Pairwise LLM-as-judge metric for comparing the empathy of two responses to the same
patient question.

## How it works

For each patient question, a judge LLM sees two responses and picks one of:

- **Response 1** is more empathetic
- **Response 2** is more empathetic
- **Equal**: both are equally empathetic

The three-way choice extends the binary EM-Rank judge. For a comparison *A vs B*
(A = Response 1, B = Response 2) the score is reported as three percentages over the
validly parsed pairs:

| metric   | meaning                                   |
|----------|-------------------------------------------|
| `A >> B` | % of pairs where A was judged more empathetic |
| `B >> A` | % of pairs where B was judged more empathetic |
| `tie`    | % of pairs judged equally empathetic       |

The paper uses two open judges, **Qwen3-8B** and **Llama-3.1-8B-Instruct**, and reports
both. The default prompt is zero-shot; `--few-shot` adds two in-context examples from
[`prompts/judge_fewshot_examples.json`](prompts/judge_fewshot_examples.json).

## Files

| file                         | purpose                                                         |
|------------------------------|-----------------------------------------------------------------|
| `generate_responses.py`      | Produce LLM-edited (or directly generated) responses            |
| `empathy_ranker.py`          | Run the 3EM-Ranker judge on two response columns                |
| `compute_empathy_score.py`   | Aggregate one or more judgment files into the score table       |
| `prompts.py`                 | All edit, generation and judge prompts                          |
| `llm_utils.py`               | Model loading, generation and JSON parsing helpers              |
| `run_example.sh`             | End-to-end example on the synthetic data                        |

## Quick start

From the repository root:

```bash
pip install -r requirements.txt
cd empathy_score

# Fast CPU/GPU smoke test with a tiny model (results are not meaningful)
EDITOR_MODEL=Qwen/Qwen2.5-0.5B-Instruct JUDGE_MODELS=Qwen/Qwen2.5-0.5B-Instruct bash run_example.sh

# Paper setting (needs a GPU and access to the gated Llama 3.1 weights)
bash run_example.sh
```

## Step by step

### 1. Generate responses (optional)

Skip this step if you already have the two responses you want to compare in one CSV.

```bash
python generate_responses.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --data-path ../data/example_qa.csv \
    --prompt refined \
    --output-column refined_edit \
    --output-path outputs/responses.csv
```

`--prompt` options (full text in [`prompts.py`](prompts.py)):

| prompt    | description                                                                  |
|-----------|------------------------------------------------------------------------------|
| `simple`  | Revise the physician response to be more empathetic, keep meaning and style  |
| `minimal` | Same as `simple`, but make only minimal revisions                            |
| `refined` | Minimal edits with explicit rules: no new medical facts, advice or follow-up; no false reassurance or added fear; keep length and style |
| `direct`  | LLM answers the patient question directly (physician response not shown)     |

The script appends `<output-column>` (and a `<output-column>_raw` column with the full
model output) to the input CSV. Run it again on its own output to add more columns, so
that all variants end up in one file.

### 2. Judge empathy

```bash
python empathy_ranker.py \
    --judge-model Qwen/Qwen3-8B \
    --data-path outputs/responses.csv \
    --response1-column physician_response \
    --response2-column refined_edit \
    --output-path outputs/judge_qwen3_physician_vs_refined.csv
```

This writes a per-row CSV (`judge_choice`, `judge_reason`, `judge_raw_output`) and a
`*_summary.json` file with the score.

Useful flags:

- `--few-shot`: use the few-shot judge prompt.
- `--exclude-column COL --exclude-value V`: drop rows where `COL == V`. For example, the
  paper ran the general-question experiment only on questions that did not need
  patient records.
- `--max-rows N`: judge only the first N rows (debugging).
- `--device`: `auto` (default, uses all visible GPUs), `cuda:0`, or `cpu`.

### 3. Aggregate scores

```bash
python compute_empathy_score.py outputs/judge_*.csv
```

```
                                         file  n_valid  n_invalid  R1>>R2 (%)  R2>>R1 (%)  tie (%)
outputs/judge_qwen3_physician_vs_refined.csv      ...
```

## Notes

- **Position bias.** LLM judges can favor one position. To check, run the judge a
  second time with `--response1-column` and `--response2-column` swapped.
- **Decoding.** Decoding is greedy (`do_sample=False`), so results do not depend on
  sampling. Results can still differ slightly across hardware and library versions.
- **Unparseable outputs.** These are labelled `invalid` and left out of the
  percentages. Their count is reported as `n_invalid`. If it is high, raise
  `--max-new-tokens`, because reasoning models such as Qwen3 can use many tokens
  thinking before they answer.
- **Hardware.** An 8B judge in bf16 needs about 16 GB of GPU memory.
