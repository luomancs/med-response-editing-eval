# Evaluating LLM-Edited Physician Responses

Code for the evaluation metrics in
**[From Generation to Collaboration: Using LLMs to Edit for Empathy in Healthcare](https://arxiv.org/abs/2601.15558)**.

The paper studies LLMs as *editors*: the model refines a physician's written reply to a
patient so that it is more empathetic, without changing its medical content. A good
edit has to do two things, so the repository has two complementary metrics:

| metric | question it answers | folder |
|--------|---------------------|--------|
| **Empathy Ranking Score** (3EM-Ranker) | Is the edited response more empathetic than the original (or than another response)? | [`empathy_score/`](empathy_score/) |
| **MedFactChecking Score** | Does the edited response keep the original's medical facts without adding new ones? | [`factuality_score/`](factuality_score/) |

Each metric is self-contained, with its own README and scripts. Both use the same input
format, described below.

## Repository layout

```
.
├── README.md                 # this file
├── requirements.txt
├── data/
│   ├── README.md             # input format (no real patient data is included)
│   └── example_qa.csv        # 5 synthetic examples for smoke tests
├── empathy_score/            # Empathy Ranking Score (3EM-Ranker)
│   ├── README.md
│   ├── generate_responses.py # LLM editing / direct-generation
│   ├── empathy_ranker.py     # pairwise empathy judge
│   ├── compute_empathy_score.py
│   ├── prompts.py
│   ├── prompts/judge_fewshot_examples.json
│   ├── llm_utils.py
│   └── run_example.sh
└── factuality_score/         # MedFactChecking Score
    └── README.md
```

## Installation

Requires Python 3.9+ and, for 7–8B models, a GPU with about 16 GB of memory.

```bash
git clone <this-repo>
cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version
(see <https://pytorch.org/get-started/locally/>) if the default wheel does not fit your
setup.

Some models, such as `meta-llama/Llama-3.1-8B-Instruct`, are gated on Hugging Face.
Request access on the model page, then authenticate:

```bash
huggingface-cli login          # or: export HF_TOKEN=<your token>
```

Never hard-code tokens in scripts or config files.

## Data

The patient data used in the paper is **not** included and is not publicly available.
To use your own data, provide a CSV with a `patient_question` column and a
`physician_response` column. See [`data/README.md`](data/README.md) for details.

## Quick start

```bash
cd empathy_score
# Smoke test with a tiny model (runs on CPU; numbers are not meaningful)
EDITOR_MODEL=Qwen/Qwen2.5-0.5B-Instruct JUDGE_MODELS=Qwen/Qwen2.5-0.5B-Instruct bash run_example.sh
```

See [`empathy_score/README.md`](empathy_score/README.md) for the full workflow and the
paper's settings, and [`factuality_score/README.md`](factuality_score/README.md) for the
factuality metric.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@misc{luo2026generation,
  title         = {From Generation to Collaboration: Using LLMs to Edit for Empathy in Healthcare},
  author        = {Luo, Man and Harandizadeh, Bahareh and Tariq, Amara and Abbas, Halim and Ghaffar, Umar and Warren, Christopher J. and Kolade, Segun O. and Abdul-Muhsin, Haidar M.},
  year          = {2026},
  eprint        = {2601.15558},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2601.15558}
}
```
