# Factuality smoke-test results

Model: `Qwen/Qwen2.5-0.5B-Instruct`, CPU, greedy decoding, seed 42.
Input: all five synthetic examples in `data/example_qa.csv`; no patient data.

## Software validation

- Six regression tests passed with the full model dependencies installed.
- `pip check`, shell syntax, and `git diff --check` passed.
- All five refined edits generated and parsed successfully.
- Fact extraction, bidirectional entailment, JSON/CSV output, and aggregation completed.
- All five input rows remained in the output, with finite scores in [0, 1].

## Model accuracy: failed

The tiny model assigned recall, precision, and F1 of 1.0 to every example.
It missed a generated edit changing a 3-month test interval to 6 months and
adding a stronger treatment-success claim. Extraction also invented claims
and sometimes failed to extract explicit recommendations.

To isolate the judge from extraction, four hand-authored controls supplied
known facts directly to `check_entailment.py`:

| Control | Expected recall | Expected precision | Actual recall | Actual precision |
|---|---:|---:|---:|---:|
| unchanged | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| deleted fact | 0.5000 | 1.0000 | 1.0000 | 1.0000 |
| changed dosage | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| added claim | 1.0000 | 0.6667 | 1.0000 | 1.0000 |

This verifies software execution only. Do not use the tiny-model scores as
validation of the metric or as medical-content safety decisions. Repeat these
controls with the intended evaluation model before claiming scoring accuracy.
The fixes and tests are local; no merge or publication was performed.

## Reproduce

From the repository root, with the repository requirements installed:

```bash
python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python empathy_score/generate_responses.py --model-name Qwen/Qwen2.5-0.5B-Instruct --device cpu --data-path data/example_qa.csv --prompt refined --output-column refined_edit --output-path empathy_score/outputs/smoke/responses.csv
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MODEL=Qwen/Qwen2.5-0.5B-Instruct DEVICE=cpu DATA=../empathy_score/outputs/smoke/responses.csv OUT=outputs/smoke PYTHON=python bash factuality_score/run_example.sh
python factuality_score/check_entailment.py --model-name Qwen/Qwen2.5-0.5B-Instruct --device cpu --data-path factuality_score/outputs/smoke/controls.csv --edited-column refined_edit --output-path factuality_score/outputs/smoke/controls_entailment.json
```

Generated artifacts are in ignored output folders. `controls.csv` contains
synthetic controls prepared for this run; preserve it when reproducing the
last command. The exact installed environment is recorded in
`factuality_score/outputs/smoke/requirements-tested.txt`.

## Installed versions

- torch: 2.15.0
- transformers: 5.22.2
- accelerate: 1.15.0
- numpy: 2.4.6
- pandas: 3.1.1
- tqdm: 4.70.1
