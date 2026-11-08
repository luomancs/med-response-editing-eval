# Data

**No patient data is included in this repository.** The dataset used in the paper
(de-identified patient portal messages collected under IRB approval) is not
publicly released.

`example_qa.csv` contains **5 synthetic** question/response pairs written for this repo.
It is only meant to show the expected format and to smoke-test the pipeline.

## Expected input format

A CSV file with one row per patient message:

| column               | required                      | description                                   |
|----------------------|-------------------------------|-----------------------------------------------|
| `patient_question`   | yes                           | The patient's message                         |
| `physician_response` | yes for editing prompts       | The physician's original reply                |
| any other columns    | no                            | Kept unchanged in all outputs (e.g. `id`)     |

The column names can be changed with `--question-column` / `--physician-column`
(generation) and `--question-column` / `--response1-column` / `--response2-column`
(judging).

To compare responses from an external source (e.g. a different system or a
human-edited version), add them as extra columns to the same CSV and pass the column
names to `empathy_ranker.py`.

> Do not commit real patient data to this repository. `data/*` is git-ignored
> except for the synthetic example and this README.
