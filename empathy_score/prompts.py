"""Prompt templates used in the paper.

Edit / generation prompts
-------------------------
- simple   : "Simple" editing prompt (make the response more empathetic).
- minimal  : Minimal-editing variant of the simple prompt.
- refined  : "Refined" editing prompt with explicit constraints on factual
             accuracy, physician intent, emotional balance and length/style.
- direct   : Baseline where the LLM answers the patient question directly
             (no physician response is given).

Judge prompts (3EM-Ranker)
--------------------------
- JUDGE_ZERO_SHOT : default judge prompt (three-way choice).
- JUDGE_FEW_SHOT  : same task with in-context examples loaded from
                    prompts/judge_fewshot_examples.json.

The judge wording is kept verbatim from the version used for the paper's
experiments (including its phrasing) so scores are comparable.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Edit / generation prompts. Placeholders: {pq} = patient question,
# {r1} = physician response. The model must answer with JSON {"edition": ...}
# (or {"response": ...} for the direct-generation prompt).
# ---------------------------------------------------------------------------

EDIT_SIMPLE = """
You will be provided with a patient's question and a physician's response. Your task is to revise the physician's response to make it more empathetic while preserving its original meaning and writing style.

Patient Question:
{pq}

Physician's response:
{r1}

Only output a JSON object with the key "edition" containing the revised response as its value. Do not include any additional text, explanations, or formatting outside the JSON object:
"""

EDIT_MINIMAL = """
You will be provided with a patient's question and a physician's response. Your task is to make minimal revisions to the physician's response to enhance its empathy, ensuring the original meaning, medical accuracy, and writing style are preserved.

Patient Question:
{pq}

Physician's response:
{r1}

Only output a JSON object with the key "edition" containing the revised response as its value. Do not include any additional text, explanations, or formatting outside the JSON object:
"""

EDIT_REFINED = """
You will be provided with a patient's question and a physician's response.
Your task is to make minimal revisions to the physician's response to enhance empathy, while preserving the original meaning, medical accuracy, tone, and writing style.

Patient Question:
{pq}

Physician's Original Response:
{r1}

Key editing principles:
1. Preserve factual and clinical accuracy.
- Do not introduce or infer medical facts or conditions not explicitly stated by the physician.
- Do not make any clinical assumptions or diagnoses that are not present in the original response.
2. Respect the physician's intent.
- Do not add follow-up recommendations or next-step suggestions unless they already appear in the physician's original response.
- Do not add new advice, warnings, or treatment instructions.
3. Maintain emotional balance.
- Do not add false reassurance or overconfidence.
- Do not introduce unnecessary doubt, fear, or alarming language.
- Empathy should be expressed through tone, acknowledgment, and understanding—not through added medical content.
4. Preserve structure and style.
- Keep the response roughly the same length as the original.
- Maintain the physician's professional tone and sentence structure where possible.
- Revise only what's necessary to make the tone warmer, more understanding, or more supportive.

Only output a JSON object with the key "edition" containing the revised response as its value. Do not include any additional text, explanations, or formatting outside the JSON object:
"""

# The paper's data is from urology patients; change "urology" for other specialties.
DIRECT_GENERATION = """
Hi, pretend that you are the urology healthcare provider of a patient, your patient come to you and ask a question, please answer the patient question as short as you can (no more than 100 words), and also shows the empathy to the patient.

Patient Question:
{pq}

Only output a JSON object with the key "response" containing the response as its value. Do not include any additional text, explanations, or formatting outside the JSON object:
"""

GENERATION_PROMPTS = {
    "simple": (EDIT_SIMPLE, "edition"),
    "minimal": (EDIT_MINIMAL, "edition"),
    "refined": (EDIT_REFINED, "edition"),
    "direct": (DIRECT_GENERATION, "response"),
}

# ---------------------------------------------------------------------------
# Judge prompts (3EM-Ranker). Placeholders: {pq}, {r1}, {r2}.
# ---------------------------------------------------------------------------

JUDGE_ZERO_SHOT = """
You are an expert in comparing the empathy level of two responses to a patient question. You will be given a patient question, and two responses, your task is to evaluate which answer is more empathy.

Patient Question:
{pq}

Response 1:
{r1}

Response 2:
{r2}

Which response is more empathy? Response 1 or 2, or Equal?
- Response 1
- Response 2
- Equal
Only response as JSON object with keys "choice" and "reason" without any other content:
"""

JUDGE_FEW_SHOT = """
You are an expert in comparing the empathy level of two responses to a patient question. You will be given a patient question, and two responses, your task is to evaluate which answer is more empathy.

{examples}

Patient Question:
{pq}

Response 1:
{r1}

Response 2:
{r2}

Which response is more empathy? Response 1 or 2?
- Response 1: More empathy
- Response 2: More empathy
- Both responses are equally empathy
Only response as JSON object with keys "choice" and "reason" without any other content:
"""

_FEW_SHOT_EXAMPLE = """Example {i}:

Patient Question:
{pq}

Response1:
{r1}

Response2:
{r2}

Which response is more empathy? Response 1 or 2?
- Response 1: More empathy
- Response 2: More empathy
- Both responses are equally empathy
Answer:
{answer}"""

JSON_SUFFIX_JUDGE = """{
    "choice": "your_answer_here",
    "reason": "your_reason_here"
    }"""

DEFAULT_FEWSHOT_FILE = Path(__file__).parent / "prompts" / "judge_fewshot_examples.json"


def load_fewshot_block(path=DEFAULT_FEWSHOT_FILE) -> str:
    """Render the few-shot examples JSON file into a prompt block."""
    with open(path) as f:
        examples = json.load(f)
    blocks = []
    for i, ex in enumerate(examples, start=1):
        answer = json.dumps({"choice": ex["choice"], "reason": ex["reason"]}, indent=4)
        blocks.append(_FEW_SHOT_EXAMPLE.format(
            i=i, pq=ex["patient_question"], r1=ex["response_1"],
            r2=ex["response_2"], answer=answer,
        ))
    return "\n\n".join(blocks)
