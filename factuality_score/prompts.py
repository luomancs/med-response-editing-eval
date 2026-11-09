"""Prompt templates for the MedFactChecking Score.

Fact-decomposition prompts
--------------------------
Four decomposition methods control which facts are extracted:
- factehr             : all atomic facts
- factehr_medical     : medical / clinical facts only
- factehr_medical_adv : stricter medical-only filter
- medscore_verifiable : verifiable medical claims (MedScore-style)

Entailment prompt
-----------------
Binary entailment: given a premise text and a hypothesis fact, decide
whether the premise entails the hypothesis.  The model must return
{"entailment_prediction": 1} or {"entailment_prediction": 0}.
"""

# ---------------------------------------------------------------------------
# System role messages
# ---------------------------------------------------------------------------

SYSTEM_MEDICAL_EXPERT = (
    "You are a medical expert who specializes in analyzing healthcare "
    "communications. You carefully examine medical texts and extract "
    "factual claims with precision."
)

SYSTEM_FACT_CHECKER = (
    "You are a careful fact-checker who evaluates whether statements "
    "are supported by given evidence. You focus on medical accuracy "
    "and provide clear judgments."
)

# ---------------------------------------------------------------------------
# Fact decomposition templates.  Placeholder: {text}, optional {context}.
# ---------------------------------------------------------------------------

FACTEHR_DECOMPOSITION = """Please breakdown the following doctor response into independent atomic facts.
Note: each atomic fact should be concise and self-contained.

Doctor response: {text}

Please list each atomic fact, separated by double slashes (//). Example format:
Fact 1 // Fact 2 // Fact 3"""

FACTEHR_MEDICAL_ONLY = """Please breakdown the following doctor response into independent atomic medical/clinical facts only.
Ignore any non-medical statements such as greetings, pleasantries, emotional expressions, or general conversational filler.
Focus exclusively on: diagnoses, symptoms, treatments, medications, test results, procedures, anatomical references, and clinical recommendations.

Doctor response: {text}

Please list each medical/clinical atomic fact, separated by double slashes (//). Example format:
Fact 1 // Fact 2 // Fact 3"""

FACTEHR_MEDICAL_ONLY_ADV = """You are a medical fact extraction expert. Extract ONLY verifiable medical/clinical facts from the following doctor response.

INCLUDE:
- Specific diagnoses or conditions mentioned
- Medications, dosages, or treatment plans
- Test results or lab values
- Specific medical procedures or surgeries referenced
- Anatomical references related to clinical findings
- Quantitative clinical data (measurements, dates, frequencies)
- Specific clinical recommendations or follow-up plans

EXCLUDE:
- Greetings, closings, or pleasantries ("I hope this helps", "Take care")
- Emotional expressions or empathy statements
- General reassurances without specific medical content
- Administrative statements ("I'll have scheduling call you")
- Vague statements without specific medical claims

Context (patient question): {context}

Doctor response: {text}

List ONLY the medical/clinical atomic facts, separated by double slashes (//):"""

MEDSCORE_VERIFIABLE = """Extract all verifiable medical claims from the following doctor response.
A verifiable claim is a statement that can be confirmed or denied using medical knowledge or the patient's medical record.

Context (patient question): {context}

Doctor response: {text}

List each claim on a new line starting with a dash (-).
If there are no verifiable claims, write: No verifiable claim"""

DECOMPOSITION_METHODS = {
    "factehr": FACTEHR_DECOMPOSITION,
    "factehr_medical": FACTEHR_MEDICAL_ONLY,
    "factehr_medical_adv": FACTEHR_MEDICAL_ONLY_ADV,
    "medscore_verifiable": MEDSCORE_VERIFIABLE,
}

# ---------------------------------------------------------------------------
# Entailment template.  Placeholders: {premise}, {hypothesis}.
# ---------------------------------------------------------------------------

ENTAILMENT_PROMPT = """Given the following premise and hypothesis, determine if the premise entails the hypothesis.
Entailment means the hypothesis can be inferred or is supported by the premise.

Premise: {premise}

Hypothesis: {hypothesis}

Respond with a JSON object containing only the key "entailment_prediction" with value 1 (entails) or 0 (does not entail).
Only output the JSON object, nothing else:"""


def get_decomposition_prompt(text: str, method: str = "factehr_medical",
                             context: str = None) -> str:
    """Return the filled-in decomposition prompt for *method*."""
    template = DECOMPOSITION_METHODS[method]
    if "{context}" in template and context:
        return template.format(text=text, context=context)
    return template.format(text=text)


def get_entailment_prompt(premise: str, hypothesis: str) -> str:
    return ENTAILMENT_PROMPT.format(premise=premise, hypothesis=hypothesis)
