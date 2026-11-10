"""Regression tests; no model downloads, GPU, or Hugging Face token required."""
import contextlib
import io
import json
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from factuality_score import check_entailment, compute_factuality_score, extract_facts
from factuality_score.prompts import DECOMPOSITION_METHODS, get_decomposition_prompt

ROOT = Path(__file__).resolve().parents[1]


class FactualityTests(unittest.TestCase):
    def test_cli_imports_from_another_directory(self):
        for script in ("extract_facts.py", "check_entailment.py", "compute_factuality_score.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "factuality_score" / script), "--help"],
                    cwd=tempfile.gettempdir(), capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_medical_numbers_survive(self):
        facts = ["5 mg daily", "10 mg nightly", "120/80 mmHg", "2026 follow-up recommended",
                 "5.5 mg daily", "-5 degrees", "0.5 mg daily"]
        self.assertEqual(extract_facts.parse_factehr_facts(" // ".join(facts)), facts)

    def test_list_markers_are_removed(self):
        self.assertEqual(
            extract_facts.parse_factehr_facts("Atomic facts:\n1. Take 5 mg\n2) Follow up\n- Monitor symptoms"),
            ["Take 5 mg", "Follow up", "Monitor symptoms"],
        )

    def test_optional_context(self):
        for method in DECOMPOSITION_METHODS:
            for context in (None, "", "Patient context"):
                with self.subTest(method=method, context=context):
                    prompt = get_decomposition_prompt("Response", method, context)
                    self.assertIn("Response", prompt)
                    self.assertNotIn("{context}", prompt)

    def test_entailment_requires_binary_integer_json(self):
        # Use the real JSON parser without importing heavyweight model dependencies.
        torch = types.ModuleType("torch")
        torch.no_grad = lambda: (lambda function: function)
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = object
        transformers.AutoTokenizer = object
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            parser_env = runpy.run_path(str(ROOT / "empathy_score" / "llm_utils.py"))
        helper = types.ModuleType("empathy_score.llm_utils")
        helper.extract_json = parser_env["extract_json"]
        cases = [('{"entailment_prediction": 1}', True),
                 ('{"entailment_prediction": 0}', False),
                 ('The hypothesis is not supported: 10 mg is not stated.', None),
                 ('{"entailment_prediction": "1"}', None),
                 ('{"entailment_prediction": true}', None),
                 ('{"entailment_prediction": 2}', None),
                 ('', None)]
        with patch.dict(sys.modules, {"empathy_score.llm_utils": helper}):
            for raw, expected in cases:
                with self.subTest(raw=raw):
                    helper.generate = lambda *a, **k: raw
                    if expected is None:
                        with self.assertRaises(ValueError):
                            check_entailment.check(None, None, "Premise", "Fact")
                    else:
                        self.assertIs(check_entailment.check(None, None, "Premise", "Fact"), expected)

    def test_pipeline_with_controlled_generations(self):
        helper = types.ModuleType("empathy_score.llm_utils")
        helper.load_model = lambda *a: (None, None)
        torch = types.ModuleType("torch")
        torch.manual_seed = lambda seed: None
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data, facts, output = [folder / name for name in ("input.csv", "facts.csv", "results.json")]
            pd.DataFrame([{"patient_question": "Question", "physician_response": "Take 5 mg. Monitor symptoms.",
                           "edited": "Please take 5 mg. Schedule a test."}]).to_csv(data, index=False)
            generations = iter(["5 mg daily // Monitor symptoms", "5 mg daily // Schedule a test"])
            helper.generate = lambda *a, **k: next(generations)
            args = ["extract_facts.py", "--model-name", "fake", "--data-path", str(data),
                    "--edited-column", "edited", "--output-path", str(facts)]
            with patch.dict(sys.modules, {"empathy_score.llm_utils": helper, "torch": torch}), \
                 patch.object(sys, "argv", args), \
                 patch.object(extract_facts, "tqdm", lambda rows, **kwargs: rows), \
                 contextlib.redirect_stdout(io.StringIO()):
                extract_facts.main()
            self.assertEqual(pd.read_csv(facts).loc[0, "original_facts"], "5 mg daily // Monitor symptoms")
            predictions = iter([1, 0, 1, 0])
            helper.generate = lambda *a, **k: json.dumps({"entailment_prediction": next(predictions)})
            helper.extract_json = lambda raw, key: json.loads(raw)
            args = ["check_entailment.py", "--model-name", "fake", "--data-path", str(facts),
                    "--edited-column", "edited", "--output-path", str(output)]
            with patch.dict(sys.modules, {"empathy_score.llm_utils": helper, "torch": torch}), \
                 patch.object(sys, "argv", args), \
                 patch.object(check_entailment, "tqdm", lambda rows, **kwargs: rows), \
                 contextlib.redirect_stdout(io.StringIO()):
                check_entailment.main()
            results = json.loads(output.read_text())
            self.assertEqual(results[0]["preserved_facts"], 1)
            self.assertEqual(results[0]["new_facts"], 1)
            scores = compute_factuality_score.score(results)
            for key in ("macro_recall", "macro_precision", "macro_f1", "micro_recall", "micro_precision", "micro_f1"):
                self.assertEqual(scores[key], 0.5)
            self.assertTrue(output.with_suffix(".csv").exists())


if __name__ == "__main__":
    unittest.main()
