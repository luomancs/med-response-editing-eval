#!/usr/bin/env python3
"""Aggregate 3EM-Ranker judgments into the Empathy Ranking Score.

For a comparison "A vs B" (A = Response 1, B = Response 2) we report
    A >> B : % of valid pairs where the judge picked Response 1
    B >> A : % of valid pairs where the judge picked Response 2
    tie    : % of valid pairs judged equally empathetic
Rows where the judge output could not be parsed are counted as "invalid"
and excluded from the percentages (their count is reported separately).

Usage (re-score one or more judgment CSVs produced by empathy_ranker.py):
    python compute_empathy_score.py outputs/judge_*.csv
"""

import argparse
import re
from collections import Counter
from typing import Iterable

import pandas as pd

LABELS = ("response_1", "response_2", "tie")


def normalize_choice(choice) -> str:
    """Map a free-form judge "choice" string to response_1 / response_2 / tie / invalid."""
    if choice is None or (isinstance(choice, float) and pd.isna(choice)):
        return "invalid"
    c = str(choice).strip().lower()
    if not c:
        return "invalid"
    if "equal" in c or "both" in c or "tie" in c or "same" in c:
        return "tie"
    m = re.search(r"response\s*([12])", c) or re.fullmatch(r"\s*([12])\s*", c)
    if m:
        return f"response_{m.group(1)}"
    return "invalid"


def summarize(labels: Iterable[str]) -> dict:
    counts = Counter(labels)
    n_valid = sum(counts[l] for l in LABELS)
    pct = lambda l: round(100.0 * counts[l] / n_valid, 2) if n_valid else None
    return {
        "n_total": sum(counts.values()),
        "n_valid": n_valid,
        "n_invalid": counts["invalid"],
        "counts": {l: counts[l] for l in LABELS},
        "R1_more_empathetic_pct": pct("response_1"),
        "R2_more_empathetic_pct": pct("response_2"),
        "tie_pct": pct("tie"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("judgment_csvs", nargs="+", help="CSV files produced by empathy_ranker.py")
    args = p.parse_args()

    rows = []
    for path in args.judgment_csvs:
        df = pd.read_csv(path)
        col = "judge_choice_raw" if "judge_choice_raw" in df.columns else "judge_choice"
        s = summarize(df[col].map(normalize_choice))
        rows.append({
            "file": path,
            "n_valid": s["n_valid"],
            "n_invalid": s["n_invalid"],
            "R1>>R2 (%)": s["R1_more_empathetic_pct"],
            "R2>>R1 (%)": s["R2_more_empathetic_pct"],
            "tie (%)": s["tie_pct"],
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
