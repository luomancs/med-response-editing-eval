#!/usr/bin/env python3
"""Aggregate MedFactChecking results into the factuality score table.

Reads one or more per-pair JSON files produced by check_entailment.py and
prints macro / micro recall, precision, and F1 for each.

Usage
-----
python compute_factuality_score.py outputs/entailment_*.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score(results: list[dict]) -> dict:
    """Compute macro and micro metrics from a list of per-pair result dicts."""
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]
    f1s = [r["f1_score"] for r in results]

    total_preserved = sum(r["preserved_facts"] for r in results)
    total_orig = sum(r["original_fact_count"] for r in results)
    total_grounded = sum(r["edited_fact_count"] - r["new_facts"] for r in results)
    total_edited = sum(r["edited_fact_count"] for r in results)

    micro_r = total_preserved / total_orig if total_orig else 1.0
    micro_p = total_grounded / total_edited if total_edited else 1.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)

    return {
        "n_pairs": len(results),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "micro_recall": round(micro_r, 4),
        "micro_precision": round(micro_p, 4),
        "micro_f1": round(micro_f1, 4),
        "mean_orig_facts": round(float(np.mean([r["original_fact_count"] for r in results])), 1),
        "mean_edit_facts": round(float(np.mean([r["edited_fact_count"] for r in results])), 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result_jsons", nargs="+",
                   help="JSON files produced by check_entailment.py")
    args = p.parse_args()

    rows = []
    for path in args.result_jsons:
        with open(path) as f:
            results = json.load(f)
        s = score(results)
        s["file"] = Path(path).name
        rows.append(s)

    df = pd.DataFrame(rows)
    cols = ["file", "n_pairs",
            "macro_recall", "macro_precision", "macro_f1",
            "micro_recall", "micro_precision", "micro_f1",
            "mean_orig_facts", "mean_edit_facts"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
