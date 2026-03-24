"""Narrative consistency experiment.

Runs the XAI narrative synthesis multiple times on the same cases
and compares the LLM-generated summaries for consistency.

Usage:
    python -m testing.evaluation.narrative_consistency
    python -m testing.evaluation.narrative_consistency --cases 6 --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    ARTICLE_CACHE_PATH,
    DATASET_PATH,
    TEMP_PATH,
    JSON_PATH,
    PRED_JSON_PATH,
)


SELECTED_CASE_IDS = [
    "MSFT_W1_bull_run_025",    # W=1,  Microsoft (Tech)
    "GOOGL_W3_bull_run_053",   # W=3,  Alphabet (Tech)
    "NVDA_W5_bull_run_105",    # W=5,  Nvidia (Tech)
    "TSLA_W7_bull_run_157",    # W=7,  Tesla (Tech)
    "BAC_W14_bull_run_209",    # W=14, Bank of America (Finance)
    "UNH_W31_bull_run_333",    # W=31, UnitedHealth (Healthcare)
]


def load_holdout_cases(n_cases: int) -> list[dict]:
    """Load selected diverse cases from the holdout set."""
    holdout_path = DATASET_PATH / "holdout_set.json"
    with open(holdout_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    selected_ids = SELECTED_CASE_IDS[:n_cases]
    cases = [c for c in data["test_cases"] if c["id"] in selected_ids]
    return cases


def find_cached_articles(case: dict) -> Path | None:
    """Find the cached article file for a given test case."""
    case_id = case["id"]
    cache_file = ARTICLE_CACHE_PATH / f"{case_id}.json"
    if cache_file.exists():
        return cache_file
    return None


def run_narrative_for_case(case: dict, articles_path: Path, run_idx: int) -> dict:
    """Run the full XAI pipeline on a case and return the narrative result."""
    import shutil

    # Copy cached articles to the temp location expected by the pipeline
    shutil.copy2(articles_path, JSON_PATH)

    # Run prediction to get prediction_result
    from predictors.zero_shot import predict_sentiment

    pred_result = predict_sentiment(str(JSON_PATH))

    # Save prediction result
    with open(PRED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(pred_result, f, indent=2, default=str)

    # Run XAI
    from xai.explainer import run_xai

    xai_result = run_xai(
        prediction_result=pred_result,
        articles_json_path=str(JSON_PATH),
        output_path=str(TEMP_PATH / f"xai_narrative_test_{run_idx}.json"),
        summary_path=str(TEMP_PATH / f"xai_narrative_test_{run_idx}.txt"),
    )

    narrative_data = xai_result.get("narrative", {})
    return {
        "summary": narrative_data.get("summary", ""),
        "fallback_used": narrative_data.get("fallback_used", None),
        "hallucination_violations": narrative_data.get("hallucination_violations", []),
        "ollama_available": narrative_data.get("ollama_available", None),
    }


def similarity(a: str, b: str) -> float:
    """Compute text similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def run_experiment(n_cases: int = 5, n_runs: int = 3):
    """Run the narrative consistency experiment."""
    cases = load_holdout_cases(n_cases)
    results = {}

    for case in cases:
        case_id = case["id"]
        articles_path = find_cached_articles(case)
        if articles_path is None:
            print(f"  SKIP {case_id}: no cached articles found")
            continue

        print(f"\n{'='*70}")
        print(f"  Case: {case_id}")
        print(f"{'='*70}")

        narratives = []
        fallback_count = 0
        violation_count = 0

        for run_idx in range(n_runs):
            print(f"  Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)
            result = run_narrative_for_case(case, articles_path, run_idx)
            narratives.append(result)

            if result["fallback_used"]:
                fallback_count += 1
                print(f"FALLBACK (violations: {result['hallucination_violations']})")
            else:
                print(f"OK (len={len(result['summary'])})")

            if result["hallucination_violations"]:
                violation_count += len(result["hallucination_violations"])

        # Compute pairwise similarity between all runs
        summaries = [n["summary"] for n in narratives]
        similarities = []
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                sim = similarity(summaries[i], summaries[j])
                similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        min_sim = min(similarities) if similarities else 0.0
        max_sim = max(similarities) if similarities else 0.0

        results[case_id] = {
            "n_runs": n_runs,
            "fallback_count": fallback_count,
            "violation_count": violation_count,
            "avg_similarity": avg_sim,
            "min_similarity": min_sim,
            "max_similarity": max_sim,
            "summaries": summaries,
        }

        print(f"\n  Results for {case_id}:")
        print(f"    Fallbacks: {fallback_count}/{n_runs}")
        print(f"    Violations: {violation_count}")
        print(f"    Avg similarity: {avg_sim:.4f}")
        print(f"    Min similarity: {min_sim:.4f}")
        print(f"    Max similarity: {max_sim:.4f}")

    # Print overall summary
    print(f"\n{'='*70}")
    print(f"  NARRATIVE CONSISTENCY — OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"  Cases tested: {len(results)}")
    print(f"  Runs per case: {n_runs}")

    total_runs = sum(r["n_runs"] for r in results.values())
    total_fallbacks = sum(r["fallback_count"] for r in results.values())
    total_violations = sum(r["violation_count"] for r in results.values())
    all_sims = [r["avg_similarity"] for r in results.values()]

    print(f"  Total runs: {total_runs}")
    print(f"  Total fallbacks: {total_fallbacks} ({100*total_fallbacks/total_runs:.1f}%)")
    print(f"  Total violations: {total_violations}")
    print(f"  Overall avg similarity: {sum(all_sims)/len(all_sims):.4f}" if all_sims else "  No results")
    print(f"  Min case similarity: {min(r['min_similarity'] for r in results.values()):.4f}" if results else "")
    print(f"  Max case similarity: {max(r['max_similarity'] for r in results.values()):.4f}" if results else "")

    # Per-case table
    print(f"\n  {'Case':<35} {'Fallbacks':>10} {'Violations':>11} {'Avg Sim':>10} {'Min Sim':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*11} {'-'*10} {'-'*10}")
    for case_id, r in results.items():
        print(f"  {case_id:<35} {r['fallback_count']:>5}/{r['n_runs']:<4} {r['violation_count']:>11} {r['avg_similarity']:>10.4f} {r['min_similarity']:>10.4f}")

    # Save full results
    output_path = TEMP_PATH / "narrative_consistency_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Narrative consistency experiment")
    parser.add_argument("--cases", type=int, default=5, help="Number of holdout cases to test")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per case")
    args = parser.parse_args()

    run_experiment(n_cases=args.cases, n_runs=args.runs)


if __name__ == "__main__":
    main()
