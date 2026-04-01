"""
XAI Evaluation — Quality flags, HHI, and flip-set analysis across the holdout set.

Computes:
  1. Evidence quality flag distributions (thin evidence, weight concentration,
     label margin, source diversity)
  2. HHI statistics across all holdout cases
  3. Flip-set analysis: minimum number of articles whose removal changes the
     predicted label

Usage:
    python -m testing.evaluation.xai_evaluation
    python -m testing.evaluation.xai_evaluation --skip-flipset   # quality flags only (no GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import logging
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    EVAL_RESULTS_PATH,
    DATASET_PATH,
    ARTICLE_CACHE_PATH,
    DECISION_THRESHOLD_POS,
    DECISION_THRESHOLD_NEG,
    DECISION_THRESHOLD_ENABLED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

HOLDOUT_PATH = DATASET_PATH / "holdout_set.json"
EVAL_DEBERTA_PATH = EVAL_RESULTS_PATH / "evaluation_results_zero_shot.json"
COVERAGE_COUNT_BOOST = True
HEADLINE_ONLY_WEIGHT = 0.5


# Helpers

def _compute_effective_weights(articles: list[dict]) -> list[float]:
    """Compute effective weights matching the predictor's logic."""
    weights = []
    for a in articles:
        w = float(a.get("final_weight", 1.0))
        if COVERAGE_COUNT_BOOST:
            cc = int(a.get("coverage_count", 1))
            w *= math.log2(1 + cc)
        content = a.get("content") or ""
        if not content.strip():
            w *= HEADLINE_ONLY_WEIGHT
        weights.append(w)
    return weights


def _compute_hhi(weights: list[float]) -> float:
    """Herfindahl-Hirschman Index from a list of weights."""
    total = sum(weights)
    if total <= 0:
        return 1.0
    shares = [w / total for w in weights]
    return sum(s * s for s in shares)


def _apply_thresholds(scores: dict[str, float]) -> str:
    """Apply decision thresholds to normalized scores."""
    pos = scores.get("positive", 0)
    neg = scores.get("negative", 0)

    if not DECISION_THRESHOLD_ENABLED:
        return max(scores, key=scores.get)

    if pos >= DECISION_THRESHOLD_POS and pos >= neg:
        return "positive"
    elif neg >= DECISION_THRESHOLD_NEG and neg >= pos:
        return "negative"
    elif pos >= DECISION_THRESHOLD_POS:
        return "positive"
    elif neg >= DECISION_THRESHOLD_NEG:
        return "negative"
    else:
        return "neutral"


# Quality Flags

def compute_quality_flags(holdout_cases, articles_base):
    """Compute quality flag distributions and HHI stats across holdout set."""
    hhi_values = []
    flag_counts = defaultdict(int)
    quality_ratings = Counter()
    article_counts = []
    rating_accuracy = {"HIGH": {"correct": 0, "total": 0},
                       "MEDIUM": {"correct": 0, "total": 0},
                       "LOW": {"correct": 0, "total": 0}}

    for case in holdout_cases:
        case_id = case["id"]
        article_file = os.path.join(articles_base, f"{case_id}.json")

        if not os.path.exists(article_file):
            logger.warning("Missing article file: %s", case_id)
            continue

        with open(article_file, encoding="utf-8") as f:
            article_data = json.load(f)

        articles = article_data.get("articles", [])
        if not articles:
            continue

        n_articles = len(articles)
        article_counts.append(n_articles)
        weights = _compute_effective_weights(articles)
        total_w = sum(weights)
        if total_w == 0:
            continue

        # HHI
        hhi = _compute_hhi(weights)
        hhi_values.append(hhi)

        # Flags
        n_flags = 0

        # 1. Thin evidence
        if n_articles < 5:
            flag_counts["thin_evidence"] += 1
            n_flags += 1

        # 2. Weight concentration
        if hhi > 0.4:
            flag_counts["weight_concentration"] += 1
            n_flags += 1

        # 3. Label margin
        scores = case.get("normalized_scores", {})
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            if margin < 0.15:
                flag_counts["label_margin"] += 1
                n_flags += 1

        # 4. Source diversity
        domains = [a.get("domain", "unknown") for a in articles]
        domain_counts = Counter(domains)
        if domain_counts:
            max_domain_share = max(domain_counts.values()) / n_articles
            if max_domain_share > 0.6:
                flag_counts["source_diversity"] += 1
                n_flags += 1

        # 5. Timing alignment
        has_market_date = any(a.get("market_date") for a in articles)
        if not has_market_date:
            flag_counts["timing_alignment"] += 1
            n_flags += 1

        # 6. Horizon coverage
        ages = [a.get("days_ago", 0) for a in articles]
        if ages:
            lookback_span = max(ages) - min(ages) + 1
            max_backward = article_data.get("max_backward_days", case.get("expected_backward_days", 0))
            if max_backward and lookback_span < max_backward:
                flag_counts["horizon_coverage"] += 1
                n_flags += 1

        # Quality rating
        if n_flags <= 1:
            rating = "HIGH"
        elif n_flags == 2:
            rating = "MEDIUM"
        else:
            rating = "LOW"
        quality_ratings[rating] += 1

        # Track accuracy per rating
        correct = case.get("actual_label") == case.get("predicted_label")
        rating_accuracy[rating]["total"] += 1
        if correct:
            rating_accuracy[rating]["correct"] += 1

    return {
        "hhi_values": hhi_values,
        "flag_counts": dict(flag_counts),
        "quality_ratings": dict(quality_ratings),
        "rating_accuracy": rating_accuracy,
        "article_counts": article_counts,
        "total_cases": len(holdout_cases),
    }


# Flip-Set Analysis

def compute_flipsets(holdout_cases, articles_base):
    """
    For each holdout case, compute the minimum number of articles whose
    removal changes the predicted label. Requires GPU inference.
    """
    from predictors.zero_shot import _batch_classify_sentiment
    from predictors.common import title_matches, build_input_text

    flipset_sizes = []
    single_flip_count = 0
    cases_processed = 0
    no_flip_count = 0

    for ci, case in enumerate(holdout_cases):
        case_id = case["id"]
        predicted_label = case["predicted_label"]
        article_file = os.path.join(articles_base, f"{case_id}.json")

        if not os.path.exists(article_file):
            continue

        with open(article_file, encoding="utf-8") as f:
            article_data = json.load(f)

        articles = article_data.get("articles", [])
        query = article_data.get("query", "")
        ticker = article_data.get("ticker", "")

        if not articles or len(articles) < 2:
            continue

        # Step 1: Get per-article sentiment scores via batch inference
        batch_texts = []
        batch_meta = []
        for i, article in enumerate(articles):
            title = article.get("title", "")
            include_title = title_matches(title, query, ticker)
            text = build_input_text(
                article, include_title=include_title,
                company_name=query, prefix="News about",
            )
            if not text:
                continue

            content_raw = article.get("content") or ""
            final_weight = float(article.get("final_weight", 1.0))
            cc = int(article.get("coverage_count", 1))
            is_headline_only = not content_raw.strip()

            batch_texts.append(text)
            batch_meta.append({
                "idx": i,
                "final_weight": final_weight,
                "coverage_count": cc,
                "is_headline_only": is_headline_only,
            })

        if not batch_texts:
            continue

        all_scores = _batch_classify_sentiment(batch_texts, batch_size=32)

        # Build per-article effective weights and raw scores
        article_data_list = []
        for meta, raw in zip(batch_meta, all_scores):
            ew = meta["final_weight"]
            if COVERAGE_COUNT_BOOST:
                ew *= math.log2(1 + meta["coverage_count"])
            if meta["is_headline_only"]:
                ew *= HEADLINE_ONLY_WEIGHT
            article_data_list.append({
                "effective_weight": ew,
                "raw_scores": raw,
            })

        # Step 2: Compute flip-set by greedy removal
        # Sort articles by their contribution to the winning label (highest first)
        winning_label = predicted_label
        if winning_label == "neutral":
            # For neutral predictions, find which directional label is closest
            scores = case["normalized_scores"]
            if scores["positive"] >= scores["negative"]:
                opposing_label = "positive"
            else:
                opposing_label = "negative"
        else:
            opposing_label = None

        # Rank articles by how much they support the current prediction
        def article_influence(ad):
            """Score how much an article supports the current prediction."""
            if winning_label == "neutral":
                # For neutral, the prediction flips if removing articles
                # pushes a directional score above its threshold
                return -ad["raw_scores"].get(opposing_label, 0) * ad["effective_weight"]
            else:
                # For directional, removing high-weight supporters flips to neutral
                return ad["raw_scores"].get(winning_label, 0) * ad["effective_weight"]

        ranked = sorted(
            range(len(article_data_list)),
            key=lambda i: article_influence(article_data_list[i]),
            reverse=True,
        )

        # Greedy removal: remove articles one by one and check if label flips
        removed = set()
        flipped = False
        for step, remove_idx in enumerate(ranked, 1):
            removed.add(remove_idx)

            # Recompute weighted scores without removed articles
            ws = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            tw = 0.0
            for j, ad in enumerate(article_data_list):
                if j in removed:
                    continue
                for lbl in ws:
                    ws[lbl] += ad["raw_scores"][lbl] * ad["effective_weight"]
                tw += ad["effective_weight"]

            if tw > 0:
                ns = {k: v / tw for k, v in ws.items()}
            else:
                ns = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            new_label = _apply_thresholds(ns)

            if new_label != predicted_label:
                flipset_sizes.append(step)
                if step == 1:
                    single_flip_count += 1
                flipped = True
                break

        if not flipped:
            no_flip_count += 1
            flipset_sizes.append(len(article_data_list))  # all articles needed

        cases_processed += 1
        if (ci + 1) % 20 == 0:
            logger.info("Flip-set progress: %d/%d cases", ci + 1, len(holdout_cases))

    return {
        "flipset_sizes": flipset_sizes,
        "single_flip_count": single_flip_count,
        "no_flip_count": no_flip_count,
        "cases_processed": cases_processed,
    }


# Reporting

def print_quality_report(results):
    """Print quality flag and HHI results."""
    total = results["total_cases"]
    hhi = results["hhi_values"]
    flags = results["flag_counts"]
    ratings = results["quality_ratings"]
    ac = results["article_counts"]

    print("\n" + "=" * 70)
    print("  XAI EVALUATION — EVIDENCE QUALITY DIAGNOSTICS")
    print("=" * 70)

    print(f"\n  Cases analysed: {total}")
    print(f"  Mean articles per case: {sum(ac)/len(ac):.1f}")
    print(f"  Min articles: {min(ac)}, Max articles: {max(ac)}")

    print(f"\n  {'Flag':<30} {'Triggered':>10} {'Rate':>10}")
    print(f"  {'-'*50}")
    for flag_name in ["thin_evidence", "weight_concentration", "label_margin", "source_diversity", "timing_alignment", "horizon_coverage"]:
        count = flags.get(flag_name, 0)
        pct = count / total * 100
        print(f"  {flag_name:<30} {count:>10} {pct:>9.1f}%")

    ra = results["rating_accuracy"]

    print(f"\n  {'Quality Rating':<20} {'Count':>10} {'Rate':>10} {'Correct':>10} {'Accuracy':>10}")
    print(f"  {'-'*60}")
    for rating in ["HIGH", "MEDIUM", "LOW"]:
        count = ratings.get(rating, 0)
        pct = count / total * 100
        correct = ra[rating]["correct"]
        r_total = ra[rating]["total"]
        acc = correct / r_total * 100 if r_total > 0 else 0
        print(f"  {rating:<20} {count:>10} {pct:>9.1f}% {correct:>10} {acc:>9.1f}%")

    print(f"\n  HHI Statistics:")
    hhi_sorted = sorted(hhi)
    print(f"    Mean:   {sum(hhi)/len(hhi):.4f}")
    print(f"    Median: {hhi_sorted[len(hhi)//2]:.4f}")
    print(f"    Min:    {min(hhi):.4f}")
    print(f"    Max:    {max(hhi):.4f}")
    print(f"    > 0.1:  {sum(1 for h in hhi if h > 0.1)} cases")
    print(f"    > 0.2:  {sum(1 for h in hhi if h > 0.2)} cases")
    print(f"    > 0.4:  {sum(1 for h in hhi if h > 0.4)} cases")
    print("=" * 70)


def print_flipset_report(results):
    """Print flip-set analysis results."""
    sizes = results["flipset_sizes"]
    total = results["cases_processed"]

    print("\n" + "=" * 70)
    print("  XAI EVALUATION — FLIP-SET SENSITIVITY ANALYSIS")
    print("=" * 70)

    print(f"\n  Cases analysed: {total}")
    print(f"  Single-article flips: {results['single_flip_count']} ({results['single_flip_count']/total*100:.1f}%)")
    print(f"  Unflippable cases: {results['no_flip_count']} ({results['no_flip_count']/total*100:.1f}%)")

    sizes_sorted = sorted(sizes)
    print(f"\n  Flip-set size statistics:")
    print(f"    Mean:   {sum(sizes)/len(sizes):.1f}")
    print(f"    Median: {sizes_sorted[len(sizes)//2]}")
    print(f"    Min:    {min(sizes)}")
    print(f"    Max:    {max(sizes)}")

    # Distribution
    print(f"\n  {'Flip-set size':<20} {'Count':>10} {'Rate':>10}")
    print(f"  {'-'*40}")
    size_counts = Counter(sizes)
    for size in sorted(size_counts.keys())[:10]:
        count = size_counts[size]
        pct = count / total * 100
        print(f"  {size:<20} {count:>10} {pct:>9.1f}%")
    if len(size_counts) > 10:
        remaining = sum(c for s, c in size_counts.items() if s > sorted(size_counts.keys())[9])
        print(f"  {'> ' + str(sorted(size_counts.keys())[9]):<20} {remaining:>10} {remaining/total*100:>9.1f}%")

    print("=" * 70)


# Main

def main():
    """Run quality flag diagnostics and flip-set analysis across the holdout set."""
    parser = argparse.ArgumentParser(description="XAI evaluation across holdout set")
    parser.add_argument("--skip-flipset", action="store_true",
                        help="Skip flip-set analysis (no GPU needed)")
    args = parser.parse_args()

    # Load holdout cases
    with open(HOLDOUT_PATH) as f:
        holdout = json.load(f)
    holdout_ids = set(c["id"] for c in holdout["test_cases"])

    with open(EVAL_DEBERTA_PATH) as f:
        eval_data = json.load(f)
    holdout_cases = [c for c in eval_data["case_results"] if c["id"] in holdout_ids]

    logger.info("Loaded %d holdout cases", len(holdout_cases))

    articles_base = str(ARTICLE_CACHE_PATH)

    # 1. Quality flags (no GPU needed)
    logger.info("Computing quality flags and HHI...")
    quality_results = compute_quality_flags(holdout_cases, articles_base)
    print_quality_report(quality_results)

    # 2. Flip-set analysis (needs GPU)
    if not args.skip_flipset:
        logger.info("Computing flip-set analysis (requires GPU inference)...")
        flipset_results = compute_flipsets(holdout_cases, articles_base)
        print_flipset_report(flipset_results)
    else:
        logger.info("Skipping flip-set analysis (--skip-flipset)")

    # Save results
    output = {
        "quality_flags": {
            "flag_counts": quality_results["flag_counts"],
            "quality_ratings": quality_results["quality_ratings"],
            "rating_accuracy": quality_results["rating_accuracy"],
            "hhi_mean": sum(quality_results["hhi_values"]) / len(quality_results["hhi_values"]),
            "hhi_median": sorted(quality_results["hhi_values"])[len(quality_results["hhi_values"]) // 2],
            "article_count_mean": sum(quality_results["article_counts"]) / len(quality_results["article_counts"]),
        },
    }
    if not args.skip_flipset:
        output["flipset"] = {
            "sizes": flipset_results["flipset_sizes"],
            "single_flip_count": flipset_results["single_flip_count"],
            "no_flip_count": flipset_results["no_flip_count"],
            "cases_processed": flipset_results["cases_processed"],
            "median_size": sorted(flipset_results["flipset_sizes"])[len(flipset_results["flipset_sizes"]) // 2],
            "mean_size": sum(flipset_results["flipset_sizes"]) / len(flipset_results["flipset_sizes"]),
        }

    out_path = EVAL_RESULTS_PATH / "xai_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
