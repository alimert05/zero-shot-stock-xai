"""Run the full pipeline against the test set and compute evaluation metrics.

For each test case:
    1. Programmatically runs Fetcher (bypassing interactive input)
    2. Runs the configured sentiment predictor
    3. Compares predicted label to ground truth
    4. Computes metrics (accuracy, precision, recall, F1, confusion matrix)

Outputs metrics grouped by:
    - Overall
    - Per prediction window (1, 3, 5, 7, 14, 31 days)
    - Per company (AAPL, MSFT, GOOGL, AMZN, NVDA)
    - Per class (positive, negative, neutral)

Usage:
    python -m testset.test_runner
    python -m testset.test_runner --test-set path/to/test_set.json
    python -m testset.test_runner --max-cases 10   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PRED_PATH,
    JSON_PATH,
    SENTIMENT_MODEL,
    ZEROSHOT_PREDS,
    FINBERT_PREDS,
    FINGPT_PREDS,
)
from fetcher.fetcher import Fetcher
from predictor.zero_shot import run_sentiment_prediction as run_zero_shot
from predictor.finbert import run_sentiment_prediction as run_finbert
from predictor.fingpt import run_sentiment_prediction as run_fingpt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TEST_SET_PATH = PRED_PATH / "test_set.json"
RESULTS_PATH = PRED_PATH / "evaluation_results.json"
LABELS = ["positive", "negative", "neutral"]


# ── Metrics Computation ──

def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute classification metrics without sklearn dependency."""
    if not y_true:
        return {
            "accuracy": 0.0,
            "total": 0,
            "per_class": {},
            "confusion_matrix": {},
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total

    labels_present = sorted(set(y_true) | set(y_pred))
    confusion = {actual: {pred: 0 for pred in labels_present} for actual in labels_present}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for label in labels_present:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in labels_present if other != label)
        fn = sum(confusion.get(label, {}).get(other, 0) for other in labels_present if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if support > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


# ── Pipeline Runner ──

def run_single_case(case: dict) -> dict:
    """Run the full pipeline for a single test case and return the result."""
    case_id = case["id"]
    company_name = case["company_name"]
    ticker = case["ticker"]
    start_date = case["start_date"]
    end_date = case["end_date"]

    logger.info("=" * 60)
    logger.info("Running case: %s", case_id)
    logger.info("  %s (%s) | %s → %s (W=%d)",
                company_name, ticker, start_date, end_date,
                case["prediction_window_days"])

    result = {
        "id": case_id,
        "actual_label": case["actual_label"],
        "predicted_label": None,
        "correct": False,
        "error": None,
        "issue": None,
        "duration_seconds": 0,
    }

    start_time = time.time()

    try:
        # Step 1: Fetch articles (programmatic, no interactive input)
        fetcher = Fetcher()
        fetcher.query = company_name
        fetcher.ticker = ticker
        fetcher.start_date = start_date
        fetcher.end_date = end_date
        fetcher.number_of_news = 250
        fetcher.search()
        has_articles = fetcher.display_results()

        if not has_articles:
            issue = "no_articles_fetched"
            result["issue"] = issue
            result["predicted_label"] = "neutral"
            result["correct"] = result["predicted_label"] == case["actual_label"]
            result["normalized_scores"] = {}
            result["articles_analyzed"] = 0
            result["final_confidence"] = 0.0
            logger.warning("  ISSUE: %s (continuing to next case)", issue)
            result["duration_seconds"] = round(time.time() - start_time, 2)
            return result

        # Step 2: Run sentiment prediction
        if SENTIMENT_MODEL == "zero-shot":
            pred_output = str(ZEROSHOT_PREDS)
            pred_result = run_zero_shot(
                articles_json_path=str(JSON_PATH),
                output_path=pred_output,
                company_name=company_name,
                ticker=ticker,
            )
        elif SENTIMENT_MODEL == "ProsusAI/finbert":
            pred_output = str(FINBERT_PREDS)
            pred_result = run_finbert(
                articles_json_path=str(JSON_PATH),
                output_path=pred_output,
            )
        elif SENTIMENT_MODEL == "fingpt":
            pred_output = str(FINGPT_PREDS)
            pred_result = run_fingpt(
                articles_json_path=str(JSON_PATH),
                output_path=pred_output,
            )
        else:
            raise ValueError(f"Unknown sentiment model: {SENTIMENT_MODEL}")

        predicted_label = pred_result.get("final_label", "neutral")
        result["predicted_label"] = predicted_label
        result["correct"] = predicted_label == case["actual_label"]
        result["normalized_scores"] = pred_result.get("normalized_scores", {})
        result["articles_analyzed"] = pred_result.get("articles_analyzed", 0)
        result["final_confidence"] = pred_result.get("final_confidence", 0.0)

        logger.info("  Predicted: %s | Actual: %s | %s",
                    predicted_label, case["actual_label"],
                    "✓ CORRECT" if result["correct"] else "✗ WRONG")

    except Exception as exc:
        logger.error("  FAILED: %s", exc)
        result["error"] = str(exc)
        result["predicted_label"] = "neutral"

    result["duration_seconds"] = round(time.time() - start_time, 2)
    return result


# ── Main Evaluation ──

def run_evaluation(test_set_path: str, max_cases: int | None = None) -> dict:
    """Run the full evaluation and compute all metrics."""
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_cases = test_data["test_cases"]
    if max_cases:
        test_cases = test_cases[:max_cases]

    logger.info("Starting evaluation: %d test cases", len(test_cases))

    all_results = []
    for i, case in enumerate(test_cases):
        logger.info("\n[%d/%d] Processing...", i + 1, len(test_cases))
        result = run_single_case(case)
        result["prediction_window_days"] = case["prediction_window_days"]
        result["ticker"] = case["ticker"]
        result["market_period"] = case["market_period"]
        all_results.append(result)

    y_true = [r["actual_label"] for r in all_results]
    y_pred = [r["predicted_label"] for r in all_results]

    overall_metrics = compute_metrics(y_true, y_pred)

    window_metrics = {}
    by_window = defaultdict(lambda: ([], []))
    for r in all_results:
        w = r["prediction_window_days"]
        by_window[w][0].append(r["actual_label"])
        by_window[w][1].append(r["predicted_label"])
    for w, (yt, yp) in sorted(by_window.items()):
        window_metrics[f"W={w}"] = compute_metrics(yt, yp)

    company_metrics = {}
    by_company = defaultdict(lambda: ([], []))
    for r in all_results:
        t = r["ticker"]
        by_company[t][0].append(r["actual_label"])
        by_company[t][1].append(r["predicted_label"])
    for t, (yt, yp) in sorted(by_company.items()):
        company_metrics[t] = compute_metrics(yt, yp)

    period_metrics = {}
    by_period = defaultdict(lambda: ([], []))
    for r in all_results:
        p = r["market_period"]
        by_period[p][0].append(r["actual_label"])
        by_period[p][1].append(r["predicted_label"])
    for p, (yt, yp) in sorted(by_period.items()):
        period_metrics[p] = compute_metrics(yt, yp)

    errors = [r for r in all_results if r.get("error")]
    issues = [r for r in all_results if r.get("issue")]

    evaluation = {
        "metadata": {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_set": str(test_set_path),
            "sentiment_model": SENTIMENT_MODEL,
            "total_cases": len(all_results),
            "total_errors": len(errors),
            "total_issues": len(issues),
            "total_duration_seconds": round(sum(r["duration_seconds"] for r in all_results), 2),
        },
        "overall_metrics": overall_metrics,
        "per_window_metrics": window_metrics,
        "per_company_metrics": company_metrics,
        "per_period_metrics": period_metrics,
        "case_results": all_results,
        "errors": [{"id": e["id"], "error": e["error"]} for e in errors],
        "issues": [{"id": i["id"], "issue": i["issue"]} for i in issues],
    }

    return evaluation


def print_report(evaluation: dict) -> None:
    """Print a human-readable evaluation report."""
    meta = evaluation["metadata"]
    overall = evaluation["overall_metrics"]

    print("\n" + "=" * 70)
    print("  EVALUATION REPORT")
    print("=" * 70)
    print(f"  Model      : {meta['sentiment_model']}")
    print(f"  Test cases : {meta['total_cases']}")
    print(f"  Errors     : {meta['total_errors']}")
    print(f"  Issues     : {meta.get('total_issues', 0)}")
    print(f"  Duration   : {meta['total_duration_seconds']:.0f}s")
    print("=" * 70)

    print(f"\n  OVERALL METRICS")
    print(f"  {'─' * 40}")
    print(f"  Accuracy        : {overall['accuracy']:.4f}  ({overall.get('correct', 0)}/{overall['total']})")
    print(f"  Macro Precision : {overall['macro_precision']:.4f}")
    print(f"  Macro Recall    : {overall['macro_recall']:.4f}")
    print(f"  Macro F1        : {overall['macro_f1']:.4f}")

    print(f"\n  PER-CLASS METRICS")
    print(f"  {'─' * 50}")
    print(f"  {'Class':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label, m in overall.get("per_class", {}).items():
        print(f"  {label:>10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")

    print(f"\n  CONFUSION MATRIX")
    print(f"  {'─' * 50}")
    cm = overall.get("confusion_matrix", {})
    labels = sorted(cm.keys())
    header = f"  {'Actual\\Pred':>12}" + "".join(f" {l:>10}" for l in labels)
    print(header)
    for actual in labels:
        row = f"  {actual:>12}" + "".join(f" {cm[actual].get(pred, 0):>10}" for pred in labels)
        print(row)

    print(f"\n  PER-WINDOW METRICS")
    print(f"  {'─' * 60}")
    print(f"  {'Window':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N':>5}")
    for window, m in evaluation.get("per_window_metrics", {}).items():
        print(f"  {window:>10} {m['accuracy']:>10.4f} {m['macro_precision']:>10.4f} {m['macro_recall']:>10.4f} {m['macro_f1']:>10.4f} {m['total']:>5}")

    print(f"\n  PER-COMPANY METRICS")
    print(f"  {'─' * 60}")
    print(f"  {'Company':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N':>5}")
    for company, m in evaluation.get("per_company_metrics", {}).items():
        print(f"  {company:>10} {m['accuracy']:>10.4f} {m['macro_precision']:>10.4f} {m['macro_recall']:>10.4f} {m['macro_f1']:>10.4f} {m['total']:>5}")

    print(f"\n  PER-PERIOD METRICS")
    print(f"  {'─' * 60}")
    print(f"  {'Period':>16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N':>5}")
    for period, m in evaluation.get("per_period_metrics", {}).items():
        print(f"  {period:>16} {m['accuracy']:>10.4f} {m['macro_precision']:>10.4f} {m['macro_recall']:>10.4f} {m['macro_f1']:>10.4f} {m['total']:>5}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline against test set")
    parser.add_argument("--test-set", type=str, default=str(TEST_SET_PATH),
                        help="Path to test_set.json")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of test cases (for quick smoke tests)")
    parser.add_argument("--output", type=str, default=str(RESULTS_PATH),
                        help="Path to save evaluation results JSON")
    args = parser.parse_args()

    evaluation = run_evaluation(args.test_set, args.max_cases)

    PRED_PATH.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", args.output)

    print_report(evaluation)


if __name__ == "__main__":
    main()
