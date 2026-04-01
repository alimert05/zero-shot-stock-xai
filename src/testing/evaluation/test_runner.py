"""
test_runner.py — Evaluation harness for the sentiment prediction pipeline.

Runs the full pipeline (fetch + predict) or cached evaluation against a
labelled test set and computes per-class precision, recall, F1, and macro
averages.  Supports multiple sentiment models via config dispatch.
"""

# ── Imports & Paths ───────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    EVAL_RESULTS_PATH,
    DATASET_PATH,
    TEMP_PATH,
    ARTICLE_CACHE_PATH,
    SENTIMENT_MODEL,
)
from news_retriever.retriever import Fetcher
from predictors.zero_shot import predict_sentiment as predict_zero_shot
from predictors.finbert import predict_sentiment as predict_finbert
from predictors.fingpt import predict_sentiment as predict_fingpt
from predictors.ollama_predictor import predict_sentiment as predict_ollama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

TEST_SET_PATH = DATASET_PATH / "test_set.json"
# Dynamic results filename based on configured model
_MODEL_TAG = SENTIMENT_MODEL.replace("/", "_").replace("-", "_")
RESULTS_PATH = EVAL_RESULTS_PATH / f"evaluation_results_{_MODEL_TAG}.json"
LABELS = ["positive", "negative", "neutral"]


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter for Finnhub API (free tier: 60 calls/min)."""

    def __init__(self, calls_per_minute: int = 55):
        self._lock = threading.Lock()
        self._timestamps: list[float] = []
        self._limit = calls_per_minute
        self._window = 60.0

    def acquire(self):
        """Block until an API call is allowed."""
        while True:
            with self._lock:
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < self._window]
                if len(self._timestamps) < self._limit:
                    self._timestamps.append(now)
                    return
            time.sleep(0.5)


_rate_limiter = RateLimiter(calls_per_minute=55)
_inference_lock = threading.Lock()


# ── Progress Tracking ─────────────────────────────────────────────────────────

class ProgressTracker:
    """Thread-safe progress reporting for parallel test execution."""

    def __init__(self, total: int):
        self._lock = threading.Lock()
        self._completed = 0
        self._correct = 0
        self._errors = 0
        self._issues = 0
        self._total = total

    def record(self, correct: bool, error: bool, issue: bool = False):
        with self._lock:
            self._completed += 1
            if correct:
                self._correct += 1
            if error:
                self._errors += 1
            if issue:
                self._issues += 1
            pct = self._completed / self._total * 100
            print(
                f"\r  Progress: {self._completed}/{self._total} "
                f"({pct:.0f}%) | Correct: {self._correct} | "
                f"Errors: {self._errors} | Issues: {self._issues}\n",
                end="", flush=True,
            )


# ── Metrics Computation ───────────────────────────────────────────────────────

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


# ── Model Pre-Warming ─────────────────────────────────────────────────────────

def _prewarm_models():
    """Pre-load all GPU models before spawning threads to avoid race conditions."""
    logger.info("Pre-loading GPU models...")

    if SENTIMENT_MODEL == "zero-shot":
        from predictors.zero_shot import _get_deberta_pipeline
        _get_deberta_pipeline()
    elif SENTIMENT_MODEL == "ProsusAI/finbert":
        from predictors.finbert import _get_sentiment_pipeline
        _get_sentiment_pipeline()
    elif SENTIMENT_MODEL == "fingpt":
        from predictors.fingpt import _get_fingpt_model
        _get_fingpt_model()
    elif SENTIMENT_MODEL in ("ollama-llama3", "ollama-mistral"):
        # Send a throwaway request to load model into Ollama memory
        import ollama
        from config import OLLAMA_SENTIMENT_MODEL
        logger.info("Pre-warming Ollama model: %s", OLLAMA_SENTIMENT_MODEL)
        ollama.chat(
            model=OLLAMA_SENTIMENT_MODEL,
            messages=[{"role": "user", "content": "test"}],
            options={"num_predict": 1},
        )

    # Fetcher pipeline models (impact horizon + noise reducer)
    from weighting.event_classifier import _get_classifier
    _get_classifier()

    logger.info("All models loaded.")


# ── Single Case Evaluation ────────────────────────────────────────────────────

def run_single_case(
    case: dict,
    progress: ProgressTracker | None = None,
    use_cache: bool = False,
) -> dict:
    """Run the full pipeline for a single test case with isolated file I/O."""
    case_id = case["id"]
    company_name = case["company_name"]
    ticker = case["ticker"]
    start_date = case["start_date"]
    end_date = case["end_date"]

    # Per-case temp directory for file isolation
    case_temp_dir = Path(TEMP_PATH) / case_id
    case_temp_dir.mkdir(parents=True, exist_ok=True)
    case_articles_path = str(case_temp_dir / "articles.json")

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
        if use_cache:
            # ── Load fully-processed articles from cache (sentiment only) ──
            cache_file = ARTICLE_CACHE_PATH / f"{case_id}.json"
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"Cache miss for '{case_id}'. Run with --mode fetch first."
                )

            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            articles = cached_data.get("articles", [])
            has_articles = len(articles) > 0

            # Write cached articles directly to temp dir for predictor
            with open(case_articles_path, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, indent=2, ensure_ascii=False)
        else:
            # ── Fetch articles from API ──
            fetcher = Fetcher(
                temp_dir=str(case_temp_dir),
                rate_limiter=_rate_limiter.acquire,
            )
            fetcher.query = company_name
            fetcher.ticker = ticker
            fetcher.start_date = start_date
            fetcher.end_date = end_date
            fetcher.quiet = True

            fetcher.search()
            has_articles = fetcher.display_results()

        if not has_articles:
            issue_type = "no_articles_in_cache" if use_cache else "no_articles_fetched"
            result["issue"] = issue_type
            result["predicted_label"] = "neutral"
            result["correct"] = result["predicted_label"] == case["actual_label"]
            result["normalized_scores"] = {}
            result["articles_analyzed"] = 0
            result["final_confidence"] = 0.0
            result["duration_seconds"] = round(time.time() - start_time, 2)
            if progress:
                progress.record(correct=result["correct"], error=False, issue=True)
            return result

        # Step 2: Run sentiment prediction (serialized — GPU model not thread-safe)
        with _inference_lock:
            if SENTIMENT_MODEL == "zero-shot":
                pred_result = predict_zero_shot(
                    articles_json_path=case_articles_path,
                    company_name=company_name,
                    ticker=ticker,
                )
            elif SENTIMENT_MODEL == "ProsusAI/finbert":
                pred_result = predict_finbert(
                    articles_json_path=case_articles_path,
                    company_name=company_name,
                    ticker=ticker,
                )
            elif SENTIMENT_MODEL == "fingpt":
                pred_result = predict_fingpt(
                    articles_json_path=case_articles_path,
                    company_name=company_name,
                    ticker=ticker,
                )
            elif SENTIMENT_MODEL in ("ollama-llama3", "ollama-mistral"):
                pred_result = predict_ollama(
                    articles_json_path=case_articles_path,
                    company_name=company_name,
                    ticker=ticker,
                )
            else:
                raise ValueError(f"Unknown sentiment model: {SENTIMENT_MODEL}")

        predicted_label = pred_result.get("final_label", "neutral")
        result["predicted_label"] = predicted_label
        result["correct"] = predicted_label == case["actual_label"]
        result["normalized_scores"] = pred_result.get("normalized_scores", {})
        result["articles_analyzed"] = pred_result.get("articles_analyzed", 0)
        result["final_confidence"] = pred_result.get("final_confidence", 0.0)

    except Exception as exc:
        logger.error("Case %s FAILED: %s", case_id, exc)
        result["error"] = str(exc)
        result["predicted_label"] = "neutral"

    finally:
        # Clean up per-case temp directory
        try:
            shutil.rmtree(case_temp_dir, ignore_errors=True)
        except Exception:
            pass

    result["duration_seconds"] = round(time.time() - start_time, 2)

    if progress:
        progress.record(
            correct=result["correct"],
            error=result.get("error") is not None,
            issue=result.get("issue") is not None,
        )

    return result


# ── Cleanup Helpers ───────────────────────────────────────────────────────────

def _clean_orphan_temp_dirs():
    """Remove leftover temp dirs from previous crashed test runner runs.

    Only removes directories whose names look like test case IDs
    (contain '_W' which all test IDs have, e.g. 'AAPL_W7_bull_run_001').
    """
    temp = Path(TEMP_PATH)
    if not temp.exists():
        return
    for child in temp.iterdir():
        if child.is_dir() and "_W" in child.name:
            shutil.rmtree(child, ignore_errors=True)


# ── Article Fetching ──────────────────────────────────────────────────────────

def fetch_single_case(
    case: dict,
    progress: ProgressTracker | None = None,
) -> dict:
    """Fetch articles for a single test case and save to permanent cache."""
    case_id = case["id"]
    company_name = case["company_name"]
    ticker = case["ticker"]
    start_date = case["start_date"]
    end_date = case["end_date"]

    cache_file = ARTICLE_CACHE_PATH / f"{case_id}.json"

    result = {
        "id": case_id,
        "ticker": ticker,
        "cached": False,
        "article_count": 0,
        "skipped": False,
        "issue": None,
        "error": None,
    }

    # Skip if already cached (enables resume after interrupted fetch)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            result["cached"] = True
            result["skipped"] = True
            result["article_count"] = cached_data.get("article_count", 0)
            if progress:
                progress.record(correct=True, error=False)
            return result
        except (json.JSONDecodeError, KeyError):
            pass  # Re-fetch if cache file is corrupt

    # Use a temporary directory for the fetcher (it writes articles.json there)
    case_temp_dir = Path(TEMP_PATH) / f"fetch_{case_id}"
    case_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        fetcher = Fetcher(
            temp_dir=str(case_temp_dir),
            rate_limiter=_rate_limiter.acquire,
        )
        fetcher.query = company_name
        fetcher.ticker = ticker
        fetcher.start_date = start_date
        fetcher.end_date = end_date
        fetcher.quiet = True

        fetcher.search()
        fetcher.display_results()

        # Read the articles.json the fetcher wrote to temp dir
        temp_articles_path = case_temp_dir / "articles.json"
        if temp_articles_path.exists():
            with open(temp_articles_path, "r", encoding="utf-8") as f:
                articles_data = json.load(f)

            # Copy to permanent cache
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(articles_data, f, indent=2, ensure_ascii=False)

            result["cached"] = True
            n_articles = articles_data.get("article_count", 0)
            result["article_count"] = n_articles
            if n_articles == 0:
                result["issue"] = "no_articles_fetched"
                logger.warning("Case %s (%s): 0 articles fetched", case_id, ticker)
        else:
            # Fetcher found no articles — cache an empty-articles marker
            empty_payload = {
                "query": company_name,
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "article_count": 0,
                "articles": [],
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(empty_payload, f, indent=2, ensure_ascii=False)

            result["cached"] = True
            result["article_count"] = 0
            result["issue"] = "no_articles_fetched"
            logger.warning("Case %s (%s): 0 articles fetched (no articles.json)", case_id, ticker)

    except Exception as exc:
        logger.error("Fetch case %s FAILED: %s", case_id, exc)
        result["error"] = str(exc)

    finally:
        shutil.rmtree(case_temp_dir, ignore_errors=True)

    if progress:
        progress.record(
            correct=result["cached"],
            error=result.get("error") is not None,
            issue=result.get("issue") is not None,
        )

    return result


def run_fetch(
    test_set_path: str,
    max_cases: int | None = None,
    workers: int = 10,
) -> None:
    """Phase 1: Fetch articles for all test cases and store in permanent cache."""
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_cases = test_data["test_cases"]
    if max_cases:
        test_cases = test_cases[:max_cases]

    total = len(test_cases)
    logger.info("FETCH MODE: %d test cases, saving to %s", total, ARTICLE_CACHE_PATH)

    ARTICLE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    _clean_orphan_temp_dirs()

    # Full pipeline: noise reduction + recency weights + impact horizon
    # Pre-warm GPU models needed for fetch pipeline
    _prewarm_models()
    logger.info("Fetch mode: full pipeline (noise reduction + recency + impact horizon)")

    # Sequential only — fetch uses GPU models (impact horizon, noise reduction)
    # that are not thread-safe
    progress = ProgressTracker(total)
    all_results: list[dict] = []

    for case in test_cases:
        result = fetch_single_case(case, progress)
        all_results.append(result)

    print()  # Newline after progress bar

    # ── Fetch Summary ──
    cached_count = sum(1 for r in all_results if r["cached"])
    skipped_count = sum(1 for r in all_results if r.get("skipped"))
    zero_article_count = sum(
        1 for r in all_results if r["cached"] and r["article_count"] == 0
    )
    error_count = sum(1 for r in all_results if r.get("error"))
    total_articles = sum(r["article_count"] for r in all_results)

    print("\n" + "=" * 60)
    print("  FETCH SUMMARY")
    print("=" * 60)
    print(f"  Total cases      : {total}")
    print(f"  Cached (new)     : {cached_count - skipped_count}")
    print(f"  Cached (skipped) : {skipped_count}")
    print(f"  Zero articles    : {zero_article_count}")
    print(f"  Errors           : {error_count}")
    print(f"  Total articles   : {total_articles}")
    print(f"  Cache dir        : {ARTICLE_CACHE_PATH}")
    print("=" * 60)

    if zero_article_count > 0:
        print(f"\n  ⚠ WARNING: {zero_article_count} cases had 0 articles fetched:")
        for r in sorted(all_results, key=lambda x: x["id"]):
            if r["cached"] and r["article_count"] == 0:
                print(f"    - {r['id']} ({r.get('ticker', '?')})")

    if error_count > 0:
        print(f"\n  ERRORS ({error_count} cases):")
        for r in all_results:
            if r.get("error"):
                print(f"    - {r['id']}: {r['error']}")


# ── Cached Evaluation ─────────────────────────────────────────────────────────

def _prewarm_cached_mode():
    """Pre-load sentiment model only for cached evaluation.

    Cached articles are already fully processed (noise reduction, recency,
    impact horizon), so only the sentiment model is needed.
    """
    logger.info("Pre-loading sentiment model (cached articles are fully processed)...")
    if SENTIMENT_MODEL == "zero-shot":
        from predictors.zero_shot import _get_deberta_pipeline
        _get_deberta_pipeline()
    elif SENTIMENT_MODEL == "ProsusAI/finbert":
        from predictors.finbert import _get_sentiment_pipeline
        _get_sentiment_pipeline()
    elif SENTIMENT_MODEL == "fingpt":
        from predictors.fingpt import _get_fingpt_model
        _get_fingpt_model()
    elif SENTIMENT_MODEL in ("ollama-llama3", "ollama-mistral"):
        import ollama
        from config import OLLAMA_SENTIMENT_MODEL
        logger.info("Pre-warming Ollama model: %s", OLLAMA_SENTIMENT_MODEL)
        ollama.chat(
            model=OLLAMA_SENTIMENT_MODEL,
            messages=[{"role": "user", "content": "test"}],
            options={"num_predict": 1},
        )
    logger.info("Sentiment model loaded for cached evaluation.")


def run_evaluation(
    test_set_path: str,
    max_cases: int | None = None,
    workers: int = 10,
    use_cache: bool = False,
) -> dict:
    """Run the full evaluation with optional parallel workers."""
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_cases = test_data["test_cases"]
    if max_cases:
        test_cases = test_cases[:max_cases]

    total = len(test_cases)
    mode_label = "EVALUATE (cached)" if use_cache else "FULL (fetch+evaluate)"
    logger.info("%s: %d test cases with %d workers", mode_label, total, workers)

    # Validate cache if needed
    if use_cache:
        if not ARTICLE_CACHE_PATH.exists():
            raise FileNotFoundError(
                f"Article cache not found: {ARTICLE_CACHE_PATH}\n"
                "Run with --mode fetch first to populate the cache."
            )
        cached_files = list(ARTICLE_CACHE_PATH.glob("*.json"))
        logger.info("Using %d cached article files from %s", len(cached_files), ARTICLE_CACHE_PATH)

        # Evaluate mode is GPU-bound: threading only adds overhead
        # (no API I/O to overlap, inference lock serialises GPU anyway).
        if workers > 1:
            logger.info(
                "Evaluate mode: forcing workers=1 (GPU-bound, threading not beneficial)"
            )
            workers = 1

    # Clean up and prepare
    _clean_orphan_temp_dirs()
    if use_cache:
        _prewarm_cached_mode()
    else:
        _prewarm_models()

    progress = ProgressTracker(total)
    all_results = []

    if workers <= 1:
        # Sequential fallback (useful for debugging)
        for case in test_cases:
            result = run_single_case(case, progress, use_cache=use_cache)
            result["prediction_window_days"] = case["prediction_window_days"]
            result["ticker"] = case["ticker"]
            result["market_period"] = case["market_period"]
            result["sector"] = case.get("sector", "Unknown")
            all_results.append(result)
    else:
        # Parallel execution
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for case in test_cases:
                future = executor.submit(run_single_case, case, progress, use_cache)
                futures[future] = case

            try:
                for future in as_completed(futures):
                    case = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "id": case["id"],
                            "actual_label": case["actual_label"],
                            "predicted_label": "neutral",
                            "correct": False,
                            "error": f"Thread exception: {exc}",
                            "issue": None,
                            "duration_seconds": 0,
                        }
                    result["prediction_window_days"] = case["prediction_window_days"]
                    result["ticker"] = case["ticker"]
                    result["market_period"] = case["market_period"]
                    result["sector"] = case.get("sector", "Unknown")
                    all_results.append(result)
            except KeyboardInterrupt:
                logger.info("Interrupted! Processing completed results...")
                executor.shutdown(wait=False, cancel_futures=True)

    print()  # Newline after progress bar

    # Sort results back to original order
    case_order = {case["id"]: i for i, case in enumerate(test_cases)}
    all_results.sort(key=lambda r: case_order.get(r["id"], 0))

    # ── Compute metrics ──
    y_true = [r["actual_label"] for r in all_results]
    y_pred = [r["predicted_label"] for r in all_results]
    overall_metrics = compute_metrics(y_true, y_pred)

    # Per-window
    window_metrics = {}
    by_window = defaultdict(lambda: ([], []))
    for r in all_results:
        w = r["prediction_window_days"]
        by_window[w][0].append(r["actual_label"])
        by_window[w][1].append(r["predicted_label"])
    for w, (yt, yp) in sorted(by_window.items()):
        window_metrics[f"W={w}"] = compute_metrics(yt, yp)

    # Per-company
    company_metrics = {}
    by_company = defaultdict(lambda: ([], []))
    for r in all_results:
        t = r["ticker"]
        by_company[t][0].append(r["actual_label"])
        by_company[t][1].append(r["predicted_label"])
    for t, (yt, yp) in sorted(by_company.items()):
        company_metrics[t] = compute_metrics(yt, yp)

    # Per-sector
    sector_metrics = {}
    by_sector = defaultdict(lambda: ([], []))
    for r in all_results:
        s = r.get("sector", "Unknown")
        by_sector[s][0].append(r["actual_label"])
        by_sector[s][1].append(r["predicted_label"])
    for s, (yt, yp) in sorted(by_sector.items()):
        sector_metrics[s] = compute_metrics(yt, yp)

    # Per-period
    period_metrics = {}
    by_period = defaultdict(lambda: ([], []))
    for r in all_results:
        p = r["market_period"]
        by_period[p][0].append(r["actual_label"])
        by_period[p][1].append(r["predicted_label"])
    for p, (yt, yp) in sorted(by_period.items()):
        period_metrics[p] = compute_metrics(yt, yp)

    # Positive rate
    pos_count = sum(1 for p in y_pred if p == "positive")
    pos_rate = pos_count / len(y_pred) if y_pred else 0.0

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
            "workers": workers,
            "positive_rate": round(pos_rate, 4),
        },
        "overall_metrics": overall_metrics,
        "per_window_metrics": window_metrics,
        "per_company_metrics": company_metrics,
        "per_sector_metrics": sector_metrics,
        "per_period_metrics": period_metrics,
        "case_results": all_results,
        "errors": [
            {"id": e["id"], "ticker": e.get("ticker"), "error": e["error"]}
            for e in errors
        ],
        "issues": [
            {"id": i["id"], "ticker": i.get("ticker"), "issue": i["issue"]}
            for i in issues
        ],
    }

    return evaluation


# ── Report Generation ─────────────────────────────────────────────────────────

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
    print(f"  Workers    : {meta.get('workers', 1)}")
    print(f"  Pos. Rate  : {meta.get('positive_rate', 0):.2%}")
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

    print(f"\n  PER-SECTOR METRICS")
    print(f"  {'─' * 60}")
    print(f"  {'Sector':>12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N':>5}")
    for sector, m in evaluation.get("per_sector_metrics", {}).items():
        print(f"  {sector:>12} {m['accuracy']:>10.4f} {m['macro_precision']:>10.4f} {m['macro_recall']:>10.4f} {m['macro_f1']:>10.4f} {m['total']:>5}")

    print(f"\n  PER-PERIOD METRICS")
    print(f"  {'─' * 60}")
    print(f"  {'Period':>16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N':>5}")
    for period, m in evaluation.get("per_period_metrics", {}).items():
        print(f"  {period:>16} {m['accuracy']:>10.4f} {m['macro_precision']:>10.4f} {m['macro_recall']:>10.4f} {m['macro_f1']:>10.4f} {m['total']:>5}")

    # ── Flagged Cases ──
    issues = evaluation.get("issues", [])
    errors = evaluation.get("errors", [])

    if issues or errors:
        print(f"\n  FLAGGED CASES")
        print(f"  {'─' * 60}")

    if issues:
        # Group issues by type
        issue_types: dict[str, list] = {}
        for iss in issues:
            itype = iss.get("issue", "unknown")
            issue_types.setdefault(itype, []).append(iss)

        for itype, cases in issue_types.items():
            tickers = [c.get("ticker", "?") for c in cases]
            ticker_counts: dict[str, int] = {}
            for t in tickers:
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
            breakdown = ", ".join(f"{t}({n})" for t, n in sorted(ticker_counts.items()))
            print(f"  {itype}: {len(cases)} cases")
            print(f"    By ticker: {breakdown}")

    if errors:
        # Group errors by pattern (first 80 chars)
        error_types: dict[str, list] = {}
        for err in errors:
            key = str(err.get("error", ""))[:80]
            error_types.setdefault(key, []).append(err)

        print(f"  Errors: {len(errors)} total")
        for etype, cases in error_types.items():
            ids = [c["id"] for c in cases[:5]]
            print(f"    [{len(cases)}x] {etype}")
            print(f"         e.g. {', '.join(ids)}")

    print("\n" + "=" * 70)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline against test set")
    parser.add_argument("--mode", type=str, choices=["full", "fetch", "evaluate"],
                        default="full",
                        help="full = fetch+evaluate (default), "
                             "fetch = populate article cache only, "
                             "evaluate = run from cached articles only")
    parser.add_argument("--test-set", type=str, default=str(TEST_SET_PATH),
                        help="Path to test_set.json")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of test cases (for quick smoke tests)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers (default: 10, use 1 for sequential)")
    parser.add_argument("--output", type=str, default=str(RESULTS_PATH),
                        help="Path to save evaluation results JSON")
    parser.add_argument("--force-refetch", action="store_true",
                        help="Delete existing article cache before fetching (for rebuilding cache)")
    args = parser.parse_args()

    # ── fetch-only mode: populate article cache and exit ──
    if args.mode == "fetch":
        if args.force_refetch and ARTICLE_CACHE_PATH.exists():
            logger.info("--force-refetch: clearing existing cache at %s", ARTICLE_CACHE_PATH)
            shutil.rmtree(ARTICLE_CACHE_PATH, ignore_errors=True)
        run_fetch(args.test_set, args.max_cases, workers=args.workers)
        return

    # ── evaluate (cached) or full (live fetch + evaluate) ──
    use_cache = (args.mode == "evaluate")
    evaluation = run_evaluation(
        args.test_set, args.max_cases, workers=args.workers, use_cache=use_cache,
    )

    EVAL_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", args.output)

    print_report(evaluation)


if __name__ == "__main__":
    main()
