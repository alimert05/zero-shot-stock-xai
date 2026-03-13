"""Retry fetching for cases that returned 0 articles.

Scans the article cache for cases with 0 articles, looks up their
details from the test set, and re-runs the fetcher pipeline.

Usage:
    python -m testset.retry_empty_cases
    python -m testset.retry_empty_cases --cache-dir data/article_cache_noise_yes
    python -m testset.retry_empty_cases --dry-run   # just list empty cases
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ARTICLE_CACHE_PATH, TEMP_PATH
from fetcher.fetcher import Fetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEST_SET_PATH = Path(__file__).parent.parent.parent / "data" / "test_results" / "test_set.json"


def _load_test_cases() -> dict[str, dict]:
    """Load test set and return a dict keyed by case ID."""
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data if isinstance(data, list) else data.get("cases", data.get("test_cases", []))
    return {c["id"]: c for c in cases}


def _find_empty_cases(cache_dir: Path) -> list[str]:
    """Return case IDs that have 0 articles in the cache."""
    empty = []
    for cache_file in sorted(cache_dir.glob("*.json")):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if len(data.get("articles", [])) == 0:
                empty.append(cache_file.stem)
        except (json.JSONDecodeError, KeyError):
            empty.append(cache_file.stem)
    return empty


def _retry_case(case: dict, cache_dir: Path) -> int:
    """Re-fetch a single case. Returns article count."""
    case_id = case["id"]
    case_temp_dir = Path(TEMP_PATH) / f"retry_{case_id}"
    case_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        fetcher = Fetcher(
            temp_dir=str(case_temp_dir),
        )
        fetcher.query = case["company_name"]
        fetcher.ticker = case["ticker"]
        fetcher.start_date = case["start_date"]
        fetcher.end_date = case["end_date"]
        fetcher.quiet = True

        fetcher.search()
        fetcher.display_results()

        # Read the articles.json the fetcher wrote
        temp_articles_path = case_temp_dir / "articles.json"
        if temp_articles_path.exists():
            with open(temp_articles_path, "r", encoding="utf-8") as f:
                articles_data = json.load(f)

            # Overwrite the cache file
            cache_file = cache_dir / f"{case_id}.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(articles_data, f, indent=2, ensure_ascii=False)

            return articles_data.get("article_count", 0)

        return 0
    finally:
        if case_temp_dir.exists():
            shutil.rmtree(case_temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Retry empty cases")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(ARTICLE_CACHE_PATH),
        help="Article cache directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list empty cases, don't retry",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error("Cache directory not found: %s", cache_dir)
        sys.exit(1)

    # Find empty cases
    empty_ids = _find_empty_cases(cache_dir)
    if not empty_ids:
        logger.info("No empty cases found in %s", cache_dir)
        return

    logger.info("Found %d empty cases:", len(empty_ids))
    for cid in empty_ids:
        logger.info("  - %s", cid)

    if args.dry_run:
        return

    # Load test set to get case details
    all_cases = _load_test_cases()

    # Prewarm models
    logger.info("Loading models...")
    from fetcher.impact_horizon import _get_classifier
    _get_classifier()

    # Retry each empty case
    retried = 0
    still_empty = 0
    total_articles = 0

    for case_id in empty_ids:
        if case_id not in all_cases:
            logger.warning("Case %s not found in test set, skipping", case_id)
            continue

        case = all_cases[case_id]
        logger.info("Retrying %s (%s, %s -> %s)...",
                     case_id, case["ticker"], case["start_date"], case["end_date"])

        count = _retry_case(case, cache_dir)
        retried += 1
        total_articles += count

        if count == 0:
            still_empty += 1
            logger.warning("  -> Still 0 articles")
        else:
            logger.info("  -> Got %d articles", count)

        # Small delay between API calls
        time.sleep(1)

    print("\n" + "=" * 60)
    print("  RETRY SUMMARY")
    print("=" * 60)
    print(f"  Retried        : {retried}")
    print(f"  Now have articles: {retried - still_empty}")
    print(f"  Still empty    : {still_empty}")
    print(f"  Total articles : {total_articles}")
    print("=" * 60)


if __name__ == "__main__":
    main()
