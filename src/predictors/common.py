"""
common.py - Shared utilities for all sentiment predictor modules.

Contains article-matching, text-preparation, and summary-printing logic
used identically across DeBERTa zero-shot, FinBERT, FinGPT, and Ollama
predictors.
"""

from __future__ import annotations

import re

# Article filtering

COMPANY_SUFFIXES = frozenset({
    "inc", "inc.", "corp", "corp.", "ltd", "ltd.", "co", "co.",
    "plc", "llc", "group", "holdings", "sa", "ag", "se", "nv",
    "the", "company",
})


def title_matches(title: str, company_name: str, ticker: str | None) -> bool:
    """Check whether an article title mentions the target company or ticker."""
    title_lower = title.lower()

    # Full company name match (e.g. "Apple Inc." in title)
    if company_name.lower() in title_lower:
        return True

    # Ticker match - case-sensitive with word boundaries to avoid
    # short tickers (V, A, T) matching inside random words
    if ticker and re.search(rf"\b{re.escape(ticker)}\b", title):
        return True

    # Core-name match: strip common suffixes like Inc., Corp., Ltd.
    # so "Apple Inc." matches a title containing just "Apple"
    core_words = [w for w in company_name.lower().split() if w not in COMPANY_SUFFIXES]
    if core_words:
        core_name = " ".join(core_words)
        if core_name in title_lower:
            return True

    return False


# Text preparation
def build_input_text(
    article: dict,
    include_title: bool,
    company_name: str,
    max_chars: int = 1500,
    prefix: str = "Sentiment for",
) -> str:
    """Build model-ready text from an article's title and content.

    Args:
        prefix: Text prepended before the company name.  DeBERTa uses
                "News about", all other models use "Sentiment for".
    """
    title = article.get("title", "").strip()
    content = (article.get("content") or "").strip()

    if include_title:
        body = f"{title}. {content}" if content else title
    elif content:
        body = content
    else:
        body = title

    if not body:
        return ""

    text = f"{prefix} {company_name}: {body}"
    return text[:max_chars]


# Summary printing
def print_summary(result: dict, model_label: str) -> None:
    """Print a human-readable sentiment prediction summary to stdout."""
    print(f"\n{'=' * 50}")
    print(f"  SENTIMENT PREDICTION RESULT ({model_label})")
    print(f"{'=' * 50}")
    print(f"  Company : {result['company_name']}")
    if result.get("ticker"):
        print(f"  Ticker  : {result['ticker']}")
    print(f"  Articles: {result['articles_analyzed']}/{result['articles_total']} matched")
    print(f"{'-' * 50}")
    print("  Normalized Scores (weighted by article importance):")
    for label in ["positive", "negative", "neutral"]:
        score = result["normalized_scores"][label]
        bar = "#" * int(score * 30)
        print(f"    {label:>8}: {score:.4f}  {bar}")
    print(f"{'-' * 50}")
    print(f"  FINAL LABEL : {result['final_label'].upper()}")
    print(f"  CONFIDENCE  : {result['final_confidence']:.4f}")

    abst = result.get("abstention_test", {})
    method = abst.get("method", "none")
    margin = abst.get("margin", 0.0)

    if method != "none":
        print(f"  ABSTAINED   : Yes ({method})")
    print(f"  MARGIN      : {margin:.4f}")
    print(f"{'=' * 50}\n")
