from __future__ import annotations

import re
from typing import Any


def herfindahl_index(weights: list[float]) -> float:
    total = sum(weights)
    if total == 0:
        return 0.0
    shares = [w / total for w in weights]
    return round(sum(s ** 2 for s in shares), 6)


def safe_round(value: float | None, digits: int = 4) -> float:
    if value is None:
        return 0.0
    return round(float(value), digits)


def label_index(label: str) -> int:
    return {"positive": 0, "negative": 1, "neutral": 2}.get(label, 0)


def top_n_by_key(
    items: list[dict],
    key: str,
    n: int,
    reverse: bool = True,
) -> list[dict]:
    return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)[:n]


def get_dominant_label(raw_scores: dict[str, float]) -> str:
    return max(raw_scores, key=raw_scores.get)


def scores_to_pct(scores: dict[str, float]) -> dict[str, float]:
    return {k: round(v * 100, 2) for k, v in scores.items()}


# ── LIME noise-token filter ──────────────────────────────────────────────────

# Words that get high LIME attribution due to input-template injection or
# grammatical structure, not genuine sentiment signal.
LIME_NOISE_WORDS: frozenset[str] = frozenset({
    # Prefix tokens injected by _build_input_text → always present in every
    # LIME perturbation, so they absorb attribution mechanically.
    "news", "about",
    # English stopwords / function words
    "the", "a", "an", "of", "in", "and", "for", "on", "with", "at", "by",
    "from", "as", "or", "be", "it", "to", "is", "this", "that", "its",
    "are", "was", "were", "has", "have", "had", "will", "would", "could",
    "should", "may", "might", "just", "also", "not", "no", "but", "so",
    "than", "then", "now", "new", "more", "most", "very", "all", "each",
    "any", "few", "many", "much", "some", "such", "own", "other",
    # Contraction fragments that LIME tokeniser sometimes isolates
    "s", "t", "re", "ve", "ll", "d", "m",
})


_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")


def build_lime_noise_set(
    company_name: str,
    ticker: str = "",
    article_titles: list[str] | None = None,
) -> frozenset[str]:
    """Return a lower-cased set of tokens to exclude from LIME top-token lists.

    Combines the static stopword list with company-specific tokens so that
    the *summary* token lists contain only sentiment-bearing words.
    The full LIME weight vector is still stored unfiltered for transparency.

    If *article_titles* are provided, ticker symbols in parentheses
    (e.g. "(AAPL)") are auto-detected and added to the noise set.
    """
    extra: set[str] = set()
    for token in company_name.split():
        extra.add(token.lower())
    if ticker:
        extra.add(ticker.lower())
    # Auto-detect ticker symbols from article titles: (AAPL), (NVDA), etc.
    if article_titles:
        for title in article_titles:
            for match in _TICKER_RE.finditer(title):
                extra.add(match.group(1).lower())
    return LIME_NOISE_WORDS | frozenset(extra)


def headline_tokens_set(title: str) -> frozenset[str]:
    """Extract lower-cased word tokens from an article headline.

    Used to filter headline words from LIME top-token lists so that only
    words from the article *body* appear in the summary display.
    """
    # Split on non-alpha, keep tokens ≥ 2 chars
    tokens = re.findall(r"[A-Za-z]{2,}", title)
    return frozenset(t.lower() for t in tokens)


def is_lime_noise_token(
    token: str,
    noise: frozenset[str],
    headline_noise: frozenset[str] | None = None,
) -> bool:
    """Return True if a token should be excluded from LIME summary lists.

    Checks: membership in noise set, headline noise, purely numeric, or
    single character.
    """
    low = token.lower()
    if low in noise:
        return True
    if headline_noise and low in headline_noise:
        return True
    if token.isdigit():
        return True
    if len(token) <= 1:
        return True
    return False
