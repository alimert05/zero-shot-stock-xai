from __future__ import annotations

import re
import logging
from typing import Iterable

logger = logging.getLogger(__name__)

def _ticker_regex(ticker: str) -> re.Pattern:
    t = re.escape(ticker.upper())
    return re.compile(rf"(?i)(\${t}\b|\({t}\)|\b{t}\b|\b{t}[-\.][A-Z]{{1,6}}\b)")

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _contains_name_or_ticker(text: str, company_name: str, ticker_re: re.Pattern | None) -> bool:
    if not text:
        return False

    txt = _normalize_spaces(text)
    low = txt.lower()

    name_phrase = _normalize_spaces(company_name).lower()
    if name_phrase and name_phrase in low:
        return True

    if ticker_re and ticker_re.search(txt):
        return True

    return False

_QUESTION_WORDS_RE = re.compile(
    r"(?i)^(who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|should|will|would)\b"
)

def _is_question_headline(title: str) -> bool:
    if not title:
        return False
    t = title.strip()
    return "?" in t or bool(_QUESTION_WORDS_RE.search(t))

def filter_company_related(articles: list[dict], company_name: str, ticker: str | None) -> list[dict]:

    kept: list[dict] = []
    ticker_re = _ticker_regex(ticker) if ticker else None

    dropped_question_no_content = 0
    dropped_no_match_no_content = 0
    dropped_no_match_in_content = 0

    for article in articles:
        try:
            title = article.get("title") or ""
            content = article.get("content")

            if _is_question_headline(title) and not content:
                dropped_question_no_content += 1
                continue

            headline_has = _contains_name_or_ticker(title, company_name, ticker_re)
            content_has = _contains_name_or_ticker(content, company_name, ticker_re)
            if headline_has and content_has:
                kept.append(article)

            # if headline_has:
            #     if _contains_name_or_ticker(content, company_name, ticker_re):
            #         kept.append(article)
            #     else:
            #         article["content"] = None
            #         kept.append(article)
            #     continue

            # if not content:
            #     dropped_no_match_no_content += 1
            #     continue

            # if _contains_name_or_ticker(content, company_name, ticker_re):
            #     kept.append(article)
            # else:
            #     dropped_no_match_in_content += 1

        except (KeyError, AttributeError, TypeError) as exc:
            logger.warning("Error processing article while company filter: %s", exc)
            continue

    logger.info(
        "filter_company_related: in=%s kept=%s | dropped(question_no_content)=%s dropped(no_match_no_content)=%s dropped(no_match_in_content)=%s",
        len(articles),
        len(kept),
        dropped_question_no_content,
        dropped_no_match_no_content,
        dropped_no_match_in_content,
    )
    return kept


def filter_financial_keywords(
    articles: list[dict], financial_keywords: Iterable[str]
) -> list[dict]:
    financial_articles: list[dict] = []
    keywords_lower = [k.lower() for k in financial_keywords]

    for article in articles:
        try:
            title = article.get("title", "")
            if title and any(k in title.lower() for k in keywords_lower):
                financial_articles.append(article)
        except (KeyError, AttributeError, TypeError) as exc:
            logger.warning("Error processing article while financial filter: %s", exc)
            continue

    logger.info(
        "Filtered by financial keywords: %s -> %s articles",
        len(articles),
        len(financial_articles),
    )
    return financial_articles


def filter_language(articles: list[dict], allowed_languages: list[str]) -> list[dict]:
    allowed = {lang.lower() for lang in allowed_languages}
    filtered = [
        article
        for article in articles
        if article.get("language", "").lower() in allowed
    ]

    logger.info("Filtered by language: %s -> %s articles", len(articles), len(filtered))
    return filtered


def remove_duplicates(articles: list[dict]) -> list[dict]:
    """De-duplicate articles by title across ALL domains.

    When the same headline appears on multiple sources (e.g. Yahoo and
    Benzinga), keep the **oldest** copy — the first to break the news is
    what the market reacted to.  If two copies have the same timestamp,
    prefer the one with longer content.

    Each surviving article carries ``coverage_count`` (how many copies
    existed) so downstream code can see syndication breadth.
    """
    if not articles:
        logger.warning("No articles to check for duplicates")
        return []

    seen: dict[str, dict] = {}          # title -> best article so far

    for article in articles:
        try:
            title = article.get("title", "").strip().lower()
            if not title:
                continue

            # Key on title only — catches cross-domain syndication
            key = title

            if key not in seen:
                article["coverage_count"] = 1
                seen[key] = article
            else:
                seen[key]["coverage_count"] += 1

                # Keep the oldest article (earliest seendate).
                # If same date, prefer the one with more content.
                existing = seen[key]
                existing_date = existing.get("seendate", "")
                new_date = article.get("seendate", "")

                if new_date < existing_date:
                    # New article is older — keep it instead
                    article["coverage_count"] = existing["coverage_count"]
                    seen[key] = article
                elif new_date == existing_date:
                    # Same date — prefer longer content
                    new_content_len = len((article.get("content") or ""))
                    old_content_len = len((existing.get("content") or ""))
                    if new_content_len > old_content_len:
                        article["coverage_count"] = existing["coverage_count"]
                        seen[key] = article

        except Exception as exc:
            logger.warning("Error processing article while dedup: %s", exc)
            continue

    unique = list(seen.values())
    logger.info(
        "Removed duplicates (cross-domain, title-based): %s -> %s",
        len(articles), len(unique),
    )
    return unique
