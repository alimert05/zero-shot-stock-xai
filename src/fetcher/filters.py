from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def filter_company_related(articles: list[dict], query: str) -> list[dict]:
    related_articles: list[dict] = []
    query_words = [w.lower() for w in query.split()] if query else []

    for article in articles:
        try:
            title = article.get("title", "")
            if title and any(word.lower() in title.lower() for word in query_words):
                related_articles.append(article)
        except (KeyError, AttributeError, TypeError) as exc:
            logger.warning("Error processing article while company filter: %s", exc)
            continue

    logger.info(
        "Filtered by company related: %s -> %s articles",
        len(articles),
        len(related_articles),
    )
    return related_articles


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
    if not articles:
        logger.warning("No articles to check for duplicates")
        return []

    seen: dict[str, dict] = {}

    for article in articles:
        try:
            title = article.get("title", "").strip().lower()
            domain = article.get("domain", "").strip().lower()

            if not title:
                continue

            key = f"{domain}||{title}"

            if key not in seen:
                article["coverage_count"] = 1
                seen[key] = article
            else:
                seen[key]["coverage_count"] += 1

        except Exception as exc:
            logger.warning("Error processing article while dedup: %s", exc)
            continue

    unique = list(seen.values())
    logger.info(
        "Removed duplicates (domain+title): %s -> %s", len(articles), len(unique)
    )
    return unique
