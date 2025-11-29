from __future__ import annotations

import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup
from requests.exceptions import Timeout, HTTPError, RequestException

logger = logging.getLogger(__name__)


def fetch_article_content(url: str, timeout: int) -> Optional[str]:
    if not url:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Timeout:
        logger.warning("Timeout while fetching article content: %s", url)
        return None
    except HTTPError as e:
        logger.warning("HTTP error while fetching article content: %s | %s", e, url)
        return None
    except RequestException as e:
        logger.warning("Request error while fetching article content: %s | %s", e, url)
        return None
    except Exception as e:
        logger.warning(
            "Unexpected error while fetching article content: %s | %s", e, url
        )
        return None

    try:
        soup = BeautifulSoup(resp.text, "lxml")
        article_tag = soup.find("article")
        if article_tag:
            text_parts = [p.get_text(strip=True) for p in article_tag.find_all("p")]
            content = "\n".join([t for t in text_parts if t])
        else:
            text_parts = [p.get_text(strip=True) for p in soup.find_all("p")]
            content = "\n".join([t for t in text_parts if t])

        if content and len(content) > 200:
            return content

        return None

    except Exception as e:
        logger.warning("Error parsing HTML for content: %s | %s", e, url)
        return None


def enrich_articles_with_content(articles: list[dict], timeout: int) -> None:
    logger.info("Enriching %s articles with full content...", len(articles))

    for idx, article in enumerate(articles, 1):
        try:
            url = article.get("url") or article.get("sourceurl")
            if not url:
                logger.debug("No URL found for article %s, skipping", idx)
                continue

            content = fetch_article_content(url, timeout=timeout)
            if content:
                article["content"] = content
                logger.info("Fetched content for article %s/%s", idx, len(articles))
            else:
                article["content"] = None

        except Exception as e:
            logger.warning("Error while enriching article %s: %s", idx, e)
            article["content"] = None
            continue
