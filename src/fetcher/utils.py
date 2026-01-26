from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import requests
from requests.exceptions import HTTPError

logger = logging.getLogger(__name__)


def resolve_ticker(
    company_name: str, max_retries: int = 5, timeout: int = 5
) -> Optional[str]:
    url = "https://query2.finance.yahoo.com/v1/finance/search"

    params = {"q": company_name, "quotesCount": 5, "newsCount": 0}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            quotes = data.get("quotes", [])
            if not quotes:
                logger.warning("No ticker found for '%s'", company_name)
                return None

            best = sorted(quotes, key=lambda x: x.get("score", 0), reverse=True)[0]
            symbol = best.get("symbol")

            return symbol

        except HTTPError as exc:
            if resp.status_code == 429:
                wait = 2**attempt
                logger.warning(
                    "Rate limited while resolving ticker… waiting %ss (attempt %s/%s)",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            logger.error("HTTP error while resolving ticker: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error resolving ticker: %s", exc)
            return None

    logger.error("Ticker resolution failed after retries for: %s", company_name)
    return None


def validate_date(date_str: str) -> None:
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        logger.error("Invalid date format: %s", date_str)
        raise ValueError(
            f"Invalid date format: {date_str}. Please use DD-MM-YYYY (e.g., 01-01-2000)"
        )

    day, month, year = map(int, date_str.split("-"))

    if not (1 <= day <= 31):
        logger.error("Invalid day: %s", day)
        raise ValueError(f"Invalid day: {day}. Day must be between 1 and 31.")

    if not (1 <= month <= 12):
        logger.error("Invalid month: %s", month)
        raise ValueError(f"Invalid month: {month}. Month must be between 1 and 12.")

    current_year = datetime.now().year
    if year < 1990 or year > current_year + 1:
        logger.error("Invalid year: %s", year)
        raise ValueError(
            f"Invalid year: {year}. Please enter a realistic year between 1990 and {current_year + 1}."
        )


def add_recency_weights(
    articles: list[dict],
    ref_date: datetime,
    backward_end_date: datetime,
    max_backward_days: int,
) -> None:
    for article in articles:
        try:
            seendate_str = article.get("seendate")
            if seendate_str:
                seen_dt = datetime.strptime(seendate_str, "%Y%m%dT%H%M%SZ")
            else:
                seen_dt = backward_end_date

            days_ago = max(0, (ref_date.date() - seen_dt.date()).days)
           

            # Recency weight rule: will be tested

            # Option 1: Rule-based
            # if days_ago <= 7:
            #     recency_weight = 2.0
            # elif days_ago <= 14:
            #     recency_weight = 1.0
            # else:
            #     recency_weight = 0.0

            # Option 2: Linear Decay
            # recency_weight = max(0.0, 1.0 - (days_ago / self.max_backward_days))

            # Option 3: Exponential Decay
            # lambda_ = 0.15 # might be different value
            # recency_weight = math.exp(-lambda_ * days_ago)

            # Option 4: Pierwise Non-Linear
            if days_ago <= 3:  # fresh news
                recency_weight = 1.5
            elif days_ago <= 7:  # still fresh but not important as first one
                recency_weight = 1.0
            elif days_ago <= 14:  # borderline
                recency_weight = 0.5
            else:  # not important
                recency_weight = 0.0

            article["days_ago"] = days_ago
            article["recency_weight"] = recency_weight

        except Exception as exc:
            logger.warning("Error computing recency weight for article: %s", exc)
            article["days_ago"] = None
            article["recency_weight"] = 1.0
