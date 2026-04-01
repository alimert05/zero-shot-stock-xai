from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import finnhub

from config import (
    LOG_PATH,
    TEMP_PATH,
    REQUEST_TIMEOUT_LIMIT,
    FINNHUB_API_KEY,
    MARKET_TIMEZONE,
)
from .fetcher_utils import resolve_ticker, validate_date, add_recency_weights, assign_market_date
from preprocessing.filters import (
    filter_company_related,
    remove_duplicates,
    filter_headline_content_duplicates,
)
from preprocessing.content_noise_reducer import clean_articles_content
from weighting.weighting import add_impact_horizon_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class Fetcher:
    """Fetches, filters, and scores financial news articles for a given company."""

    def run_fetcher(self) -> bool:
        self.get_input()
        self.search()
        return self.display_results()

    def run_pipeline(
        self,
        company_name: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        """Programmatic entry point -- sets inputs directly, then runs search + save.

        Parameters
        ----------
        company_name : str
            e.g. "Apple Inc." or "AAPL"
        start_date / end_date : str
            DD-MM-YYYY format
        """
        self.query = company_name.strip()
        self.start_date = start_date
        self.end_date = end_date
        self.search()
        return self.display_results()

    def __init__(self, temp_dir=None, rate_limiter=None) -> None:
        self.start_date: str | None = None
        self.end_date: str | None = None
        self.backward_start_date: datetime | None = None
        self.backward_end_date: datetime | None = None
        self.query: str | None = None
        self.ticker: str | None = None

        self.data: dict | None = None
        self.temp_dir = temp_dir if temp_dir is not None else TEMP_PATH
        self.output_file = os.path.join(self.temp_dir, "articles.json")
        self.quiet: bool = False

        self.timeout = REQUEST_TIMEOUT_LIMIT
        self.max_backward_days: int | None = None
        self.prediction_window_days: int | None = None

        # Optional callable invoked before every Finnhub API call.
        # Should block until a call is allowed (e.g. RateLimiter.acquire).
        self._rate_limiter = rate_limiter

    def get_input(self) -> str:
        """Prompt the user for company name, start date, and end date via stdin."""
        try:
            self.query = input("Enter company's name: ").strip()
            if not self.query:
                raise ValueError("Company name cannot be empty.")

            self.start_date = input("Enter start date(DD-MM-YYYY): ").strip()
            if not self.start_date:
                raise ValueError("Start date cannot be empty.")

            self.end_date = input("Enter end date(DD-MM-YYYY): ").strip()
            if not self.end_date:
                raise ValueError("End date cannot be empty.")

            return self.query

        except Exception as exc:
            logger.error("Unexpected error during input: %s", exc)
            raise

    def _fetch_finnhub_news(self, symbol: str, from_date: str, to_date: str) -> list[dict]:
        """Fetch company news from Finnhub and normalise into a standard article format."""
        if not FINNHUB_API_KEY:
            raise RuntimeError(
                "FINNHUB_API_KEY is not set in config.py. "
                "Get a free key at https://finnhub.io and add it to config.py."
            )

        # Per-call rate limiting (blocks until a slot is available)
        if self._rate_limiter is not None:
            self._rate_limiter()

        client = finnhub.Client(api_key=FINNHUB_API_KEY)

        logger.info("Finnhub request: symbol=%s from=%s to=%s", symbol, from_date, to_date)

        raw_articles = client.company_news(symbol, _from=from_date, to=to_date)

        if not isinstance(raw_articles, list):
            logger.warning("Unexpected Finnhub response type: %s", type(raw_articles))
            return []

        logger.info("Finnhub returned %s raw articles", len(raw_articles))

        eastern = ZoneInfo(MARKET_TIMEZONE)

        normalised: list[dict] = []
        for item in raw_articles:
            try:
                unix_ts = item.get("datetime", 0)
                seen_dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
                seendate = seen_dt.strftime("%Y%m%dT%H%M%SZ")

                # Convert to ET and assign market trading session
                et_dt = seen_dt.astimezone(eastern)
                seendate_et = et_dt.strftime("%Y%m%dT%H%M%S")
                market_date_dt = assign_market_date(seen_dt)
                market_date_str = market_date_dt.strftime("%Y-%m-%d")

                source = item.get("source", "")
                article_url = item.get("url", "")
                domain = source if source else ""
                if not domain and article_url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(article_url).netloc
                    except Exception:
                        domain = ""

                normalised.append({
                    "title": item.get("headline", ""),
                    "sourceurl": article_url,
                    "seendate": seendate,
                    "seendate_et": seendate_et,
                    "market_date": market_date_str,
                    "domain": domain,
                    "language": "English",
                    "content": item.get("summary", ""),
                })
            except Exception as exc:
                logger.warning("Error normalising Finnhub article: %s", exc)
                continue

        return normalised

    def _fetch_chunked(
        self,
        symbol: str,
        window_start: datetime,
        window_end: datetime,
        chunk_days: int = 7,
    ) -> list[dict]:
        """
        Split [window_start, window_end] into weekly chunks and make
        separate Finnhub API calls for each, then merge all results.
        This ensures temporal coverage even when Finnhub returns only
        the most recent articles for a wide date range.
        """
        all_articles: list[dict] = []
        chunk_start = window_start

        while chunk_start <= window_end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), window_end)

            try:
                chunk_articles = self._fetch_finnhub_news(
                    symbol=symbol,
                    from_date=chunk_start.strftime("%Y-%m-%d"),
                    to_date=chunk_end.strftime("%Y-%m-%d"),
                )
                logger.info(
                    "Chunk %s to %s: fetched %d articles",
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    len(chunk_articles),
                )
                all_articles.extend(chunk_articles)
            except Exception as exc:
                logger.warning(
                    "Chunk %s to %s failed: %s — continuing with remaining chunks",
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    exc,
                )

            chunk_start = chunk_end + timedelta(days=1)

        logger.info(
            "Chunked fetch complete: %d total articles from %d-day window in %d-day chunks",
            len(all_articles),
            (window_end - window_start).days + 1,
            chunk_days,
        )
        return all_articles

    def search(self, raw_fetch_only: bool = False) -> None:
        """Fetch and process articles.

        Args:
            raw_fetch_only: If True, stop after API fetch + dedup + company
                filter.  Skips recency weighting, impact horizon classification,
                and final_weight computation.  Used by the test-runner cache
                mode so that those tunable stages run at evaluate time.
        """
        if self.start_date is None or self.end_date is None or self.query is None:
            raise RuntimeError("search() called before get_input()")

        validate_date(self.start_date)
        validate_date(self.end_date)

        start_dt = datetime.strptime(self.start_date, "%d-%m-%Y")
        end_dt = datetime.strptime(self.end_date, "%d-%m-%Y")

        if end_dt < start_dt:
            raise ValueError("End date cannot be before start date.")

        self.prediction_window_days = max(1, (end_dt - start_dt).days)
        logger.info("Prediction window: %d days", self.prediction_window_days)

        # backward_days = clamp(⌈α × √W⌉, min=7, max=90), α=5
        self.max_backward_days = max(7, min(90, math.ceil(5 * math.sqrt(self.prediction_window_days))))
        logger.info("Computed max_backward_days: %d (from √W scaling, W=%d)", self.max_backward_days, self.prediction_window_days)

        self.backward_start_date = start_dt - timedelta(days=self.max_backward_days)
        self.backward_end_date = start_dt - timedelta(days=1)

        logger.info(
            "Backward search window: %s to %s (max %s days)",
            self.backward_start_date.strftime("%d-%m-%Y"),
            self.backward_end_date.strftime("%d-%m-%Y"),
            self.max_backward_days,
        )

        company_name = self.query.strip()
        if self.ticker:
            ticker = self.ticker
        else:
            ticker = resolve_ticker(company_name)
            self.ticker = ticker

        if ticker:
            logger.info("Resolved ticker for '%s' -> %s", company_name, ticker)
        else:
            logger.warning("No ticker resolved for '%s', cannot query Finnhub without a symbol", company_name)
            raise RuntimeError(
                f"Could not resolve a stock ticker for '{company_name}'. "
                "Finnhub requires a valid ticker symbol. Please try again with the ticker directly."
            )

        try:
            all_articles = self._fetch_chunked(
                symbol=ticker,
                window_start=self.backward_start_date,
                window_end=self.backward_end_date,
                chunk_days=7,
            )
        except finnhub.FinnhubAPIException as exc:
            logger.error("Finnhub API error: %s", exc)
            raise
        except Exception as exc:
            logger.error("Error fetching from Finnhub: %s", exc)
            raise

        filtered_articles = remove_duplicates(all_articles)
        filtered_articles = filter_headline_content_duplicates(filtered_articles)

        logger.info(
            "After dedup + headline filter: Raw=%s, Filtered=%s",
            len(all_articles),
            len(filtered_articles),
        )

        # Sort by seendate descending and use all articles
        candidates = sorted(
            filtered_articles,
            key=lambda a: a.get("seendate", ""),
            reverse=True,
        )

        logger.info(
            "Using all %d articles after dedup (no article cap)",
            len(candidates),
        )

        # Log actual temporal span vs intended
        if candidates:
            seendates = [a.get("seendate", "") for a in candidates if a.get("seendate")]
            if seendates:
                logger.info(
                    "Temporal coverage: articles span %s to %s (intended window: %s to %s)",
                    min(seendates)[:8], max(seendates)[:8],
                    self.backward_start_date.strftime("%Y%m%d"),
                    self.backward_end_date.strftime("%Y%m%d"),
                )

        add_recency_weights(
            candidates,
            ref_date=start_dt,
            backward_end_date=self.backward_end_date,
            prediction_window_days=self.prediction_window_days,
        )

        filtered_after_rules = filter_company_related(
            candidates,
            company_name=company_name,
            ticker=ticker,
        )

        filtered_after_rules = clean_articles_content(
            filtered_after_rules,
            company_name=company_name,
            ticker=ticker)

        # ── raw_fetch_only: stop here (no recency, no impact horizon, no weighting) ──
        if raw_fetch_only:
            # Clean content whitespace before caching
            for article in filtered_after_rules:
                a = article.get("content")
                if isinstance(a, str) and a:
                    a = a.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("\"", " ")
                    a = re.sub(r"\s+", " ", a).strip()
                    article["content"] = a

            self.data = {"articles": filtered_after_rules}
            logger.info(
                "Raw fetch complete: %d articles (from %d-day backward window)",
                len(filtered_after_rules),
                self.max_backward_days,
            )
            return

        add_impact_horizon_data(
            filtered_after_rules,
            prediction_window_days=self.prediction_window_days)

        for article in filtered_after_rules:
            a = article.get("content")
            if isinstance(a, str) and a:
                a = a.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("\"", " ")
                a = re.sub(r"\s+", " ", a).strip()
                article["content"] = a

        if filtered_after_rules and "final_weight" in filtered_after_rules[0]:
            filtered_after_rules = sorted(
                filtered_after_rules,
                key=lambda x: x.get("final_weight", 0),
                reverse=True,
            )
            logger.info("Articles sorted by final_weight (recency + impact horizon)")

        self.data = {"articles": filtered_after_rules}

        logger.info(
            "Final article count: %d (from %d-day backward window)",
            len(filtered_after_rules),
            self.max_backward_days,
        )

    def _build_output_payload(self, articles: list[dict]) -> dict:
        """Build the full JSON-serialisable output dict including query metadata."""
        return {
            "query": self.query,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "backward_start_date": (
                self.backward_start_date.strftime("%d-%m-%Y")
                if self.backward_start_date
                else None
            ),
            "backward_end_date": (
                self.backward_end_date.strftime("%d-%m-%Y")
                if self.backward_end_date
                else None
            ),
            "prediction_window_days": self.prediction_window_days,
            "max_backward_days": self.max_backward_days,
            "timestamp_alignment": "ET_market_close",
            "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "article_count": len(articles),
            "articles": articles,
        }

    def save_articles(self, articles: list[dict]) -> None:
        """Serialise articles and metadata to the output JSON file."""
        try:
            output_data = self._build_output_payload(articles)

            with open(self.output_file, "w", encoding="utf-8") as file:
                json.dump(output_data, file, indent=2, ensure_ascii=False)

            logger.info("Saved %s articles to %s", len(articles), self.output_file)

        except OSError as exc:
            msg = f"Failed to save articles to {self.output_file}: {exc}"
            logger.error(msg)
            raise Exception(msg) from exc

        except Exception as exc:
            msg = f"Unexpected error while saving articles: {exc}"
            logger.error(msg)
            raise Exception(msg) from exc

    def display_results(self) -> bool:
        """Display fetched articles to stdout and persist them to disk."""
        if not self.data:
            logger.warning("No data to display")
            if not self.quiet:
                print("No data available to display.")
            return False

        articles = self.data.get("articles", [])
        if not articles:
            try:
                # Overwrite stale file so downstream steps never reuse old articles.
                self.save_articles([])
            except Exception as exc:
                logger.error("Failed to save empty article payload: %s", exc)
            logger.warning("No articles to display")
            if not self.quiet:
                print("No articles found in the response.")
            return False

        try:
            self.save_articles(articles)
        except Exception as exc:
            logger.error("Failed to save articles: %s", exc)
            if not self.quiet:
                print("Warning: Could not save articles to file")

        if not self.quiet:
            print(f"\n{'='*50}")
            print(f"Displaying {len(articles)} articles:")
            print(f"{'='*50}\n")

            for idx, article in enumerate(articles, 1):
                try:
                    title = article.get("title", "No title")
                    final_weight = article.get("final_weight")
                    impact_horizon = article.get("impact_horizon", {})

                    print(f"[{idx}] Title: {title}")
                    if final_weight is not None:
                        horizon_cat = impact_horizon.get("category", "N/A")
                        horizon_days = impact_horizon.get("horizon_days", "N/A")
                        print(f"     Weight: {final_weight:.3f} | Horizon: {horizon_cat} ({horizon_days}d)")
                    print("-" * 40)
                except Exception as exc:
                    logger.warning("Error displaying article %s: %s", idx, exc)
                    continue

        logger.info("Saved %s articles to %s", len(articles), self.output_file)
        return True