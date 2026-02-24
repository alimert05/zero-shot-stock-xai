import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import logging
import sys
import os
import re
import math

import finnhub

from config import (
    LOG_PATH,
    TEMP_PATH,
    REQUEST_TIMEOUT_LIMIT,
    FINNHUB_API_KEY,
    MARKET_TIMEZONE,
)
from .utils import resolve_ticker, validate_date, add_recency_weights, assign_market_date
from .filters import (
    filter_company_related,
    remove_duplicates
)
from .content import enrich_articles_with_content
from .content_noise_reducer import clean_articles_content
from .impact_horizon import add_impact_horizon_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class Fetcher:

    def run_fetcher(self) -> None:
        self.get_input()
        self.search()
        self.display_results()

    def __init__(self) -> None:
        self.start_date: str | None = None
        self.end_date: str | None = None
        self.backward_start_date: datetime | None = None
        self.backward_end_date: datetime | None = None
        self.query: str | None = None
        self.ticker: str | None = None

        self.data: dict | None = None
        self.temp_dir = TEMP_PATH
        self.output_file = os.path.join(self.temp_dir, "articles.json")

        self.timeout = REQUEST_TIMEOUT_LIMIT
        self.number_of_news: int | None = None
        self.max_backward_days: int | None = None
        self.prediction_window_days: int | None = None

    def get_input(self) -> str:

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

            number_input = input("Enter number of news(default 250): ").strip()
            self.number_of_news = int(number_input) if number_input else 250

            return self.query

        except Exception as exc:
            logger.error("Unexpected error during input: %s", exc)
            raise

    def _fetch_finnhub_news(self, symbol: str, from_date: str, to_date: str) -> list[dict]:
        if not FINNHUB_API_KEY:
            raise RuntimeError(
                "FINNHUB_API_KEY is not set in config.py. "
                "Get a free key at https://finnhub.io and add it to config.py."
            )

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
                    "url": article_url,
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

    def search(self) -> None:

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

        logger.info(
            "After dedup: Raw=%s, Filtered=%s",
            len(all_articles),
            len(filtered_articles),
        )

        # ── Time-bucketed prefetch ────────────────────────────────
        # Instead of sorting by recency and taking the top N (which
        # discards older articles), divide articles into weekly time
        # buckets and allocate the prefetch budget proportionally.
        prefetch_n = max(self.number_of_news * 3, 50)

        def _bucket_key(a: dict) -> str:
            sd = a.get("seendate", "")
            if sd:
                try:
                    dt = datetime.strptime(sd, "%Y%m%dT%H%M%SZ")
                    return dt.strftime("%Y-W%W")
                except ValueError:
                    pass
            return "unknown"

        buckets: dict[str, list[dict]] = defaultdict(list)
        for art in filtered_articles:
            buckets[_bucket_key(art)].append(art)

        # Sort within each bucket by seendate descending
        for key in buckets:
            buckets[key].sort(key=lambda a: a.get("seendate", ""), reverse=True)

        # Allocate prefetch budget proportionally across buckets
        n_buckets = len(buckets) or 1
        base_per_bucket = prefetch_n // n_buckets
        remainder = prefetch_n % n_buckets

        candidates: list[dict] = []
        for i, (key, arts) in enumerate(sorted(buckets.items())):
            quota = base_per_bucket + (1 if i < remainder else 0)
            candidates.extend(arts[:quota])

        # Re-sort by seendate for downstream compatibility
        candidates.sort(key=lambda a: a.get("seendate", ""), reverse=True)

        logger.info(
            "Time-bucketed prefetch: %d buckets, %d candidates from %d total",
            n_buckets, len(candidates), len(filtered_articles),
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

        # enrich_articles_with_content(candidates, timeout=self.timeout)


        candidates = clean_articles_content(
            candidates,
            company_name=company_name,
            ticker=ticker)

        filtered_after_rules = filter_company_related(
            candidates,
            company_name=company_name,
            ticker=ticker,
        )

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

        final_articles = filtered_after_rules[: self.number_of_news]

        self.data = {"articles": final_articles}

        if len(final_articles) < self.number_of_news:
            logger.info(
                "Only %s articles matched filters within %s days, but %s were requested.",
                len(final_articles),
                self.max_backward_days,
                self.number_of_news,
            )

    def _build_output_payload(self, articles: list[dict]) -> dict:
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

    def display_results(self) -> None:

        if not self.data:
            logger.warning("No data to display")
            print("No data available to display.")
            return

        articles = self.data.get("articles", [])
        if not articles:
            logger.warning("No articles to display")
            print("No articles found in the response.")
            return

        try:
            self.save_articles(articles)
        except Exception as exc:
            logger.error("Failed to save articles: %s", exc)
            print("Warning: Could not save articles to file")

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

        logger.info("Displayed %s articles", len(articles))
