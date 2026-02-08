import json
import time
from datetime import datetime, timedelta
import logging
import sys
import os
import re
import math
 
from gdeltdoc import GdeltDoc, Filters
 
from config import (
    LOG_PATH,
    TEMP_PATH,
    REQUEST_TIMEOUT_LIMIT,
    THEMES,
    WEIGHT_COMBINE_METHOD,
)
from .utils import resolve_ticker, validate_date, add_recency_weights
from .filters import (
    filter_company_related,
    filter_language,
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
 
        self.gd = GdeltDoc()
 
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
 
        all_articles: list[dict] = []
        filtered_articles: list[dict] = []
 
        current_day = self.backward_end_date
        request_count = 0
 
        company_name = self.query.strip()
        ticker = resolve_ticker(company_name)
        self.ticker = ticker
 
        if ticker:
            logger.info("Resolved ticker for '%s' -> %s", company_name, ticker)
            gdelt_query = f"{company_name} {ticker}"
        else:
            logger.info("Using raw company name (no ticker found) → %s", company_name)
            gdelt_query = company_name
 
        logger.info("GDELT query: %s", gdelt_query)
 
        while (
            current_day >= self.backward_start_date
            and len(filtered_articles) < self.number_of_news
        ):
            day_start = current_day
            day_end = current_day + timedelta(days=1)
 
            logger.info(
                "Fetching day: %s to %s (current collected: %s/%s)",
                day_start.strftime("%Y-%m-%d"),
                day_end.strftime("%Y-%m-%d"),
                len(filtered_articles),
                self.number_of_news,
            )
 
            f = Filters(
                keyword=gdelt_query,
                start_date=day_start,
                end_date=day_end,
                num_records=min(250, self.number_of_news),
                language="English",
                theme=THEMES,
            )
 
            try:
                request_count += 1
                df = self.gd.article_search(f)
 
            except Exception as e:
                logger.error("Error during GDELT request: %s", e)
                current_day -= timedelta(days=1)
                if request_count % 3 == 0:
                    time.sleep(2)
                continue
 
            if df is None or df.empty:
                logger.info(
                    "No articles returned for %s - %s.",
                    day_start.strftime("%Y-%m-%d"),
                    day_end.strftime("%Y-%m-%d"),
                )
                current_day -= timedelta(days=1)
                if request_count % 3 == 0:
                    time.sleep(2)
                continue
 
            batch_articles = df.to_dict(orient="records")
            logger.info(
                "Received %s raw articles for %s.",
                len(batch_articles),
                day_start.strftime("%Y-%m-%d"),
            )
 
            all_articles.extend(batch_articles)
 
            unique_articles = remove_duplicates(all_articles)
            new_filtered = filter_language(unique_articles, ["English"])
            filtered_articles[:] = new_filtered
 
            logger.info(
                "Progress after %s: Raw=%s, Filtered=%s/%s",
                day_start.strftime("%Y-%m-%d"),
                len(all_articles),
                len(filtered_articles),
                self.number_of_news,
            )
 
            if len(filtered_articles) >= self.number_of_news:
                logger.info("Reached target of %s filtered articles.", self.number_of_news)
                break
 
            current_day -= timedelta(days=1)
            if request_count % 3 == 0:
                time.sleep(2)
 
        def _sort_key(a: dict) -> str:
            return a.get("seendate") or ""
 
        filtered_articles = sorted(filtered_articles, key=_sort_key, reverse=True)
 
        prefetch_n = max(self.number_of_news * 3, 50)
 
        candidates = filtered_articles[:prefetch_n]
 
        add_recency_weights(
            candidates,
            ref_date=start_dt,
            backward_end_date=self.backward_end_date,
            prediction_window_days=self.prediction_window_days,
        )
 
        enrich_articles_with_content(candidates, timeout=self.timeout)


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