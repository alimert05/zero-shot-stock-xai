import json
import time
import math
import requests
from gdeltdoc import GdeltDoc, Filters

from bs4 import BeautifulSoup

# import trafilatura
from config import (
    BASE_URL,
    LOG_PATH,
    TEMP_PATH,
    FINANCIAL_KEYWORDS,
    REQUEST_TIMEOUT_LIMIT,
    THEMES,
)
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException
from datetime import datetime, timedelta
import yfinance as yf
import logging
import re
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)


def resolve_ticker(company_name: str) -> str:
    url = "https://query2.finance.yahoo.com/v1/finance/search"

    params = {"q": company_name, "quotesCount": 5, "newsCount": 0}

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            quotes = data.get("quotes", [])
            if not quotes:
                logging.warning(f"No ticker found for '{company_name}'")
                return None

            best = sorted(quotes, key=lambda x: x.get("score", 0), reverse=True)[0]
            symbol = best.get("symbol")

            return symbol

        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = 2**attempt
                logging.warning(f"Rate limited while resolving ticker… waiting {wait}s")
                time.sleep(wait)
                continue
            else:
                logging.error(f"HTTP error while resolving ticker: {e}")
                return None
        except Exception as e:
            logging.error(f"Unexpected error resolving ticker: {e}")
            return None

    logging.error(f"Ticker resolution failed after retries for: {company_name}")
    return None


class Fetcher:

    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.backward_start_date = None
        self.backward_end_date = None
        self.query = None

        self.data = None
        self.temp_dir = TEMP_PATH
        self.output_file = os.path.join(self.temp_dir, "articles.json")

        self.financial_keywords = FINANCIAL_KEYWORDS
        self.timeout = REQUEST_TIMEOUT_LIMIT
        self.number_of_news: int = None
        self.max_backward_days: int = None

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

            # input number of news
            number_input = input("Enter number of news(default 250): ").strip()
            if number_input:
                self.number_of_news = int(number_input)
            else:
                self.number_of_news = 250

            # max days go backward
            max_days_input = input(
                f"Enter maximum backward search days (default 14): "
            ).strip()
            if max_days_input:
                self.max_backward_days = int(max_days_input)
            else:
                self.max_backward_days = 14

            return self.query

        except Exception as e:
            logging.error(f"Unexpected error during input: {e}")
            raise

    def validate_date(self, date_str: str) -> None:
        try:
            dt = datetime.strptime(date_str, "%d-%m-%Y")
        except ValueError:
            logging.error(f"Invalid date format: {date_str}")
            raise ValueError(
                f"Invalid date format: {date_str}. Please use DD-MM-YYYY (e.g., 01-01-2000)"
            )

        day, month, year = map(int, date_str.split("-"))

        if not (1 <= day <= 31):
            logging.error(f"Invalid day: {day}")
            raise ValueError(f"Invalid day: {day}. Day must be between 1 and 31.")

        if not (1 <= month <= 12):
            logging.error(f"Invalid month: {month}")
            raise ValueError(f"Invalid month: {month}. Month must be between 1 and 12.")

        if year < 1990 or year > datetime.now().year + 1:
            logging.error(f"Invalid year: {year}")
            raise ValueError(f"Invalid year: {year}. Please enter a realistic year.")

    def search(self) -> dict:
        self.validate_date(self.start_date)
        self.validate_date(self.end_date)

        start_dt = datetime.strptime(self.start_date, "%d-%m-%Y")
        end_dt = datetime.strptime(self.end_date, "%d-%m-%Y")

        if end_dt < start_dt:
            raise ValueError("End date cannot be before start date.")

        self.backward_start_date = start_dt - timedelta(days=self.max_backward_days)
        self.backward_end_date = start_dt - timedelta(days=1)

        logging.info(
            f"Backward search window: "
            f"{self.backward_start_date.strftime('%d-%m-%Y')} "
            f"to {self.backward_end_date.strftime('%d-%m-%Y')} "
            f"(max {self.max_backward_days} days)"
        )

        all_articles = []
        filtered_articles = []

        current_day = self.backward_end_date
        request_count = 0

        resolved = resolve_ticker(self.query)
        if resolved:
            logging.info(f"Resolved ticker for '{self.query}' → {resolved}")
            self.query = f"{self.query} {resolved}"
        else:
            logging.info(f"Using raw company name (no ticker found) → {self.query}")

        print(self.query)

        while (
            current_day >= self.backward_start_date
            and len(filtered_articles) < self.number_of_news
        ):
            day_start = current_day
            day_end = current_day + timedelta(days=1)

            logging.info(
                f"Fetching day: {day_start.strftime('%Y-%m-%d')} "
                f"to {day_end.strftime('%Y-%m-%d')} "
                f"(current collected: {len(filtered_articles)}/{self.number_of_news})"
            )

            f = Filters(
                keyword=self.query,
                start_date=day_start,
                end_date=day_end,
                num_records=min(250, self.number_of_news),
                language="English",
                theme="ECON_STOCKMARKET",
            )

            try:
                request_count += 1
                df = self.gd.article_search(f)

            except Timeout:
                logging.error("Request timed out")
                current_day -= timedelta(days=1)
                if request_count % 3 == 0:
                    time.sleep(2)
                continue

            except HTTPError as e:
                logging.error(f"HTTP error: {e}")
                current_day -= timedelta(days=1)
                if request_count % 3 == 0:
                    time.sleep(2)
                continue

            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                current_day -= timedelta(days=1)
                if request_count % 3 == 0:
                    time.sleep(2)
                continue

            if df is None or df.empty:
                logging.info(
                    f"No articles returned for {day_start.strftime('%Y-%m-%d')} - {day_end.strftime('%Y-%m-%d')}."
                )
                current_day -= timedelta(days=1)
                if request_count % 3 == 0:
                    time.sleep(2)
                continue

            batch_articles = df.to_dict(orient="records")
            logging.info(
                f"Received {len(batch_articles)} raw articles for "
                f"{day_start.strftime('%Y-%m-%d')}."
            )

            all_articles.extend(batch_articles)

            unique_articles = self.remove_duplicates({"articles": all_articles})
            english_articles = self.filter_language(unique_articles, ["English"])
            # financial_articles = self.filter_financial_keywords(english_articles) # will be tested if improve accuracy or not
            new_filtered = self.filter_company_related(english_articles)

            filtered_articles[:] = new_filtered

            logging.info(
                f"Progress after {day_start.strftime('%Y-%m-%d')}: "
                f"Raw={len(all_articles)}, "
                f"Filtered={len(filtered_articles)}/{self.number_of_news}"
            )

            if len(filtered_articles) >= self.number_of_news:
                logging.info(
                    f"Reached target of {self.number_of_news} filtered articles."
                )
                break

            current_day -= timedelta(days=1)
            if request_count % 3 == 0:
                time.sleep(2)

        final_articles = filtered_articles[: self.number_of_news]
        self.add_recency_weights(final_articles, ref_date=start_dt)
        self.enrich_articles_with_content(final_articles)
        self.data = {"articles": final_articles}

        if len(final_articles) < self.number_of_news:
            logging.info(
                f"Only {len(final_articles)} articles matched filters "
                f"within {self.max_backward_days} days, "
                f"but {self.number_of_news} were requested."
            )

    # Filters

    # Filtering news by company related
    def filter_company_related(self, articles: list) -> list:
        related_articles = []

        for article in articles:
            try:
                title = article.get("title", "")
                if title and any(
                    word.lower() in title.lower() for word in self.query.split()
                ):
                    related_articles.append(article)
            except (KeyError, AttributeError, TypeError) as e:
                logging.warning(f"Error processing article: {e}")
                continue

        logging.info(
            f"Filtered by company related: {len(articles)} -> {len(related_articles)} articles"
        )
        return related_articles

    # Filtering articles by financial keywords
    def filter_financial_keywords(self, articles: list) -> list:
        # TODO: improve the accuracy
        financial_articles = []

        for article in articles:
            try:
                title = article.get("title", "")
                if title and any(
                    keyword in title.lower() for keyword in self.financial_keywords
                ):
                    financial_articles.append(article)
            except (KeyError, AttributeError, TypeError) as e:
                logging.warning(f"Error processing article: {e}")
                continue

        logging.info(
            f"Filtered by financial keywords: {len(articles)} -> {len(financial_articles)} articles"
        )
        return financial_articles

    # Filtering articles by english
    def filter_language(self, articles: list, allowed_languages: list) -> list:
        filtered = [
            article
            for article in articles
            if article.get("language", "").lower()
            in [l.lower() for l in allowed_languages]
        ]
        logging.info(
            f"Filtered by language: {len(articles)} -> {len(filtered)} articles"
        )
        return filtered

    # Filtering duplicates
    def remove_duplicates(self, data):
        try:
            articles = data.get("articles", [])
            if not articles:
                logging.warning("No articles to check for duplicates")
                return []

            seen = {}
            unique = []

            for article in articles:
                try:
                    title = article.get("title", "").strip().lower()
                    domain = article.get("domain", "").strip().lower()

                    if not title:
                        continue

                    # Dedupe key: domain + title
                    key = f"{domain}||{title}"

                    if key not in seen:
                        article["coverage_count"] = 1
                        seen[key] = article
                    else:
                        seen[key]["coverage_count"] += 1

                except Exception as e:
                    logging.warning(f"Error processing article: {e}")
                    continue

            unique = list(seen.values())
            logging.info(
                f"Removed duplicates (domain+title): {len(articles)} -> {len(unique)}"
            )
            return unique

        except Exception as e:
            logging.error(f"Unexpected error in duplicates: {e}")
            return []

    # Save & display results
    def save_articles(self, articles: list) -> None:
        try:
            output_data = {
                "query": self.query,
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
                "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "article_count": len(articles),
                "articles": articles,
            }

            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logging.info(f"Saved {len(articles)} articles to {self.output_file}")

        except IOError as e:
            error_msg = f"Failed to save articles to {self.output_file}: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error while saving articles: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)

    def display_results(self) -> None:
        """
        Display and save the already-filtered articles from search()
        """
        if not self.data:
            logging.warning("No data to display")
            print("No data available to display.")
            return

        articles = self.data.get("articles", [])
        if not articles:
            logging.warning("No articles to display")
            print("No articles found in the response.")
            return

        try:
            self.save_articles(articles)
        except Exception as e:
            logging.error(f"Failed to save articles: {e}")
            print(f"Warning: Could not save articles to file")

        print(f"\n{'='*50}")
        print(f"Displaying {len(articles)} articles:")
        print(f"{'='*50}\n")

        for i, article in enumerate(articles, 1):
            try:
                title = article.get("title", "No title")
                print(f"[{i}] Title: {title}")
                print("-" * 40)
            except Exception as e:
                logging.warning(f"Error displaying article {i}: {e}")
                continue

        logging.info(f"Displayed {len(articles)} articles")

    def add_recency_weights(self, articles: list, ref_date: datetime) -> None:
        for article in articles:
            try:
                seendate_str = article.get("seendate")
                if seendate_str:
                    seen_dt = datetime.strptime(seendate_str, "%Y%m%dT%H%M%SZ")
                else:
                    seen_dt = self.backward_end_date

                days_ago = (ref_date.date() - seen_dt.date()).days

                if days_ago < 0:
                    days_ago = 0

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

            except Exception as e:
                logging.warning(f"Error computing recency weight for article: {e}")
                article["days_ago"] = None
                article["recency_weight"] = 1.0

    def fetch_article_content(self, url: str) -> str | None:
        if not url:
            return None

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
        except Timeout:
            logging.warning(f"Timeout while fetching article content: {url}")
            return None
        except HTTPError as e:
            logging.warning(f"HTTP error while fetching article content: {e} | {url}")
            return None
        except RequestException as e:
            logging.warning(
                f"Request error while fetching article content: {e} | {url}"
            )
            return None
        except Exception as e:
            logging.warning(
                f"Unexpected error while fetching article content: {e} | {url}"
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
            else:
                return None

        except Exception as e:
            logging.warning(f"Error parsing HTML for content: {e} | {url}")
            return None

    def enrich_articles_with_content(self, articles: list) -> None:
        """Noisy and null data returns -> future improvement is solving this issue"""
        logging.info(f"Enriching {len(articles)} articles with full content...")
        for i, article in enumerate(articles, 1):
            try:
                url = article.get("url") or article.get("sourceurl")
                if not url:
                    logging.debug(f"No URL found for article {i}, skipping")
                    continue

                content = self.fetch_article_content(url)
                if content:
                    article["content"] = content
                    logging.info(f"Fetched content for article {i}/{len(articles)}")
                else:
                    article["content"] = None

            except Exception as e:
                logging.warning(f"Error while enriching article {i}: {e}")
                article["content"] = None
                continue
