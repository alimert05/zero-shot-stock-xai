import json
import time
import requests
from gdeltdoc import GdeltDoc, Filters
from config import BASE_URL, LOG_PATH, TEMP_PATH, FINANCIAL_KEYWORDS, REQUEST_TIMEOUT_LIMIT, THEMES
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException
from datetime import datetime, timedelta
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

"""
Dynamic Selection for Fetching Data:
- Read URL to reach contents: on/off
- How many headlines/news to fetch: {number_of_news} -> density-aware pagination
"""

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
            max_days_input = input(f"Enter maximum backward search days (default 14): ").strip()
            if max_days_input:
                self.max_backward_days = int(max_days_input)
            else:
                self.max_backward_days = 14
            
            return self.query
        
        except Exception as e:
            logging.error(f"Unexpected error during input: {e}")
            raise
    
    def validate_date(self, date_str : str) -> None:
        try: 
            dt = datetime.strptime(date_str, "%d-%m-%Y")
        except ValueError:
            logging.error(f"Invalid date format: {date_str}")
            raise ValueError(f"Invalid date format: {date_str}. Please use DD-MM-YYYY (e.g., 01-01-2000)")
        
        day, month, year = map(int, date_str.split('-'))

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

        while current_day >= self.backward_start_date and len(filtered_articles) < self.number_of_news:
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
                    theme = THEMES
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
                logging.info(f"No articles returned for {day_start.strftime('%Y-%m-%d')} - {day_end.strftime('%Y-%m-%d')}.")
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
            new_filtered = self.filter_financial_keywords(english_articles)

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
        
        final_articles = filtered_articles[:self.number_of_news]
        self.data = {"articles": final_articles}

        if len(final_articles) < self.number_of_news:
            logging.info(
                f"Only {len(final_articles)} articles matched filters "
                f"within {self.max_backward_days} days, "
                f"but {self.number_of_news} were requested."
            )

    
    # Filters                
    def filter_financial_keywords(self, articles : list) -> list:
        # TODO: improve the accuracy
        financial_articles = []

        for article in articles:
            try:
                title = article.get('title', '')
                if title and any(keyword in title.lower() for keyword in self.financial_keywords):
                    financial_articles.append(article)
            except (KeyError, AttributeError, TypeError) as e:
                logging.warning(f"Error processing article: {e}")
                continue
        
        logging.info(f"Filtered by financial keywords: {len(articles)} -> {len(financial_articles)} articles")
        return financial_articles

    def filter_language(self, articles: list, allowed_languages: list) -> list:
        filtered = [
            article for article in articles
            if article.get('language', '').lower() in [l.lower() for l in allowed_languages]
            ]
        logging.info(f"Filtered by language: {len(articles)} -> {len(filtered)} articles")
        return filtered
      
                
    def remove_duplicates(self, data):
        try:

            articles = data.get("articles", [])
            if not articles:
                logging.warning("No articles to check for duplicates")
                return []
            
            seen = set()
            unique = []

            for article in articles:
                try:
                    title = article['title'].strip().lower()
                    if title and title not in seen:
                        seen.add(title)
                        unique.append(article)
                except (AttributeError, TypeError) as e:
                    logging.warning(f"Error processing article title: {e}")
                    continue

            logging.info(f"Removed duplicates: {len(articles) - len(unique)} duplicates found.")
            return unique
        
        except Exception as e:
            logging.error(f"Unexpected error in removing duplicates: {e}")
            return[]
    
    # Save & display results
    def save_articles(self, articles: list) -> None:
        try:
            output_data = {
                "query": self.query,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "backward_start_date": self.backward_start_date.strftime("%d-%m-%Y")
                if self.backward_start_date
                else None,
                "backward_end_date": self.backward_end_date.strftime("%d-%m-%Y")
                if self.backward_end_date
                else None,
                "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "article_count": len(articles),
                "articles": articles,
            }
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved {len(articles)} articles to {self.output_file}")

        except IOError as e:
            error_msg = (f"Failed to save articles to {self.output_file}: {e}")
            logging.error(error_msg)
            raise Exception(error_msg)
        
        except Exception as e:
            error_msg = (f"Unexpected error while saving articles: {e}")
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
                title = article.get('title', 'No title')
                print(f"[{i}] Title: {title}")
                print("-" * 40)
            except Exception as e:
                logging.warning(f"Error displaying article {i}: {e}")
                continue
        
        logging.info(f"Displayed {len(articles)} articles")