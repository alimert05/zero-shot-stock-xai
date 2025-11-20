import json
import requests
from config import BASE_URL, LOG_PATH, TEMP_PATH, FINANCIAL_KEYWORDS, REQUEST_TIMEOUT_LIMIT
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
        

    def normalise_date(self, date_str: str) -> str:
        self.validate_date(date_str)
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.strftime("%Y%m%d000000")
     

    def search(self) -> dict:
        self.validate_date(self.start_date)
        self.validate_date(self.end_date)

        start_dt = datetime.strptime(self.start_date, "%d-%m-%Y")
        end_dt = datetime.strptime(self.end_date, "%d-%m-%Y")

        self.backward_start_date = start_dt - timedelta(days=self.max_backward_days)
        self.backward_end_date = start_dt - timedelta(days=1)

        backward_start = self.backward_start_date
        backward_end = self.backward_end_date


        if end_dt < start_dt:
            raise ValueError("End date cannot be before start date.")

        all_articles = []
        filtered_articles = []

        self._fetch_window(backward_start, backward_end, all_articles, filtered_articles, depth=0)

        final_articles = filtered_articles[:self.number_of_news]
        self.data = {"articles": final_articles}

        if len(final_articles) < self.number_of_news:
            logging.info(
                f"Only {len(final_articles)} articles matched filters, "
                f"but {self.number_of_news} were requested."
            )
    
    def _fetch_window(self, start_dt, end_dt, all_articles, filtered_articles, depth=0):

        if len(filtered_articles) >= self.number_of_news:
            return

        if end_dt < start_dt:
            return

        MAX_DEPTH = 6
        if depth > MAX_DEPTH:
            logging.info("Max recursion depth reached, stopping further splitting.")
            return

        window_days = (end_dt - start_dt).days

        start_str = start_dt.strftime("%d-%m-%Y")
        end_str = end_dt.strftime("%d-%m-%Y")

        start_norm = self.normalise_date(start_str)
        end_norm = self.normalise_date(end_str).replace("000000", "235959")



        logging.info(
            f"[depth={depth}] Fetching window: {start_str} to {end_str} "
            f"({window_days + 1} days)"
        )


        params = {
            "query": self.query,
            "mode": "artlist",
            "maxrecords": 250,
            "STARTDATETIME": start_norm,
            "ENDDATETIME": end_norm,
            "sourcelang": "english",
            "format": "json",
        }

        try:
            logging.info(f"Sending request to {BASE_URL} with timeout={self.timeout}s")
            response = requests.get(BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()

            try:
                batch_data = response.json()
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e}")
                logging.error(f"Response content: {response.text}")
                raise Exception("API returned invalid JSON response.")

            batch_articles = batch_data.get("articles", [])
            logging.info(
                f"[depth={depth}] Received {len(batch_articles)} raw articles "
                f"for this window."
            )

            if not batch_articles:
                return

            all_articles.extend(batch_articles)

            unique_articles = self.remove_duplicates({"articles": all_articles})
            english_articles = self.filter_language(unique_articles, ["English"])
            new_filtered = self.filter_financial_keywords(english_articles)

            filtered_articles[:] = new_filtered

            logging.info(
                f"[depth={depth}] Progress: Raw={len(all_articles)}, "
                f"Unique={len(unique_articles)}, "
                f"Filtered={len(filtered_articles)}/{self.number_of_news}"
            )

            if len(filtered_articles) >= self.number_of_news:
                return

            if len(batch_articles) == 250 and window_days > 0:
                print("250 den fazla")
                mid_dt = start_dt + (end_dt - start_dt) / 2
                mid_dt = datetime(mid_dt.year, mid_dt.month, mid_dt.day)

                if mid_dt <= start_dt or mid_dt >= end_dt:
                    return

                logging.info(
                    f"[depth={depth}] Window appears dense (250 records). "
                    f"Splitting into [{start_dt.strftime('%d-%m-%Y')} - "
                    f"{mid_dt.strftime('%d-%m-%Y')}] and "
                    f"[{(mid_dt + timedelta(days=1)).strftime('%d-%m-%Y')} - "
                    f"{end_dt.strftime('%d-%m-%Y')}]"
                )

                self._fetch_window(start_dt, mid_dt, all_articles, filtered_articles, depth + 1)

                if len(filtered_articles) < self.number_of_news:
                    self._fetch_window(
                        mid_dt + timedelta(days=1),
                        end_dt,
                        all_articles,
                        filtered_articles,
                        depth + 1,
                    )

        except Timeout:
            error_msg = f"Request timed out after {self.timeout} seconds."
            logging.error(error_msg)
            raise Exception(error_msg)

        except ConnectionError as e:
            error_msg = (
                f"Connection error: Unable to connect to {BASE_URL}. "
                f"Please check your internet connection."
            )
            logging.error(f"{error_msg} Details: {e}")
            raise Exception(error_msg)

        except HTTPError as e:
            status_code = e.response.status_code
            if status_code == 400:
                error_msg = "Bad Request (400): Invalid query parameters. Please check your inputs."
            elif status_code == 401:
                error_msg = "Unauthorized (401): API key may be missing or invalid."
            elif status_code == 403:
                error_msg = "Forbidden (403): Access denied. Check API permissions."
            elif status_code == 404:
                error_msg = "Not Found (404): API endpoint not found."
            elif status_code == 429:
                error_msg = "Too Many Requests (429): Rate limit exceeded. Please wait and try again."
            elif status_code == 500:
                error_msg = "Internal Server Error (500): API server error. Try again later."
            elif status_code == 502:
                error_msg = "Bad Gateway (502): API gateway error. Try again later."
            elif status_code == 503:
                error_msg = "Service Unavailable (503): API temporarily unavailable. Try again later."
            elif status_code == 504:
                error_msg = "Gateway Timeout (504): API request timed out. Try again later."
            else:
                error_msg = f"HTTP Error {status_code}: {e}"

            logging.error(error_msg)
            logging.error(f"Response content: {e.response.text}")
            raise Exception(error_msg)

        except RequestException as e:
            error_msg = f"Request failed: {type(e).__name__} - {str(e)}"
            logging.error(error_msg)
            raise Exception(f"An unexpected error occurred while fetching data: {e}")


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
    
    def save_articles(self, articles: list) -> None:
        try:
            output_data = {
                "query": self.query,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "article_count": len(articles),
                "articles": articles
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