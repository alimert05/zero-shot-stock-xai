import json
import requests
from config import BASE_URL, LOG_PATH, TEMP_PATH, FINANCIAL_KEYWORDS, REQUEST_TIMEOUT_LIMIT
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException
from datetime import datetime
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

class Fetcher:
    
    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.query = None
        self.data = None
        self.temp_dir = TEMP_PATH
        self.output_file = os.path.join(self.temp_dir, "articles.json")
        self.financial_keywords = FINANCIAL_KEYWORDS
        self.timeout = REQUEST_TIMEOUT_LIMIT
    
    def filter_financial_keywords(self, articles : list) -> list:
        financial_articles = []

        for article in articles:
            if any(keyword in article.get('title', '').lower() for keyword in self.financial_keywords):
                financial_articles.append(article)
        logging.info(f"Filtered by financial keywords: {len(articles)} -> {len(financial_articles)} articles")
        return financial_articles


    def get_input(self) -> str:
        self.query = input("Enter company's name: ")
        self.start_date = input("Enter start date(DD-MM-YYYY): ")
        self.end_date = input("Enter end date(DD-MM-YYYY): ")
        return self.query
    
    def validate_date(self, date_str : str) -> None:
        # TO DO: Februrary scenario
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
        start_norm = self.normalise_date(self.start_date)
        end_norm = self.normalise_date(self.end_date)
            
        params = {
            "query": self.query,
            "mode" : "artlist",  
            "maxrecords" : 250,
            # "timespan" : "24h",
            "STARTDATETIME" : start_norm,
            "ENDDATETIME" : end_norm,
            "sourcelang" : "english",
            "format" : "json"
        }

        try: 
            logging.info(f"Sending request to {BASE_URL} with timeout={self.timeout}s")

            response = requests.get(BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()

            try: 
                self.data = response.json()
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e}")
                logging.error(f"Response content: {response.text}")
                raise Exception("API returned invalid JSON response.")
            
            if self.data:
                logging.info(f"Fetched data for {self.query} between {self.start_date} and {self.end_date}")
            else: 
                logging.warning("API returned empty response")
            return self.data

        except Timeout:
            error_msg = f"Request timed out after {self.timeout} seconds."
            logging.error(error_msg)
            raise Exception(error_msg)
        
        except ConnectionError as e:
            error_msg = f"Connection error: Unable to connect to {BASE_URL}. Please check your internet connection."
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
    
    def remove_duplicates(self):
        articles = self.data.get("articles", [])
        seen = set()
        unique = []

        for article in articles:
            title = article['title'].strip().lower()
            if title and title not in seen:
                seen.add(title)
                unique.append(article)
        logging.info(f"Removed duplicates: {len(articles) - len(unique)} duplicates found.")
        return unique
    
    def save_articles(self, articles: list) -> None:
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

    def display_results(self) -> None:
        if not self.data:
            logging.warning("No data to display")
            return ("No data found.")

        articles = self.data.get("articles", [])
        if not articles:
            logging.warning("No articles to display")
            return ("No articles found.")
        
        unique_art = self.remove_duplicates()
        english_articles = self.filter_language(unique_art, ['English'])
        financial_art = self.filter_financial_keywords(english_articles)
    
        if not english_articles:
            logging.warning("No UK articles found after filtering.")
            print("No UK articles found. Showing all English articles instead.")
            english_articles = unique_art

        if not financial_art:
            logging.warning("No financial articles found after filtering.")
            print("No financial articles found.")
            financial_art = english_articles

        self.save_articles(financial_art)

        for article in financial_art:
            print(f"Title: {article['title']}")
            print("-" * 40)
        logging.info(f"Displayed {len(unique_art)}")

 


       
            