import requests
from config import BASE_URL
from datetime import datetime


class Fetcher:
    
    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.query = None
        self.data = None

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
            raise ValueError(f"Invalid date format: {date_str}. Please use DD-MM-YYYY (e.g., 01-01-2000)")
        
        day, month, year = map(int, date_str.split('-'))
        if not (1 <= day <= 31):
            raise ValueError(f"Invalid day: {day}. Day must be between 1 and 31.")
        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month: {month}. Month must be between 1 and 12.")
        if year < 1990 or year > datetime.now().year + 1:
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
            "maxrecords" : 10,
            # "timespan" : "24h",
            "STARTDATETIME" : start_norm,
            "ENDDATETIME" : end_norm,
            "sourcelang" : "english",
            "sourcecountry" : "UK",
            "format" : "json"
        }

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        self.data = response.json()
        return self.data

    def display_results(self) -> None:
        if not self.data:
            return ("No data found.")

        articles = self.data.get("articles", [])
        if not articles:
            return ("No articles found.")
        
        for article in articles[:10]:
            print(f"Title: {article['title']}")

 


       
            