import requests
from config import BASE_URL


class Fetcher:
    
    def __init__(self):
        self.query = None
        self.data = None

    def get_input(self) -> str:
        self.query = input("Enter company's name: ")
        return self.query

    def search(self) -> dict:
            
        params = {
            "query": self.query,
            "mode" : "artlist",  
            "maxrecords" : 10,
            "timespan" : "24h",
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

 


       
            