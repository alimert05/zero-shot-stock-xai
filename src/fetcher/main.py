import requests
from config import BASE_URL

query = input("Enter company's name: ")

params = {
    "query": query,
    "mode" : "artlist",  
    "maxrecords" : 10,
    "timespan" : "24h",
    "sourcelang" : "english",
    "sourcecountry" : "UK",
    "format" : "json"
}

response = requests.get(BASE_URL, params=params)
data = response.json()

articles = data.get("articles", [])
for article in articles[:5]:
    print(f"Title: {article['title']}")