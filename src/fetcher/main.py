import requests

query = input("Enter company's name: ")

url = "https://api.gdeltproject.org/api/v2/doc/doc"
params = {
    "query": query,
    "mode" : "artlist",  
    "maxrecords" : 10,
    "timespan" : "24h",
    "sourcelang" : "english",
    "sourcecountry" : "UK",
    "format" : "json"
}

response = requests.get(url, params=params)
data = response.json()

articles = data.get("articles", [])
for article in articles[:5]:
    print(f"Title: {article['title']}")