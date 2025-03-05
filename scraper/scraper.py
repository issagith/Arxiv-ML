import requests
import feedparser
import time
import csv
from constants import subcats


class ArxivScraper:
    def __init__(self, base_url="http://export.arxiv.org/api/query?", rate_limit=3):
        self.base_url = base_url
        self.rate_limit = rate_limit

    def build_query(self, subcat, start, batch_size):
        return f"search_query=cat:{subcat}&start={start}&max_results={batch_size}"

    def fetch_feed(self, url):
        print(f"Fetching: {url}")
        response = requests.get(url)
        return feedparser.parse(response.text)

    def process_entry(self, entry):
        return {
            "id": entry.get("id"),
            "title": entry.get("title"),
            "summary": entry.get("summary"),
            "authors": ", ".join(author.name for author in entry.get("authors", [])),
            "published": entry.get("published"),
            "updated": entry.get("updated"),
            "link": next(
                (link.href for link in entry.get("links", []) if link.get("rel") == "alternate"), None
            )
        }

    def scrape_subcategory(self, subcat, max_articles=10000, batch_size=100):
        articles = []
        for start in range(0, max_articles, batch_size):
            query = self.build_query(subcat, start, batch_size)
            url = self.base_url + query
            feed = self.fetch_feed(url)

            if not feed.entries:
                print(f"Aucune nouvelle entrée trouvée pour {subcat} à start={start}.")
                break

            for entry in feed.entries:
                article = self.process_entry(entry)
                articles.append(article)

            print(f"{len(articles)} articles récupérés jusqu'à présent pour la sous-catégorie {subcat}")
            time.sleep(self.rate_limit)
        return articles




