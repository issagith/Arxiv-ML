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


def write_csv(articles, output_file):
    if not articles:
        print("Aucun article à écrire.")
        return
    fieldnames = ["id", "title", "summary", "authors", "published", "updated", "link"]
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)
    print(f"{len(articles)} articles sauvegardés dans {output_file}")


def main():
    TEST = True  # Définir à True pour ne traiter que les 3 premières catégories
    scraper = ArxivScraper()
    items = list(subcats.items())
    if TEST:
        items = items[:3]

    all_articles = []
    # Pour chaque catégorie principale dans subcats
    for main_cat, subcat_list in items:
        # Si la catégorie possède des sous-catégories, on les traite
        if subcat_list:
            for subcat in subcat_list:
                print(f"Scraping de la sous-catégorie : {subcat}")
                try:
                    articles = scraper.scrape_subcategory(subcat, max_articles=500)
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"Erreur lors du scraping de la sous-catégorie {subcat} : {e}")
        else:
            # Sinon, on traite la catégorie principale directement
            print(f"Scraping de la catégorie : {main_cat}")
            try:
                articles = scraper.scrape_subcategory(main_cat, max_articles=500)
                all_articles.extend(articles)
            except Exception as e:
                print(f"Erreur lors du scraping de la catégorie {main_cat} : {e}")

    # Écriture de tous les articles dans un seul fichier CSV
    write_csv(all_articles, "all_articles.csv")


if __name__ == "__main__":
    main()