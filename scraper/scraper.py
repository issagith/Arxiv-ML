import requests
import feedparser
import time
import csv

class ArxivScraper:
    def __init__(self, base_url="http://export.arxiv.org/api/query?", rate_limit=3, csv_file=None, csv_mode="at_end"):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.csv_file = csv_file
        self.csv_mode = csv_mode
        self.fieldnames = ["id", "title", "summary", "category", "authors", "published", "updated"]

    def build_query(self, search_query="", start=0, max_results=100, sortBy=None, sortOrder=None):
        params = []
        if search_query:
            params.append(f"search_query={search_query}")
        params.append(f"start={start}")
        params.append(f"max_results={max_results}")
        if sortBy:
            params.append(f"sortBy={sortBy}")
        if sortOrder:
            params.append(f"sortOrder={sortOrder}")
        query_string = "&".join(params)
        return self.base_url + query_string

    def fetch_data(self, url):
        print(f"Fetching: {url}")
        response = requests.get(url)
        return feedparser.parse(response.text)

    def process_entry(self, entry):
        return {
            "id": entry.get("id"),
            "title": entry.get("title"),
            "summary": entry.get("summary"),
            "category": entry.get("arxiv_primary_category", {}).get("term"),
            "authors": ", ".join(author.name for author in entry.get("authors", [])),
            "published": entry.get("published"),
            "updated": entry.get("updated"),
        }

    def scrape_query(self, search_query="", max_articles=10000, batch_size=100,
                     sortBy=None, sortOrder=None, max_retries=3, retry_wait=10):
        total = 0
        start = 0
        # Tant qu'on n'a pas atteint le nombre max d'articles demandé
        while start < max_articles:
            query_url = self.build_query(
                search_query=search_query, 
                start=start, 
                max_results=batch_size, 
                sortBy=sortBy, 
                sortOrder=sortOrder
            )
            data = self.fetch_data(query_url)
            
            # Si aucune entrée n'est retournée, vérifier le total d'articles disponibles
            if not data.entries:
                total_results = int(data.feed.get("opensearch_totalresults", 0))
                if total_results > total:
                    attempt = 1
                    while attempt <= max_retries:
                        print(f"No new entries found at start={start}. Retry {attempt}/{max_retries} in {retry_wait} seconds...")
                        time.sleep(retry_wait)
                        data = self.fetch_data(query_url)
                        if data.entries:
                            break
                        attempt += 1
                    if not data.entries:
                        print(f"Max retries reached at start={start} with no new entries. Moving on.")
                        break
                else:
                    print(f"No new entries found at start={start}.")
                    break

            for entry in data.entries:
                article = self.process_entry(entry)
                total += 1
                yield article
            print(f"{total} articles retrieved for query '{search_query}'")
            start += batch_size
            time.sleep(self.rate_limit)

    def scrape(self, *, categories=None, keyword=None, chronological=False, max_articles=10000, batch_size=100,
               sortOrder="descending", max_retries=3, retry_wait=10):
        # Construction des requêtes par catégories/sous-catégories (comportement initial)
        if categories is not None:
            queries = []
            if isinstance(categories, dict):
                for main_cat, subcat_list in categories.items():
                    if subcat_list:
                        for subcat in subcat_list:
                            queries.append(f"cat:{subcat}")
                    else:
                        queries.append(f"cat:{main_cat}")
            else:
                queries = [f"cat:{cat}" for cat in categories]
        elif keyword is not None:
            queries = [f"all:{keyword}"]
        else:
            queries = ["all:*"]

        # Si l'option chronologique est activée, on définit les paramètres de tri
        sortBy = "submittedDate" if chronological else None
        sortOrder_final = sortOrder if chronological else None

        all_articles = []
        writer = None
        csvfile = None
        if self.csv_file:
            csvfile = open(self.csv_file, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

        for query in queries:
            print(f"Scraping for query: '{query}'")
            articles = []
            for article in self.scrape_query(
                search_query=query,
                max_articles=max_articles,
                batch_size=batch_size,
                sortBy=sortBy,
                sortOrder=sortOrder_final,
                max_retries=max_retries,
                retry_wait=retry_wait
            ):
                articles.append(article)
                if writer and self.csv_mode == "per_article":
                    writer.writerow(article)
                    csvfile.flush()
            if writer and self.csv_mode == "per_subquery":
                for article in articles:
                    writer.writerow(article)
                csvfile.flush()
            all_articles.extend(articles)

        if writer and self.csv_mode == "at_end":
            for article in all_articles:
                writer.writerow(article)
            csvfile.flush()

        if csvfile:
            csvfile.close()
            print(f"Articles have been saved to {self.csv_file}.")

        print(f"Scraping completed. {len(all_articles)} articles retrieved.")
        return all_articles
