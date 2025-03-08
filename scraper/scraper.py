import requests
import feedparser
import time
import csv
from tqdm import tqdm

class ArxivScraper:
    def __init__(self, base_url="http://export.arxiv.org/api/query?", rate_limit=3, csv_file=None, csv_mode="at_end",
                 verbose=True, debug=False, max_retries=3, retry_wait=10):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.csv_file = csv_file
        self.csv_mode = csv_mode
        self.verbose = verbose    
        self.debug = debug        
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.fieldnames = ["id", "title", "summary", "category", "authors", "published", "updated"]

    def log(self, message):
        if self.verbose:
            # Utilise tqdm.write pour ne pas interférer avec la barre de progression
            tqdm.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

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
        if self.debug:
            self.log(f"Fetching: {url}")
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

    def scrape_query(self, search_query="", max_articles=1000, batch_size=10, sortBy=None, sortOrder=None):
        total = 0
        start = 0
        query_start_time = time.time()  # Début de la requête globale
        while start < max_articles:
            query_url = self.build_query(
                search_query=search_query,
                start=start,
                max_results=batch_size,
                sortBy=sortBy,
                sortOrder=sortOrder
            )
            data = self.fetch_data(query_url)
            
            if not data.entries:
                total_results = int(data.feed.get("opensearch_totalresults", 0))
                if total_results > total:
                    attempt = 1
                    while attempt <= self.max_retries:
                        self.log(f"No new entries at start={start}. Retry {attempt}/{self.max_retries} in {self.retry_wait} seconds...")
                        time.sleep(self.retry_wait)
                        data = self.fetch_data(query_url)
                        if data.entries:
                            break
                        attempt += 1
                    if not data.entries:
                        self.log(f"Max retries reached at start={start}. Moving on.")
                        break
                else:
                    self.log(f"No new entries at start={start}.")
                    break

            for entry in data.entries:
                article = self.process_entry(entry)
                total += 1
                yield article
            start += batch_size
            time.sleep(self.rate_limit)
        query_end_time = time.time()
        self.log(f"Query '{search_query}' finished in {query_end_time - query_start_time:.2f} seconds.")

    def scrape(self, *, categories=None, keyword=None, chronological=False, max_articles=10000, batch_size=100, sortOrder="descending"):
        # Construction des requêtes par catégories ou mots-clés
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

        sortBy = "submittedDate" if chronological else None
        sortOrder_final = sortOrder if chronological else None

        all_articles = []
        writer = None
        csvfile = None
        # Si un chemin CSV est fourni, on crée le fichier
        if self.csv_file is not None:
            csvfile = open(self.csv_file, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

        global_start_time = time.time()
        for query in queries:
            self.log(f"Starting scraping for query: '{query}'")
            articles = []
            # Barre de progression pour la requête en cours
            with tqdm(total=max_articles, desc=f"Query: {query}", disable=not self.verbose) as pbar:
                for article in self.scrape_query(
                    search_query=query,
                    max_articles=max_articles,
                    batch_size=batch_size,
                    sortBy=sortBy,
                    sortOrder=sortOrder_final
                ):
                    articles.append(article)
                    pbar.update(1)
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
            self.log(f"Articles saved to {self.csv_file}.")

        global_end_time = time.time()
        duration_hours = (global_end_time - global_start_time) / 3600
        self.log(f"Scraping completed. {len(all_articles)} articles retrieved in {duration_hours:.2f} hours.")
        return all_articles
