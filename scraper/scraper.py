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
        self.fieldnames = ["id", "title", "summary", "authors", "published", "updated", "link"]

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
            "authors": ", ".join(author.name for author in entry.get("authors", [])),
            "published": entry.get("published"),
            "updated": entry.get("updated"),
            "link": next((link.href for link in entry.get("links", []) if link.get("rel") == "alternate"), None)
        }

    def scrape_query(self, search_query="", max_articles=10000, batch_size=100, sortBy=None, sortOrder=None):
        total = 0
        for start in range(0, max_articles, batch_size):
            query_url = self.build_query(
                search_query=search_query, 
                start=start, 
                max_results=batch_size, 
                sortBy=sortBy, 
                sortOrder=sortOrder
            )
            data = self.fetch_data(query_url)
            if not data.entries:
                print(f"No new entries found at start={start}.")
                break
            for entry in data.entries:
                article = self.process_entry(entry)
                total += 1
                yield article
            print(f"{total} articles retrieved for query '{search_query}'")
            time.sleep(self.rate_limit)

    def scrape(self, *, categories=None, keyword=None, chronological=False, max_articles=10000, batch_size=100, sortOrder="descending"):
        
        mode_count = 0
        if categories is not None:
            mode_count += 1
        if keyword is not None:
            mode_count += 1
        if chronological:
            mode_count += 1
        if mode_count != 1:
            raise ValueError("Please specify exactly one type of query: either 'categories', 'keyword', or 'chronological'.")

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
            sortBy = None
            sortOrder_final = None
        elif keyword is not None:
            queries = [f"all:{keyword}"]
            sortBy = None
            sortOrder_final = None
        elif chronological:
            queries = ["all:*"]
            sortBy = "submittedDate"
            sortOrder_final = sortOrder

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
                sortOrder=sortOrder_final
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
