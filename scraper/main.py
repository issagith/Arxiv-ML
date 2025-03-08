from scraper import ArxivScraper
from constants import subcats, subcats_from_mathOC
from datetime import datetime

def main():
    test_cats = {
        "math": [
                "math.AG",
                "math.AT",
                "math.AP",
                "math.CT",
                "math.CA",
            ]
    }
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = "extractions/CSV/BySubCats/20250307_094141.csv"
    csv_filename = f"extractions/CSV/BySubCats/{current_time}.csv"
    scraper = ArxivScraper(rate_limit=3, csv_file=file_name, csv_mode="per_article", verbose=True, debug=True, max_retries=3, retry_wait=30)
    articles = scraper.scrape(categories=subcats_from_mathOC, chronological=True, max_articles=20000, batch_size=1000, sortOrder="descending")
    print(f"Total articles scrapés par catégories/sous-catégories : {len(articles)}")
    
if __name__ == "__main__":
    main()
