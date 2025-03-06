from scraper import ArxivScraper
from constants import subcats
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
    csv_filename = f"extractions/CSV/BySubCats/{current_time}.csv"
    scraper = ArxivScraper(rate_limit=3, retry_wait=30, debug=True)
    articles = scraper.scrape(chronological=True, categories=test_cats, max_articles=5, batch_size=5)
    print(f"Total articles scrapés par catégories/sous-catégories : {len(articles)}")
    
if __name__ == "__main__":
    main()
