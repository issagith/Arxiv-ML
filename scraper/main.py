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
    scraper = ArxivScraper(csv_file=csv_filename, csv_mode="per_article", rate_limit=3)
    articles = scraper.scrape(categories=subcats, max_articles=10000, batch_size=200)
    print(f"Total articles scrapés par catégories/sous-catégories : {len(articles)}")
    
if __name__ == "__main__":
    main()
