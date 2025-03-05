from scraper import ArxivScraper
from constants import subcats

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
    TEST = True 
    scraper = ArxivScraper()
    items = list(subcats.items())
    if TEST:
        items = items[:3]

    all_articles = []
    for main_cat, subcat_list in items:
        if subcat_list:
            for subcat in subcat_list:
                print(f"Scraping de la sous-catégorie : {subcat}")
                try:
                    articles = scraper.scrape_subcategory(subcat, max_articles=500)
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"Erreur lors du scraping de la sous-catégorie {subcat} : {e}")
        else:
            print(f"Scraping de la catégorie : {main_cat}")
            try:
                articles = scraper.scrape_subcategory(main_cat, max_articles=500)
                all_articles.extend(articles)
            except Exception as e:
                print(f"Erreur lors du scraping de la catégorie {main_cat} : {e}")

    write_csv(all_articles, "all_articles.csv")


if __name__ == "__main__":
    main()