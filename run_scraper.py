from ecb_scraper.scraper import load_ecb_conferences

if __name__ == "__main__":
    df = load_ecb_conferences(start_year=1998, end_year=2025)
    df.to_csv("ecb_conferences_1998_2025.csv", index=False)
    print("✅ Scraping terminé : fichier ecb_conferences_1998_2025.csv créé !")
