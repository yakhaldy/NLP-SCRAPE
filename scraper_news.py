# ============================================================
# scraper_news.py — Collect articles from BBC News
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
from datetime import datetime
import os

# --- Settings ---
BASE_URL = "https://www.bbc.com"

# More sections = more articles
SECTIONS = [
    "/news/technology",
    "/news/science_and_environment",
    "/news/business",
    "/news/world",
    "/news/health",
    "/news/entertainment_and_arts",
    "/news/uk",
    "/news/us-and-canada",
    "/news/world/asia",
    "/news/world/europe",
    "/news/world/africa",
    "/news/world/latin_america",
    "/sport",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OUTPUT_FILE = "data/articles.csv"


# ============================================================
# Function 1: Get article links from a section page
# ============================================================
def get_article_links(section_url):
    links = []
    try:
        response = requests.get(BASE_URL + section_url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"  WARNING: Failed to connect to {section_url} — code: {response.status_code}")
            return links

        soup = BeautifulSoup(response.text, "html.parser")
        all_anchors = soup.find_all("a", href=True)

        for anchor in all_anchors:
            href = anchor["href"]
            if href.startswith("/news/") and href.count("/") >= 3:
                full_url = BASE_URL + href
                if full_url not in links:
                    links.append(full_url)

            # Also catch /sport/ articles
            if href.startswith("/sport/") and href.count("/") >= 3:
                full_url = BASE_URL + href
                if full_url not in links:
                    links.append(full_url)

    except Exception as e:
        print(f"  ERROR in {section_url}: {e}")

    return links


# ============================================================
# Function 2: Scrape a single article
# ============================================================
def scrape_article(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract headline
        headline_tag = soup.find("h1")
        if not headline_tag:
            return None
        headline = headline_tag.get_text(strip=True)

        # Extract body
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        body = " ".join(p.get_text(strip=True) for p in paragraphs)

        if len(body) < 100:
            return None

        date_today = datetime.now().strftime("%Y-%m-%d")
        article_id = str(uuid.uuid4())

        return {
            "id":       article_id,
            "url":      url,
            "date":     date_today,
            "headline": headline,
            "body":     body,
        }

    except Exception as e:
        print(f"  ERROR parsing {url}: {e}")
        return None


# ============================================================
# Function 3: Main — coordinates everything
# ============================================================
def main():
    print("=" * 60)
    print("Starting BBC News scraper")
    print("=" * 60)

    all_articles = []
    seen_urls = set()

    # --- Step 1: Collect links from all sections ---
    all_links = []
    for section in SECTIONS:
        print(f"\nCollecting links from: {section}")
        links = get_article_links(section)
        all_links.extend(links)
        print(f"  Found {len(links)} links")

    # Remove duplicates
    all_links = list(set(all_links))
    print(f"\nTotal unique links: {len(all_links)}")

    # --- Step 2: Scrape each article ---
    print("\n" + "=" * 60)
    print("Scraping articles...")
    print("=" * 60)

    for i, url in enumerate(all_links, start=1):
        if url in seen_urls:
            continue
        seen_urls.add(url)

        print(f"\n{i}. scraping {url}")
        print("   requesting ...")

        article = scrape_article(url)

        if article:
            print("   parsing ...")
            all_articles.append(article)
            print(f"   saved in {OUTPUT_FILE} — total: {len(all_articles)}")
        else:
            print("   skipped (no content)")

        time.sleep(0.5)  # Reduced to 0.5s to collect faster

        if len(all_articles) >= 350:
            print("\nTarget reached: 350 articles!")
            break

    # --- Step 3: Save to CSV ---
    print("\n" + "=" * 60)
    print("Saving results...")

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(all_articles)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Saved {len(df)} articles to: {OUTPUT_FILE}")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()