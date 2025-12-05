import requests
from bs4 import BeautifulSoup, Tag
import csv
from pathlib import Path
import time

def crawl_quotes(output_file="data/quotes_raw.csv"):
    # pastikan folder ada
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    base_url = "https://quotes.toscrape.com/page/{}/"
    all_data = []

    page = 1
    while True:
        url = base_url.format(page)
        print(f"[INFO] Crawling page {page} -> {url}")

        try:
            r = requests.get(url, timeout=10)
        except requests.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            break

        if r.status_code != 200:
            print(f"[DONE] Status code {r.status_code} — no more pages or blocked.")
            break

        # gunakan content untuk parse
        soup = BeautifulSoup(r.content, "html.parser")
        quotes = soup.find_all("div", class_="quote")

        if not quotes:
            print("[DONE] No data found on this page.")
            break

        for i, q in enumerate(quotes, start=1):
            # pastikan q adalah Tag yang punya method .find
            if not isinstance(q, Tag):
                print(f"[WARN] Skipping non-Tag element at page {page} index {i}")
                continue

            # ambil elemen dengan pengecekan None-safe
            text_tag = q.find("span", class_="text")
            author_tag = q.find("small", class_="author")
            tag_tags = q.find_all("a", class_="tag")

            text = text_tag.get_text(strip=True) if text_tag else ""
            author = author_tag.get_text(strip=True) if author_tag else ""
            tags = [t.get_text(strip=True) for t in tag_tags] if tag_tags else []

            # optional: skip jika text kosong
            if not text:
                print(f"[WARN] Empty text at page {page} index {i} — skipping")
                continue

            all_data.append([text, author, ", ".join(tags)])

        page += 1
        time.sleep(0.5)  # jeda kecil agar tidak membebani server

    # tulis CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "author", "tags"])
        writer.writerows(all_data)

    print(f"[DONE] {len(all_data)} quotes saved to {output_file}")


if __name__ == "__main__":
    crawl_quotes()
