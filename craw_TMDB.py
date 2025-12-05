import requests
import pandas as pd
import time

API_KEY = "20b3f92ba0d825a596bca80edf913009"
BASE_URL = "https://api.themoviedb.org/3"

MAX_DATA = 1000
PER_PAGE = 20
MAX_PAGE = MAX_DATA // PER_PAGE

def get_page(page):
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&page={page}"
    response = requests.get(url)
    return response.json()

def get_detail(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    return response.json()

film_data = []
page = 1

print(f"Mengambil maksimal {MAX_DATA} film dari TMDb\n")

while page <= MAX_PAGE:
    print(f"Mengambil halaman {page}...")
    data = get_page(page)

    if "results" not in data:
        break

    for m in data["results"]:
        detail = get_detail(m["id"])

        film_data.append({
            "id": m["id"],
            "judul_asli": detail.get("original_title", ""),
            "judul_display": detail.get("title", ""),
            "sinopsis_asli": detail.get("overview", ""),
            "genre": ", ".join([g["name"] for g in detail.get("genres", [])]),
            "tanggal_rilis": detail.get("release_date", ""),
            "rating": detail.get("vote_average", 0),
            "popularitas": detail.get("popularity", 0),
            "bahasa_asli": detail.get("original_language", "")
        })

        if len(film_data) >= MAX_DATA:
            break

    if len(film_data) >= MAX_DATA:
        break

    page += 1
    time.sleep(0.25)

df = pd.DataFrame(film_data)
df.to_csv("tmdb_1000_film.csv", index=False, encoding="utf-8")

print("\n SELESAI! File disimpan sebagai: tmdb_1000_film.csv")
print(f"Total data diambil: {len(df)}")
