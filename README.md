# 🎬 IMDbMoviesAnalysis

Scrape movie data from **IMDb**, store it in **MongoDB**, analyze it with **pandas/scikit‑learn**, and present interactive visualizations in a **Streamlit** web app. Scraping combines **Selenium WebDriver** (for dynamic pages) and **IMDbPY** (structured data API). Data can also be exported to **JSON** and loaded back into the app.

---

## ✨ Key Features

- **Data acquisition**
  - Selenium-based scraping for dynamic content (ratings/votes/runtime/people lists).
  - **IMDbPY** for stable, structured lookups (titles, IDs, full credits, genres, etc.).
- **Cleaning & preprocessing**
  - Numeric imputation, list parsing (genres, cast, languages…), runtime normalization, decade bucketing, age certification unification.
- **Exploratory analysis & visuals (Streamlit)**
  - Overview metrics, distributions, treemaps, trends over time, top actors/directors, collaboration network graph, word cloud.
  - ML demo: RandomForest classifier to predict **age_group** from plot TF‑IDF + genres.
- **Storage & export**
  - Save filtered dataframe to **MongoDB** collections and export back to **JSON** from within the UI (dedicated tab). fileciteturn0file0
- **Simple, responsive UI**
  - Sidebar filters (year/rating), tabbed analytics, and downloadable cleaned CSV from the app. fileciteturn0file0

---

## 🧱 Tech Stack

- **Python 3.10+**
- **Selenium** (with Chrome/Edge/Firefox WebDriver)
- **IMDbPY**
- **pandas, numpy, scikit‑learn, matplotlib, seaborn, wordcloud, squarify, networkx**
- **MongoDB** + **PyMongo**
- **Streamlit** web app (runs locally or on a server) fileciteturn0file0

---

## 📁 Project Structure (suggested)

```
IMDbMoviesAnalysis/
├─ scraping/
│  ├─ selenium_scraper.py
│  ├─ imdbpy_harvester.py
│  └─ utils.py
├─ data/
│  ├─ raw/                 # raw dumps, HTML snapshots (optional)
│  └─ processed/           # cleaned CSV / JSON
├─ app/
│  └─ Streamlit_Application.py   # Streamlit dashboard
├─ notebooks/              # optional EDA
├─ requirements.txt
└─ README.md

## ⚙️ Setup

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
Either:
```bash
pip install -r requirements.txt
```
or install key libs directly:
```bash
pip install streamlit selenium imdbpy pandas numpy scikit-learn matplotlib seaborn \
            wordcloud squarify networkx pymongo python-dotenv
```

### 3) WebDriver for Selenium
Install a driver matching your browser version:
- **Chrome**: chromedriver (or use `webdriver-manager`)
- **Edge**: msedgedriver
- **Firefox**: geckodriver

Quick option:
```bash
pip install webdriver-manager
```
and use it in your Selenium script to auto-download a compatible driver.

### 4) MongoDB
- Install and run **MongoDB** locally or use a hosted cluster.
- Example connection string: `mongodb://localhost:27017/`

Create a `.env` (optional):
```
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=MoviesDB
MONGODB_COLLECTION=movies
```

---

## 🕸️ Scraping

### Selenium (dynamic pages)
```python
# skeleton (scraping/selenium_scraper.py)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time, json

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
try:
    driver.get("https://www.imdb.com/chart/top/")
    time.sleep(2)
    items = []
    rows = driver.find_elements(By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")
    for r in rows:
        title = r.find_element(By.CSS_SELECTOR, "h3").text
        year = r.text.split("(")[-1].split(")")[0]
        link = r.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
        items.append({"title": title, "year": year, "url": link})
finally:
    driver.quit()

with open("data/raw/imdb_seed.json", "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
```

### IMDbPY (structured details)
```python
# skeleton (scraping/imdbpy_harvester.py)
from imdb import IMDb
import json

ia = IMDb()
enriched = []
for seed in json.load(open("data/raw/imdb_seed.json", encoding="utf-8")):
    results = ia.search_movie(seed["title"])
    if results:
        m = ia.get_movie(results[0].movieID)
        enriched.append({
            "imdb_id": m.movieID,
            "title": m.get("title"),
            "year": m.get("year"),
            "rating": m.get("rating"),
            "votes": m.get("votes"),
            "runtime": (m.get("runtimes") or [None])[0],
            "genres": m.get("genres") or [],
            "directors": [p["name"] for p in (m.get("directors") or [])],
            "cast": [p["name"] for p in (m.get("cast") or [])[:10]],
            "plot": (m.get("plot") or [""])[0],
            "countries": m.get("countries") or [],
            "languages": m.get("languages") or [],
            "age_group": m.get("certificates") or []
        })
json.dump(enriched, open("data/processed/imdb_movies.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
```

> Merge/join Selenium seeds with IMDbPY enrichment as needed, then convert to CSV for analysis.

---

## 🧽 Cleaning & Schema (typical)

Expected columns used by the app:
```
title, year, rating, votes, runtime, genres, directors, cast,
plot, countries, languages, age_group
```
During load, the app performs:
- Numeric imputation for `rating`, `votes`, `runtime`
- String/list parsing for multi-value fields
- Runtime normalization and `runtime_minutes`
- **decade** derivation, **age_group** standardization, and UI-ready formats for votes/runtime. fileciteturn0file0

Save your final table as CSV at `data/processed/imdb_movies.csv` (or upload it via the app).

---

## ▶️ Run the Streamlit App

From the project root (where the app file lives):
```bash
streamlit run Streamlit_AppLication.py
```
Then:
1. Upload your CSV in the UI.
2. Adjust sidebar filters (year/rating).
3. Explore tabs (overview, people, geography/genre, trends, ML, **MongoDB & JSON**). fileciteturn0file0
4. **Save to MongoDB** or **Export JSON** directly from the app. fileciteturn0file0

---

## 💾 MongoDB & JSON (in-app)

- Enter **MongoDB URI**, **Database**, **Collection**.
- Click **Save to MongoDB** to upsert the filtered dataset.
- Click **Export to JSON** to dump collection to a downloadable JSON file.
- View collection stats and sample documents from the same tab. fileciteturn0file0

---

## 🧪 ML Demo

- TF‑IDF on `plot` + multi‑label binarized `genres` → feature matrix.
- RandomForestClassifier → predicts `age_group`; classification report & feature importances shown in app. fileciteturn0file0

---

## 🧰 Troubleshooting

- **Selenium driver** mismatch → use `webdriver-manager` or install matching driver.
- **IMDbPY throttling** → add sleeps/backoff; cache requests.
- **Missing fields** → keep parsers tolerant (e.g., empty genres/certificates).
- **MongoDB auth/connection** → verify URI, firewall, and database user roles.
- **Streamlit large CSVs** → pre‑filter rows/columns to speed up load.

---

## 📦 Minimal `requirements.txt`

```
streamlit
selenium
webdriver-manager
imdbpy
pandas
numpy
scikit-learn
matplotlib
seaborn
wordcloud
squarify
networkx
pymongo
python-dotenv
```

---

## 📝 License (MIT)

---

## 🙌 Acknowledgements

- IMDb content and trademarks belong to IMDb.
- Thanks to the maintainers of Selenium and IMDbPY.
