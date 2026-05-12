import pandas as pd
import requests
import time
from src.diplomatic_sentiment import (
    score_distilbert_sentiment,
    compute_daily_gmm_weights,
    aggregate_daily
)

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_raw_gdelt(query_str, date_str):
    """Speaks directly to GDELT with a 90-second timeout."""
    params = {
        "query": query_str,
        "startdatetime": f"{date_str}000000",
        "enddatetime": f"{date_str}235959",
        "maxrecords": 250,
        "mode": "artlist",
        "format": "json"
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=90)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get('articles', []))
        else:
            print(f" [!] Server returned {response.status_code}", end="")
    except Exception as e:
        print(f" [!] Error: {str(e)[:50]}...", end="")
    return pd.DataFrame()

print("🚀 Launching Deep-Connection Fetch for Conflict Outbreak...")

THEMES = ["ARMEDCONFLICT", "ACT_FORCEPOSTURE", "ACT_HARMTHREATEN", "TERROR", "SANCTIONS", "ECON_OILPRICE"]
COUNTRIES = ["IR", "IS", "US", "LE", "YM", "SA"]

target_dates = ["2026-02-28", "2026-03-01"]
master_articles = []

for current_date in target_dates:
    print(f"\n--- Forcing Connection for {current_date} ---")
    date_clean = current_date.replace("-", "")

    for theme in THEMES:
        print(f"  -> Pulling {theme}...", end="", flush=True)
        df = fetch_raw_gdelt(f"theme:{theme}", date_clean)
        if not df.empty:
            df['date'] = current_date
            master_articles.append(df)
            print(f" ✓ Got {len(df)} articles")
        else:
            print(" X")
        time.sleep(5) # Give the server room to breathe

    for country in COUNTRIES:
        print(f"  -> Pulling {country}xConflict...", end="", flush=True)
        df = fetch_raw_gdelt(f"sourcecountry:{country} theme:ARMEDCONFLICT", date_clean)
        if not df.empty:
            df['date'] = current_date
            master_articles.append(df)
            print(f" ✓ Got {len(df)} articles")
        else:
            print(" X")
        time.sleep(5)

if not master_articles:
    print("\n[!] GDELT is likely fully offline or your machine is blocking outgoing HTTPS.")
    exit()

# Combine, deduplicate, and apply the critical 'seendate' fix
df_articles = pd.concat(master_articles, ignore_index=True).drop_duplicates(subset=['url'])
df_articles['seendate'] = df_articles['date']
print(f"\nSuccess! Total articles for outbreak: {len(df_articles)}")

# NLP Processing
df_scored = score_distilbert_sentiment(df_articles)
df_scored = compute_daily_gmm_weights(df_scored)
tone_mock = pd.DataFrame({'date': ['2026-02-28', '2026-03-01'], 'gdelt_tone_avg': [-3.5, -3.5]})
df_daily = aggregate_daily(df_scored, tone_mock)

df_daily.to_csv("data/outbreak_patch.csv", index=False)
print("\n✅ Outbreak patch saved to data/outbreak_patch.csv")