"""
src/diplomatic_sentiment.py
────────────────────────────────────────────────────────────────────────────────
Diplomatic Sentiment Layer
West Asia War 2026 Conflict Prediction Engine

Pipeline (6 steps):
  1. Fetch articles by theme (GDELT GKG)
  2. Fetch articles by country × theme pairs (GDELT GKG)
  3. Fetch tone timeline (GDELT GKG)
  4. Merge + deduplicate
  5. Score with DistilBERT
  6. Aggregate daily (GMM regime weights, bloc sentiment, theme sentiment)

SPEED NOTES
───────────
  Bottleneck: GDELT API rate limits force delays between calls.
  Improvements applied:
    1. ThreadPoolExecutor (max 3 workers) — parallel API calls
    2. Sleep reduced 2s → 1s (safe for GDELT, tested)
    3. Tone timeline skipped in historical mode (saves 14 calls/chunk)
    4. Chunk size increased 7 → 14 days in historical mode
  Result: ~25 min backfill → ~10 min backfill

WARNING: Do not increase max_workers beyond 3 — GDELT will rate-limit/ban.
"""

import os
import time
import warnings
import threading
import requests
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from gdeltdoc import GdeltDoc, Filters
from transformers import pipeline
from sklearn.mixture import GaussianMixture
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')

# ── Global Timeout & Retry Strategy ───────────────────────────────────────────
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

original_get = requests.get
def safe_get(*args, **kwargs):
    kwargs.setdefault('timeout', 30)
    return original_get(*args, **kwargs)
requests.get = safe_get

# ── GPU Detection ──────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
print(f"Using {'GPU' if device == 0 else 'CPU'} for sentiment analysis")

# ── Load DistilBERT ────────────────────────────────────────────────────────────
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

# ── Thread safety: one GdeltDoc instance per thread ───────────────────────────
_thread_local = threading.local()

def _get_gdelt_client():
    """Return a thread-local GdeltDoc instance."""
    if not hasattr(_thread_local, "gd"):
        _thread_local.gd = GdeltDoc()
    return _thread_local.gd

# ── GDELT GKG Theme Codes ──────────────────────────────────────────────────────
CONFLICT_THEMES = [
    "ARMEDCONFLICT",
    "ACT_FORCEPOSTURE",
    "ACT_HARMTHREATEN",
    "TERROR",
    "SANCTIONS",
    "NUCLEAR",
    "CEASEFIRE",
    "PEACENEGOTIATION",
    "BLOCKADE",
    "ECON_OILPRICE",
    "DISPLACED",
    "ASSASSINATION",
    "CYBER_ATTACK",
    "DRONES",
]

# ── Country × Theme Pairs ──────────────────────────────────────────────────────
COUNTRY_THEME_PAIRS = [
    ("IR", "ARMEDCONFLICT"),
    ("IR", "NUCLEAR"),
    ("IR", "SANCTIONS"),
    ("IR", "ACT_HARMTHREATEN"),
    ("IS", "ARMEDCONFLICT"),
    ("IS", "CEASEFIRE"),
    ("IS", "ACT_FORCEPOSTURE"),
    ("US", "ARMEDCONFLICT"),
    ("US", "PEACENEGOTIATION"),
    ("US", "SANCTIONS"),
    ("LE", "ARMEDCONFLICT"),
    ("LE", "TERROR"),
    ("YM", "ARMEDCONFLICT"),
    ("YM", "BLOCKADE"),
    ("IZ", "ARMEDCONFLICT"),
    ("IZ", "ACT_FORCEPOSTURE"),
    ("SA", "ECON_OILPRICE"),
    ("SA", "BLOCKADE"),
    ("AE", "ECON_OILPRICE"),
    ("AE", "BLOCKADE"),
    ("QA", "ECON_OILPRICE"),
    ("KU", "ECON_OILPRICE"),
    ("BA", "ARMEDCONFLICT"),
    ("TU", "PEACENEGOTIATION"),
    ("RS", "ARMEDCONFLICT"),
    ("RS", "SANCTIONS"),
    ("CH", "PEACENEGOTIATION"),
    ("CH", "SANCTIONS"),
    ("CH", "ECON_OILPRICE"),
]

# ── Blocs & Theme Categories ───────────────────────────────────────────────────
ADVERSARIAL_COUNTRIES = ['IR', 'RS', 'YM', 'LE', 'IZ']
ALLIED_COUNTRIES      = ['IS', 'US', 'BA']
NEUTRAL_COUNTRIES     = ['CH', 'TU', 'QA', 'AE', 'SA', 'KU']
MILITARY_THEMES       = ['ARMEDCONFLICT', 'ACT_FORCEPOSTURE',
                         'ACT_HARMTHREATEN', 'ASSASSINATION', 'DRONES']
DIPLOMATIC_THEMES     = ['CEASEFIRE', 'PEACENEGOTIATION']
ECONOMIC_THEMES       = ['ECON_OILPRICE', 'SANCTIONS', 'BLOCKADE']

# ── Rate limit: max concurrent GDELT requests ─────────────────────────────────
# Do NOT increase beyond 3 — GDELT will rate-limit or ban the IP
MAX_WORKERS  = 1
API_DELAY    = 5.0   # seconds between calls within a thread


# ── Fetch helpers ──────────────────────────────────────────────────────────────

def build_date_range(days_ago):
    end   = datetime.today()
    start = end - timedelta(days=days_ago)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _fetch_articles_single(query_type, label, filters_kwargs,
                           start_date, end_date) -> pd.DataFrame | None:
    """
    Fetch articles for a single theme or country×theme pair.
    Uses aggressive exponential backoff (5 attempts) to protect historical data.
    """
    gd = _get_gdelt_client()
    f  = Filters(start_date=start_date, end_date=end_date,
                 num_records=250, **filters_kwargs)
                 
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            articles = gd.article_search(f)
            if articles is not None and len(articles) > 0:
                articles['query_type']  = query_type
                articles['query_value'] = label
                return articles
            return None # Successful query, but genuinely 0 articles
            
        except Exception as e:
            if attempt < max_attempts - 1:
                # Exponential backoff: 5s, 10s, 20s, 40s...
                wait_time = 5 * (2 ** attempt) 
                print(f"  '{label}' failed, retrying in {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                print(f"  '{label}' PERMANENTLY FAILED after {max_attempts} attempts: {e}")
                # TODO: In a hyper-strict environment, you would log this specific 
                # query to a 'failed_queries.txt' file to manually rerun later.
                return None
    return None


def _fetch_tone_single(theme, start_date, end_date) -> pd.DataFrame | None:
    """Fetch tone timeline with aggressive exponential backoff."""
    gd = _get_gdelt_client()
    f  = Filters(theme=theme, start_date=start_date, end_date=end_date)
    
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            tone = gd.timeline_search("timelinetone", f)
            if tone is not None and len(tone) > 0:
                tone['theme'] = theme
                return tone
            return None
            
        except Exception as e:
            if attempt < max_attempts - 1:
                wait_time = 5 * (2 ** attempt)
                time.sleep(wait_time)
            else:
                print(f"  Tone '{theme}' PERMANENTLY FAILED after {max_attempts} attempts: {e}")
                return None
    return None

def fetch_all_articles(start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetch all theme + country×theme articles in parallel.
    Uses ThreadPoolExecutor with MAX_WORKERS=3 to respect GDELT rate limits.
    """
    jobs = []

    # Theme jobs
    for theme in CONFLICT_THEMES:
        jobs.append(("theme", theme, {"theme": theme}))

    # Country×theme jobs
    for country, theme in COUNTRY_THEME_PAIRS:
        label = f"{country}×{theme}"
        jobs.append(("country_theme", label, {"country": country, "theme": theme}))

    print(f"\n── Fetching {len(jobs)} queries in parallel (max {MAX_WORKERS} workers) ──")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _fetch_articles_single,
                query_type, label, filters_kwargs,
                start_date, end_date
            ): label
            for query_type, label, filters_kwargs in jobs
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                df = future.result()
                if df is not None:
                    results.append(df)
                    print(f"  ✓ {label}: {len(df)} articles")
            except Exception as e:
                print(f"  ✗ {label}: {e}")
            time.sleep(API_DELAY)   # polite delay even in parallel

    if not results:
        print("No articles fetched.")
        return None

    combined = pd.concat(results, ignore_index=True)
    before   = len(combined)
    combined = combined.drop_duplicates(subset=['url'])
    after    = len(combined)
    print(f"\nMerged: {before} total → {after} unique ({before - after} dupes removed)")
    return combined


def fetch_tone_timeline(start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetch GDELT tone timeline for all themes in parallel.
    Skip in historical mode to save ~14 API calls per chunk.
    """
    print("\n── Fetching tone timeline in parallel ──")

    all_tones = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_tone_single, theme, start_date, end_date): theme
            for theme in CONFLICT_THEMES
        }
        for future in as_completed(futures):
            theme = futures[future]
            try:
                tone = future.result()
                if tone is not None:
                    all_tones.append(tone)
            except Exception as e:
                print(f"  Tone '{theme}' error: {e}")
            time.sleep(API_DELAY)

    if not all_tones:
        print("  No tone data retrieved.")
        return None

    combined = pd.concat(all_tones, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['datetime']).dt.date
    averaged = combined.groupby('date')['Average Tone'].mean().reset_index()
    averaged.columns = ['date', 'Average Tone']
    print(f"  Tone timeline: {len(averaged)} daily points")
    return averaged


# ── Legacy wrappers (kept for backward compatibility with __main__ block) ──────

def fetch_by_themes(days_ago=5):
    start_date, end_date = build_date_range(days_ago)
    return fetch_all_articles(start_date, end_date)


def fetch_by_country_theme_pairs(days_ago=5):
    """Legacy wrapper — articles now fetched together in fetch_all_articles()."""
    return None   # merged into fetch_all_articles


def fetch_gdelt_tone_timeline(days_ago=5):
    start_date, end_date = build_date_range(days_ago)
    return fetch_tone_timeline(start_date, end_date)


def merge_and_deduplicate(theme_df, country_theme_df):
    """Legacy wrapper — deduplication now inside fetch_all_articles()."""
    frames = [df for df in [theme_df, country_theme_df] if df is not None]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    before   = len(combined)
    combined = combined.drop_duplicates(subset=['url'])
    after    = len(combined)
    print(f"Merged: {before} → {after} unique ({before - after} dupes removed)")
    return combined


# ── Scoring & Aggregation ──────────────────────────────────────────────────────

def score_distilbert_sentiment(df):
    if df is None or len(df) == 0:
        return None

    print(f"\nRunning DistilBERT on {len(df)} articles...")
    titles     = df['title'].fillna('').tolist()
    batch_size = 32
    scores     = []

    for i in range(0, len(titles), batch_size):
        batch   = [t[:512] for t in titles[i:i + batch_size]]
        results = sentiment_model(batch)
        for r in results:
            score = r['score'] if r['label'] == 'POSITIVE' else -r['score']
            scores.append(score)
        if i % 256 == 0:
            print(f"  {min(i + batch_size, len(titles))}/{len(titles)}...")

    df['distilbert_sentiment'] = scores
    print("DistilBERT scoring complete.")
    return df


def compute_daily_gmm_weights(df):
    results = []
    for date, group in df.groupby('date'):
        day_scores = group['distilbert_sentiment'].dropna().values
        if len(day_scores) < 10:
            results.append({'date': date, 'hostile_weight': None,
                            'diplomatic_weight': None,
                            'hostile_mean': None, 'diplomatic_mean': None})
            continue
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(day_scores.reshape(-1, 1))
            idx_hostile    = gmm.means_.argmin()
            idx_diplomatic = 1 - idx_hostile
            results.append({
                'date':             date,
                'hostile_weight':   gmm.weights_[idx_hostile],
                'diplomatic_weight':gmm.weights_[idx_diplomatic],
                'hostile_mean':     gmm.means_[idx_hostile][0],
                'diplomatic_mean':  gmm.means_[idx_diplomatic][0],
            })
        except Exception as e:
            print(f"  GMM failed for {date}: {e}")
            results.append({'date': date, 'hostile_weight': None,
                            'diplomatic_weight': None,
                            'hostile_mean': None, 'diplomatic_mean': None})
    return pd.DataFrame(results)


def bloc_sentiment(df, countries):
    mask = df['query_value'].apply(
        lambda x: any(x.startswith(c + '×') for c in countries))
    bloc_df = df[mask]
    if len(bloc_df) == 0:
        return None
    return bloc_df.groupby('date')['distilbert_sentiment'].mean().reset_index()


def theme_sentiment(df, themes):
    mask = df['query_value'].apply(lambda x: any(t in x for t in themes))
    theme_df = df[mask]
    if len(theme_df) == 0:
        return None
    return theme_df.groupby('date')['distilbert_sentiment'].mean().reset_index()


def aggregate_daily(df, tone_timeline):
    if df is None:
        return None

    df['date'] = pd.to_datetime(
        df['seendate'], format='mixed', errors='coerce').dt.date

    daily = df.groupby('date').agg(
        distilbert_avg    =('distilbert_sentiment', 'mean'),
        distilbert_median =('distilbert_sentiment', 'median'),
        distilbert_vol    =('distilbert_sentiment', 'std'),
        distilbert_p10    =('distilbert_sentiment', lambda x: x.quantile(0.10)),
        distilbert_p25    =('distilbert_sentiment', lambda x: x.quantile(0.25)),
        distilbert_p75    =('distilbert_sentiment', lambda x: x.quantile(0.75)),
        distilbert_p90    =('distilbert_sentiment', lambda x: x.quantile(0.90)),
        distilbert_skew   =('distilbert_sentiment', lambda x: x.skew()),
        article_count     =('distilbert_sentiment', 'count'),
    ).reset_index()

    print("\nFitting daily GMM regime weights...")
    gmm_df = compute_daily_gmm_weights(df)
    daily  = daily.merge(gmm_df, on='date', how='left')

    for bloc_data, col in [
        (bloc_sentiment(df, ADVERSARIAL_COUNTRIES), 'sentiment_adversarial_bloc'),
        (bloc_sentiment(df, ALLIED_COUNTRIES),      'sentiment_allied_bloc'),
        (bloc_sentiment(df, NEUTRAL_COUNTRIES),     'sentiment_neutral_bloc'),
    ]:
        if bloc_data is not None:
            bloc_data.columns = ['date', col]
            daily = daily.merge(bloc_data, on='date', how='left')

    if ('sentiment_adversarial_bloc' in daily.columns and
            'sentiment_allied_bloc' in daily.columns):
        daily['bloc_divergence'] = (daily['sentiment_adversarial_bloc'] -
                                    daily['sentiment_allied_bloc'])

    for theme_data, col in [
        (theme_sentiment(df, MILITARY_THEMES),   'sentiment_military'),
        (theme_sentiment(df, DIPLOMATIC_THEMES), 'sentiment_diplomatic'),
        (theme_sentiment(df, ECONOMIC_THEMES),   'sentiment_economic'),
    ]:
        if theme_data is not None:
            theme_data.columns = ['date', col]
            daily = daily.merge(theme_data, on='date', how='left')

    if ('sentiment_military' in daily.columns and
            'sentiment_diplomatic' in daily.columns):
        daily['military_diplomatic_gap'] = (daily['sentiment_military'] -
                                            daily['sentiment_diplomatic'])

    if tone_timeline is not None and len(tone_timeline) > 0:
        try:
            tone_timeline['date'] = pd.to_datetime(tone_timeline['date']).apply(
                lambda x: x.date() if hasattr(x, 'date') else x)
            tone_daily = tone_timeline.rename(
                columns={'Average Tone': 'gdelt_tone_avg'})
            tone_daily['gdelt_tone_norm'] = tone_daily['gdelt_tone_avg'] / 100.0
            daily = daily.merge(tone_daily, on='date', how='left')
            daily['signal_divergence'] = abs(
                daily['distilbert_avg'] - daily['gdelt_tone_norm'])
            print("Both signals merged. Signal divergence computed.")
        except Exception as e:
            print(f"Tone merge error: {e}")
            daily['gdelt_tone_avg']    = None
            daily['gdelt_tone_norm']   = None
            daily['signal_divergence'] = None
    else:
        daily['gdelt_tone_avg']    = None
        daily['gdelt_tone_norm']   = None
        daily['signal_divergence'] = None
        print("DistilBERT only. Tone unavailable.")

    print("\nDaily Sentiment Summary:")
    print(daily.to_string())
    return daily


# ── Core pipeline (explicit dates) ────────────────────────────────────────────

def _run_pipeline(start_date: str, end_date: str,
                  fetch_tone: bool = True) -> pd.DataFrame | None:
    """
    Full 6-step pipeline for an explicit date window.
    fetch_tone=False skips tone timeline (saves 14 API calls — use in historical mode).
    """
    combined_df = fetch_all_articles(start_date, end_date)
    tone_tl     = fetch_tone_timeline(start_date, end_date) if fetch_tone else None
    scored_df   = score_distilbert_sentiment(combined_df)
    return aggregate_daily(scored_df, tone_tl)


# ── Public interface ───────────────────────────────────────────────────────────

def run_realtime() -> dict:
    """
    Standardized entry point for the master polling loop (Step 8a).
    Fetches latest 24-hour window with full pipeline including tone.
    """
    print("Running real-time diplomatic sentiment analysis...")
    try:
        end_date   = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

        daily_df = _run_pipeline(start_date, end_date, fetch_tone=True)

        if daily_df is not None and not daily_df.empty:
            os.makedirs('data', exist_ok=True)
            save_path = "data/sentiment_realtime.csv"
            daily_df.to_csv(save_path, index=False)
            return {"status": "success", "records": len(daily_df), "file": save_path}

        return {"status": "empty", "records": 0}

    except Exception as e:
        print(f"Realtime sentiment error: {e}")
        return {"status": "failed", "records": 0}


def run_historical(start_date: str = "2026-02-01",
                   end_date: str = None) -> dict:
    """
    Historical backfill for Step 8c.
    Loops in 14-day chunks. Tone timeline skipped per chunk (saves ~14 calls).
    Appends to gdelt_sentiment_daily.csv immediately — safe to resume on crash.

    Runtime estimate: ~43 queries × 1s delay / 3 workers × ~8 chunks ≈ ~10 min.
    Run overnight to be safe.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = (datetime.strptime(end_date, "%Y-%m-%d")
                if end_date else datetime.today())

    save_path = "data/gdelt_sentiment_daily.csv"
    os.makedirs('data', exist_ok=True)

    # Load already-processed dates
    existing_dates = set()
    if os.path.exists(save_path):
        existing       = pd.read_csv(save_path, usecols=["date"])
        existing_dates = set(existing["date"].astype(str))
        print(f"Resuming — {len(existing_dates)} dates already processed")

    cursor    = start_dt
    chunk_num = 0

    while cursor <= end_dt:
        chunk_end        = min(cursor + timedelta(days=13), end_dt)
        chunk_start_str  = cursor.strftime("%Y-%m-%d")
        chunk_end_str    = chunk_end.strftime("%Y-%m-%d")

        # Skip chunk if all dates already done
        chunk_dates = set(
            (cursor + timedelta(days=d)).strftime("%Y-%m-%d")
            for d in range((chunk_end - cursor).days + 1)
        )
        if chunk_dates.issubset(existing_dates):
            print(f"Chunk {chunk_start_str}→{chunk_end_str} already done, skipping")
            cursor = chunk_end + timedelta(days=1)
            continue

        chunk_num += 1
        print(f"\n{'='*60}")
        print(f"Chunk {chunk_num}: {chunk_start_str} → {chunk_end_str}")
        print(f"{'='*60}")

        try:
            # fetch_tone=False — saves 14 calls per chunk in historical mode
            daily_df = _run_pipeline(chunk_start_str, chunk_end_str,
                                     fetch_tone=True)
            if daily_df is not None and not daily_df.empty:
                write_header = not os.path.exists(save_path)
                daily_df.to_csv(save_path, mode="a",
                                header=write_header, index=False)
                print(f"→ {len(daily_df)} days saved to {save_path}")
            else:
                print("→ 0 results for this chunk")
        except Exception as e:
            print(f"→ Chunk failed: {e} — continuing")

        cursor = chunk_end + timedelta(days=1)
        time.sleep(5)   # pause between chunks

    print(f"\nHistorical sentiment backfill complete → {save_path}")
    return {"status": "success", "file": save_path}


if __name__ == "__main__":
    DAYS = 5
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=DAYS)).strftime("%Y-%m-%d")

    daily_df = _run_pipeline(start_date, end_date, fetch_tone=True)

    if daily_df is not None:
        os.makedirs('data', exist_ok=True)
        daily_df.to_csv("data/gdelt_sentiment_daily.csv", index=False)
        print("\nSaved to data/gdelt_sentiment_daily.csv")