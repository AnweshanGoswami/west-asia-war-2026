import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from transformers import pipeline
from sklearn.mixture import GaussianMixture
import torch
import time
import warnings
import requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
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
    kwargs.setdefault('timeout', 30) # Only apply if not already specified!
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

# ── GDELT GKG Theme Codes (verified from GDELT master theme list) ──────────────
CONFLICT_THEMES = [
    "ARMEDCONFLICT",      # Any armed conflict coverage
    "ACT_FORCEPOSTURE",   # Military force positioning and posture
    "ACT_HARMTHREATEN",   # Threats of violence or attack
    "TERROR",             # Terrorism and militant group activity
    "SANCTIONS",          # Economic sanctions
    "NUCLEAR",            # Nuclear-related coverage
    "CEASEFIRE",          # Ceasefire coverage
    "PEACENEGOTIATION",   # Diplomatic talks and negotiations
    "BLOCKADE",           # Maritime blockade / Strait of Hormuz
    "ECON_OILPRICE",      # Oil price disruption
    "DISPLACED",          # Refugee and displacement crisis
    "ASSASSINATION",      # Leadership elimination events
    "CYBER_ATTACK",       # Cyber warfare dimension
    "DRONES",             # Drone warfare coverage
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

# ── Source Blocs ───────────────────────────────────────────────────────────────
ADVERSARIAL_COUNTRIES = ['IR', 'RS', 'YM', 'LE', 'IZ']
ALLIED_COUNTRIES      = ['IS', 'US', 'BA']
NEUTRAL_COUNTRIES     = ['CH', 'TU', 'QA', 'AE', 'SA', 'KU']

# ── Theme Categories ───────────────────────────────────────────────────────────
MILITARY_THEMES   = ['ARMEDCONFLICT', 'ACT_FORCEPOSTURE',
                     'ACT_HARMTHREATEN', 'ASSASSINATION', 'DRONES']
DIPLOMATIC_THEMES = ['CEASEFIRE', 'PEACENEGOTIATION']
ECONOMIC_THEMES   = ['ECON_OILPRICE', 'SANCTIONS', 'BLOCKADE']


def build_date_range(days_ago):
    end = datetime.today()
    start = end - timedelta(days=days_ago)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def safe_article_search(gd, f, label):
    try:
        articles = gd.article_search(f)
        return articles
    except Exception as e:
        print(f"  '{label}' failed, retrying in 5s... ({e})")
        time.sleep(5)
        try:
            articles = gd.article_search(f)
            print(f"  '{label}' retry succeeded.")
            return articles
        except Exception as e2:
            print(f"  '{label}' permanently failed: {e2}")
            return None


def safe_timeline_search(gd, f, label):
    try:
        tone = gd.timeline_search("timelinetone", f)
        return tone
    except Exception as e:
        print(f"  '{label}' tone failed, retrying in 5s... ({e})")
        time.sleep(5)
        try:
            tone = gd.timeline_search("timelinetone", f)
            print(f"  '{label}' tone retry succeeded.")
            return tone
        except Exception as e2:
            print(f"  '{label}' tone permanently failed: {e2}")
            return None


def fetch_by_themes(days_ago=5):
    gd = GdeltDoc()
    start_date, end_date = build_date_range(days_ago)
    all_articles = []

    print("\n── Fetching by Theme Codes (Global) ──")
    for theme in CONFLICT_THEMES:
        f = Filters(
            theme=theme,
            start_date=start_date,
            end_date=end_date,
            num_records=250
        )
        articles = safe_article_search(gd, f, theme)
        if articles is not None and len(articles) > 0:
            articles['query_type'] = 'theme'
            articles['query_value'] = theme
            all_articles.append(articles)
            print(f"  Theme '{theme}': {len(articles)} articles")
        time.sleep(2)

    return pd.concat(all_articles, ignore_index=True) if all_articles else None


def fetch_by_country_theme_pairs(days_ago=5):
    gd = GdeltDoc()
    start_date, end_date = build_date_range(days_ago)
    all_articles = []

    print("\n── Fetching by Country × Theme Pairs ──")
    for country, theme in COUNTRY_THEME_PAIRS:
        label = f"{country}×{theme}"
        f = Filters(
            country=country,
            theme=theme,
            start_date=start_date,
            end_date=end_date,
            num_records=250
        )
        articles = safe_article_search(gd, f, label)
        if articles is not None and len(articles) > 0:
            articles['query_type'] = 'country_theme'
            articles['query_value'] = label
            all_articles.append(articles)
            print(f"  {label}: {len(articles)} articles")
        time.sleep(2)

    return pd.concat(all_articles, ignore_index=True) if all_articles else None


def fetch_gdelt_tone_timeline(days_ago=5):
    gd = GdeltDoc()
    start_date, end_date = build_date_range(days_ago)
    all_tones = []

    print("\n── Fetching GDELT Tone Timeline (per theme) ──")
    for theme in CONFLICT_THEMES:
        f = Filters(
            theme=theme,
            start_date=start_date,
            end_date=end_date
        )
        tone = safe_timeline_search(gd, f, theme)
        if tone is not None and len(tone) > 0:
            tone['theme'] = theme
            all_tones.append(tone)
            print(f"  Theme '{theme}': {len(tone)} tone data points")
        time.sleep(2)

    if not all_tones:
        print("  No tone timeline data retrieved.")
        return None

    combined = pd.concat(all_tones, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['datetime']).dt.date
    averaged = combined.groupby('date')['Average Tone'].mean().reset_index()
    averaged.columns = ['date', 'Average Tone']

    print(f"\n  Combined tone timeline: {len(averaged)} daily data points "
          f"across {len(all_tones)} themes")
    return averaged


def merge_and_deduplicate(theme_df, country_theme_df):
    frames = [df for df in [theme_df, country_theme_df] if df is not None]
    if not frames:
        print("No articles fetched from any source.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=['url'])
    after = len(combined)
    print(f"\nMerged: {before} total → {after} unique articles "
          f"({before - after} duplicates removed)")
    return combined


def score_distilbert_sentiment(df):
    if df is None or len(df) == 0:
        return None

    print(f"\nRunning DistilBERT sentiment on {len(df)} articles...")
    titles = df['title'].fillna('').tolist()
    batch_size = 32
    scores = []

    for i in range(0, len(titles), batch_size):
        batch = [t[:512] for t in titles[i:i + batch_size]]
        results = sentiment_model(batch)
        for r in results:
            score = r['score'] if r['label'] == 'POSITIVE' else -r['score']
            scores.append(score)

        if i % 256 == 0:
            print(f"  Processed {min(i + batch_size, len(titles))}"
                  f"/{len(titles)} articles...")

    df['distilbert_sentiment'] = scores
    print("DistilBERT scoring complete.")
    return df


def compute_daily_gmm_weights(df):
    results = []

    for date, group in df.groupby('date'):
        day_scores = group['distilbert_sentiment'].dropna().values

        if len(day_scores) < 10:
            results.append({
                'date':             date,
                'hostile_weight':   None,
                'diplomatic_weight':None,
                'hostile_mean':     None,
                'diplomatic_mean':  None
            })
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
                'diplomatic_mean':  gmm.means_[idx_diplomatic][0]
            })
        except Exception as e:
            print(f"  GMM failed for {date}: {e}")
            results.append({
                'date':             date,
                'hostile_weight':   None,
                'diplomatic_weight':None,
                'hostile_mean':     None,
                'diplomatic_mean':  None
            })

    return pd.DataFrame(results)


def bloc_sentiment(df, countries):
    mask = df['query_value'].apply(
        lambda x: any(x.startswith(c + '×') for c in countries)
    )
    bloc_df = df[mask]
    if len(bloc_df) == 0:
        return None
    return bloc_df.groupby('date')['distilbert_sentiment'].mean().reset_index()


def theme_sentiment(df, themes):
    mask = df['query_value'].apply(
        lambda x: any(t in x for t in themes)
    )
    theme_df = df[mask]
    if len(theme_df) == 0:
        return None
    return theme_df.groupby('date')['distilbert_sentiment'].mean().reset_index()


def aggregate_daily(df, tone_timeline):
    if df is None:
        return None

    df['date'] = pd.to_datetime(
        df['seendate'], format='mixed', errors='coerce'
    ).dt.date

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
    print("GMM weights computed.")

    adv  = bloc_sentiment(df, ADVERSARIAL_COUNTRIES)
    ally = bloc_sentiment(df, ALLIED_COUNTRIES)
    neut = bloc_sentiment(df, NEUTRAL_COUNTRIES)

    for bloc_df, col_name in [
        (adv,  'sentiment_adversarial_bloc'),
        (ally, 'sentiment_allied_bloc'),
        (neut, 'sentiment_neutral_bloc'),
    ]:
        if bloc_df is not None:
            bloc_df.columns = ['date', col_name]
            daily = daily.merge(bloc_df, on='date', how='left')

    if ('sentiment_adversarial_bloc' in daily.columns and
            'sentiment_allied_bloc' in daily.columns):
        daily['bloc_divergence'] = (
            daily['sentiment_adversarial_bloc'] -
            daily['sentiment_allied_bloc']
        )

    mil  = theme_sentiment(df, MILITARY_THEMES)
    dip  = theme_sentiment(df, DIPLOMATIC_THEMES)
    econ = theme_sentiment(df, ECONOMIC_THEMES)

    for theme_df, col_name in [
        (mil,  'sentiment_military'),
        (dip,  'sentiment_diplomatic'),
        (econ, 'sentiment_economic'),
    ]:
        if theme_df is not None:
            theme_df.columns = ['date', col_name]
            daily = daily.merge(theme_df, on='date', how='left')

    if ('sentiment_military' in daily.columns and
            'sentiment_diplomatic' in daily.columns):
        daily['military_diplomatic_gap'] = (
            daily['sentiment_military'] -
            daily['sentiment_diplomatic']
        )

    if tone_timeline is not None and len(tone_timeline) > 0:
        try:
            tone_timeline['date'] = pd.to_datetime(
                tone_timeline['date']
            ).apply(lambda x: x.date() if hasattr(x, 'date') else x)

            tone_daily = tone_timeline.rename(
                columns={'Average Tone': 'gdelt_tone_avg'}
            )
            tone_daily['gdelt_tone_norm'] = (
                tone_daily['gdelt_tone_avg'] / 100.0
            )

            daily = daily.merge(tone_daily, on='date', how='left')
            daily['signal_divergence'] = abs(
                daily['distilbert_avg'] - daily['gdelt_tone_norm']
            )
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
        print("Only DistilBERT signal available. "
              "Epistemic uncertainty elevated — GDELT tone unavailable.")

    print("\nDaily Sentiment Summary:")
    print(daily.to_string())
    return daily


def run_realtime():
    """
    Standardized entry point for the master polling loop.
    Runs the 6-step sentiment pipeline for the latest 24 hours.
    """
    print("Running real-time diplomatic sentiment analysis...")
    try:
        # Polling window: 1 day to ensure it runs quickly but catches new articles
        DAYS = 1 
        
        # 1. Fetch articles
        theme_df         = fetch_by_themes(days_ago=DAYS)
        country_theme_df = fetch_by_country_theme_pairs(days_ago=DAYS)

        # 2. Fetch Tone timeline
        tone_tl = fetch_gdelt_tone_timeline(days_ago=DAYS)

        # 3. Merge
        combined_df = merge_and_deduplicate(theme_df, country_theme_df)

        # 4. Score DistilBERT
        scored_df = score_distilbert_sentiment(combined_df)

        # 5. Aggregate Daily stats (GMM, blocs, themes)
        daily_df = aggregate_daily(scored_df, tone_tl)

        # 6. Save receipt
        if daily_df is not None and not daily_df.empty:
            save_path = "data/sentiment_realtime.csv"
            daily_df.to_csv(save_path, index=False)
            return {"status": "success", "records": len(daily_df), "file": save_path}
        
        return {"status": "empty", "records": 0}
        
    except Exception as e:
        print(f"Error during sentiment realtime poll: {e}")
        return {"status": "failed", "records": 0}


if __name__ == "__main__":
    # Standard 5-day historical run for manual script execution
    DAYS = 5

    theme_df         = fetch_by_themes(days_ago=DAYS)
    country_theme_df = fetch_by_country_theme_pairs(days_ago=DAYS)
    tone_tl          = fetch_gdelt_tone_timeline(days_ago=DAYS)
    
    combined_df = merge_and_deduplicate(theme_df, country_theme_df)
    scored_df   = score_distilbert_sentiment(combined_df)
    daily_df    = aggregate_daily(scored_df, tone_tl)

    if scored_df is not None:
        scored_df.to_csv("data/gdelt_raw.csv", index=False)
        print("\nRaw articles saved to data/gdelt_raw.csv")

    if daily_df is not None:
        daily_df.to_csv("data/gdelt_sentiment_daily.csv", index=False)
        print("Daily sentiment saved to data/gdelt_sentiment_daily.csv")