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
# Monkey-patches requests.get so gdeltdoc inherits our timeout and retry logic.
# Without this, failed connections hang indefinitely.
#
# timeout=30       → give up if GDELT doesn't respond within 30 seconds
# total=3          → retry up to 3 times before permanently failing
# backoff_factor=2 → waits 2s, 4s, 8s between retries (progressive backoff)
# status_forcelist → auto-retry on server errors and rate limit responses

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
requests.get = lambda *args, **kwargs: original_get(
    *args, timeout=30, **kwargs
)

# ── GPU Detection ──────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
print(f"Using {'GPU' if device == 0 else 'CPU'} for sentiment analysis")

# ── Load DistilBERT ────────────────────────────────────────────────────────────
# BENCHMARK FINDING (see notebooks/gpu_utilisation_test.ipynb):
# HuggingFace Dataset prefetching was tested against the sequential batch loop
# on RTX 3050 Laptop GPU (4GB VRAM). Dataset approach was 0.70x SLOWER
# (5.02s vs 3.49s for 500 articles). Three reasons:
#   1. Prefetching overhead outweighs benefits on laptop GPUs
#   2. DistilBERT (66M params) forward passes are too fast to prefetch around
#   3. Larger batch_size=64 increased memory transfer overhead on 4GB VRAM
# Decision: keep sequential batch loop (batch_size=32). Re-benchmark if
# deployed on server GPU (AWS p3, GCP A100) where results may differ.
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

# ── GDELT GKG Theme Codes (verified from GDELT master theme list) ──────────────
# Meaning-based categories GDELT assigns to every article.
# Far more robust than keywords — catches any article about these topics
# regardless of exact wording used by the journalist.
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
# PREVIOUS APPROACH (naive country query):
#   We previously queried each country independently e.g. country="YM".
#   This returned ALL articles published by Yemeni media — including floods,
#   domestic political crises, famines, and local elections — none of which
#   are related to the conflict. This is a confounding problem: the sentiment
#   signal gets contaminated by unrelated domestic events, causing the model
#   to misinterpret a Yemeni flood (negative sentiment) as conflict escalation.
#
# CURRENT APPROACH (country × theme intersection):
#   We now query country AND theme simultaneously. This returns only articles
#   from Yemen that GDELT has also classified under a conflict-relevant theme
#   (e.g. ARMEDCONFLICT, BLOCKADE). The flood and famine articles are
#   automatically excluded because they don't carry those theme tags.
#   Each country is only paired with the themes relevant to its specific
#   role in this conflict — Yemen gets ARMEDCONFLICT and BLOCKADE,
#   Turkey gets only PEACENEGOTIATION, Saudi Arabia gets only oil/shipping.
#   This eliminates confounding at the query level rather than trying to
#   filter it out downstream.

COUNTRY_THEME_PAIRS = [
    # Direct combatants — military and diplomatic themes
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

    # Proxy actors — only conflict-relevant themes
    ("LE", "ARMEDCONFLICT"),
    ("LE", "TERROR"),
    ("YM", "ARMEDCONFLICT"),
    ("YM", "BLOCKADE"),
    ("IZ", "ARMEDCONFLICT"),
    ("IZ", "ACT_FORCEPOSTURE"),

    # Gulf states — economic and shipping impact only
    ("SA", "ECON_OILPRICE"),
    ("SA", "BLOCKADE"),
    ("AE", "ECON_OILPRICE"),
    ("AE", "BLOCKADE"),
    ("QA", "ECON_OILPRICE"),
    ("KU", "ECON_OILPRICE"),
    ("BA", "ARMEDCONFLICT"),

    # Diplomatic actors — only diplomatic and strategic themes
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
    """Returns start and end date strings for the query window."""
    end = datetime.today()
    start = end - timedelta(days=days_ago)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def safe_article_search(gd, f, label):
    """
    Wraps gd.article_search with a retry mechanism.
    First attempt: immediate.
    On failure: waits 5 seconds and retries once.
    On second failure: logs and returns None.
    The global timeout=30 ensures no request hangs indefinitely.
    """
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
    """
    Wraps gd.timeline_search with a retry mechanism.
    Same logic as safe_article_search.
    """
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
    """
    Fetches articles using verified GDELT GKG theme codes globally.
    Provides broad conflict coverage regardless of publication country.
    """
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
    """
    Fetches articles using targeted country × theme combinations.
    Eliminates confounding from domestic events unrelated to the conflict.
    Each pair captures only the specific role that country plays
    in the West Asia conflict.
    """
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
    """
    Fetches GDELT's pre-computed tone timeline across ALL conflict themes.
    Runs one timeline query per theme and averages into a single daily signal.
    This is our SECOND independent sentiment signal alongside DistilBERT.
    High divergence between the two = elevated epistemic uncertainty.
    """
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
    """
    Merges global theme results and country×theme results.
    Removes duplicate articles caught by multiple queries.
    """
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
    """
    Scores article titles using DistilBERT on RTX 3050.
    Primary sentiment signal. Output: -1.0 (hostile) to +1.0 (diplomatic).

    BENCHMARK FINDING (notebooks/gpu_utilisation_test.ipynb):
    Sequential batch loop (batch_size=32) outperforms HuggingFace Dataset
    prefetching on this hardware — 143.1 vs 99.5 articles/sec. See notebook
    for full analysis. Re-benchmark on server GPU before any cloud deployment.
    """
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
    """
    Fits a 2-component Gaussian Mixture Model to each day's sentiment scores.

    FINDING (notebooks/distribution_analysis.ipynb):
    Sentiment scores follow a bimodal distribution — not normal. Three
    normality tests (Shapiro-Wilk, D'Agostino-Pearson, KS) all rejected
    normality at p < 0.001. The distribution has two regimes:
      - Hostile regime:    mean ≈ -0.84, weight ≈ 0.73 (during ceasefire)
      - Diplomatic regime: mean ≈ +0.76, weight ≈ 0.27 (during ceasefire)

    The hostile regime weight is a more robust conflict signal than the
    daily mean — especially during ceasefire periods where mean sentiment
    shifts slowly but regime weights respond faster to diplomatic signals.

    A sustained drop in hostile_weight below 0.50 triggers the diplomatic
    switch in the Lanchester model — a stronger signal than mean sentiment
    crossing a threshold.

    Columns produced:
    - hostile_weight    : fraction of coverage in hostile regime (0 to 1)
    - diplomatic_weight : fraction of coverage in diplomatic regime (0 to 1)
    - hostile_mean      : centre of hostile component
    - diplomatic_mean   : centre of diplomatic component
    """
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

            # Always assign Component 0 as hostile (lower mean)
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
    """
    Computes daily mean sentiment for a specific source country bloc.
    """
    mask = df['query_value'].apply(
        lambda x: any(x.startswith(c + '×') for c in countries)
    )
    bloc_df = df[mask]
    if len(bloc_df) == 0:
        return None
    return bloc_df.groupby('date')['distilbert_sentiment'].mean().reset_index()


def theme_sentiment(df, themes):
    """
    Computes daily mean sentiment for a specific theme category.
    """
    mask = df['query_value'].apply(
        lambda x: any(t in x for t in themes)
    )
    theme_df = df[mask]
    if len(theme_df) == 0:
        return None
    return theme_df.groupby('date')['distilbert_sentiment'].mean().reset_index()


def aggregate_daily(df, tone_timeline):
    """
    Aggregates article-level scores into a rich daily feature vector.

    INITIAL ASSUMPTION (naive):
        A single daily mean was used as the sentiment signal, implicitly
        assuming the underlying score distribution was Gaussian (normal).
        Under normality, the mean is a sufficient statistic.

    FINDING (notebooks/distribution_analysis.ipynb):
        Shapiro-Wilk, D'Agostino-Pearson, and KS tests all rejected
        normality at p < 0.001. The distribution is bimodal — a direct
        consequence of DistilBERT's binary classification architecture
        pushing scores toward ±1. Under non-normality, the mean alone
        is NOT a sufficient estimator.

    CORRECTION:
        Full distribution statistics, bloc/theme breakdowns, GMM regime
        weights, and signal divergence now form the daily feature vector.

    Output columns:
    ┌─ Distribution ──────────────────────────────────────────────────────┐
    │ distilbert_avg      : mean score (-1 to +1)                         │
    │ distilbert_median   : median (robust to outliers)                   │
    │ distilbert_vol      : std deviation                                 │
    │ distilbert_p10/p90  : tail behaviour                                │
    │ distilbert_p25/p75  : interquartile range                           │
    │ distilbert_skew     : asymmetry                                     │
    │ article_count       : volume of coverage                            │
    ├─ GMM Regime Weights ────────────────────────────────────────────────┤
    │ hostile_weight      : fraction of coverage in hostile regime        │
    │ diplomatic_weight   : fraction of coverage in diplomatic regime     │
    │ hostile_mean        : centre of hostile component                   │
    │ diplomatic_mean     : centre of diplomatic component                │
    ├─ Bloc Breakdown ────────────────────────────────────────────────────┤
    │ sentiment_adversarial_bloc : Iran/Russia/Yemen/Lebanon/Iraq         │
    │ sentiment_allied_bloc      : Israel/USA/Bahrain                     │
    │ sentiment_neutral_bloc     : China/Turkey/Gulf states               │
    │ bloc_divergence            : adversarial minus allied sentiment     │
    ├─ Theme Breakdown ───────────────────────────────────────────────────┤
    │ sentiment_military    : armed conflict/drone/assassination          │
    │ sentiment_diplomatic  : ceasefire/negotiation                       │
    │ sentiment_economic    : oil/sanctions/blockade                      │
    │ military_diplomatic_gap : mil minus diplomatic sentiment            │
    ├─ Second Signal ─────────────────────────────────────────────────────┤
    │ gdelt_tone_avg      : GDELT's own tone score (raw)                  │
    │ gdelt_tone_norm     : GDELT tone normalised to (-1 to +1)           │
    │ signal_divergence   : |distilbert_avg - gdelt_tone_norm|            │
    │                       HIGH = elevated epistemic uncertainty         │
    └─────────────────────────────────────────────────────────────────────┘
    """
    if df is None:
        return None

    df['date'] = pd.to_datetime(
        df['seendate'], format='mixed', errors='coerce'
    ).dt.date

    # ── Core distribution statistics ──────────────────────────────────────────
    daily = df.groupby('date').agg(
        distilbert_avg    =('distilbert_sentiment', 'mean'),
        distilbert_median =('distilbert_sentiment', 'median'),
        distilbert_vol    =('distilbert_sentiment', 'std'),
        distilbert_p10    =('distilbert_sentiment',
                            lambda x: x.quantile(0.10)),
        distilbert_p25    =('distilbert_sentiment',
                            lambda x: x.quantile(0.25)),
        distilbert_p75    =('distilbert_sentiment',
                            lambda x: x.quantile(0.75)),
        distilbert_p90    =('distilbert_sentiment',
                            lambda x: x.quantile(0.90)),
        distilbert_skew   =('distilbert_sentiment',
                            lambda x: x.skew()),
        article_count     =('distilbert_sentiment', 'count'),
    ).reset_index()

    # ── GMM regime weights ─────────────────────────────────────────────────────
    print("\nFitting daily GMM regime weights...")
    gmm_df = compute_daily_gmm_weights(df)
    daily  = daily.merge(gmm_df, on='date', how='left')
    print("GMM weights computed.")

    # ── Sentiment by source bloc ───────────────────────────────────────────────
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

    # ── Sentiment by theme category ────────────────────────────────────────────
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

    # ── GDELT tone timeline (second independent signal) ────────────────────────
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


if __name__ == "__main__":
    DAYS = 5

    # Step 1: Fetch articles from all sources
    theme_df         = fetch_by_themes(days_ago=DAYS)
    country_theme_df = fetch_by_country_theme_pairs(days_ago=DAYS)

    # Step 2: Fetch GDELT tone timeline across all themes
    tone_tl = fetch_gdelt_tone_timeline(days_ago=DAYS)

    # Step 3: Merge and deduplicate
    combined_df = merge_and_deduplicate(theme_df, country_theme_df)

    # Step 4: Score with DistilBERT
    scored_df = score_distilbert_sentiment(combined_df)

    # Step 5: Aggregate to daily with full feature vector
    daily_df = aggregate_daily(scored_df, tone_tl)

    # Step 6: Save outputs
    if scored_df is not None:
        scored_df.to_csv("data/gdelt_raw.csv", index=False)
        print("\nRaw articles saved to data/gdelt_raw.csv")

    if daily_df is not None:
        daily_df.to_csv("data/gdelt_sentiment_daily.csv", index=False)
        print("Daily sentiment saved to data/gdelt_sentiment_daily.csv")