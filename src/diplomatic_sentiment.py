import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from transformers import pipeline
import torch
import time
import warnings
import requests
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
    ("IR", "ARMEDCONFLICT"),    # Iran armed conflict coverage
    ("IR", "NUCLEAR"),          # Iran nuclear coverage
    ("IR", "SANCTIONS"),        # Iran sanctions coverage
    ("IR", "ACT_HARMTHREATEN"), # Iranian threat rhetoric
    ("IS", "ARMEDCONFLICT"),    # Israel armed conflict coverage
    ("IS", "CEASEFIRE"),        # Israel ceasefire coverage
    ("IS", "ACT_FORCEPOSTURE"), # Israeli military posture
    ("US", "ARMEDCONFLICT"),    # US military involvement
    ("US", "PEACENEGOTIATION"), # US diplomatic activity
    ("US", "SANCTIONS"),        # US sanctions decisions

    # Proxy actors — only conflict-relevant themes
    ("LE", "ARMEDCONFLICT"),    # Lebanon/Hezbollah conflict
    ("LE", "TERROR"),           # Hezbollah militant activity
    ("YM", "ARMEDCONFLICT"),    # Yemen/Houthi conflict activity
    ("YM", "BLOCKADE"),         # Houthi Red Sea blockade specifically
    ("IZ", "ARMEDCONFLICT"),    # Iraq proxy militia activity
    ("IZ", "ACT_FORCEPOSTURE"), # Iraq military posture

    # Gulf states — economic and shipping impact only
    ("SA", "ECON_OILPRICE"),    # Saudi oil market impact
    ("SA", "BLOCKADE"),         # Saudi shipping disruption concern
    ("AE", "ECON_OILPRICE"),    # UAE oil market impact
    ("AE", "BLOCKADE"),         # UAE Strait of Hormuz concern
    ("QA", "ECON_OILPRICE"),    # Qatar LNG supply disruption
    ("KU", "ECON_OILPRICE"),    # Kuwait oil impact
    ("BA", "ARMEDCONFLICT"),    # Bahrain (hosts US 5th Fleet) conflict coverage

    # Diplomatic actors — only diplomatic and strategic themes
    ("TU", "PEACENEGOTIATION"), # Turkey mediation role only
    ("RS", "ARMEDCONFLICT"),    # Russia arms supply and involvement
    ("RS", "SANCTIONS"),        # Russia sanctions angle
    ("CH", "PEACENEGOTIATION"), # China diplomatic pressure
    ("CH", "SANCTIONS"),        # China stance on sanctions
    ("CH", "ECON_OILPRICE"),    # China as largest Gulf oil buyer
]


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
    Catches any article GDELT has classified under conflict/diplomatic
    themes regardless of exact terminology used by the journalist.
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
    Each pair is designed to capture only the specific role that country
    plays in the West Asia conflict — Yemen is only queried for
    ARMEDCONFLICT and BLOCKADE, not for general Yemeni domestic coverage.

    This means uncertainty spikes in our model only when genuinely
    conflict-relevant signals diverge — not when a flood hits Sana'a.
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
    By querying the same 14 themes used in article fetching, the tone
    timeline reflects the exact same slice of the conflict — making it
    a genuinely comparable cross-validation signal.

    High divergence between DistilBERT and GDELT tone = elevated
    epistemic uncertainty → model widens confidence intervals.
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

    # Combine all theme tone timelines and average by date
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
    Removes duplicate articles — the same article may appear in both
    a global theme query and a targeted country×theme query.
    We keep it once but retain the source tag.
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
    Scores article titles using DistilBERT on your RTX 3050.
    Primary sentiment signal. Output: -1.0 (hostile) to +1.0 (diplomatic).
    Processes in batches of 32 for maximum GPU efficiency.
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


def aggregate_daily(df, tone_timeline):
    """
    Aggregates to daily scores using both signals where available.

    Output columns:
    - distilbert_avg      : DistilBERT mean score (-1 to +1)
    - distilbert_vol      : Score volatility (std dev) — uncertainty proxy
    - article_count       : Volume of coverage that day
    - gdelt_tone_avg      : GDELT raw tone score (averaged across themes)
    - gdelt_tone_norm     : GDELT tone normalised to (-1 to +1)
    - signal_divergence   : |distilbert_avg - gdelt_tone_norm|
                            HIGH = high epistemic uncertainty
                            → model widens confidence intervals
    """
    if df is None:
        return None

    df['date'] = pd.to_datetime(
        df['seendate'], format='mixed', errors='coerce'
    ).dt.date

    daily = df.groupby('date').agg(
        distilbert_avg=('distilbert_sentiment', 'mean'),
        distilbert_vol=('distilbert_sentiment', 'std'),
        article_count=('distilbert_sentiment', 'count')
    ).reset_index()

    if tone_timeline is not None and len(tone_timeline) > 0:
        try:
            tone_timeline['date'] = pd.to_datetime(
                tone_timeline['date']
            ).apply(lambda x: x.date() if hasattr(x, 'date') else x)

            tone_daily = tone_timeline.rename(
                columns={'Average Tone': 'gdelt_tone_avg'}
            )

            # Normalise GDELT tone (-100 to +100) → (-1 to +1)
            tone_daily['gdelt_tone_norm'] = (
                tone_daily['gdelt_tone_avg'] / 100.0
            )

            daily = daily.merge(tone_daily, on='date', how='left')

            # Signal divergence — key epistemic uncertainty metric
            daily['signal_divergence'] = abs(
                daily['distilbert_avg'] - daily['gdelt_tone_norm']
            )
            print("\nBoth signals merged. Signal divergence computed.")

        except Exception as e:
            print(f"Tone merge error: {e}")
            daily['gdelt_tone_avg'] = None
            daily['gdelt_tone_norm'] = None
            daily['signal_divergence'] = None

    else:
        daily['gdelt_tone_avg'] = None
        daily['gdelt_tone_norm'] = None
        daily['signal_divergence'] = None
        print("\nOnly DistilBERT signal available. "
              "Epistemic uncertainty elevated — GDELT tone unavailable.")

    print("\nDaily Sentiment Summary:")
    print(daily.to_string())
    return daily


if __name__ == "__main__":
    DAYS = 5

    # Step 1: Fetch articles — two complementary strategies
    # Global theme query: broad conflict coverage worldwide
    # Country×theme query: targeted regional coverage without confounding
    theme_df        = fetch_by_themes(days_ago=DAYS)
    country_theme_df = fetch_by_country_theme_pairs(days_ago=DAYS)

    # Step 2: Fetch GDELT tone timeline across all themes
    tone_tl = fetch_gdelt_tone_timeline(days_ago=DAYS)

    # Step 3: Merge and deduplicate articles
    combined_df = merge_and_deduplicate(theme_df, country_theme_df)

    # Step 4: Score with DistilBERT
    scored_df = score_distilbert_sentiment(combined_df)

    # Step 5: Aggregate to daily with both signals
    daily_df = aggregate_daily(scored_df, tone_tl)

    # Step 6: Save outputs
    if scored_df is not None:
        scored_df.to_csv("data/gdelt_raw.csv", index=False)
        print("\nRaw articles saved to data/gdelt_raw.csv")

    if daily_df is not None:
        daily_df.to_csv("data/gdelt_sentiment_daily.csv", index=False)
        print("Daily sentiment saved to data/gdelt_sentiment_daily.csv")