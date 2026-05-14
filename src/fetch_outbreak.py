grep -E '^def ' src/diplomatic_sentiment.py
def safe_get(*args, **kwargs):
def _get_gdelt_client():
def build_date_range(days_ago):
def _fetch_articles_single(query_type, label, filters_kwargs,
def _fetch_tone_single(theme, start_date, end_date) -> pd.DataFrame | None:
def fetch_all_articles(start_date: str, end_date: str) -> pd.DataFrame | None:
def fetch_tone_timeline(start_date: str, end_date: str) -> pd.DataFrame | None:
def fetch_by_themes(days_ago=5):
def fetch_by_country_theme_pairs(days_ago=5):
def fetch_gdelt_tone_timeline(days_ago=5):
def merge_and_deduplicate(theme_df, country_theme_df):
def score_distilbert_sentiment(df):
def compute_daily_gmm_weights(df):
def bloc_sentiment(df, countries):
def theme_sentiment(df, themes):
def aggregate_daily(df, tone_timeline):
def _run_pipeline(start_date: str, end_date: str,
def run_realtime() -> dict:
def run_historical(start_date: str = "2026-02-01",
(venv) 
anwes@ASUS MINGW64 /d/Projects/west-asia-war-2026 (main)
$ 