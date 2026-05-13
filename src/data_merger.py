"""
src/data_merger.py
────────────────────────────────────────────────────────────────────────────────
Data Merging — Step 9
West Asia War 2026 Conflict Prediction Engine

Joins all 4 data sources on `date` into a single master DataFrame.

KEY OPERATIONS
──────────────
  1. Spatial Anchoring via BallTree (The Ground Truth Filter)
     GDELT locations are plagued by capital-city bias. We cross-reference
     every GDELT event against NASA FIRMS thermal anomalies (±2 day window).
     If a GDELT event is NOT within 50km of a real fire, it is discarded.
     
  2. GDELT kinetic 6-day lag correction
     Applied dynamically during the Spatial Anchoring phase to align news
     reports with physical reality before the distance calculations occur.

  3. Weekend forward-fill
     Markets close on weekends. Economic signals forward-filled from Friday.

  4. Missing data flagging
     Missing days are flagged (e.g., `firms_data_missing = True`) for 
     uncertainty propagation in the Lanchester ODEs, rather than dropped.

OUTPUT
──────
  data/master_df.csv — single merged DataFrame, one row per day
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.neighbors import BallTree

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT_DIR / "data"
FIRMS_CSV     = DATA_DIR / "firms_raw.csv"
ECONOMIC_CSV  = DATA_DIR / "economic_raw.csv"
GDELT_KIN_CSV = DATA_DIR / "gdelt_kinetic_raw.csv"
SENTIMENT_CSV = DATA_DIR / "gdelt_sentiment_daily.csv"
OUTPUT_CSV    = DATA_DIR / "master_df.csv"

# ── Config ────────────────────────────────────────────────────────────────────
GDELT_LAG_DAYS  = 6
ANCHOR_RADIUS_KM = 50.0
EARTH_RADIUS_KM  = 6371.0
BACKFILL_START  = "2026-02-01"

# Model feature columns — S&P 500 explicitly excluded
MODEL_FEATURES = [
    "firms_frp_mean",
    "firms_anomaly_count",
    "gdelt_event_count",
    "gdelt_avg_goldstein",
    "gdelt_total_mentions",
    "gdelt_avg_tone",
    "brent_crude_change",
    "vix_change",
    "usd_ils_change",
    "gold_change",
    "distilbert_avg",
    "hostile_weight",
    "diplomatic_weight",
    "signal_divergence",
    "bloc_divergence",
    "military_diplomatic_gap",
]


# ── Loaders & Anchoring ───────────────────────────────────────────────────────

def _load_firms() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FIRMS raw CSV.
    Returns:
        raw_df: Unaggregated rows (needed for spatial anchoring).
        daily_df: Aggregated daily means/counts (for master merge).
    """
    if not FIRMS_CSV.exists():
        log.warning("firms_raw.csv not found")
        empty = pd.DataFrame(columns=["date", "firms_frp_mean", "firms_anomaly_count"])
        return pd.DataFrame(), empty

    raw_df = pd.read_csv(FIRMS_CSV)
    date_col = "acq_date" if "acq_date" in raw_df.columns else "date"
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce").dt.date
    raw_df = raw_df.rename(columns={date_col: "date"}).dropna(subset=["latitude", "longitude", "date"])

    if "frp" not in raw_df.columns:
        empty = pd.DataFrame(columns=["date", "firms_frp_mean", "firms_anomaly_count"])
        return raw_df, empty

    daily_df = (
        raw_df.groupby("date")
        .agg(firms_frp_mean=("frp", "mean"), firms_anomaly_count=("frp", "count"))
        .reset_index()
    )
    log.info("FIRMS loaded: %d raw anomalies, %d daily aggregates", len(raw_df), len(daily_df))
    return raw_df, daily_df


def _load_and_anchor_gdelt(raw_firms: pd.DataFrame) -> pd.DataFrame:
    """
    Load GDELT kinetic CSV, apply 6-day lag, spatially anchor to FIRMS, and aggregate.
    """
    if not GDELT_KIN_CSV.exists():
        log.warning("gdelt_kinetic_raw.csv not found")
        return pd.DataFrame()

    gdelt_df = pd.read_csv(GDELT_KIN_CSV)
    gdelt_df = gdelt_df.dropna(subset=["event_id", "action_lat", "action_lon", "date"])
    
    if gdelt_df.empty or raw_firms.empty:
        log.warning("Cannot anchor: Missing GDELT or FIRMS data. Returning empty.")
        return pd.DataFrame()

    log.info("Applying 6-day lag and executing Spatial Anchoring via BallTree...")
    
    # 1. Apply 6-day temporal shift
    gdelt_df["date"] = pd.to_datetime(gdelt_df["date"]) - pd.Timedelta(days=GDELT_LAG_DAYS)
    gdelt_df["date"] = gdelt_df["date"].dt.date
    
    anchored_indices = []
    max_dist_rad = ANCHOR_RADIUS_KM / EARTH_RADIUS_KM

    # 2. Group FIRMS by date for fast temporal slicing
    firms_by_date = dict(list(raw_firms.groupby("date")))

    # 3. Anchor day by day using BallTree
    for current_date, g_group in gdelt_df.groupby("date"):
        # Gather FIRMS fires within ±2 days of the shifted GDELT date
        temporal_firms = []
        for delta in range(-2, 3):
            target_date = current_date + timedelta(days=delta)
            if target_date in firms_by_date:
                temporal_firms.append(firms_by_date[target_date])

        if not temporal_firms:
            continue # No fires within the time window = no anchor

        t_firms_df = pd.concat(temporal_firms)

        # Build rapid spatial index
        firms_rad = np.deg2rad(t_firms_df[["latitude", "longitude"]].values)
        tree = BallTree(firms_rad, metric="haversine")

        # Query GDELT events against the tree
        g_rad = np.deg2rad(g_group[["action_lat", "action_lon"]].values)
        ind = tree.query_radius(g_rad, r=max_dist_rad)

        # Retain GDELT events that matched at least one fire
        for idx, matches in zip(g_group.index, ind):
            if len(matches) > 0:
                anchored_indices.append(idx)

    anchored_gdelt = gdelt_df.loc[anchored_indices].copy()
    anchor_rate = (len(anchored_gdelt) / len(gdelt_df)) * 100
    log.info("Anchored %d / %d GDELT events (%.1f%%)", len(anchored_gdelt), len(gdelt_df), anchor_rate)

    # 4. Aggregate cleanly anchored events to daily level
    daily = (
        anchored_gdelt.groupby("date")
        .agg(
            gdelt_event_count   =("event_id",        "count"),
            gdelt_avg_goldstein =("goldstein_scale", "mean"),
            gdelt_total_mentions=("num_mentions",    "sum"),
            gdelt_avg_tone      =("avg_tone",        "mean"),
        )
        .reset_index()
    )
    return daily


def _load_economic() -> pd.DataFrame:
    if not ECONOMIC_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(ECONOMIC_CSV)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.drop(columns=["Date"], errors="ignore").sort_values("date")
    
    # --- FIX: Weekend Forward-Fill ---
    # Reindex to a continuous daily frequency to expose weekend gaps
    df.set_index("date", inplace=True)
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D").date
    df = df.reindex(all_days)
    
    # Forward-fill prices (Friday's close carries through the weekend)
    df = df.ffill()
    df.index.name = "date"
    df = df.reset_index()

    # NOW calculate the daily differences. 
    # Saturday/Sunday diffs will be 0. Monday diff will be (Monday - Friday).
    for col, out_col in [("Brent_Crude", "brent_crude_change"), ("VIX", "vix_change"),
                         ("USD_ILS", "usd_ils_change"), ("Gold", "gold_change")]:
        if col in df.columns:
            df[out_col] = df[col].diff()

    if "SP500" in df.columns:
        sp500_peak_date = pd.to_datetime("2026-02-27").date()
        peak_mask = df["date"] <= sp500_peak_date
        if peak_mask.any():
            sp500_peak = df.loc[peak_mask, "SP500"].iloc[-1]
            df["sp500_drawdown_pct"] = (df["SP500"] - sp500_peak) / sp500_peak * 100
        else:
            df["sp500_drawdown_pct"] = None
        df = df.drop(columns=["SP500"], errors="ignore")

    log.info("Economic loaded and forward-filled: %d days", len(df))
    return df


def _load_sentiment() -> pd.DataFrame:
    if not SENTIMENT_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(SENTIMENT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # --- FIX: Handle Duplicates by Maximum Data Availability ---
    # Count how many non-null values exist in each row
    df['valid_data_count'] = df.notna().sum(axis=1)
    
    # Sort by date, and then by data count (descending) so the richest row is first
    df = df.sort_values(by=['date', 'valid_data_count'], ascending=[True, False])
    
    # Drop duplicates, keeping that first (richest) row
    df = df.drop_duplicates(subset=['date'], keep='first')
    df = df.drop(columns=['valid_data_count'])

    keep = ["date", "distilbert_avg", "distilbert_vol", "article_count", "hostile_weight", 
            "diplomatic_weight", "hostile_mean", "diplomatic_mean", "sentiment_adversarial_bloc", 
            "sentiment_allied_bloc", "sentiment_neutral_bloc", "bloc_divergence", 
            "sentiment_military", "sentiment_diplomatic", "sentiment_economic", 
            "military_diplomatic_gap", "gdelt_tone_avg", "gdelt_tone_norm", "signal_divergence"]
    df = df[[c for c in keep if c in df.columns]]

    log.info("Sentiment loaded and deduplicated: %d days", len(df))
    return df


# ── Master merge ──────────────────────────────────────────────────────────────

def build_master_df(start_date: str = BACKFILL_START, end_date: str = None, save: bool = True) -> pd.DataFrame:
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else (datetime.utcnow() - timedelta(days=1)).date()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()

    log.info("Building master DataFrame: %s → %s", start_dt, end_dt)

    all_dates = pd.DataFrame({"date": pd.date_range(str(start_dt), str(end_dt), freq="D").date})

    raw_firms, daily_firms = _load_firms()
    gdelt_kin = _load_and_anchor_gdelt(raw_firms)
    economic  = _load_economic()
    sentiment = _load_sentiment()

    master = all_dates.copy()

    for df, label in [(daily_firms, "FIRMS"), (gdelt_kin, "GDELT kinetic"), 
                      (economic, "Economic"), (sentiment, "Sentiment")]:
        if not df.empty:
            master = master.merge(df, on="date", how="left")
            log.info("Merged %s", label)

    master["firms_data_missing"]     = master["firms_frp_mean"].isna()
    master["gdelt_data_missing"]     = master["gdelt_event_count"].isna()
    master["sentiment_data_missing"] = master["distilbert_avg"].isna()
    master["economic_data_missing"]  = master["brent_crude_change"].isna()

    if save:
        master.to_csv(OUTPUT_CSV, index=False)
        log.info("Master DataFrame saved → %s (%d rows × %d cols)", OUTPUT_CSV, len(master), len(master.columns))

    return master


if __name__ == "__main__":
    master = build_master_df()
    print("\nMissing data flags:")
    for col in ["firms_data_missing", "gdelt_data_missing", "sentiment_data_missing", "economic_data_missing"]:
        if col in master.columns:
            print(f"  {col}: {master[col].sum()} days")