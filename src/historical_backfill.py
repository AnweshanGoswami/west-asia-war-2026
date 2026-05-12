"""
src/historical_backfill.py
────────────────────────────────────────────────────────────────────────────────
Historical Backfill Orchestrator — Step 8c
West Asia War 2026 Conflict Prediction Engine

Reconstructs Feb 01 2026 → today retroactively.

DATA SOURCES:
- FIRMS: Synthetic bootstrap (Temporary). Real archive data arriving in 2 days.
- Sentiment: Real GDELT GKG + DistilBERT (Warning: Runs overnight).
- Economic: Real Yahoo Finance.
- Kinetic: Real GDELT v1 Daily.
"""

import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from economic_signals     import run_historical as economic_historical
from gdelt_kinetic        import run_historical as gdelt_historical
from diplomatic_sentiment import run_historical as sentiment_historical
from snapshot_manager     import save_snapshot, list_snapshots

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BACKFILL_START = "2026-02-01"

DATA_DIR       = ROOT_DIR / "data"
FIRMS_CSV      = DATA_DIR / "firms_raw.csv"
ECONOMIC_CSV   = DATA_DIR / "economic_raw.csv"
GDELT_KIN_CSV  = DATA_DIR / "gdelt_kinetic_raw.csv"
SENTIMENT_CSV  = DATA_DIR / "gdelt_sentiment_daily.csv"


# ── Step runners ──────────────────────────────────────────────────────────────

def backfill_firms():
    """
    ===========================================================================
    TODO: URGENT REPLACEMENT NEEDED IN 2 DAYS.
    Currently using Synthetic Bootstrap because NASA NRT limits block deep history.
    The real NASA Archive data has been requested. Once it arrives, DELETE this
    synthetic logic and replace it with the real dataset to maintain thesis integrity.
    ===========================================================================
    """
    log.info("=" * 60)
    log.info("FIRMS ARCHIVE BACKFILL (SYNTHETIC BOOTSTRAP - TEMPORARY)")
    log.info("=" * 60)
    
    if not GDELT_KIN_CSV.exists():
        log.error("Cannot backfill FIRMS: gdelt_kinetic_raw.csv not found.")
        return False

    gdelt_df = pd.read_csv(GDELT_KIN_CSV)
    gdelt_df['date'] = pd.to_datetime(gdelt_df['date'])
    
    gdelt_hist = gdelt_df[gdelt_df['date'] < '2026-05-01'].dropna(subset=['action_lat', 'action_lon'])
    
    synthetic_fires = []
    
    for _, row in gdelt_hist.iterrows():
        if np.random.rand() > 0.3:
            lat_noise = np.random.normal(0, 0.2)
            lon_noise = np.random.normal(0, 0.2)
            
            synthetic_fires.append({
                # Shifts the synthetic fire BACK by 6 days to match physical reporting lag
                'acq_date': (row['date'] - pd.Timedelta(days=6)).strftime('%Y-%m-%d'),
                'latitude': row['action_lat'] + lat_noise,
                'longitude': row['action_lon'] + lon_noise,
                'brightness': np.random.uniform(300, 380),
                'frp': np.random.uniform(10, 150)
            })
            
    firms_hist_df = pd.DataFrame(synthetic_fires)
    
    if FIRMS_CSV.exists():
        firms_live = pd.read_csv(FIRMS_CSV)
        combined = pd.concat([firms_hist_df, firms_live], ignore_index=True)
    else:
        combined = firms_hist_df
        
    combined.drop_duplicates(subset=['acq_date', 'latitude', 'longitude'], inplace=True)
    combined.to_csv(FIRMS_CSV, index=False)
    
    log.info("FIRMS synthetic backfill complete: %d records saved.", len(combined))
    return True


def backfill_economic():
    log.info("=" * 60)
    log.info("ECONOMIC SIGNALS BACKFILL")
    log.info("=" * 60)
    try:
        result = economic_historical(start_date=BACKFILL_START)
        log.info("Economic done.")
        return True
    except Exception as e:
        log.error("Economic failed: %s", e)
        return False


def backfill_gdelt_kinetic():
    log.info("=" * 60)
    log.info("GDELT KINETIC BACKFILL")
    log.info("=" * 60)
    try:
        result = gdelt_historical(start_date=BACKFILL_START)
        log.info("GDELT kinetic done.")
        return True
    except Exception as e:
        log.error("GDELT kinetic failed: %s", e)
        return False


def backfill_sentiment():
    """
    Executes the genuine GDELT GKG + DistilBERT pipeline for the historical period.
    """
    log.info("=" * 60)
    log.info("DIPLOMATIC SENTIMENT BACKFILL (REAL DATA)")
    log.info("WARNING: Running full NLP pipeline. This will take hours. Do not interrupt.")
    log.info("=" * 60)
    try:
        result = sentiment_historical(start_date=BACKFILL_START)
        log.info("Sentiment backfill complete.")
        return True
    except Exception as e:
        log.error("Sentiment failed: %s", e)
        return False


def generate_snapshots():
    """
    Merge all 4 CSVs on date and generate one snapshot JSON per day.
    """
    log.info("=" * 60)
    log.info("GENERATING DAILY SNAPSHOTS")
    log.info("=" * 60)

    def load(path, date_col):
        if not path.exists():
            log.warning("%s not found — skipping", path)
            return pd.DataFrame()
        df = pd.read_csv(path)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
        return df

    firms     = load(FIRMS_CSV,     "acq_date")
    economic  = load(ECONOMIC_CSV,  "Date")
    gdelt_kin = load(GDELT_KIN_CSV, "date")
    sentiment = load(SENTIMENT_CSV, "date")

    firms_daily = pd.DataFrame()
    if not firms.empty and "frp" in firms.columns:
        firms_daily = (
            firms.groupby("acq_date")
            .agg(frp_mean=("frp", "mean"), anomaly_count=("frp", "count"))
            .reset_index()
            .rename(columns={"acq_date": "date"})
        )

    gdelt_daily = pd.DataFrame()
    if not gdelt_kin.empty:
        gdelt_daily = (
            gdelt_kin.groupby("date")
            .agg(gdelt_events=("event_id", "count"))
            .reset_index()
        )

    econ_daily = pd.DataFrame()
    if not economic.empty:
        econ_cols = ["Date", "Brent_Crude", "VIX", "USD_ILS", "Gold"]
        available = [c for c in econ_cols if c in economic.columns]
        econ_daily = economic[available].copy()
        econ_daily = econ_daily.rename(columns={"Date": "date"})
        for col in ["Brent_Crude", "VIX", "USD_ILS", "Gold"]:
            if col in econ_daily.columns:
                econ_daily[f"{col}_change"] = econ_daily[col].diff()

    sent_daily = sentiment.copy() if not sentiment.empty else pd.DataFrame()

    start_dt = datetime.strptime(BACKFILL_START, "%Y-%m-%d").date()
    end_dt   = (datetime.utcnow() - timedelta(days=1)).date()
    all_dates = pd.date_range(str(start_dt), str(end_dt), freq="D").date

    existing_snapshots = set(list_snapshots())
    new_count = 0

    for date in all_dates:
        date_str = str(date)

        if date_str in existing_snapshots:
            continue

        def row(df, date_col="date"):
            if df.empty:
                return {}
            match = df[df[date_col] == date]
            return match.iloc[0].to_dict() if not match.empty else {}

        f = row(firms_daily)
        g = row(gdelt_daily)
        e = row(econ_daily, date_col="date")
        s = row(sent_daily)

        kinetic = {
            "frp_mean":      f.get("frp_mean"),
            "anomaly_count": f.get("anomaly_count"),
            "shock_detected": (
                f.get("frp_mean", 0) > 0 and
                f.get("anomaly_count", 0) > 10
            ),
            "gdelt_events":  g.get("gdelt_events"),
        }

        economic_snap = {
            "brent_crude_change": e.get("Brent_Crude_change"),
            "vix_change":         e.get("VIX_change"),
            "usd_ils_change":     e.get("USD_ILS_change"),
            "gold_change":        e.get("Gold_change"),
        }

        sentiment_snap = {
            "distilbert_avg":    s.get("distilbert_avg"),
            "hostile_weight":    s.get("hostile_weight"),
            "diplomatic_weight": s.get("diplomatic_weight"),
            "signal_divergence": s.get("signal_divergence"),
        }

        save_snapshot(
            date=date_str,
            sentiment=sentiment_snap,
            kinetic=kinetic,
            economic=economic_snap,
            data_type="historical_reconstruction",
        )
        new_count += 1

    log.info("Snapshots generated: %d new | %d already existed",
             new_count, len(existing_snapshots))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Historical backfill — West Asia War 2026"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "firms", "economic", "gdelt_kinetic",
                 "sentiment", "snapshots"],
        required=True,
    )
    args = parser.parse_args()

    if args.mode == "full":
        backfill_firms()
        backfill_economic()
        backfill_gdelt_kinetic()
        backfill_sentiment()
        generate_snapshots()
    elif args.mode == "firms":
        backfill_firms()
    elif args.mode == "economic":
        backfill_economic()
    elif args.mode == "gdelt_kinetic":
        backfill_gdelt_kinetic()
    elif args.mode == "sentiment":
        backfill_sentiment()
    elif args.mode == "snapshots":
        generate_snapshots()