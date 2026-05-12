"""
src/historical_backfill.py
────────────────────────────────────────────────────────────────────────────────
Historical Backfill Orchestrator — Step 8c
West Asia War 2026 Conflict Prediction Engine

Reconstructs Feb 01 2026 → today retroactively.
Calls all 4 modules in historical mode.
Generates one snapshot per day via snapshot_manager.

RUN ORDER (important)
─────────────────────
  Fast backfills first (~10 minutes total):
    1. FIRMS archive       — chunked, 10 days/call
    2. Economic signals    — single yfinance call
    3. GDELT kinetic       — chunked, 1 file/day

  Slow backfill last (~10-15 minutes):
    4. Diplomatic sentiment — chunked, 14 days/chunk, parallel API calls

  Then snapshots:
    5. Generate one snapshot JSON per day from merged CSVs

USAGE
─────
  # Full backfill — all modules
  python src/historical_backfill.py --mode full

  # Individual modules (for resuming after crash)
  python src/historical_backfill.py --mode firms
  python src/historical_backfill.py --mode economic
  python src/historical_backfill.py --mode gdelt_kinetic
  python src/historical_backfill.py --mode sentiment
  python src/historical_backfill.py --mode snapshots
"""

import sys
import logging
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from kinetic_pulse        import run_historical as firms_historical
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

# Data paths
DATA_DIR       = ROOT_DIR / "data"
FIRMS_CSV      = DATA_DIR / "firms_raw.csv"
ECONOMIC_CSV   = DATA_DIR / "economic_raw.csv"
GDELT_KIN_CSV  = DATA_DIR / "gdelt_kinetic_raw.csv"
SENTIMENT_CSV  = DATA_DIR / "gdelt_sentiment_daily.csv"


# ── Step runners ──────────────────────────────────────────────────────────────

def backfill_firms():
    log.info("=" * 60)
    log.info("FIRMS ARCHIVE BACKFILL")
    log.info("=" * 60)
    result = firms_historical(start_date=BACKFILL_START)
    log.info("FIRMS done: %s", result)
    return result


def backfill_economic():
    log.info("=" * 60)
    log.info("ECONOMIC SIGNALS BACKFILL")
    log.info("=" * 60)
    result = economic_historical(start_date=BACKFILL_START)
    log.info("Economic done: %s", result)
    return result


def backfill_gdelt_kinetic():
    log.info("=" * 60)
    log.info("GDELT KINETIC BACKFILL")
    log.info("=" * 60)
    result = gdelt_historical(start_date=BACKFILL_START)
    log.info("GDELT kinetic done: %s", result)
    return result


def backfill_sentiment():
    log.info("=" * 60)
    log.info("DIPLOMATIC SENTIMENT BACKFILL")
    log.info("WARNING: slow — ~10-15 minutes. Do not interrupt.")
    log.info("=" * 60)
    result = sentiment_historical(start_date=BACKFILL_START)
    log.info("Sentiment done: %s", result)
    return result


def generate_snapshots():
    """
    Merge all 4 CSVs on date and generate one snapshot JSON per day.
    Skips dates that already have a snapshot file.
    """
    log.info("=" * 60)
    log.info("GENERATING DAILY SNAPSHOTS")
    log.info("=" * 60)

    # ── Load all CSVs ─────────────────────────────────────────────────────────
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

    # ── Daily aggregations ────────────────────────────────────────────────────

    # FIRMS: mean FRP + anomaly count per day
    firms_daily = pd.DataFrame()
    if not firms.empty and "frp" in firms.columns:
        firms_daily = (
            firms.groupby("acq_date")
            .agg(frp_mean=("frp", "mean"), anomaly_count=("frp", "count"))
            .reset_index()
            .rename(columns={"acq_date": "date"})
        )

    # GDELT kinetic: event count per day
    gdelt_daily = pd.DataFrame()
    if not gdelt_kin.empty:
        gdelt_daily = (
            gdelt_kin.groupby("date")
            .agg(gdelt_events=("event_id", "count"))
            .reset_index()
        )

    # Economic: daily changes (already computed as first differences)
    econ_daily = pd.DataFrame()
    if not economic.empty:
        econ_cols = ["Date", "Brent_Crude", "VIX", "USD_ILS", "Gold"]
        available = [c for c in econ_cols if c in economic.columns]
        econ_daily = economic[available].copy()
        econ_daily = econ_daily.rename(columns={"Date": "date"})
        # Compute daily changes
        for col in ["Brent_Crude", "VIX", "USD_ILS", "Gold"]:
            if col in econ_daily.columns:
                econ_daily[f"{col}_change"] = econ_daily[col].diff()

    # Sentiment: already daily
    sent_daily = sentiment.copy() if not sentiment.empty else pd.DataFrame()

    # ── Build full date index ─────────────────────────────────────────────────
    start_dt = datetime.strptime(BACKFILL_START, "%Y-%m-%d").date()
    end_dt   = (datetime.utcnow() - timedelta(days=1)).date()
    all_dates = pd.date_range(str(start_dt), str(end_dt), freq="D").date

    existing_snapshots = set(list_snapshots())
    new_count = 0

    for date in all_dates:
        date_str = str(date)

        if date_str in existing_snapshots:
            log.info("Snapshot %s already exists — skipping", date_str)
            continue

        # Pull data for this date
        def row(df, date_col="date"):
            if df.empty:
                return {}
            match = df[df[date_col] == date]
            return match.iloc[0].to_dict() if not match.empty else {}

        f = row(firms_daily)
        g = row(gdelt_daily)
        e = row(econ_daily, date_col="date")
        s = row(sent_daily)

        # Build kinetic dict
        kinetic = {
            "frp_mean":      f.get("frp_mean"),
            "anomaly_count": f.get("anomaly_count"),
            "shock_detected": (
                f.get("frp_mean", 0) > 0 and
                f.get("anomaly_count", 0) > 10
            ),
            "gdelt_events":  g.get("gdelt_events"),
        }

        # Build economic dict
        economic_snap = {
            "brent_crude_change": e.get("Brent_Crude_change"),
            "vix_change":         e.get("VIX_change"),
            "usd_ils_change":     e.get("USD_ILS_change"),
            "gold_change":        e.get("Gold_change"),
        }

        # Build sentiment dict
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
        description="Historical backfill — West Asia War 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python src/historical_backfill.py --mode full
  python src/historical_backfill.py --mode firms
  python src/historical_backfill.py --mode economic
  python src/historical_backfill.py --mode gdelt_kinetic
  python src/historical_backfill.py --mode sentiment
  python src/historical_backfill.py --mode snapshots
        """,
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