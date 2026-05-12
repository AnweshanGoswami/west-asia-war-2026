"""
src/kinetic_pulse.py
────────────────────────────────────────────────────────────────────────────────
NASA FIRMS Thermal Anomaly Layer
West Asia War 2026 Conflict Prediction Engine

ROLE IN ARCHITECTURE
────────────────────
Physical proof half of the Dual-Signal Veto System.
FIRMS thermal anomalies = confirmed fires/strikes on the ground.
Cannot trigger a Kinetic Shock alone — must cross-validate with
GDELT kinetic events (the narrative layer) in Step 12.

TWO ENDPOINTS
─────────────
  Realtime  →  VIIRS_SNPP_NRT  — last 1–10 days, updates every few hours
  Archive   →  VIIRS_SNPP_SP   — historical standard processing, max 10 days/call
               Used in Step 8c to backfill Feb 01 → today.

SILENT FAILURE WARNING
──────────────────────
NASA returns HTTP 200 with plain-text error when API key is invalid or
rate-limited. Pandas silently reads this as an empty CSV, making it look
like zero anomalies. Raw response is validated before parsing.

OUTPUT
──────
  data/firms_raw.csv        ← historical archive (Feb 01 → today)
  data/firms_realtime.csv   ← latest NRT window (polling loop)
"""

import os
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NASA_FIRMS_KEY")

# Bounding box: lon_min, lat_min, lon_max, lat_max
# Covers Iran, Israel, Lebanon, Syria, Iraq, Jordan, West Bank, Gulf
REGION = "34.0,29.0,60.0,38.0"

# FIRMS endpoints
NRT_URL     = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{region}/{days}"
ARCHIVE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_SP/{region}/{days}/{date}"

# Paths
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
ARCHIVE_CSV    = DATA_DIR / "firms_raw.csv"
REALTIME_CSV   = DATA_DIR / "firms_realtime.csv"


# ── Core helper ───────────────────────────────────────────────────────────────

def _fetch_and_parse(url: str) -> pd.DataFrame:
    """
    Download a FIRMS CSV URL and return parsed DataFrame.

    Includes silent-failure guard: NASA returns HTTP 200 with plain-text
    error on bad API keys. We check the raw response before parsing.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  FIRMS request failed: {e}")
        return pd.DataFrame()

    # Silent failure guard — valid CSV always starts with "latitude" header
    raw = response.text.strip()
    if not raw.startswith("latitude"):
        print(f"  FIRMS API error (silent failure detected):")
        print(f"  Raw response: {raw[:300]}")
        print("  Check NASA_FIRMS_KEY in .env — key may be invalid or rate-limited.")
        return pd.DataFrame()

    df = pd.read_csv(StringIO(raw))
    return df


# ── Public interface ──────────────────────────────────────────────────────────

def get_firms_data(days_ago: int = 1) -> pd.DataFrame | None:
    """
    Fetch NRT FIRMS data for the last N days (max 10).
    Used for manual testing and realtime polling.
    """
    url = NRT_URL.format(key=API_KEY, region=REGION, days=days_ago)
    print(f"Fetching NASA FIRMS NRT data for the last {days_ago} day(s)...")

    df = _fetch_and_parse(url)

    if df.empty:
        print("0 thermal anomalies detected (or API error — see above).")
        return None

    print(f"Success. {len(df)} thermal anomalies detected.")
    return df


def run_historical(
    start_date: str = "2026-02-01",
    end_date:   str = None,
    append:     bool = True,
) -> pd.DataFrame:
    """
    Historical backfill using FIRMS Archive API (VIIRS_SNPP_SP).
    Archive endpoint maxes at 10 days per call — chunks automatically.

    Args:
        start_date : ISO date string (default "2026-02-01")
        end_date   : ISO date string (default yesterday)
        append     : Skip date ranges already in ARCHIVE_CSV (default True)

    Returns:
        DataFrame of all newly fetched anomalies.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date
        else (datetime.utcnow() - timedelta(days=1)).date()
    )

    print(f"FIRMS archive backfill: {start_dt} → {end_dt}")

    # Load existing dates to skip if appending
    existing_dates = set()
    if append and ARCHIVE_CSV.exists():
        existing = pd.read_csv(ARCHIVE_CSV, usecols=["acq_date"])
        existing_dates = set(existing["acq_date"].astype(str))
        print(f"Resuming — {len(existing_dates)} dates already in archive CSV")

    # Build 10-day chunks
    chunks = []
    cursor = start_dt
    while cursor <= end_dt:
        chunk_end = min(cursor + timedelta(days=9), end_dt)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)

    print(f"{len(chunks)} chunks to download (10 days each)")

    all_frames = []

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        # Skip if all dates in this chunk already downloaded
        chunk_dates = set(
            (chunk_start + timedelta(days=d)).strftime("%Y-%m-%d")
            for d in range((chunk_end - chunk_start).days + 1)
        )
        if chunk_dates.issubset(existing_dates):
            print(f"[{i}/{len(chunks)}] {chunk_start} → {chunk_end} — already downloaded, skipping")
            continue

        days_in_chunk = (chunk_end - chunk_start).days + 1
        url = ARCHIVE_URL.format(
            key=API_KEY,
            region=REGION,
            days=days_in_chunk,
            date=chunk_start.strftime("%Y-%m-%d"),
        )

        print(f"[{i}/{len(chunks)}] {chunk_start} → {chunk_end} ({days_in_chunk} days)...")
        df = _fetch_and_parse(url)

        if df.empty:
            print(f"  → 0 anomalies (or API error)")
        else:
            print(f"  → {len(df)} anomalies retained")
            all_frames.append(df)

            # Append to archive CSV immediately (safe resume on crash)
            write_header = not ARCHIVE_CSV.exists()
            df.to_csv(ARCHIVE_CSV, mode="a", header=write_header, index=False)

    if not all_frames:
        print("Archive backfill complete — no new anomalies found.")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    print(f"Archive backfill complete. {len(result)} new anomalies → {ARCHIVE_CSV}")
    return result


def run_realtime() -> dict:
    """
    Standardized entry point for the master polling loop (Step 8a).
    Fetches the latest 24-hour NRT window and appends to firms_realtime.csv.
    """
    df = get_firms_data(days_ago=1)

    if df is not None and not df.empty:
        df.to_csv(REALTIME_CSV, index=False)
        return {"status": "success", "records": len(df), "file": str(REALTIME_CSV)}

    return {"status": "empty", "records": 0}


if __name__ == "__main__":
    df = get_firms_data(days_ago=5)
    if df is not None:
        print(df.head())
        df.to_csv(ARCHIVE_CSV, index=False)
        print(f"Data saved to {ARCHIVE_CSV}")