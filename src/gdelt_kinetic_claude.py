"""
src/gdelt_kinetic.py
────────────────────────────────────────────────────────────────────────────────
GDELT Event V2 — Kinetic Events Layer
West Asia War 2026 Conflict Prediction Engine

ROLE IN ARCHITECTURE
────────────────────
This module is one half of the Dual-Signal Veto System. It provides the
*narrative* layer — who fired at whom, using GDELT CAMEO conflict codes.
It cannot trigger a Kinetic Shock alone. It must be cross-validated against
NASA FIRMS thermal anomalies (the *physical proof* layer) in Step 12.

ACLED NOTE
──────────
ACLED was the original data source for this layer. It was deprecated due to
enterprise paywalls blocking real-time crisis data access. GDELT Event V2 is
the drop-in replacement: open-access, updating every 15 minutes, using
standardised CAMEO codes.

CAMEO CODES INGESTED
────────────────────
  18 — ASSAULT        (armed attacks, bombings, shelling)
  19 — FIGHT          (armed clashes, firefights)
  20 — USE UNCONVENTIONAL MASS VIOLENCE (WMD use, massacres)

KNOWN LIMITATIONS (tested in notebooks/gdelt_kinetic_analysis.ipynb)
──────────────────────────────────────────────────────────────────────
  1. Spatial bias: GDELT defaults unknown locations to country capital
     coordinates. Corrected in Step 9 via Spatial Anchoring against FIRMS.
  2. No exact casualty counts: Lanchester k-coefficients are approximated
     via Negative Binomial distributions mapped from CAMEO codes in Step 14.
  3. Reporting lag: GDELT ingests news articles — events may appear 1–6 hours
     after occurrence. Measured formally in the analysis notebook.

ASSUMPTION (to be tested in notebook)
──────────────────────────────────────
Sampling 4 GDELT 15-minute export files per day (one per 6-hour window)
captures sufficient event density for daily aggregation without requiring
full download of all 96 files/day (~19GB total for Feb 1 – present).

DATA FLOW
─────────
  Historical mode  →  masterfilelist.txt  →  sample 4 files/day  →  filter  →  CSV
  Realtime mode    →  lastupdate.txt      →  latest file          →  filter  →  CSV

OUTPUT
──────
  data/gdelt_kinetic_raw.csv
    Columns: date, event_id, cameo_code, cameo_root, goldstein_scale,
             num_mentions, num_articles, avg_tone, action_lat, action_lon,
             action_country, action_fullname, actor1_country, actor2_country,
             source_url, gdelt_file_timestamp

USAGE
─────
  # Historical backfill (Feb 1 2026 → today)
  python src/gdelt_kinetic.py --mode historical --start 2026-02-01

  # Real-time (latest 15-minute window only)
  python src/gdelt_kinetic.py --mode realtime

  # Full density test (all 96 files for one day — for notebook comparison)
  python src/gdelt_kinetic.py --mode density_test --date 2026-03-15
"""

import os
import io
import logging
import argparse
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = DATA_DIR / "gdelt_kinetic_raw.csv"

# ── GDELT V2 endpoints ────────────────────────────────────────────────────────
GDELT_MASTER     = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_LASTUPDATE = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# ── Filter config ─────────────────────────────────────────────────────────────
# CAMEO root codes to retain (kinetic conflict events only)
CAMEO_KINETIC_ROOTS = {"18", "19", "20"}

# FIPS 10-4 country codes for the conflict region
# IR=Iran  IS=Israel  LE=Lebanon  SY=Syria  IZ=Iraq
# YM=Yemen  SA=Saudi Arabia  AE=UAE  JO=Jordan  WE=West Bank  GZ=Gaza
CONFLICT_COUNTRIES = {
    "IR", "IS", "LE", "SY", "IZ",
    "YM", "SA", "AE", "JO", "WE", "GZ",
}

# ── GDELT V2 export column names (58 columns, tab-separated) ──────────────────
GDELT_COLUMNS = [
    "GlobalEventID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

# Columns to retain in output (discard unused actor/geo metadata)
OUTPUT_COLUMNS = [
    "date", "event_id", "cameo_code", "cameo_root", "goldstein_scale",
    "num_mentions", "num_articles", "avg_tone",
    "action_lat", "action_lon", "action_country", "action_fullname",
    "actor1_country", "actor2_country",
    "source_url", "gdelt_file_timestamp",
]


# ── Core helpers ──────────────────────────────────────────────────────────────

def _download_and_filter(url: str, file_timestamp: str) -> pd.DataFrame:
    """
    Download one GDELT V2 export zip, unpack it in memory, apply filters,
    and return a tidy DataFrame. Returns empty DataFrame on any failure.
    """
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Download failed: %s  (%s)", url, e)
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".export.CSV")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    names=GDELT_COLUMNS,
                    dtype=str,
                    on_bad_lines="skip",
                )
    except Exception as e:
        log.warning("Parse failed: %s  (%s)", url, e)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # ── Filter 1: kinetic CAMEO root codes ────────────────────────────────────
    df = df[df["EventRootCode"].isin(CAMEO_KINETIC_ROOTS)]
    if df.empty:
        return pd.DataFrame()

    # ── Filter 2: conflict region (action geography, primary filter) ──────────
    # ActionGeo is where the event physically occurred — more reliable than
    # Actor geo for spatial anchoring against FIRMS.
    df = df[df["ActionGeo_CountryCode"].isin(CONFLICT_COUNTRIES)]
    if df.empty:
        return pd.DataFrame()

    # ── Reshape to output schema ───────────────────────────────────────────────
    out = pd.DataFrame()
    out["date"]               = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce").dt.date
    out["event_id"]           = df["GlobalEventID"].values
    out["cameo_code"]         = df["EventCode"].values
    out["cameo_root"]         = df["EventRootCode"].values
    out["goldstein_scale"]    = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    out["num_mentions"]       = pd.to_numeric(df["NumMentions"],    errors="coerce")
    out["num_articles"]       = pd.to_numeric(df["NumArticles"],    errors="coerce")
    out["avg_tone"]           = pd.to_numeric(df["AvgTone"],        errors="coerce")
    out["action_lat"]         = pd.to_numeric(df["ActionGeo_Lat"],  errors="coerce")
    out["action_lon"]         = pd.to_numeric(df["ActionGeo_Long"], errors="coerce")
    out["action_country"]     = df["ActionGeo_CountryCode"].values
    out["action_fullname"]    = df["ActionGeo_FullName"].values
    out["actor1_country"]     = df["Actor1CountryCode"].values
    out["actor2_country"]     = df["Actor2CountryCode"].values
    out["source_url"]         = df["SOURCEURL"].values
    out["gdelt_file_timestamp"] = file_timestamp

    out = out.dropna(subset=["date"])
    return out


def _get_master_file_index() -> pd.DataFrame:
    """
    Fetch and parse the GDELT V2 master file list.
    Returns DataFrame with columns: size, md5, url, timestamp (parsed from filename).
    """
    log.info("Fetching GDELT master file index...")
    resp = requests.get(GDELT_MASTER, timeout=60)
    resp.raise_for_status()

    rows = []
    for line in resp.text.strip().split("\n"):
        parts = line.strip().split(" ")
        if len(parts) == 3:
            size, md5, url = parts
            # Extract timestamp from filename: .../20260301120000.export.CSV.zip
            fname = url.split("/")[-1]
            ts_str = fname.split(".")[0]  # "20260301120000"
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                rows.append({"size": int(size), "md5": md5, "url": url, "timestamp": ts})
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    log.info("Master index loaded: %d files", len(df))
    return df


def _sample_urls_for_range(
    master_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    samples_per_day: int = 4,
) -> list[dict]:
    """
    Select `samples_per_day` evenly-spaced files per calendar day in range.
    Returns list of dicts: {url, timestamp_str}.

    Default 4 samples/day = one file per 6-hour window.
    Assumption: This is sufficient for daily aggregation.
    Tested against full-density (96/day) in notebooks/gdelt_kinetic_analysis.ipynb.
    """
    mask = (master_df["timestamp"] >= start_date) & (master_df["timestamp"] <= end_date)
    # Only export files (not mentions or GKG files)
    mask &= master_df["url"].str.contains("export.CSV.zip")
    window = master_df[mask].copy()

    if window.empty:
        log.warning("No files found in master index for the requested date range.")
        return []

    # Group by calendar date, sample evenly within each day
    window["date"] = window["timestamp"].dt.date
    sampled = []
    for date, group in window.groupby("date"):
        group = group.sort_values("timestamp")
        indices = [int(i * (len(group) - 1) / (samples_per_day - 1))
                   for i in range(samples_per_day)]
        indices = sorted(set(indices))  # deduplicate edge cases
        for idx in indices:
            row = group.iloc[idx]
            sampled.append({
                "url": row["url"],
                "timestamp_str": row["timestamp"].strftime("%Y%m%d%H%M%S"),
            })

    log.info("Sampled %d files across %d days (%d samples/day)",
             len(sampled), window["date"].nunique(), samples_per_day)
    return sampled


# ── Public interface ──────────────────────────────────────────────────────────

def run_historical(
    start_date: str = "2026-02-01",
    end_date:   str = None,
    samples_per_day: int = 4,
    append: bool = True,
) -> pd.DataFrame:
    """
    Historical backfill: download sampled GDELT files from start_date to end_date,
    filter for kinetic events in the conflict region, and save to CSV.

    Args:
        start_date:      ISO date string, default "2026-02-01"
        end_date:        ISO date string, default today
        samples_per_day: Files per day (4 = one per 6hr window). Tested in notebook.
        append:          If True and output CSV exists, append new rows only.

    Returns:
        DataFrame of all fetched events.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()

    log.info("Historical mode: %s → %s (%d samples/day)",
             start_date, end_dt.date(), samples_per_day)

    # Load existing data to avoid re-downloading
    existing_timestamps = set()
    if append and OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, usecols=["gdelt_file_timestamp"])
        existing_timestamps = set(existing["gdelt_file_timestamp"].astype(str))
        log.info("Existing CSV found: %d unique file timestamps already loaded",
                 len(existing_timestamps))

    master_df = _get_master_file_index()
    targets   = _sample_urls_for_range(master_df, start_dt, end_dt, samples_per_day)

    if not targets:
        log.error("No files to download. Exiting.")
        return pd.DataFrame()

    # Filter out already-downloaded files
    targets = [t for t in targets if t["timestamp_str"] not in existing_timestamps]
    log.info("%d files to download after deduplication", len(targets))

    all_frames = []
    for i, target in enumerate(targets, 1):
        log.info("[%d/%d] Downloading %s", i, len(targets), target["timestamp_str"])
        df = _download_and_filter(target["url"], target["timestamp_str"])
        if not df.empty:
            all_frames.append(df)
            log.info("  → %d kinetic events retained", len(df))
        else:
            log.info("  → 0 events (no matching records in this file)")

        # Polite delay — GDELT asks for reasonable crawl rates
        sleep(0.5)

    if not all_frames:
        log.warning("No kinetic events found in any downloaded file.")
        return pd.DataFrame()

    new_data = pd.concat(all_frames, ignore_index=True)
    log.info("Total new events fetched: %d", len(new_data))

    # Append to CSV or write fresh
    if append and OUTPUT_CSV.exists():
        new_data.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        log.info("Appended to %s", OUTPUT_CSV)
    else:
        new_data.to_csv(OUTPUT_CSV, index=False)
        log.info("Wrote %s", OUTPUT_CSV)

    return new_data


def run_realtime() -> pd.DataFrame:
    """
    Real-time mode: fetch the latest GDELT 15-minute export file only.
    Used by the polling loop (Step 8a) every 15 minutes.

    Returns:
        DataFrame of kinetic events in the latest 15-minute window.
    """
    log.info("Realtime mode: fetching latest GDELT update...")

    resp = requests.get(GDELT_LASTUPDATE, timeout=30)
    resp.raise_for_status()

    # lastupdate.txt has 3 lines: export, mentions, GKG — we want export
    export_url = None
    for line in resp.text.strip().split("\n"):
        parts = line.strip().split(" ")
        if len(parts) == 3 and "export.CSV.zip" in parts[2]:
            export_url = parts[2]
            break

    if not export_url:
        log.error("Could not find export URL in lastupdate.txt")
        return pd.DataFrame()

    ts_str = export_url.split("/")[-1].split(".")[0]
    log.info("Latest file: %s", ts_str)

    df = _download_and_filter(export_url, ts_str)

    if df.empty:
        log.info("No kinetic events in latest 15-minute window.")
        return pd.DataFrame()

    log.info("%d kinetic events in latest window", len(df))

    # Append to CSV
    write_header = not OUTPUT_CSV.exists()
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
    log.info("Appended %d rows to %s", len(df), OUTPUT_CSV)

    return df


def run_density_test(date: str) -> dict:
    """
    Full-density test mode: download ALL 96 files for a single day.
    Used in notebooks/gdelt_kinetic_analysis.ipynb to test whether
    4 samples/day is sufficient vs full 96 files/day.

    Args:
        date: ISO date string e.g. "2026-03-15"

    Returns:
        Dict with keys "full_density_df", "sampled_df" for notebook comparison.
    """
    target_dt = datetime.strptime(date, "%Y-%m-%d")
    end_dt    = target_dt + timedelta(days=1)

    log.info("Density test mode for %s — downloading all 96 files...", date)
    master_df = _get_master_file_index()

    # Full density: all files for the day
    all_targets = _sample_urls_for_range(master_df, target_dt, end_dt, samples_per_day=96)

    full_frames = []
    for i, t in enumerate(all_targets, 1):
        log.info("[%d/%d] %s", i, len(all_targets), t["timestamp_str"])
        df = _download_and_filter(t["url"], t["timestamp_str"])
        if not df.empty:
            full_frames.append(df)
        sleep(0.5)

    full_df    = pd.concat(full_frames, ignore_index=True) if full_frames else pd.DataFrame()
    full_path  = DATA_DIR / f"gdelt_density_full_{date}.csv"
    full_df.to_csv(full_path, index=False)
    log.info("Full density: %d events → %s", len(full_df), full_path)

    # 4-sample version for comparison
    sampled_targets = _sample_urls_for_range(master_df, target_dt, end_dt, samples_per_day=4)
    sampled_frames  = []
    for t in sampled_targets:
        df = _download_and_filter(t["url"], t["timestamp_str"])
        if not df.empty:
            sampled_frames.append(df)
        sleep(0.5)

    sampled_df   = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else pd.DataFrame()
    sampled_path = DATA_DIR / f"gdelt_density_sampled_{date}.csv"
    sampled_df.to_csv(sampled_path, index=False)
    log.info("4-sample: %d events → %s", len(sampled_df), sampled_path)

    coverage = len(sampled_df) / len(full_df) * 100 if len(full_df) > 0 else 0
    log.info("Coverage: %.1f%% of full-density events captured by 4-sample approach", coverage)

    return {"full_density_df": full_df, "sampled_df": sampled_df}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GDELT Kinetic Events ingestion — West Asia War 2026"
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "realtime", "density_test"],
        required=True,
        help="historical: backfill from start date | realtime: latest 15-min window | density_test: sampling assumption test",
    )
    parser.add_argument(
        "--start",
        default="2026-02-01",
        help="Start date for historical mode (ISO format, default: 2026-02-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for historical mode (ISO format, default: today)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Files per day for historical mode (default: 4)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Single date for density_test mode (ISO format)",
    )
    args = parser.parse_args()

    if args.mode == "historical":
        run_historical(
            start_date=args.start,
            end_date=args.end,
            samples_per_day=args.samples,
        )
    elif args.mode == "realtime":
        run_realtime()
    elif args.mode == "density_test":
        if not args.date:
            parser.error("--date is required for density_test mode")
        run_density_test(date=args.date)