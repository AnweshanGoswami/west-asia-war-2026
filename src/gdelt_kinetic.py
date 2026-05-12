"""
src/gdelt_kinetic.py
────────────────────────────────────────────────────────────────────────────────
GDELT Kinetic Events Layer
West Asia War 2026 Conflict Prediction Engine

ROLE IN ARCHITECTURE
────────────────────
Narrative half of the Dual-Signal Veto System.
Provides WHO fired at WHOM and WHERE (from news articles via CAMEO codes).
Cannot trigger a Kinetic Shock alone — must cross-validate with NASA FIRMS
thermal anomalies (the physical proof layer) in Step 12.

ACLED NOTE
──────────
ACLED was the original source for this layer. Deprecated due to enterprise
paywalls blocking real-time crisis data. GDELT is the open-access replacement.

DATA SOURCES — TWO GDELT VERSIONS
───────────────────────────────────
  Historical mode  →  GDELT v1 Daily Export
    URL: data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
    One complete file per calendar day. Available ~6am UTC the following day.
    100% event coverage — no sampling assumption required.

  Realtime mode    →  GDELT v2 15-Minute Export
    URL: data.gdeltproject.org/gdeltv2/lastupdate.txt
    Updates every 15 minutes. Used by the Step 8a polling loop.

  This split eliminates the sampling assumption entirely from historical data
  while preserving real-time capability for production polling.

CAMEO CODES INGESTED
────────────────────
  18 — ASSAULT        (armed attacks, bombings, shelling)
  19 — FIGHT          (armed clashes, firefights)
  20 — USE UNCONVENTIONAL MASS VIOLENCE (WMD use, massacres)

KNOWN LIMITATIONS (tested in notebooks/gdelt_kinetic_analysis.ipynb)
──────────────────────────────────────────────────────────────────────
  1. Spatial bias   : GDELT defaults unknown locations to country capital.
                      Corrected in Step 9 via Spatial Anchoring against FIRMS.
  2. No casualties  : Lanchester k-coefficients approximated via Negative
                      Binomial distributions mapped from CAMEO codes (Step 14).
  3. Reporting lag  : GDELT ingests news — events appear 1–6 hours after
                      occurrence. Lag measured formally in analysis notebook.

OUTPUT
──────
  data/gdelt_kinetic_raw.csv
    date, event_id, cameo_code, cameo_root, goldstein_scale,
    num_mentions, num_articles, avg_tone,
    action_lat, action_lon, action_country, action_fullname,
    actor1_country, actor2_country, source_url, gdelt_file_date

USAGE
─────
  # Historical backfill (Feb 1 2026 → yesterday)
  python src/gdelt_kinetic.py --mode historical --start 2026-02-01

  # Resume interrupted backfill (skips already-downloaded dates)
  python src/gdelt_kinetic.py --mode historical --start 2026-02-01 --append

  # Historical with explicit end date
  python src/gdelt_kinetic.py --mode historical --start 2026-02-01 --end 2026-04-01

  # Real-time (latest 15-minute window — called by polling loop)
  python src/gdelt_kinetic.py --mode realtime
"""

import io
import logging
import argparse
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = DATA_DIR / "gdelt_kinetic_raw.csv"

# ── GDELT endpoints ───────────────────────────────────────────────────────────
GDELT_V1_DAILY      = "http://data.gdeltproject.org/events/{date}.export.CSV.zip"
GDELT_V2_LASTUPDATE = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# ── Filter config ─────────────────────────────────────────────────────────────
CAMEO_KINETIC_ROOTS = {"18", "19", "20"}

# FIPS 10-4 country codes for the conflict region
CONFLICT_COUNTRIES = {
    "IR",   # Iran
    "IS",   # Israel
    "LE",   # Lebanon
    "SY",   # Syria
    "IZ",   # Iraq
    "YM",   # Yemen
    "SA",   # Saudi Arabia
    "AE",   # UAE
    "JO",   # Jordan
    "WE",   # West Bank
    "GZ",   # Gaza
}

# ── GDELT column schema (58 cols, tab-separated, no header row) ───────────────
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


# ── Core helpers ──────────────────────────────────────────────────────────────

def _parse_and_filter(raw_bytes: bytes, file_label: str) -> pd.DataFrame:
    """
    Unzip raw bytes from any GDELT export file, apply CAMEO + country filters,
    and return a clean DataFrame. Works for both v1 daily and v2 15-min files.
    Returns empty DataFrame on any failure — caller continues to next file.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
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
        log.warning("Parse failed [%s]: %s", file_label, e)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Filter 1: kinetic CAMEO root codes only
    df = df[df["EventRootCode"].isin(CAMEO_KINETIC_ROOTS)]
    if df.empty:
        return pd.DataFrame()

    # Filter 2: conflict region by action geography
    df = df[df["ActionGeo_CountryCode"].isin(CONFLICT_COUNTRIES)]
    if df.empty:
        return pd.DataFrame()

    # Reshape to clean output schema
    out = pd.DataFrame({
        "date":            pd.to_datetime(
                               df["SQLDATE"], format="%Y%m%d", errors="coerce"
                           ).dt.date,
        "event_id":        df["GlobalEventID"].values,
        "cameo_code":      df["EventCode"].values,
        "cameo_root":      df["EventRootCode"].values,
        "goldstein_scale": pd.to_numeric(df["GoldsteinScale"], errors="coerce"),
        "num_mentions":    pd.to_numeric(df["NumMentions"],    errors="coerce"),
        "num_articles":    pd.to_numeric(df["NumArticles"],    errors="coerce"),
        "avg_tone":        pd.to_numeric(df["AvgTone"],        errors="coerce"),
        "action_lat":      pd.to_numeric(df["ActionGeo_Lat"],  errors="coerce"),
        "action_lon":      pd.to_numeric(df["ActionGeo_Long"], errors="coerce"),
        "action_country":  df["ActionGeo_CountryCode"].values,
        "action_fullname": df["ActionGeo_FullName"].values,
        "actor1_country":  df["Actor1CountryCode"].values,
        "actor2_country":  df["Actor2CountryCode"].values,
        "source_url":      df["SOURCEURL"].values,
        "gdelt_file_date": file_label,
    })

    return out.dropna(subset=["date"])


def _download(url: str, timeout_sec: int = 60) -> bytes | None:
    """Download URL, return raw bytes. Returns None on any failure."""
    try:
        # Fixed keyword argument mapping to prevent namespace collision
        resp = requests.get(url, timeout=timeout_sec)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        log.warning("Download failed [%s]: %s", url, e)
        return None


def _load_existing_labels() -> set:
    """
    Return set of gdelt_file_date strings already saved to OUTPUT_CSV.
    Used to skip dates already downloaded in both historical and realtime modes.
    """
    if not OUTPUT_CSV.exists():
        return set()
    try:
        existing = pd.read_csv(OUTPUT_CSV, usecols=["gdelt_file_date"])
        return set(existing["gdelt_file_date"].astype(str))
    except Exception:
        return set()


def _write(df: pd.DataFrame) -> None:
    """Append DataFrame to OUTPUT_CSV. Writes header only if file is new."""
    write_header = not OUTPUT_CSV.exists()
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)


# ── Public interface ──────────────────────────────────────────────────────────

def run_historical(
    start_date: str = "2026-02-01",
    end_date:   str = None,
    append:     bool = True,
) -> pd.DataFrame:
    """Historical backfill logic remains unchanged."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date
        else (datetime.utcnow() - timedelta(days=1)).date()
    )

    log.info("Historical mode (GDELT v1 daily): %s → %s", start_dt, end_dt)

    existing_labels = _load_existing_labels() if append else set()
    if existing_labels:
        log.info("Resuming — %d date labels already in CSV", len(existing_labels))

    date_range = pd.date_range(str(start_dt), str(end_dt), freq="D")
    targets    = [
        d.strftime("%Y%m%d") for d in date_range
        if d.strftime("%Y%m%d") not in existing_labels
    ]

    skipped = len(date_range) - len(targets)
    log.info("%d days to download (%d already present, skipped)",
             len(targets), skipped)

    if not targets:
        log.info("Nothing to download — all dates already present.")
        return pd.DataFrame()

    all_frames = []

    for i, date_str in enumerate(targets, 1):
        url = GDELT_V1_DAILY.format(date=date_str)
        log.info("[%d/%d]  %s", i, len(targets), date_str)

        # Updated to use timeout_sec
        raw = _download(url, timeout_sec=30)

        if raw is None:
            log.info("  → skipped (file unavailable — may not be published yet)")
            sleep(1)
            continue

        df = _parse_and_filter(raw, file_label=date_str)

        if df.empty:
            log.info("  → 0 kinetic events in conflict region")
            sentinel = pd.DataFrame([{
                "date": date_str, "event_id": None, "cameo_code": None,
                "cameo_root": None, "goldstein_scale": None,
                "num_mentions": None, "num_articles": None, "avg_tone": None,
                "action_lat": None, "action_lon": None,
                "action_country": None, "action_fullname": None,
                "actor1_country": None, "actor2_country": None,
                "source_url": None, "gdelt_file_date": date_str,
            }])
            _write(sentinel)
        else:
            log.info("  → %d kinetic events retained", len(df))
            all_frames.append(df)
            _write(df)

        sleep(1)

    if not all_frames:
        log.info("Backfill complete — no new kinetic events found.")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    log.info("Backfill complete. %d new events → %s", len(result), OUTPUT_CSV)
    return result


def run_realtime() -> dict:
    """
    Standardized entry point for the master polling loop.
    Fetches only the latest 15-minute window from lastupdate.txt.
    Called by the Step 8a polling loop (src/data_collector.py) every 15 min.

    Returns:
        Standardized dictionary receipt for the orchestrator.
    """
    log.info("Realtime mode (GDELT v2): fetching latest 15-minute window...")
    
    try:
        # Updated to use timeout_sec
        raw_index = _download(GDELT_V2_LASTUPDATE, timeout_sec=15)
        if raw_index is None:
            log.error("Could not reach GDELT v2 lastupdate.txt")
            return {"status": "failed", "records": 0}

        # lastupdate.txt: 3 lines — export, mentions, GKG. We want export.
        export_url = None
        for line in raw_index.decode("utf-8").strip().split("\n"):
            parts = line.strip().split(" ")
            if len(parts) == 3 and "export.CSV.zip" in parts[2]:
                export_url = parts[2]
                break

        if not export_url:
            log.error("Export URL not found in lastupdate.txt")
            return {"status": "failed", "records": 0}

        # File timestamp from filename e.g. "20260512143000.export.CSV.zip"
        file_ts = export_url.split("/")[-1].split(".")[0]
        log.info("Latest v2 file: %s", file_ts)

        # Skip if already processed
        if file_ts in _load_existing_labels():
            log.info("Already processed — no new data in this 15-minute window.")
            return {"status": "empty", "records": 0}

        # Updated to use timeout_sec
        raw = _download(export_url, timeout_sec=60)
        if raw is None:
            return {"status": "failed", "records": 0}

        df = _parse_and_filter(raw, file_label=file_ts)

        if df.empty:
            log.info("No kinetic events in latest window.")
            return {"status": "empty", "records": 0}

        log.info("%d kinetic events in latest window → appended to %s",
                 len(df), OUTPUT_CSV)
        _write(df)
        
        return {"status": "success", "records": len(df), "file": str(OUTPUT_CSV)}
        
    except Exception as e:
        log.error("Error during GDELT realtime poll: %s", e)
        return {"status": "failed", "records": 0}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GDELT Kinetic Events ingestion — West Asia War 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python src/gdelt_kinetic.py --mode historical --start 2026-02-01
  python src/gdelt_kinetic.py --mode historical --start 2026-02-01 --end 2026-04-01
  python src/gdelt_kinetic.py --mode realtime
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "realtime"],
        required=True,
        help="historical: v1 daily backfill | realtime: v2 15-min polling",
    )
    parser.add_argument(
        "--start",
        default="2026-02-01",
        help="Start date for historical mode (YYYY-MM-DD, default: 2026-02-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for historical mode (YYYY-MM-DD, default: yesterday)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        default=False,
        help="Re-download all dates even if already present (default: skip existing)",
    )
    args = parser.parse_args()

    if args.mode == "historical":
        run_historical(
            start_date=args.start,
            end_date=args.end,
            append=not args.no_append,
        )
    elif args.mode == "realtime":
        result = run_realtime()
        print(f"\nRealtime execution receipt: {result}")