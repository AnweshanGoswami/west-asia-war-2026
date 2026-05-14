"""
src/data_merger.py
────────────────────────────────────────────────────────────────────────────────
Data Merging — Step 9
West Asia War 2026 Conflict Prediction Engine

Joins all 4 data sources on `date` into single master DataFrame.

KEY OPERATIONS
──────────────
  1. Spatial Anchoring via BallTree (100km geocoding error budget)
     Cross-references GDELT events against NASA FIRMS thermal anomalies
     within a ±2 day temporal window. Logs exact match distance.
     IRAN EXCEPTION: IR events outside 100km NOT dropped — marked
     'iran_unverified' for downstream uncertainty scaling in Phase 3.

  2. GDELT kinetic 6-day lag correction
     Narrative reports trail physical thermal detections by 6 days.
     Applied before spatial anchoring so causality is preserved.

  3. Weekend forward-fill
     Economic signals forward-filled from Friday close.

  4. Missing data flagging
     Missing days flagged for uncertainty propagation in Lanchester ODEs.

OUTPUT
──────
  data/master_df.csv — one row per day, Feb 01 2026 → yesterday
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
FIRMS_CSV     = DATA_DIR / "firms_compiled.csv"      # compiled physical layer
ECONOMIC_CSV  = DATA_DIR / "economic_raw.csv"
GDELT_KIN_CSV = DATA_DIR / "gdelt_kinetic_raw.csv"
OUTPUT_CSV    = DATA_DIR / "master_df.csv"

SENTIMENT_FILES = [
    DATA_DIR / "gdelt_sentiment_daily.csv",
    DATA_DIR / "outbreak_patch.csv",
    DATA_DIR / "sentiment_realtime.csv",
]

# ── Config ────────────────────────────────────────────────────────────────────
GDELT_LAG_DAYS   = 6
ANCHOR_RADIUS_KM = 100.0       # geocoding error budget for West Asia theater
EARTH_RADIUS_KM  = 6371.0
BACKFILL_START   = "2026-02-01"

# Historically unbackfilled — all 107 rows null; excluded from model features
NULL_SENTIMENT_COLS = ["gdelt_tone_avg", "gdelt_tone_norm", "signal_divergence"]

# Explicit GDELT V1 column map — prevents silent row-swallowing header bug
GDELT_COLS = [
    "date", "event_id", "event_root_code", "cameo_code",
    "goldstein_scale", "num_mentions", "num_sources", "avg_tone",
    "action_lat", "action_lon", "action_country_code", "action_geo_fullname",
    "actor1_country", "actor2_country", "source_url", "date_added",
]

# Model features — S&P 500 excluded; gdelt_avg_anchor_dist diagnostic only
MODEL_FEATURES = [
    "firms_frp_mean",
    "firms_brightness_mean",
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
    "bloc_divergence",
    "military_diplomatic_gap",
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_firms() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load firms_compiled.csv.
    Returns:
        raw_df   — unaggregated rows for BallTree spatial anchoring
        daily_df — daily aggregates for master merge
    """
    if not FIRMS_CSV.exists():
        log.warning("firms_compiled.csv not found — run firms_compiler.py first")
        return pd.DataFrame(), pd.DataFrame()

    raw_df = pd.read_csv(FIRMS_CSV)
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.date
    raw_df = raw_df.dropna(subset=["latitude", "longitude", "date"])

    # Safety net: reject any pre-Feb rows if old compiled file used
    raw_df = raw_df[pd.to_datetime(raw_df["date"]) >= pd.Timestamp(BACKFILL_START)]

    daily_df = (
        raw_df.groupby("date")
        .agg(
            firms_frp_mean       =("frp",               "mean"),
            firms_brightness_mean=("unified_brightness", "mean"),
            firms_anomaly_count  =("date",               "count"),
        )
        .reset_index()
    )
    log.info("FIRMS loaded: %d anomalies → %d daily aggregates", len(raw_df), len(daily_df))
    return raw_df, daily_df


def _load_and_anchor_gdelt(raw_firms: pd.DataFrame) -> pd.DataFrame:
    """
    Load GDELT kinetic CSV, apply 6-day lag, anchor via BallTree.
    Iran exception: IR events beyond 100km kept as 'iran_unverified'.
    gdelt_avg_anchor_dist retained as diagnostic column (not a model feature).
    """
    if not GDELT_KIN_CSV.exists():
        log.warning("gdelt_kinetic_raw.csv not found")
        return pd.DataFrame()

    # header=None + explicit names prevents first-row-as-header silent kill
    gdelt_df = pd.read_csv(
        GDELT_KIN_CSV,
        header=None,
        names=GDELT_COLS,
        low_memory=False,
    )

    # Enforce numeric geometry — coerce bad geocodes to NaN then drop
    gdelt_df["action_lat"] = pd.to_numeric(gdelt_df["action_lat"], errors="coerce")
    gdelt_df["action_lon"] = pd.to_numeric(gdelt_df["action_lon"], errors="coerce")
    gdelt_df = gdelt_df.dropna(subset=["event_id", "action_lat", "action_lon", "date"])

    if gdelt_df.empty or raw_firms.empty:
        log.warning("Cannot anchor: GDELT or FIRMS empty")
        return pd.DataFrame()

    # Apply 6-day lag before anchoring — causality preserved
    gdelt_df["date"] = (
        pd.to_datetime(gdelt_df["date"]) - pd.Timedelta(days=GDELT_LAG_DAYS)
    ).dt.date

    log.info(
        "Anchoring %d GDELT events (100km radius, ±2 day window, 6-day lag)...",
        len(gdelt_df),
    )

    anchored_indices   = []
    distances_dict     = {}
    anchor_status_dict = {}

    max_dist_rad  = ANCHOR_RADIUS_KM / EARTH_RADIUS_KM
    firms_by_date = dict(list(raw_firms.groupby("date")))

    for current_date, g_group in gdelt_df.groupby("date"):

        # Gather FIRMS fires within ±2 days
        temporal_firms = [
            firms_by_date[current_date + timedelta(days=d)]
            for d in range(-2, 3)
            if (current_date + timedelta(days=d)) in firms_by_date
        ]

        # No fires anywhere in window — only IR events survive
        if not temporal_firms:
            for idx, row in g_group.iterrows():
                if row.get("action_country_code") == "IR":
                    anchored_indices.append(idx)
                    anchor_status_dict[idx] = "iran_unverified"
                    distances_dict[idx]     = np.nan
            continue

        t_firms_df = pd.concat(temporal_firms, ignore_index=True)
        firms_rad  = np.deg2rad(t_firms_df[["latitude", "longitude"]].values)
        tree       = BallTree(firms_rad, metric="haversine")

        g_rad      = np.deg2rad(g_group[["action_lat", "action_lon"]].values)
        distances, _ = tree.query(g_rad, k=1)  # nearest-neighbor distance

        for idx_pos, (idx, row) in enumerate(g_group.iterrows()):
            dist_km = distances[idx_pos][0] * EARTH_RADIUS_KM

            if dist_km <= ANCHOR_RADIUS_KM:
                anchored_indices.append(idx)
                anchor_status_dict[idx] = "verified"
                distances_dict[idx]     = dist_km
            elif row.get("action_country_code") == "IR":
                anchored_indices.append(idx)
                anchor_status_dict[idx] = "iran_unverified"
                distances_dict[idx]     = dist_km
            # else: discard — no physical corroboration, not Iran

    anchored_gdelt = gdelt_df.loc[anchored_indices].copy()
    anchored_gdelt["gdelt_anchor_dist_km"] = anchored_gdelt.index.map(distances_dict)
    anchored_gdelt["anchor_status"]        = anchored_gdelt.index.map(anchor_status_dict)

    verified   = (anchored_gdelt["anchor_status"] == "verified").sum()
    iran_unver = (anchored_gdelt["anchor_status"] == "iran_unverified").sum()
    log.info(
        "Anchored %d / %d GDELT events (verified=%d, iran_unverified=%d, discarded=%d)",
        len(anchored_gdelt), len(gdelt_df), verified, iran_unver,
        len(gdelt_df) - len(anchored_gdelt),
    )

    daily = (
        anchored_gdelt.groupby("date")
        .agg(
            gdelt_event_count    =("event_id",             "count"),
            gdelt_avg_goldstein  =("goldstein_scale",      "mean"),
            gdelt_total_mentions =("num_mentions",         "sum"),
            gdelt_avg_tone       =("avg_tone",             "mean"),
            gdelt_avg_anchor_dist=("gdelt_anchor_dist_km", "mean"),  # diagnostic
        )
        .reset_index()
    )
    return daily


def _load_economic() -> pd.DataFrame:
    """
    Load economic_raw.csv.
    Weekend gaps forward-filled from Friday close.
    Absolute prices dropped post-differencing (anti-leakage).
    SP500 walled off as drawdown_pct display-only.
    """
    if not ECONOMIC_CSV.exists():
        log.warning("economic_raw.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(ECONOMIC_CSV)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.drop(columns=["Date"], errors="ignore").sort_values("date")

    # Reindex to continuous daily spine to expose weekend gaps
    df.set_index("date", inplace=True)
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D").date
    df = df.reindex(all_days).ffill().reset_index()
    df.rename(columns={"index": "date"}, inplace=True)

    # First differences — Sat/Sun diffs = 0, Mon diff = Mon - Fri
    for raw_col, diff_col in [
        ("Brent_Crude", "brent_crude_change"),
        ("VIX",         "vix_change"),
        ("USD_ILS",     "usd_ils_change"),
        ("Gold",        "gold_change"),
    ]:
        if raw_col in df.columns:
            df[diff_col] = df[raw_col].diff()

    # SP500: drawdown from Feb 27 peak, then drop absolute price
    if "SP500" in df.columns:
        peak_date = pd.to_datetime("2026-02-27").date()
        peak_mask = df["date"] <= peak_date
        if peak_mask.any():
            sp500_peak = df.loc[peak_mask, "SP500"].iloc[-1]
            df["sp500_drawdown_pct"] = (df["SP500"] - sp500_peak) / sp500_peak * 100
        df = df.drop(columns=["SP500"], errors="ignore")

    # Drop absolute prices — anti-leakage guard for Phase 3 stationarity
    df = df.drop(columns=["Brent_Crude", "VIX", "USD_ILS", "Gold"], errors="ignore")

    log.info("Economic signals loaded: %d days (weekend-filled)", len(df))
    return df


def _load_sentiment() -> pd.DataFrame:
    """
    Stack all sentiment shards, purge historically-null columns, deduplicate.
    Shards: gdelt_sentiment_daily + outbreak_patch + sentiment_realtime.
    Richest row wins on duplicate dates.
    """
    dfs = [pd.read_csv(p) for p in SENTIMENT_FILES if p.exists()]
    if not dfs:
        log.warning("No sentiment files found")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Drop columns that are 100% null in historical backfill
    # (gdelt_tone_avg, gdelt_tone_norm, signal_divergence never backfilled)
    df = df.drop(
        columns=[c for c in NULL_SENTIMENT_COLS if c in df.columns],
        errors="ignore",
    )

    # Richest-row dedup: prefer row with most non-null values per date
    df["_valid_count"] = df.notna().sum(axis=1)
    df = (
        df.sort_values(["date", "_valid_count"], ascending=[True, False])
        .drop_duplicates(subset=["date"], keep="first")
        .drop(columns=["_valid_count"])
    )

    log.info("Sentiment stacked: %d days from %d shard(s)", len(df), len(dfs))
    return df


# ── Master merge ──────────────────────────────────────────────────────────────

def build_master_df(
    start_date: str = BACKFILL_START,
    end_date:   str = None,
) -> pd.DataFrame:
    """
    Joins all sources onto a continuous daily spine.
    Missing days flagged for uncertainty propagation, not dropped.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date
        else (datetime.utcnow() - timedelta(days=1)).date()
    )

    log.info("Building master timeline: %s → %s", start_dt, end_dt)
    master = pd.DataFrame(
        {"date": pd.date_range(str(start_dt), str(end_dt), freq="D").date}
    )

    raw_firms, daily_firms = _load_firms()

    sources = [
        (daily_firms,                        "FIRMS"),
        (_load_and_anchor_gdelt(raw_firms),  "GDELT kinetic"),
        (_load_economic(),                   "Economic"),
        (_load_sentiment(),                  "Sentiment"),
    ]

    for df, label in sources:
        if not df.empty:
            master = master.merge(df, on="date", how="left")
            log.info("Merged: %s", label)
        else:
            log.warning("Skipped (empty): %s", label)

    # Uncertainty propagation flags for Lanchester ODEs
    master["firms_data_missing"]     = master["firms_frp_mean"].isna()
    master["gdelt_data_missing"]     = master["gdelt_event_count"].isna()
    master["sentiment_data_missing"] = master["distilbert_avg"].isna()
    master["economic_data_missing"]  = master["brent_crude_change"].isna()

    master.to_csv(OUTPUT_CSV, index=False)
    log.info(
        "Master DF saved → %s  (%d rows × %d cols)",
        OUTPUT_CSV, len(master), len(master.columns),
    )

    # Quick sanity report
    print("\n── Missing data summary ──")
    for col in ["firms_data_missing", "gdelt_data_missing",
                "sentiment_data_missing", "economic_data_missing"]:
        if col in master.columns:
            print(f"  {col}: {master[col].sum()} days")

    return master


if __name__ == "__main__":
    build_master_df()