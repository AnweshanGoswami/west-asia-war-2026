"""
data_merger.py - Step 9: Data Merging
Creates master_df by joining all 4 data layers on date.

Key Operations:
1. GDELT 6-day lag correction (kinetic + sentiment)
2. FIRMS daily aggregation (12 features)
3. GDELT kinetic daily aggregation (6 features)
4. Economic data merge (raw + realtime)
5. Sentiment data merge (daily + patch + realtime)
6. Missing data flagging
7. Date range: 2026-02-08 → 2026-05-11

Author: Anweshan Goswami
Date: 2026-05-16
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "master_df.csv"

# Date range for master dataframe (after 6-day GDELT lag correction)
START_DATE = "2026-02-08"  # Start after lag correction buffer
END_DATE = "2026-05-11"    # Latest date with FIRMS/Economic coverage

# GDELT REPORTING LAG CORRECTION
# Finding: GDELT news trails FIRMS thermal detection by 6 days
# Peak CCF at lag = +6 days → subtract 6 days from GDELT dates
GDELT_LAG_DAYS = 6

print("="*80)
print("DATA MERGER - STEP 9")
print("="*80)

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================

print("\n[1/8] Loading raw datasets...")

# Physical layer: FIRMS thermal detections
firms_raw = pd.read_csv(DATA_DIR / "firms_compiled.csv", low_memory=False)
firms_raw['date'] = pd.to_datetime(firms_raw['date'])
print(f"  ✓ FIRMS: {len(firms_raw):,} detections")

# Narrative layer: GDELT kinetic events (not uploaded, will be loaded from data/)
# Expected columns: date, event_id, cameo_code, goldstein_scale, num_mentions, 
#                   num_articles, avg_tone, action_lat, action_lon
gdelt_kinetic_raw = pd.read_csv(DATA_DIR / "gdelt_kinetic_raw.csv", low_memory=False)
gdelt_kinetic_raw['date'] = pd.to_datetime(gdelt_kinetic_raw['date'], format='%d-%m-%Y')
print(f"  ✓ GDELT Kinetic: {len(gdelt_kinetic_raw):,} events")

# Economic layer
econ_raw = pd.read_csv(DATA_DIR / "economic_raw.csv")
econ_raw['Date'] = pd.to_datetime(econ_raw['Date'])
econ_raw = econ_raw.rename(columns={'Date': 'date'})  # Standardize date column

econ_realtime = pd.read_csv(DATA_DIR / "economic_realtime.csv")
econ_realtime['Date'] = pd.to_datetime(econ_realtime['Date'])
econ_realtime = econ_realtime.rename(columns={'Date': 'date'})
print(f"  ✓ Economic raw: {len(econ_raw)} days")
print(f"  ✓ Economic realtime: {len(econ_realtime)} days")

# Sentiment layer
sent_daily = pd.read_csv(DATA_DIR / "gdelt_sentiment_daily.csv")
sent_daily['date'] = pd.to_datetime(sent_daily['date'])

outbreak_patch = pd.read_csv(DATA_DIR / "outbreak_patch.csv")
outbreak_patch['date'] = pd.to_datetime(outbreak_patch['date'])

sent_realtime = pd.read_csv(DATA_DIR / "sentiment_realtime.csv")
sent_realtime['date'] = pd.to_datetime(sent_realtime['date'])
print(f"  ✓ Sentiment daily: {len(sent_daily)} days")
print(f"  ✓ Outbreak patch: {len(outbreak_patch)} days")
print(f"  ✓ Sentiment realtime: {len(sent_realtime)} days")

# ============================================================================
# 2. HANDLE DUPLICATE DATES IN SENTIMENT
# ============================================================================

print("\n[2/8] Cleaning sentiment duplicates...")

# Keep row with higher article_count (real data vs artifacts)
sent_daily = sent_daily.sort_values(['date', 'article_count'], ascending=[True, False])
sent_daily_dedup = sent_daily.drop_duplicates(subset='date', keep='first')

duplicates_dropped = len(sent_daily) - len(sent_daily_dedup)
print(f"  ✓ Dropped {duplicates_dropped} low-count duplicate dates")
print(f"  ✓ Sentiment daily after dedup: {len(sent_daily_dedup)} days")

# ============================================================================
# 3. APPLY 6-DAY LAG CORRECTION TO GDELT DATA
# ============================================================================

print(f"\n[3/8] Applying {GDELT_LAG_DAYS}-day GDELT lag correction...")
print("  Rationale: GDELT news trails FIRMS thermal detection by 6 days")
print("  → Subtract 6 days from GDELT event dates to align with actual occurrence")

# Kinetic events: shift dates back by 6 days
gdelt_kinetic_raw['date'] = gdelt_kinetic_raw['date'] - pd.Timedelta(days=GDELT_LAG_DAYS)
print(f"  ✓ Kinetic events date range after lag: {gdelt_kinetic_raw['date'].min().date()} → {gdelt_kinetic_raw['date'].max().date()}")

# Sentiment: shift dates back by 6 days
sent_daily_dedup['date'] = sent_daily_dedup['date'] - pd.Timedelta(days=GDELT_LAG_DAYS)
outbreak_patch['date'] = outbreak_patch['date'] - pd.Timedelta(days=GDELT_LAG_DAYS)
sent_realtime['date'] = sent_realtime['date'] - pd.Timedelta(days=GDELT_LAG_DAYS)
print(f"  ✓ Sentiment date range after lag: {sent_daily_dedup['date'].min().date()} → {sent_daily_dedup['date'].max().date()}")

# ============================================================================
# 4. AGGREGATE FIRMS BY DAY
# ============================================================================

print("\n[4/8] Aggregating FIRMS thermal detections by day...")

firms_daily = firms_raw.groupby('date').agg({
    # Count metrics
    'frp': [
        ('firms_detection_count', 'count'),
        ('firms_high_intensity_count', lambda x: (x > 50).sum()),  # FRP > 50 MW
        # Intensity metrics
        ('firms_total_frp', 'sum'),
        ('firms_mean_frp', 'mean'),
        ('firms_max_frp', 'max'),
    ],
    'unified_brightness': [
        ('firms_mean_brightness', 'mean'),
        ('firms_max_brightness', 'max'),
    ],
    # Spatial metrics (for BallTree spatial anchoring later)
    'latitude': [
        ('firms_centroid_lat', 'mean'),
        ('firms_spatial_std_lat', 'std'),
    ],
    'longitude': [
        ('firms_centroid_lon', 'mean'),
        ('firms_spatial_std_lon', 'std'),
    ],
}).reset_index()

# Flatten multi-level column names
firms_daily.columns = [col[1] if col[1] else col[0] for col in firms_daily.columns]
firms_daily = firms_daily.rename(columns={'': 'date'})

# Calculate brightness anomaly (where background data exists)
firms_with_bg = firms_raw[firms_raw['unified_background'].notna()].copy()
firms_with_bg['brightness_delta'] = firms_with_bg['unified_brightness'] - firms_with_bg['unified_background']

brightness_anomaly = firms_with_bg.groupby('date').agg({
    'brightness_delta': [('firms_mean_brightness_delta', 'mean')],
    'unified_background': [('firms_pct_with_background', 'count')]
}).reset_index()
brightness_anomaly.columns = [col[1] if col[1] else col[0] for col in brightness_anomaly.columns]
brightness_anomaly = brightness_anomaly.rename(columns={'': 'date'})

# Calculate percentage with background data
total_detections = firms_raw.groupby('date').size().reset_index(name='total_count')
brightness_anomaly = brightness_anomaly.merge(total_detections, on='date', how='left')
brightness_anomaly['firms_pct_with_background'] = (
    brightness_anomaly['firms_pct_with_background'] / brightness_anomaly['total_count'] * 100
)
brightness_anomaly = brightness_anomaly.drop(columns=['total_count'])

# Merge brightness anomaly into main FIRMS daily
firms_daily = firms_daily.merge(brightness_anomaly, on='date', how='left')

print(f"  ✓ FIRMS aggregated to {len(firms_daily)} days with 12 features")
print(f"    Features: detection_count, high_intensity_count, total_frp, mean_frp,")
print(f"              max_frp, mean_brightness, max_brightness, centroid_lat/lon,")
print(f"              spatial_std_lat/lon, mean_brightness_delta, pct_with_background")

# ============================================================================
# 5. AGGREGATE GDELT KINETIC BY DAY
# ============================================================================

print("\n[5/8] Aggregating GDELT kinetic events by day...")

gdelt_daily = gdelt_kinetic_raw.groupby('date').agg({
    'event_id': [('gdelt_event_count', 'count')],
    'num_mentions': [('gdelt_total_mentions', 'sum')],
    'num_articles': [('gdelt_total_articles', 'sum')],
    'goldstein_scale': [
        ('gdelt_mean_goldstein', 'mean'),
        ('gdelt_min_goldstein', 'min'),  # Most negative = most hostile
    ],
    'avg_tone': [('gdelt_mean_tone', 'mean')],
}).reset_index()

# Flatten column names
gdelt_daily.columns = [col[1] if col[1] else col[0] for col in gdelt_daily.columns]
gdelt_daily = gdelt_daily.rename(columns={'': 'date'})

print(f"  ✓ GDELT kinetic aggregated to {len(gdelt_daily)} days with 6 features")
print(f"    Features: event_count, total_mentions, total_articles,")
print(f"              mean_goldstein, min_goldstein, mean_tone")

# ============================================================================
# 6. MERGE ECONOMIC DATA
# ============================================================================

print("\n[6/8] Merging economic datasets...")

# Strategy: Use raw as base, append realtime dates not in raw
# (economic_realtime has NaN values, so we prioritize raw where overlap exists)

econ_raw_dates = set(econ_raw['date'])
econ_realtime_new = econ_realtime[~econ_realtime['date'].isin(econ_raw_dates)]

econ_merged = pd.concat([econ_raw, econ_realtime_new], ignore_index=True)
econ_merged = econ_merged.sort_values('date').reset_index(drop=True)

print(f"  ✓ Economic raw: {len(econ_raw)} days")
print(f"  ✓ New dates from realtime: {len(econ_realtime_new)} days")
print(f"  ✓ Economic merged: {len(econ_merged)} days ({econ_merged['date'].min().date()} → {econ_merged['date'].max().date()})")

# Note: Weekends already forward-filled in Step 6
print(f"  ✓ Weekends already forward-filled (no gaps in date sequence)")

# ============================================================================
# 7. MERGE SENTIMENT DATA
# ============================================================================

print("\n[7/8] Merging sentiment datasets...")

# Start with daily, apply outbreak patch (overwrite), then realtime (overwrite)
sent_merged = sent_daily_dedup.copy()

# Apply outbreak patch (overwrites Feb 22, Feb 24 after lag correction)
patch_dates = set(outbreak_patch['date'])
sent_merged = sent_merged[~sent_merged['date'].isin(patch_dates)]
sent_merged = pd.concat([sent_merged, outbreak_patch], ignore_index=True)
print(f"  ✓ Applied outbreak patch (overwrote {len(patch_dates)} dates)")

# Apply realtime (overwrites May 5 after lag correction)
realtime_dates = set(sent_realtime['date'])
sent_merged = sent_merged[~sent_merged['date'].isin(realtime_dates)]
sent_merged = pd.concat([sent_merged, sent_realtime], ignore_index=True)
print(f"  ✓ Applied sentiment realtime (overwrote {len(realtime_dates)} dates)")

sent_merged = sent_merged.sort_values('date').reset_index(drop=True)
print(f"  ✓ Sentiment merged: {len(sent_merged)} days ({sent_merged['date'].min().date()} → {sent_merged['date'].max().date()})")

# ============================================================================
# 8. CREATE MASTER DATAFRAME
# ============================================================================

print(f"\n[8/8] Creating master dataframe ({START_DATE} → {END_DATE})...")

# Create full date range
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
master_df = pd.DataFrame({'date': date_range})

print(f"  ✓ Master date range: {len(master_df)} days")

# Left join all datasets
print("  → Joining FIRMS...")
master_df = master_df.merge(firms_daily, on='date', how='left')

print("  → Joining GDELT kinetic...")
master_df = master_df.merge(gdelt_daily, on='date', how='left')

print("  → Joining Economic...")
master_df = master_df.merge(econ_merged, on='date', how='left')

print("  → Joining Sentiment...")
master_df = master_df.merge(sent_merged, on='date', how='left')

# ============================================================================
# CREATE MISSING DATA FLAGS
# ============================================================================

print("\n  → Creating missing data flags...")

# FIRMS missing flag
master_df['firms_data_missing'] = master_df['firms_detection_count'].isna()

# GDELT kinetic missing flag
master_df['gdelt_data_missing'] = master_df['gdelt_event_count'].isna()

# Sentiment missing flag
master_df['sentiment_data_missing'] = master_df['article_count'].isna()

# Economic missing flag (should be rare since weekends forward-filled)
master_df['economic_data_missing'] = master_df['Brent_Crude'].isna()

missing_summary = {
    'FIRMS': master_df['firms_data_missing'].sum(),
    'GDELT kinetic': master_df['gdelt_data_missing'].sum(),
    'Sentiment': master_df['sentiment_data_missing'].sum(),
    'Economic': master_df['economic_data_missing'].sum(),
}

print("\n  Missing data summary:")
for layer, count in missing_summary.items():
    pct = count / len(master_df) * 100
    print(f"    {layer:20s}: {count:3d} days ({pct:5.1f}%)")

# ============================================================================
# SAVE MASTER DATAFRAME
# ============================================================================

print(f"\n  → Saving to {OUTPUT_FILE}...")
master_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "="*80)
print("MASTER DATAFRAME CREATED")
print("="*80)
print(f"\nShape: {master_df.shape}")
print(f"Date range: {master_df['date'].min().date()} → {master_df['date'].max().date()}")
print(f"Total columns: {len(master_df.columns)}")
print(f"\nColumn groups:")
print(f"  - Physical layer (FIRMS): 12 features")
print(f"  - Narrative layer (GDELT): 6 features")
print(f"  - Economic layer: 5 signals (Brent, Gold, USD/ILS, SP500, VIX)")
print(f"  - Sentiment layer: {len([c for c in sent_merged.columns if c != 'date'])} features")
print(f"  - Missing flags: 4 boolean columns")
print(f"\nOutput: {OUTPUT_FILE}")
print("\n✓ Step 9 complete. Ready for Step 10 (Feature Engineering).")
print("="*80)