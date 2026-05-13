"""
src/firms_compiler.py
────────────────────────────────────────────────────────────────────────────────
NASA FIRMS Staging Compiler
Merges MODIS legacy, VIIRS Archive, and VIIRS NRT into a single physical layer.
Cleans up mismatched NASA column headers and removes pre-2026 noise.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def compile_firms_data():
    print("Compiling NASA FIRMS raw files...")
    
    modis_path = DATA_DIR / "firms_raw.csv"
    viirs_archive_path = DATA_DIR / "fire_archive_SV-C2_749347.csv"
    viirs_nrt_path = DATA_DIR / "fire_nrt_SV-C2_749347.csv"

    dfs = []
    
    if modis_path.exists():
        dfs.append(pd.read_csv(modis_path).assign(source_file='raw_mixed'))
        
    if viirs_archive_path.exists():
        archive_df = pd.read_csv(viirs_archive_path).assign(source_file='viirs_archive')
        if 'type' in archive_df.columns:
            archive_df = archive_df.drop(columns=['type']) 
        dfs.append(archive_df)
        
    if viirs_nrt_path.exists():
        dfs.append(pd.read_csv(viirs_nrt_path).assign(source_file='viirs_nrt'))

    if not dfs:
        print(" ✗ No FIRMS data found in data/ directory.")
        return

    raw_df = pd.concat(dfs, ignore_index=True)

    # 1. Standardize Dates & Filter out pre-2026 noise
    date_col = "acq_date" if "acq_date" in raw_df.columns else "date"
    raw_df["date"] = pd.to_datetime(raw_df[date_col], errors="coerce").dt.date
    raw_df = raw_df.dropna(subset=["latitude", "longitude", "date"])
    
    # Strictly limit to 2026 conflict timeline
    raw_df = raw_df[pd.to_datetime(raw_df['date']).dt.year >= 2026]

    # 2. Fix the NASA Column Name Mismatch
    # M-Band uses 'brightness', I-Band uses 'bright_ti4'. Combine them seamlessly.
    if 'bright_ti4' in raw_df.columns and 'brightness' in raw_df.columns:
        raw_df["unified_brightness"] = raw_df["bright_ti4"].combine_first(raw_df["brightness"])
    elif 'brightness' in raw_df.columns:
        raw_df["unified_brightness"] = raw_df["brightness"]
    else:
        raw_df["unified_brightness"] = raw_df["bright_ti4"]

    # 3. Drop the confusing, NaN-filled legacy columns
    cols_to_drop = ['bright_ti4', 'bright_ti5', 'bright_t31', 'brightness', 'acq_date', 'acq_time']
    raw_df = raw_df.drop(columns=[c for c in cols_to_drop if c in raw_df.columns])

    # 4. Spatial-Temporal Deduplication (Prevents double-counting overlapping satellites)
    raw_df = raw_df.sort_values(by=["date", "latitude", "longitude", "frp"], ascending=[True, True, True, False])
    raw_df = raw_df.drop_duplicates(subset=["latitude", "longitude", "date"], keep="first")

    # 5. Export
    out_path = DATA_DIR / "firms_compiled.csv"
    raw_df.to_csv(out_path, index=False)
    
    print(f"\nCompilation successful! Cleaned schema applied.")
    print(f"Total unique 2026 anomalies: {len(raw_df)}")
    print(f"Saved perfectly clean file to: {out_path}")

if __name__ == "__main__":
    compile_firms_data()