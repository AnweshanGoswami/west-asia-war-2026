"""
src/firms_compiler.py
────────────────────────────────────────────────────────────────────────────────
NASA FIRMS Staging Compiler
Merges MODIS legacy, VIIRS Archive, and VIIRS NRT into a single physical layer.
Resolves column mismatches and compiles unified brightness and background channels.
Filters strictly to Feb 01 2026+.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def compile_firms_data():
    print("Compiling NASA FIRMS raw files...")

    modis_path         = DATA_DIR / "firms_raw.csv"
    viirs_archive_path = DATA_DIR / "fire_archive_SV-C2_749347.csv"
    viirs_nrt_path     = DATA_DIR / "fire_nrt_SV-C2_749347.csv"

    dfs = []

    if modis_path.exists():
        dfs.append(pd.read_csv(modis_path).assign(source_file="raw_mixed"))

    if viirs_archive_path.exists():
        archive_df = pd.read_csv(viirs_archive_path).assign(source_file="viirs_archive")
        if "type" in archive_df.columns:
            archive_df = archive_df.drop(columns=["type"])
        dfs.append(archive_df)

    if viirs_nrt_path.exists():
        dfs.append(pd.read_csv(viirs_nrt_path).assign(source_file="viirs_nrt"))

    if not dfs:
        print("✗ No FIRMS data found in data/ directory.")
        return

    raw_df = pd.concat(dfs, ignore_index=True)

    # 1. Standardize date column
    date_col = "acq_date" if "acq_date" in raw_df.columns else "date"
    raw_df["date"] = pd.to_datetime(raw_df[date_col], errors="coerce").dt.date
    raw_df = raw_df.dropna(subset=["latitude", "longitude", "date"])

    # 2. Hard filter — Feb 01 2026 is the timeline anchor
    raw_df = raw_df[pd.to_datetime(raw_df["date"]) >= pd.Timestamp("2026-02-01")]

    # 3. Resolve NASA column mismatch
    
    # A. The Primary Fire Channel (~4.0 µm) -> Peak Explosion Temp
    if "bright_ti4" in raw_df.columns and "brightness" in raw_df.columns:
        raw_df["unified_brightness"] = raw_df["bright_ti4"].combine_first(raw_df["brightness"])
    elif "brightness" in raw_df.columns:
        raw_df["unified_brightness"] = raw_df["brightness"]
    else:
        raw_df["unified_brightness"] = raw_df.get("bright_ti4")

    # B. The Background Channel (~11.0 µm) -> Ambient Desert Temp for Cross-Verification
    if "bright_ti5" in raw_df.columns and "bright_t31" in raw_df.columns:
        raw_df["unified_background"] = raw_df["bright_ti5"].combine_first(raw_df["bright_t31"])
    elif "bright_t31" in raw_df.columns:
        raw_df["unified_background"] = raw_df["bright_t31"]
    elif "bright_ti5" in raw_df.columns:
        raw_df["unified_background"] = raw_df["bright_ti5"]
    else:
        raw_df["unified_background"] = None

    # 4. Drop legacy/redundant columns
    cols_to_drop = ["bright_ti4", "bright_ti5", "bright_t31", "brightness", "acq_date", "acq_time"]
    raw_df = raw_df.drop(columns=[c for c in cols_to_drop if c in raw_df.columns])

    # 5. Spatial-temporal dedup — highest FRP wins when satellites overlap
    raw_df = raw_df.sort_values(
        by=["date", "latitude", "longitude", "frp"],
        ascending=[True, True, True, False]
    )
    raw_df = raw_df.drop_duplicates(subset=["latitude", "longitude", "date"], keep="first")

    # 6. Export
    out_path = DATA_DIR / "firms_compiled.csv"
    raw_df.to_csv(out_path, index=False)

    print(f"\n✓ Compilation successful.")
    print(f"  Unique anomalies (Feb 01 2026+): {len(raw_df)}")
    print(f"  Saved → {out_path}")

if __name__ == "__main__":
    compile_firms_data()