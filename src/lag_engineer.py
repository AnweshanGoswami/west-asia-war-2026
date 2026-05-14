"""
src/lag_engineer.py
────────────────────────────────────────────────────────────────────────────────
Lag Engineering — Step 11
West Asia War 2026 Conflict Prediction Engine

Applies temporal lags to narrative and economic signals to align their causal 
impact with the physical ground truth (NASA FIRMS).

Note: The lag periods below are not assumptions. They were empirically verified 
via Cross-Correlation Function (CCF) against the `firms_frp_mean` target variable.
"""

import logging
import pandas as pd
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths & Config ────────────────────────────────────────────────────────────
DATA_PATH = Path("data/master_df.csv")

# Empirically locked lags via CCF Peak Analysis
EMPIRICAL_LAGS = {
    "distilbert_avg": 2,       # Peak CCF (-0.328): Fast narrative deterioration
    "brent_crude_change": 24,  # Peak CCF (+0.226): Long shadow market pricing
    "hostile_weight": 24       # Peak CCF (-0.427): Macro strategic alignment
}

def engineer_lags():
    if not DATA_PATH.exists():
        log.error(f"Cannot find {DATA_PATH}. Please run data_merger.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    initial_cols = len(df.columns)
    
    log.info("Applying empirically verified lags to causal features...")
    
    for col, lag in EMPIRICAL_LAGS.items():
        if col in df.columns:
            # Create new column, preserving the original un-shifted data
            lag_col_name = f"{col}_lag{lag}"
            df[lag_col_name] = df[col].shift(lag)
            log.info(f" ✔ Created {lag_col_name} (Shift: {lag} days)")
        else:
            log.warning(f" ✗ Target column '{col}' missing from DataFrame.")

    # Save the updated DataFrame in place
    df.to_csv(DATA_PATH, index=False)
    
    added_cols = len(df.columns) - initial_cols
    log.info(f"Lag engineering complete. Added {added_cols} new features to {DATA_PATH.name}.")

if __name__ == "__main__":
    engineer_lags()