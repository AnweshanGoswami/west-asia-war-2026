"""
src/shock_detector.py
────────────────────────────────────────────────────────────────────────────────
Shock Detection — Step 12
Implements the Dual-Signal Veto. Kinetic shocks are only registered when 
physical FIRMS data > 20 FRP and spatially-anchored GDELT events coexist.
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
DATA_PATH = Path("data/master_df.csv")

def detect_shocks():
    df = pd.read_csv(DATA_PATH)
    
    # 1. Dual-Signal Veto Logic
    # Missing FIRMS data explicitly guarantees False (veto fires)
    df['kinetic_shock'] = (
        (df['firms_frp_mean'] > 20) & 
        (df['gdelt_event_count'] > 0) & 
        (df['firms_data_missing'] == False)
    ).fillna(False)

    # 2. Shock Magnitude Calculation
    # Inverse of Goldstein (so hostile/negative events create a positive magnitude score)
    df['shock_magnitude'] = (df['firms_frp_mean'] * df['gdelt_avg_goldstein'] * -1).fillna(0)
    
    # Floor magnitude at 0 (we do not care about highly diplomatic/cooperative events here)
    df.loc[df['shock_magnitude'] < 0, 'shock_magnitude'] = 0
    # NEW: Zero out magnitude when shock not detected (prevent false positives)
    df.loc[df['kinetic_shock'] == False, 'shock_magnitude'] = 0
    
    shocks_detected = df['kinetic_shock'].sum()
    
    df.to_csv(DATA_PATH, index=False)
    logging.info(f"Step 12 Complete: Veto system successfully registered {shocks_detected} verifiable kinetic shocks.")

if __name__ == "__main__":
    detect_shocks()