import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DATA_PATH = Path("data/master_df.csv")
SCALER_PATH = Path("models/scaler.pkl")
TRAIN_CUTOFF = pd.Timestamp("2026-03-31")

MODEL_FEATURES = [
    "firms_frp_mean", "firms_brightness_mean", "firms_anomaly_count",
    "gdelt_event_count", "gdelt_avg_goldstein", "gdelt_total_mentions", "gdelt_avg_tone",
    "brent_crude_change", "vix_change", "usd_ils_change", "gold_change",
    "distilbert_avg", "hostile_weight", "diplomatic_weight",
    "bloc_divergence", "military_diplomatic_gap",
]

def run_normalization():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Dynamically grab the engineered lags from Step 11
    lag_cols = [c for c in df.columns if '_lag' in c]
    features_to_scale = MODEL_FEATURES + lag_cols
    
    train_df = df[df['date'] <= TRAIN_CUTOFF]
    fit_data = train_df[features_to_scale].dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(fit_data)
    
    nan_mask = df[features_to_scale].isna()
    transformed_array = scaler.transform(df[features_to_scale].fillna(0))
    transformed_df = pd.DataFrame(transformed_array, columns=features_to_scale, index=df.index)
    
    transformed_df[nan_mask] = np.nan
    df[features_to_scale] = transformed_df
    
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    df.to_csv(DATA_PATH, index=False)
    logging.info(f"Step 10 Complete: Scaler fitted on {len(features_to_scale)} features (including lags).")

if __name__ == "__main__":
    run_normalization()