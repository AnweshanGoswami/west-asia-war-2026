import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DATA_PATH = Path("data/master_df.csv")
PCA_PATH = Path("models/pca.pkl")
TRAIN_CUTOFF = pd.Timestamp("2026-03-31")

# Must match normalizer
MODEL_FEATURES = [
    "firms_frp_mean", "firms_brightness_mean", "firms_anomaly_count",
    "gdelt_event_count", "gdelt_avg_goldstein", "gdelt_total_mentions", "gdelt_avg_tone",
    "brent_crude_change", "vix_change", "usd_ils_change", "gold_change",
    "distilbert_avg", "hostile_weight", "diplomatic_weight",
    "bloc_divergence", "military_diplomatic_gap",
]

def run_pca():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Strictly define PCA features to ONLY include scaled data
    lag_cols = [c for c in df.columns if '_lag' in c]
    pca_features = MODEL_FEATURES + lag_cols
    
    train_df = df[df['date'] <= TRAIN_CUTOFF]
    fit_data = train_df[pca_features].dropna()
    
    # CHANGE THIS:
    pca = PCA(n_components=6)
    pca.fit(fit_data)
    
    explained_var = pca.explained_variance_ratio_
    cum_var = explained_var.cumsum()
    logging.info(f"Scree Plot: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, PC3={explained_var[2]:.2%}, PC4={explained_var[3]:.2%}, PC5={explained_var[4]:.2%}, PC6={explained_var[5]:.2%}")
    logging.info(f"Total Variance Captured by 6 PCs: {cum_var[-1]:.2%}")
    
    nan_mask = df[pca_features].isna().any(axis=1)
    temp_filled = df[pca_features].fillna(0)
    
    pc_array = pca.transform(temp_filled)
    # UPDATE THIS LOOP to 6:
    for i in range(6):
        df[f'PC{i+1}'] = pc_array[:, i]
        df.loc[nan_mask, f'PC{i+1}'] = np.nan
        
    PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)
        
    df.to_csv(DATA_PATH, index=False)
    logging.info(f"Step 13 Complete: PCA applied and model saved to {PCA_PATH}")

if __name__ == "__main__":
    run_pca()