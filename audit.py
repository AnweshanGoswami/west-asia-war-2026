import pandas as pd

file_path = 'data/gdelt_sentiment_daily.csv'
print("🚀 Launching Ultra Audit: Checking for Data Integrity...")

df = pd.read_csv(file_path)
initial_len = len(df)

# All critical parameters required for Phase 3 Modeling
critical_params = [
    'distilbert_avg', 'hostile_weight', 'diplomatic_weight', 
    'signal_divergence', 'bloc_divergence', 'military_diplomatic_gap',
    'sentiment_military', 'sentiment_diplomatic', 'sentiment_economic',
    'gdelt_tone_norm'
]

# FILTER 1: Any missing (NaN) values in critical model parameters
mask_nan = df[critical_params].isna().any(axis=1)

# FILTER 2: "Thin Data" (Days where < 100 articles were captured due to API failure)
mask_thin = df['article_count'] < 100

# FILTER 3: "Zeroed Data" (Cases where sentiments were captured as 0.0, indicating a math failure)
mask_zero = (df['sentiment_military'] == 0) & (df['sentiment_diplomatic'] == 0) & (df['sentiment_economic'] == 0)

# Combine all filters
poisoned_mask = mask_nan | mask_thin | mask_zero
poisoned_df = df[poisoned_mask]

if len(poisoned_df) > 0:
    print(f"\n[!] Found {len(poisoned_df)} dates with integrity issues:")
    for idx, row in poisoned_df.iterrows():
        reasons = []
        if any(pd.isna(row[col]) for col in critical_params): reasons.append("Missing Parameters (NaN)")
        if row['article_count'] < 100: reasons.append(f"Thin Data ({row['article_count']} articles)")
        if (row['sentiment_military'] == 0) and (row['sentiment_diplomatic'] == 0): reasons.append("Zeroed Sentiments")
        print(f"  - {row['date']}: {', '.join(reasons)}")
        
    # Drop the poisoned rows
    clean_df = df[~poisoned_mask]
    clean_df.to_csv(file_path, index=False)
    print(f"\n✅ Cleaned! CSV shrunk from {initial_len} to {len(clean_df)} days.")
    print("Restarting the backfill will now force-repair these 17 specific gaps.")
else:
    print("\n✅ Your data baseline is 100% pristine. No issues found.")