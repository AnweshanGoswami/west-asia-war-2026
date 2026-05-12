import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# ── Target Tickers ───────────────────────────────────────────────────────────
TICKERS = {
    "Brent_Crude": "BZ=F",
    "VIX": "^VIX",
    "USD_ILS": "USDILS=X",
    "Gold": "GC=F",
    "SP500": "^GSPC"
}

# War began Feb 28, 2026. We pull from Feb 15 to allow for trailing moving averages and lags.
START_DATE = "2026-02-01"

def fetch_economic_data(start_date=START_DATE, end_date=None):
    """
    Fetches daily closing prices for predefined economic indicators.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    print(f"Fetching economic signals from {start_date} to {end_date}...")
    
    # Download data
    df_raw = yf.download(
        list(TICKERS.values()), 
        start=start_date, 
        end=end_date, 
        progress=False
    )
    
    # Extract just the 'Close' prices
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_close = df_raw['Close'].copy()
    else:
        df_close = df_raw.copy()
        
    # Rename columns to our readable names
    inv_map = {v: k for k, v in TICKERS.items()}
    df_close.rename(columns=inv_map, inplace=True)
    
    # Reset index to make 'Date' a standard column
    df_close.reset_index(inplace=True)
    df_close['Date'] = pd.to_datetime(df_close['Date']).dt.date
    
    return df_close

def handle_market_closures(df):
    """
    Financial markets close on weekends and holidays.
    Wars do not. We must forward-fill missing days so economic data
    can be joined cleanly with continuous daily kinetic/sentiment data.
    """
    # Create a complete date range from min to max date
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    full_date_range = pd.date_range(start=min_date, end=max_date).date
    
    # Reindex the dataframe to include all days
    df_complete = df.set_index('Date').reindex(full_date_range)
    
    # Forward-fill the missing prices (assuming Friday's price holds over the weekend)
    df_complete.ffill(inplace=True)
    
    df_complete.reset_index(inplace=True)
    df_complete.rename(columns={'index': 'Date'}, inplace=True)
    
    print(f"Filled market closures: expanded from {len(df)} to {len(df_complete)} days.")
    return df_complete

def run_realtime():
    """
    Standardized entry point for the master polling loop.
    Fetches the latest economic data, forward-fills market closures,
    and updates the live CSV.
    """
    print("Running real-time economic signals update...")
    try:
        raw_df = fetch_economic_data()
        
        if raw_df is not None and not raw_df.empty:
            processed_df = handle_market_closures(raw_df)
            
            # Save to data folder as a live update file
            os.makedirs('data', exist_ok=True)
            save_path = "data/economic_realtime.csv"
            processed_df.to_csv(save_path, index=False)
            
            return {"status": "success", "records": len(processed_df), "file": save_path}
            
        return {"status": "empty", "records": 0}
        
    except Exception as e:
        print(f"Error fetching economic signals: {e}")
        return {"status": "failed", "records": 0}

def run_historical(start_date="2026-02-01", end_date=None):
    """
    Historical backfill for Step 8c.
    yfinance pulls any date range — no chunking needed.
    """
    print(f"Running historical economic signals backfill from {start_date}...")
    try:
        raw_df = fetch_economic_data(start_date=start_date, end_date=end_date)
        if raw_df is not None and not raw_df.empty:
            processed_df = handle_market_closures(raw_df)
            os.makedirs('data', exist_ok=True)
            save_path = "data/economic_raw.csv"
            processed_df.to_csv(save_path, index=False)
            print(f"Historical economic data saved → {save_path}")
            return {"status": "success", "records": len(processed_df), "file": save_path}
        return {"status": "empty", "records": 0}
    except Exception as e:
        print(f"Error during historical economic backfill: {e}")
        return {"status": "failed", "records": 0}
    
if __name__ == "__main__":
    # 1. Fetch raw data
    raw_df = fetch_economic_data()
    
    # 2. Handle missing weekend data
    processed_df = handle_market_closures(raw_df)
    
    # 3. Save to data directory
    os.makedirs('data', exist_ok=True)
    output_path = 'data/economic_raw.csv'
    processed_df.to_csv(output_path, index=False)
    
    print("\nSample Output:")
    print(processed_df.tail())
    print(f"\nSaved to {output_path}")