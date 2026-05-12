import requests
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("NASA_FIRMS_KEY")

# Bounding box covering Iran, Israel, and surrounding region
REGION = "34.0,29.0,60.0,38.0"

def get_firms_data(days_ago=1):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/VIIRS_SNPP_NRT/{REGION}/{days_ago}"
    print(f"Fetching NASA FIRMS data for the last {days_ago} day(s)...")
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    
    print(f"Success. {len(df)} thermal anomalies detected.")
    return df

def run_realtime():
    """
    Standardized entry point for the master polling loop.
    Fetches the latest FIRMS data and updates the live CSV.
    """
    df = get_firms_data(days_ago=1)
    
    if df is not None and not df.empty:
        # Save to data folder as a live update file
        save_path = "data/firms_realtime.csv"
        df.to_csv(save_path, index=False)
        return {"status": "success", "records": len(df), "file": save_path}
    
    return {"status": "empty", "records": 0}

if __name__ == "__main__":
    # This block remains for manual standalone testing
    df = get_firms_data(days_ago=5)
    if df is not None:
        print(df.head())
        df.to_csv("data/firms_raw.csv", index=False)
        print("Data saved to data/firms_raw.csv")