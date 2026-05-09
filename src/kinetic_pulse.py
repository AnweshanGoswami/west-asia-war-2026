import requests
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("NASA_FIRMS_KEY")

# Bounding box covering Iran, Israel, and surrounding region
# Format: west, south, east, north
REGION = "34.0,29.0,60.0,38.0"

def get_firms_data(days_ago=1):
    """
    Fetches thermal anomaly data from NASA FIRMS API.
    Returns a DataFrame of fire/heat detections in the conflict region.
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/VIIRS_SNPP_NRT/{REGION}/{days_ago}"
    
    print(f"Fetching NASA FIRMS data for the last {days_ago} day(s)...")
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    # Parse the CSV response
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    
    print(f"Success. {len(df)} thermal anomalies detected.")
    return df

if __name__ == "__main__":
    df = get_firms_data(days_ago=5)
    
    if df is not None:
        print(df.head())
        print(f"\nColumns: {list(df.columns)}")
        
        # Save to data folder
        df.to_csv("data/firms_raw.csv", index=False)
        print("Data saved to data/firms_raw.csv")