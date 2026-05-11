import pandas as pd
import requests
import zipfile
import io
import time

# GDELT Event Database Daily Export URL format
GDELT_DAILY_URL = "http://data.gdeltproject.org/events/{}.export.CSV.zip"

# FIPS 10-4 Country Codes for the theater
TARGET_COUNTRIES = ["IR", "IS", "LE", "SY", "IZ", "YM", "WE", "GZ"] 

# GDELT raw files have no headers, so we must map them explicitly
GDELT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate", 
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", 
    "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", 
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode", 
    "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", 
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", 
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", 
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID", 
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", 
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", 
    "DATEADDED", "SOURCEURL"
]

def fetch_gdelt_daily_kinetic(date_str: str) -> pd.DataFrame:
    """
    Fetches a single daily GDELT Event CSV and filters for kinetic strikes.
    """
    url = GDELT_DAILY_URL.format(date_str)
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"  -> Skipping {date_str}: Not found or error (HTTP {response.status_code})")
            return pd.DataFrame()
            
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, sep='\t', header=None, names=GDELT_COLUMNS, low_memory=False)
                
        df['EventRootCode'] = df['EventRootCode'].astype(str)
        kinetic_codes = ['18', '19', '20']
        df = df[df['EventRootCode'].isin(kinetic_codes)]
        df = df[df['ActionGeo_CountryCode'].isin(TARGET_COUNTRIES)]
        
        return df
        
    except Exception as e:
        print(f"  -> Error processing {date_str}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Define the historical window (Feb 1 to Today)
    start_date = "2026-02-01"
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    # Generate a list of dates in YYYYMMDD format
    date_range = pd.date_range(start_date, end_date).strftime('%Y%m%d')
    
    print(f"Bootstrapping GDELT History from {start_date} to {end_date}...")
    print(f"Total days to process: {len(date_range)}\n")
    
    all_daily_data = []
    
    # Loop through every day in the date range
    for i, date_str in enumerate(date_range):
        print(f"[{i+1}/{len(date_range)}] Pulling {date_str}...")
        daily_df = fetch_gdelt_daily_kinetic(date_str)
        
        if not daily_df.empty:
            all_daily_data.append(daily_df)
            
        # Polite delay to avoid hammering the GDELT server and getting IP-banned
        time.sleep(1)
        
    if all_daily_data:
        # Combine all the daily chunks into one massive master dataframe
        master_df = pd.concat(all_daily_data, ignore_index=True)
        print(f"\nSUCCESS: Retrieved {len(master_df)} total kinetic events.")
        
        # Save directly to the data folder from the project root
        save_path = "data/gdelt_raw_history.csv"
        master_df.to_csv(save_path, index=False)
        print(f"Master file saved to: {save_path}")
    else:
        print("\nFAILED: No data retrieved.")