import os
import fastf1
import pandas as pd

# Get project root from current file location
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Define paths
CACHE_DIR = os.path.join(ROOT_DIR, 'fastf1_cache')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_CSV_PATH = os.path.join(DATA_DIR, 'fastf1_training_data.csv')

# Ensure required directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Enable data caching to avoid repeated downloads
fastf1.Cache.enable_cache(CACHE_DIR) 

data = []
for year in range(2019, 2024):
    for rnd in range(1, 23):
        try:
            session = fastf1.get_session(year, rnd, 'R') # R for race session
            session.load()
            for i, row in session.results.iterrows():
                data.append({
                    'driver': row.Abbreviation,
                    'team': row.TeamName,
                    'quali_pos': row.GridPosition,
                    'track': session.event['Location'],
                    'year': year,
                    'points': row['Points'],
                    'laps': row['Laps'],
                    'time': row['Time'],
                    'winner': 1 if row.Position == 1 else 0
                })
        except Exception as e:
            print(f"Skipped {year} round {rnd}: {e}")
df = pd.DataFrame(data)
df.to_csv(DATA_CSV_PATH, index=False)
print(f"✅ Saved training data to: {DATA_CSV_PATH}")