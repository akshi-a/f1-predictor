import fastf1
import pandas as pd

# Enable data caching to avoid repeated downloads
fastf1.Cache.enable_cache('fastf1_cache')

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
df.to_csv('data/fastf1_data.csv', index=False)
print("Training data saved to data/fastf1_data.csv")