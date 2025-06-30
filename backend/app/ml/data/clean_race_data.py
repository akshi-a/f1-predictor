import os
import pickle
import pandas as pd
import app.ml.utils


from app.ml.utils import get_driver_map

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def clean_race_data(race_dir: str, year: int, round_name: str) -> pd.DataFrame:
    print(f"🔧 Cleaning race: {round_name} ({year})")

    # --- Load raw pickles ---
    car_data = load_pickle(os.path.join(race_dir, 'car_data.ff1pkl'))['data']
    laps_df = load_pickle(os.path.join(race_dir, 'laps.ff1pkl'))
    weather_df = load_pickle(os.path.join(race_dir, 'weather_data.ff1pkl'))

    # --- Combine car_data (per driver) ---
    car_data_dfs = []
    for driver_id, df in car_data.items():
        df = df.copy()
        df['driver_id'] = driver_id
        car_data_dfs.append(df)
    car_df = pd.concat(car_data_dfs, ignore_index=True)

    # Optional: downsample telemetry to 1Hz or last entry per lap
    car_df = car_df.groupby(['driver_id']).tail(1)

    # --- Clean laps ---
    if 'DriverNumber' in laps_df.columns:
        laps_df['driver_id'] = laps_df['DriverNumber'].astype(str)

    useful_lap_cols = ['driver_id', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'PitOutTime', 'PitInTime', 'Position']
    laps_df = laps_df[useful_lap_cols]

    # --- Clean weather ---
    weather_df = weather_df.sort_values('Time').drop_duplicates('Time')
    last_weather = weather_df.iloc[-1]  # use last available value
    for col in ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']:
        car_df[col] = last_weather[col]

    # --- Merge everything ---
    merged_df = pd.merge(car_df, laps_df, on='driver_id', how='inner')

    # --- Add race context ---
    merged_df['race_name'] = round_name
    merged_df['year'] = year

    # --- Add driver names ---
    driver_map = get_driver_map()
    merged_df['driver_name'] = merged_df['driver_id'].map(driver_map)

    return merged_df


# Test the function
if __name__ == "__main__":
    test_dir = os.path.abspath("backend/data/fastf1_cache/2019/2019-08-04_Hungarian_Grand_Prix")
    df = clean_race_data(
        race_dir=test_dir,
        year=2019,
        round_name="Hungarian Grand Prix"
    )
    print(df.head())    
