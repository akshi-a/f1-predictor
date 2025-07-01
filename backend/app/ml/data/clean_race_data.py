import os
import pickle
import pandas as pd
import app.ml.utils


from app.ml.utils import get_driver_map

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], pd.DataFrame):
            return obj['data']
        return obj



def clean_race_data(race_dir: str, year: int, round_name: str) -> pd.DataFrame:
    print(f"🔧 Cleaning race: {round_name} ({year})")

    # --- Load pickled data ---
    laps_obj = load_pickle(os.path.join(race_dir, 'lap_count.ff1pkl'))
    weather_obj = load_pickle(os.path.join(race_dir, 'weather_data.ff1pkl'))
    results_obj = load_pickle(os.path.join(race_dir, 'position_data.ff1pkl'))

    # --- Extract DataFrames ---
    laps_df = laps_obj.get('data') if isinstance(laps_obj, dict) else laps_obj
    weather_df = weather_obj.get('data') if isinstance(weather_obj, dict) else weather_obj
    results_df = results_obj.get('data') if isinstance(results_obj, dict) else results_obj

    # --- Clean laps ---
    if 'DriverNumber' in laps_df.columns:
        laps_df['driver_id'] = laps_df['DriverNumber'].astype(str)

    lap_cols = ['driver_id', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'PitOutTime', 'PitInTime', 'Position']
    laps_df = laps_df[lap_cols]

    # --- Get last known weather ---
    weather_df = weather_df.sort_values('Time').drop_duplicates('Time')
    last_weather = weather_df.iloc[-1]
    for col in ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']:
        laps_df[col] = last_weather[col]

    # --- Add race context ---
    laps_df['race_name'] = round_name
    laps_df['year'] = year

    # --- Add driver names ---
    driver_map = get_driver_map()
    laps_df['driver_name'] = laps_df['driver_id'].map(driver_map)

    return laps_df


# Test the function
if __name__ == "__main__":
    test_dir = os.path.abspath("app/ml/data/fastf1_cache/2019/2019-08-04_Hungarian_Grand_Prix/2019-08-04_Race")
    #C:\CS Projects\f1-predictor\backend\app\ml\data\fastf1_cache\2019\2019-08-04_Hungarian_Grand_Prix\2019-08-04_Race\car_data.ff1pkl
    df = clean_race_data(
        race_dir=test_dir,
        year=2019,
        round_name="Hungarian Grand Prix"
    )
    print(df.head())    
