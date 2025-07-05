import os
import pickle
import pandas as pd
from app.ml.utils import get_driver_map


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)



def clean_race_data(race_root: str, year: int, round_name: str) -> pd.DataFrame:
    print(f"🔧 Cleaning race: {round_name} ({year})")

    session_dir = race_root
    if not session_dir:
        raise FileNotFoundError(f"No session subdirectory found in {race_root}")

    driver_map = get_driver_map()
    merged_df = pd.DataFrame()

    # Try loading position data
    pos_path = os.path.join(session_dir, 'position_data.ff1pkl')
    if os.path.exists(pos_path):
        pos_data = load_pickle(pos_path)['data']
        pos_rows = []
        for driver_id, telemetry in pos_data.items():
            df = telemetry.copy()
            df['driver_id'] = driver_id
            pos_rows.append(df)
        position_df = pd.concat(pos_rows, ignore_index=True)
        position_df['driver_name'] = position_df['driver_id'].map(driver_map)
        merged_df = position_df
    else:
        print("⚠️ No position_data.ff1pkl found")

    # Try loading weather data
    weather_path = os.path.join(session_dir, 'weather_data.ff1pkl')
    if os.path.exists(weather_path):
        # Convert weather data to DataFrame
        # Load weather
        weather_raw = load_pickle(weather_path)
        print(f"🔍 weather_raw type: {type(weather_raw)}")

        # Check if it's a dict with a 'data' field
        if isinstance(weather_raw, dict) and 'data' in weather_raw:
            weather_df = pd.DataFrame(weather_raw['data'])  # ✅ this is the real weather data
        elif isinstance(weather_raw, pd.DataFrame):
            weather_df = weather_raw
        else:
            print(f"⚠️ Unexpected weather data format: {type(weather_raw)}. Skipping.")
            weather_df = None

        # Safely process weather
        if weather_df is not None and 'Time' in weather_df.columns:
            latest_weather = weather_df.sort_values('Time').iloc[-1]
            for col in ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']:
                merged_df[col] = latest_weather.get(col, None)
        else:
            print("⚠️ Weather data malformed or missing 'Time'.")
    else:
        print("⚠️ No weather_data.ff1pkl found")

    # Try loading session results
    results_path = os.path.join(session_dir, 'position_data.ff1pkl')
    if os.path.exists(results_path):
        results_raw = load_pickle(results_path)
        print(f"🔍 results_df type: {type(results_raw)}")
        

        # Extract inner DataFrame
        if isinstance(results_raw, dict) and 'data' in results_raw:
            results_df = results_raw['data']
        elif isinstance(results_raw, pd.DataFrame):
            results_df = results_raw
        else:
            print(f"⚠️ Unexpected results data format: {type(results_raw)}. Skipping session results.")
            results_df = None

        # Merge if possible
        if results_df is not None and not merged_df.empty:
            # Ensure matching types for merge
            results_df['DriverNumber'] = results_df['DriverNumber'].astype(str)
            merged_df['driver_id'] = merged_df['driver_id'].astype(str)

            merged_df = merged_df.merge(
                results_df[['DriverNumber', 'TeamName', 'Position']],
                left_on='driver_id',
                right_on='DriverNumber',
                how='left'
            )
        else:
            print("⚠️ Skipping session results merge.")

    else:
        print("⚠️ No session_results.ff1pkl found")

    # Add context info
    merged_df['race_name'] = round_name
    merged_df['year'] = year

    return merged_df


if __name__ == '__main__':
    df = clean_race_data(
        race_root=r"C:\CS Projects\f1-predictor\backend\app\ml\data\fastf1_cache\2019\2019-08-04_Hungarian_Grand_Prix\2019-08-04_Race",
        year=2019,
        round_name='Hungarian Grand Prix'
    )
    print(df.head())

