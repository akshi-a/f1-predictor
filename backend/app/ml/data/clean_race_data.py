import os
import pickle
import pandas as pd
from app.ml.utils import get_driver_map
from pathlib import Path
import re

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def slugify(name: str):
    return re.sub(r'\W+', '_', name.strip().lower())

def clean_race_data(race_root: str, year: int, round_name: str) -> pd.DataFrame:
    print(f"🔧 Cleaning race: {round_name} ({year})")

    driver_map = get_driver_map()
    merged_df = pd.DataFrame()

    # === Load car_data ===
    car_path = os.path.join(race_root, 'car_data.ff1pkl')
    car_df = pd.DataFrame()
    if os.path.exists(car_path):
        car_raw = load_pickle(car_path)
        car_dfs = []
        for driver_id, telemetry in car_raw['data'].items():
            df = telemetry.copy()
            df['driver_id'] = driver_id
            car_dfs.append(df)
        car_df = pd.concat(car_dfs, ignore_index=True)
        car_df['driver_name'] = car_df['driver_id'].map(driver_map)

        # Downsample to 1Hz
        car_df['Date'] = pd.to_datetime(car_df['Date'])
        car_df = car_df.sort_values('Date')
        car_df = car_df.groupby('driver_id').apply(lambda g: g.resample('1S', on='Date').nearest().dropna(how='all')).reset_index(drop=True)
    else:
        print("⚠️ No car_data.ff1pkl found")

    # === Load position_data ===
    pos_path = os.path.join(race_root, 'position_data.ff1pkl')
    pos_df = pd.DataFrame()
    if os.path.exists(pos_path):
        pos_raw = load_pickle(pos_path)
        pos_dfs = []
        for driver_id, telemetry in pos_raw['data'].items():
            df = telemetry.copy()
            df['driver_id'] = driver_id
            pos_dfs.append(df)
        pos_df = pd.concat(pos_dfs, ignore_index=True)
        pos_df['driver_name'] = pos_df['driver_id'].map(driver_map)

        pos_df['Date'] = pd.to_datetime(pos_df['Date'])
        pos_df = pos_df.sort_values('Date')
        pos_df = pos_df.groupby('driver_id').apply(lambda g: g.resample('1S', on='Date').nearest().dropna(how='all')).reset_index(drop=True)
    else:
        print("⚠️ No position_data.ff1pkl found")

    # === Merge car + position ===
    if not car_df.empty and not pos_df.empty:
        merged_df = pd.merge_asof(
            car_df.sort_values('Date'),
            pos_df.sort_values('Date'),
            on='Date',
            by='driver_id',
            suffixes=('_car', '_pos')
        )
    elif not pos_df.empty:
        merged_df = pos_df.copy()
    elif not car_df.empty:
        merged_df = car_df.copy()

    # === Weather data ===
    weather_path = os.path.join(race_root, 'weather_data.ff1pkl')
    if os.path.exists(weather_path):
        weather_raw = load_pickle(weather_path)
        if isinstance(weather_raw, dict) and 'data' in weather_raw:
            weather_df = pd.DataFrame(weather_raw['data'])
        elif isinstance(weather_raw, pd.DataFrame):
            weather_df = weather_raw
        else:
            weather_df = None

        if weather_df is not None and 'Time' in weather_df.columns:
            weather_df['Date'] = pd.to_datetime(weather_df['Time'])
            weather_df = weather_df.sort_values('Date')

            merged_df = pd.merge_asof(
                merged_df.sort_values('Date'),
                weather_df[['Date', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']],
                on='Date',
                direction='nearest'
            )
    else:
        print("⚠️ No weather_data.ff1pkl found")

    # Add context
    merged_df['race_name'] = round_name
    merged_df['year'] = year

    # Output as CSV
    out_dir = Path("app/ml/data/cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{year}_{slugify(round_name)}.csv"
    out_path = out_dir / filename
    merged_df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned data to {out_path}")

    return merged_df


if __name__ == '__main__':
    df = clean_race_data(
        race_root = r"C:\CS Projects\f1-predictor\backend\app\ml\data\fastf1_cache\2019\2019-08-04_Hungarian_Grand_Prix\2019-08-04_Race",
        year=2019,
        round_name='Hungarian Grand Prix'
    )
    print(df.head())
