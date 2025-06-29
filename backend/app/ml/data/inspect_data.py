import pickle
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

PICKLE_PATH = os.path.join(BACKEND_DIR, 'data', 'fastf1_cache', '2019', '2019-08-04_Hungarian_Grand_Prix', 'car_data.ff1pkl')

with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

driver_data_dict = data['data']

print(f"Found data for {len(driver_data_dict)} drivers.")

# Add a column to identify the driver number, then concatenate all DataFrames
dfs = []
for driver_id, df in driver_data_dict.items():
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df['driver_id'] = driver_id
        dfs.append(df)
    else:
        print(f"Warning: data for driver {driver_id} is not a DataFrame, but {type(df)}")

combined_df = pd.concat(dfs, ignore_index=True)

print("Combined DataFrame preview:")
print(combined_df.head())

print("\nData info:")
print(combined_df.info())

print("\nMissing values:")
print(combined_df.isnull().sum())

# You can now continue your inspection on combined_df,
# for example, looking at unique drivers, summary stats, etc.
print("\nUnique drivers in combined data:", combined_df['driver_id'].nunique())
print("Columns available:", combined_df.columns)
