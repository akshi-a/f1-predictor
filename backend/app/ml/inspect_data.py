import pickle
import pandas as pd
import os

# Get the directory of this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

# Construct path relative to this script location, for example:
PICKLE_PATH = os.path.join(BACKEND_DIR, 'backend', 'fastf1_cache', '2019', '2019-08-04_Hungarian_Grand_Prix', '2019-08-04_Race', 'car_data.ff1pkl')

with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

print("Loaded pickle data type:", type(data))

print("Keys in top-level dict:", list(data.keys()))

print("Type of data['version']:", type(data['version']))
print("Value of data['version']:", data['version'])

print("Type of data['data']:", type(data['data']))

# If 'data' is a dict, show its keys
if isinstance(data['data'], dict):
    print("Keys inside data['data']:", list(data['data'].keys()))

# If 'data' is a DataFrame, show head
elif isinstance(data['data'], pd.DataFrame):
    print(data['data'].head())
else:
    print("Sample of data['data']:", str(data['data'])[:500])  # print a snippet for inspection


# Lewis Hamilton data
driver_data = data['data']['44']
print("Type of driver_data:", type(driver_data))

# If it is a dict or custom object, print keys or attributes
if isinstance(driver_data, dict):
    print("Keys inside driver_data:", list(driver_data.keys()))
elif hasattr(driver_data, '__dict__'):
    print("Attributes of driver_data:", dir(driver_data))
else:
    print("Sample content:", str(driver_data)[:500])


# If it's a dict, check keys and try to convert relevant part to DataFrame
if isinstance(data, dict):
    print("🔑 Keys in the pickle data:", list(data.keys()))
    
    # Try to find a DataFrame-like object inside the dict, for example 'laps', 'results', or similar
    for key in data.keys():
        if isinstance(data[key], pd.DataFrame):
            df = data[key]
            print(f"\n🧩 DataFrame preview from key '{key}':")
            print(df.head(), "\n")
            
            print("🔍 Data Info:")
            print(df.info(), "\n")

            print("📈 Summary Stats:")
            print(df.describe(include='all'), "\n")

            print("❗ Missing Values:")
            print(df.isnull().sum(), "\n")

            # Check if 'winner' column exists before summarizing
            if 'winner' in df.columns:
                print("⚖️ Winner Label Distribution:")
                print(df['winner'].value_counts())
                print(df['winner'].value_counts(normalize=True).rename('percentage'), "\n")
            else:
                print("⚠️ No 'winner' column found in this DataFrame.")

            print("🧠 Unique Values:")
            for col in ['driver', 'team', 'track']:
                if col in df.columns:
                    print(f"- {col}: {df[col].nunique()} unique values")
                else:
                    print(f"- {col}: Not found in DataFrame columns.")
            
            # Optional: check duplicates
            dupes = df.duplicated().sum()
            if dupes:
                print(f"⚠️ Warning: {dupes} duplicated rows found.")
            break  # Stop after the first DataFrame is inspected
    else:
        print("⚠️ No DataFrame found inside pickle data.")
else:
    print("⚠️ Pickle data is not a dict; it is a:", type(data))
    # If data is already a DataFrame, you could inspect it directly:
    if isinstance(data, pd.DataFrame):
        df = data
        print(df.head())
