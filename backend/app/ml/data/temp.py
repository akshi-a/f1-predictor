import os

path = r"C:\CS Projects\f1-predictor\backend\app\ml\data\fastf1_cache\2019\2019-08-04_Hungarian_Grand_Prix\2019-08-04_Race"
files = os.listdir(path)
print("📂 Found files in race_root:")
for f in files:
    print(" -", f)
