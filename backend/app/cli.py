import typer
from pathlib import Path
from app.ml.data.clean_race_data import clean_race_data
from app.ml.data.build_dataset import build_dataset

app = typer.Typer()

DATA_DIR = Path("backend/data/fastf1_cache")
PROCESSED_DIR = Path("backend/data/processed")

@app.command()
def clean_all():
    """Clean all cached races and save to CSV."""
    for year_dir in DATA_DIR.iterdir():
        if not year_dir.is_dir():
            continue
        for race_dir in year_dir.iterdir():
            if not race_dir.is_dir():
                continue
            year = int(year_dir.name)
            round_name = race_dir.name.split('_', 1)[-1]
            try:
                df = clean_race_data(
                    race_dir=str(race_dir),
                    year=year,
                    round_name=round_name
                )
                output_path = PROCESSED_DIR / f"{year}_{round_name}.csv"
                df.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path}")
            except Exception as e:
                print(f"❌ Failed: {race_dir} ({e})")

@app.command()
def build():
    """Build dataset from processed CSVs."""
    X, y = build_dataset(PROCESSED_DIR)
    print("✅ Dataset built successfully")
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

if __name__ == "__main__":
    app()
