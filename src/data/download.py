"""
Downloads the Egyptian Real Estate Listings dataset from Kaggle so the repo
never needs to commit the 24MB raw CSV -- keeps the repo lightweight and
the data provenance explicit and reproducible, rather than just "a file
that happened to be there."

Setup (one-time):
    1. Create a Kaggle account if you don't have one: kaggle.com
    2. Go to Account Settings -> API -> "Create New Token"
       This downloads a kaggle.json file containing your username + key.
    3. Install the kagglehub package (already in requirements.txt):
           pip install kagglehub
    4. Set your credentials as environment variables (add to your .env,
       same file as your Azure keys):
           KAGGLE_USERNAME=your_username
           KAGGLE_KEY=your_key_from_kaggle.json

Usage:
    python -m src.data.download
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
import kagglehub

load_dotenv()

DATASET_SLUG = "hassankhaled21/egyptian-real-estate-listings"
RAW_DATA_DIR = Path("data/raw")


def download_dataset() -> Path:
    """
    Downloads the dataset via kagglehub (caches it in kagglehub's own cache
    dir), then copies the CSV into our own data/raw/ folder so the rest of
    the pipeline has a stable, predictable path to read from.
    """
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in your .env file. "
            "See the setup instructions at the top of this file."
        )

    print(f"Downloading {DATASET_SLUG} from Kaggle...")
    cache_path = kagglehub.dataset_download(DATASET_SLUG)
    cache_path = Path(cache_path)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(cache_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {cache_path}")

    source_csv = csv_files[0]
    dest_csv = RAW_DATA_DIR / "egypt_real_estate_listings.csv"
    shutil.copy(source_csv, dest_csv)

    print(f"Dataset ready at: {dest_csv}")
    return dest_csv


if __name__ == "__main__":
    download_dataset()
    