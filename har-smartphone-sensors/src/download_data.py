"""Download and extract the UCI HAR dataset into ./data/"""
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    target = DATA_DIR / "UCI HAR Dataset"
    if target.exists():
        print(f"Dataset already present at {target}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI HAR dataset (~60 MB)...")
    outer = zipfile.ZipFile(io.BytesIO(urlopen(URL).read()))
    inner_name = next((n for n in outer.namelist() if n.endswith(".zip")), None)
    if inner_name:  # UCI wraps the dataset in a nested zip
        zipfile.ZipFile(io.BytesIO(outer.read(inner_name))).extractall(DATA_DIR)
    else:
        outer.extractall(DATA_DIR)
    print(f"Extracted to {target}")


if __name__ == "__main__":
    main()
