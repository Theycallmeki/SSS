# constants.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "tweets_100k.csv"   # <-- use your actual filename here
MODELS_DIR = ROOT / "models"

# Create models/ folder if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)
