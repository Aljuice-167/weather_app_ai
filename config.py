# config.py
import os

# Detect if we're in Colab and set paths accordingly
try:
    from google.colab import drive
    BASE_DIR = '/content/drive/MyDrive/weather_project'
    COLAB = True
except ImportError:
    # Local machine paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COLAB = False

# CDS API Configuration
CDS_API_URL = "https://cds.climate.copernicus.eu/api"
CDS_API_KEY = "4bfcf961-e5e2-4653-85b9-90e4c2a74798"

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "python", "models")

# File Names
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "ghana_weather_era5.nc")
RAW_DATA_FILE_PATTERN = os.path.join(RAW_DATA_DIR, "era5_ghana_{year}_{month}.nc")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "ghana_weather_features.csv")

# Model output files
DROUGHT_MODEL_FILE = os.path.join(MODELS_DIR, "drought_model.pkl")
FLOOD_MODEL_FILE = os.path.join(MODELS_DIR, "flood_model.pkl")
RAINY_MODEL_FILE = os.path.join(MODELS_DIR, "rainy_model.pkl")   # NEW

# Ghana Coordinates
GHANA_BOUNDS = {
    "lat_min": 4.7,
    "lat_max": 11.1,
    "lon_min": -3.3,
    "lon_max": 1.2
}

# Model Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training Options
SAMPLE_FRAC = None  # e.g., 0.1 for 10% of data (useful for testing); None = full dataset

