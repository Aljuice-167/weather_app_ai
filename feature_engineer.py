
# feature_engineer.py
import os
import pandas as pd
import numpy as np

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import PROCESSED_DATA_FILE
except ImportError:
    from config import PROCESSED_DATA_FILE


class WeatherFeatureEngineer:
    def __init__(self):
        self.data = None

    def load_data(self):
        """Load processed data from CSV."""
        if not os.path.exists(PROCESSED_DATA_FILE):
            raise FileNotFoundError(f"Processed data file not found: {PROCESSED_DATA_FILE}")
        
        self.data = pd.read_csv(PROCESSED_DATA_FILE)
        
        # Ensure time column exists and is datetime
        if "time" not in self.data.columns:
            raise KeyError("'time' column is required but not found in dataset.")
        
        self.data["time"] = pd.to_datetime(self.data["time"], errors="coerce")
        if self.data["time"].isnull().any():
            print("⚠️ Warning: Some invalid timestamps found and set to NaT.")
        
        return self.data

    def _add_missing_columns(self, df):
        """Add missing required columns with default values."""
        defaults = {
            "tp": 0.0,          # Precipitation
            "sp": 101325.0,     # Surface pressure (Pa)
            "u10": 0.0,         # Eastward wind
            "v10": 0.0          # Northward wind
        }

        for col, default in defaults.items():
            if col not in df.columns:
                print(f"ℹ️ Adding missing column: {col} (default={default})")
                df[col] = default
        return df

    def create_features(self):
        """Generate weather-related engineered features."""
        if self.data is None:
            self.load_data()

        df = self.data.copy()
        df = self._add_missing_columns(df)

        # Sort by location and time (important for rolling/lag features)
        df = df.sort_values(["latitude", "longitude", "time"])

        # Rolling averages
        df["precip_7d_avg"] = df.groupby(["latitude", "longitude"])["tp"] \
            .transform(lambda x: x.rolling(7, min_periods=1).mean())

        df["temp_7d_avg"] = df.groupby(["latitude", "longitude"])["t2m"] \
            .transform(lambda x: x.rolling(7, min_periods=1).mean()) if "t2m" in df.columns else 25.0

        # Extreme weather indicators
        df["heavy_rain"] = (df["tp"] > 0.05).astype(int)         # > 5mm/hour
        df["heat_wave"] = (df.get("t2m", 25) > 35).astype(int)   # > 35°C

        # Soil moisture proxy (simplified)
        df["soil_moisture"] = df["tp"] - 0.1 * df.get("t2m", 25)

        # Wind speed magnitude
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)

        # Ensure 'month' column exists
        if "month" not in df.columns:
            df["month"] = df["time"].dt.month
        
        df["is_rainy_season"] = df["month"].isin(range(3, 11)).astype(int)

        # Lag features
        df["precip_lag24h"] = df.groupby(["latitude", "longitude"])["tp"].shift(1).fillna(0.0)
        df["temp_lag24h"] = (
            df.groupby(["latitude", "longitude"])["t2m"].shift(1)
            if "t2m" in df.columns else pd.Series(25.0, index=df.index)
        )
        df["temp_lag24h"] = df["temp_lag24h"].fillna(df["t2m"].mean() if "t2m" in df.columns else 25.0)

        # Reset index for cleanliness
        df = df.reset_index(drop=True)

        return df

    def save_features(self, features):
        """Save engineered features to CSV."""
        os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
        features.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"✅ Features saved to {PROCESSED_DATA_FILE}")

