# feature_engineer.py
import os
import pandas as pd
import numpy as np
from config import PROCESSED_DATA_FILE

FEATURES_FILE = PROCESSED_DATA_FILE.replace(".csv", "_features.csv")


class WeatherFeatureEngineer:
    def __init__(self, chunk_size=500000):
        self.chunk_size = chunk_size  # process in chunks for Colab stability

    def process_chunk(self, df):
        """Apply feature engineering to a single chunk of data."""

        # Ensure time column
        if "time" not in df.columns:
            if "valid_time" in df.columns:
                df["time"] = pd.to_datetime(df["valid_time"], errors="coerce")
            else:
                raise KeyError("No 'time' or 'valid_time' column found in dataset.")

        # Add missing columns with defaults
        defaults = {
            "tp": 0.0,          # precipitation
            "sp": 101325.0,     # surface pressure
            "u10": 0.0,         # east wind
            "v10": 0.0          # north wind
        }
        for col, default in defaults.items():
            if col not in df.columns:
                print(f"ℹ️ Adding missing column: {col} (default={default})")
                df[col] = default

        # Sort by location & time (needed for lag/rolling)
        df = df.sort_values(["latitude", "longitude", "time"])

        # Rolling averages
        df["precip_7d_avg"] = (
            df.groupby(["latitude", "longitude"])["tp"]
              .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
        if "t2m" in df.columns:
            df["temp_7d_avg"] = (
                df.groupby(["latitude", "longitude"])["t2m"]
                  .transform(lambda x: x.rolling(7, min_periods=1).mean())
            )
        else:
            df["temp_7d_avg"] = 25.0

        # Extreme indicators
        df["heavy_rain"] = (df["tp"] > 0.05).astype(int)
        if "t2m" in df.columns:
            df["heat_wave"] = (df["t2m"] > 35).astype(int)
        else:
            df["heat_wave"] = 0

        # Soil moisture proxy
        if "t2m" in df.columns:
            df["soil_moisture"] = df["tp"] - 0.1 * df["t2m"]
        else:
            df["soil_moisture"] = df["tp"]

        # Wind speed
        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

        # Month & rainy season
        df["month"] = df["time"].dt.month
        df["is_rainy_season"] = df["month"].isin(range(3, 11)).astype(int)

        # Lag features
        df["precip_lag24h"] = (
            df.groupby(["latitude", "longitude"])["tp"].shift(1).fillna(0.0)
        )
        if "t2m" in df.columns:
            df["temp_lag24h"] = (
                df.groupby(["latitude", "longitude"])["t2m"].shift(1)
            ).fillna(df["t2m"].mean())
        else:
            df["temp_lag24h"] = 25.0

        return df.reset_index(drop=True)

    def create_and_save_features(self):
        """Load data in chunks, process features, and save incrementally."""
        if os.path.exists(FEATURES_FILE):
            os.remove(FEATURES_FILE)

        chunk_iter = pd.read_csv(PROCESSED_DATA_FILE, chunksize=self.chunk_size)

        for i, chunk in enumerate(chunk_iter, start=1):
            print(f"⚙️ Processing chunk {i} ...")
            features_chunk = self.process_chunk(chunk)

            # Save incrementally
            write_header = not os.path.exists(FEATURES_FILE)
            features_chunk.to_csv(FEATURES_FILE, mode="a",
                                  index=False, header=write_header)

            print(f"✅ Saved {len(features_chunk)} rows to {FEATURES_FILE}")

        print("🎉 Feature engineering completed!")


if __name__ == "__main__":
    fe = WeatherFeatureEngineer()
    fe.create_and_save_features()
