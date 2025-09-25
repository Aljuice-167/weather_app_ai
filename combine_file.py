# combine_file.py
import os
import pandas as pd
import xarray as xr
from glob import glob
from config import RAW_DATA_DIR, PROCESSED_DATA_FILE, GHANA_BOUNDS

# Vars we care about
REQUIRED_VARS = ["t2m", "d2m", "tp", "sp", "u10", "v10"]


def process_single_file(file_path):
    """Load one NetCDF file safely and return cleaned DataFrame."""
    try:
        # Try multiple engines for robustness
        engines = ["netcdf4", "h5netcdf", "scipy"]
        ds = None
        for engine in engines:
            try:
                ds = xr.open_dataset(file_path, engine=engine)
                break
            except Exception as e:
                print(f"⚠️ Failed to open {file_path} with {engine}: {e}")
        if ds is None:
            return None

        # Only keep required vars that exist
        available_vars = [v for v in REQUIRED_VARS if v in ds.variables]
        ds = ds[available_vars]

        df = ds.to_dataframe().reset_index()

        # Handle time column
        time_col = None
        for col in ["time", "valid_time", "date", "datetime"]:
            if col in df.columns:
                time_col = col
                break
        if time_col:
            df["time"] = pd.to_datetime(df[time_col])
            if time_col != "time":
                df = df.drop(columns=[time_col])
        else:
            df["time"] = pd.Timestamp("2022-06-01 12:00:00")

        # Ghana filtering
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df[
                (df["latitude"] >= GHANA_BOUNDS["lat_min"]) &
                (df["latitude"] <= GHANA_BOUNDS["lat_max"]) &
                (df["longitude"] >= GHANA_BOUNDS["lon_min"]) &
                (df["longitude"] <= GHANA_BOUNDS["lon_max"])
            ]

        # Drop NaNs
        df = df.dropna()

        # Drop unnecessary cols
        for col in ["number", "expver"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Add temporal features
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day
        df["hour"] = df["time"].dt.hour

        ds.close()
        return df

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def process_era5_data():
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    nc_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*.nc")))

    if not nc_files:
        print(f"❌ No NetCDF files found in {RAW_DATA_DIR}")
        return

    print(f"📂 Found {len(nc_files)} NetCDF files")

    if os.path.exists(PROCESSED_DATA_FILE):
        os.remove(PROCESSED_DATA_FILE)

    for idx, f in enumerate(nc_files, start=1):
        print(f"⚙️ Processing {idx}/{len(nc_files)}: {os.path.basename(f)}")
        df = process_single_file(f)
        if df is None or df.empty:
            continue

        # Append to CSV
        write_header = not os.path.exists(PROCESSED_DATA_FILE)
        df.to_csv(PROCESSED_DATA_FILE,
                  mode="a",
                  header=write_header,
                  index=False)

        print(f"✅ Appended {len(df)} rows")

    print("🎉 All files combined into CSV")


if __name__ == "__main__":
    process_era5_data()
