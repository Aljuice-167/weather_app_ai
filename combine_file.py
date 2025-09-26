# combine_file.py - Clean version with instant+accum merge
import os
import pandas as pd
import xarray as xr
from glob import glob
from config import RAW_DATA_DIR, PROCESSED_DATA_FILE, GHANA_BOUNDS


# Map CDS variable names to our standard names
VARIABLE_MAP = {
    "t2m": "t2m",
    "d2m": "d2m",
    "tp": "tp",
    "sp": "sp",
    "u10": "u10",
    "v10": "v10",
    # CDS long names
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "total_precipitation": "tp",
    "surface_pressure": "sp",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}


def load_dataset(file_path):
    """Open NetCDF with fallback engines."""
    for engine in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            return xr.open_dataset(file_path, engine=engine)
        except Exception:
            continue
    raise OSError(f"Could not open {file_path} with any engine")


def process_single_file(file_path):
    """Load one NetCDF file and return DataFrame with mapped variables."""
    try:
        ds = load_dataset(file_path)

        # Map variables
        available_data = {}
        for orig, std in VARIABLE_MAP.items():
            if orig in ds.variables:
                available_data[std] = ds[orig]

        if not available_data:
            print(f"❌ No recognized variables in {os.path.basename(file_path)}")
            return None

        new_ds = xr.Dataset(available_data, coords=ds.coords)
        df = new_ds.to_dataframe().reset_index()

        # Handle time column
        time_cols = ["time", "valid_time", "date", "datetime"]
        time_col = next((col for col in time_cols if col in df.columns), None)
        if time_col:
            df["time"] = pd.to_datetime(df[time_col], errors="coerce")
            if time_col != "time":
                df = df.drop(columns=[time_col])
        else:
            df["time"] = pd.Timestamp("2023-01-01 00:00:00")

        # Ghana bounds filter
        if "latitude" in df.columns and "longitude" in df.columns:
            before = len(df)
            df = df[
                (df["latitude"] >= GHANA_BOUNDS["lat_min"])
                & (df["latitude"] <= GHANA_BOUNDS["lat_max"])
                & (df["longitude"] >= GHANA_BOUNDS["lon_min"])
                & (df["longitude"] <= GHANA_BOUNDS["lon_max"])
            ]
            print(f"   📍 Ghana filter: {before} -> {len(df)} rows")

        df = df.dropna()
        df = df.drop_duplicates()

        # Remove unnecessary cols
        for col in ["number", "expver", "step", "surface"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Temporal features
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day
        df["hour"] = df["time"].dt.hour

        # Unit conversions
        if "tp" in df.columns:
            df["tp"] = df["tp"] * 1000  # m → mm
        if "t2m" in df.columns:
            df["t2m"] = df["t2m"] - 273.15  # K → °C
        if "d2m" in df.columns:
            df["d2m"] = df["d2m"] - 273.15

        new_ds.close()
        ds.close()
        return df

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def process_era5_data():
    """Process and merge instant+accum ERA5 NetCDF files into one CSV."""
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)

    instant_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*instant.nc")))
    accum_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*accum.nc")))

    if not instant_files or not accum_files:
        print(f"❌ Did not find both instant and accum files in {RAW_DATA_DIR}")
        return

    print(f"📂 Found {len(instant_files)} instant and {len(accum_files)} accum files")

    if os.path.exists(PROCESSED_DATA_FILE):
        os.remove(PROCESSED_DATA_FILE)

    total_rows = 0
    successful_files = 0

    for instant_path, accum_path in zip(instant_files, accum_files):
        print(f"\n⚙️ Processing: {os.path.basename(instant_path)} + {os.path.basename(accum_path)}")

        df_instant = process_single_file(instant_path)
        df_accum = process_single_file(accum_path)

        if df_instant is None or df_accum is None:
            print("   ❌ Skipping pair - no data")
            continue

        # Merge on time + lat/lon
        merge_cols = ["time", "latitude", "longitude"]
        df = pd.merge(df_instant, df_accum, on=merge_cols, how="outer")

        if df.empty:
            print("   ❌ Empty merged dataframe")
            continue

        # 🚑 Safe CSV writing
        write_header = not os.path.exists(PROCESSED_DATA_FILE)
        df.to_csv(PROCESSED_DATA_FILE, mode="a", header=write_header, index=False)

        total_rows += len(df)
        successful_files += 1
        print(f"   ✅ Appended {len(df)} rows (total: {total_rows})")

    print(f"\n🎉 Processing completed!")
    print(f"📊 Successfully processed {successful_files}/{len(instant_files)} monthly pairs")
    print(f"📊 Total rows: {total_rows}")
    if os.path.exists(PROCESSED_DATA_FILE):
        size = os.path.getsize(PROCESSED_DATA_FILE) / (1024 * 1024)
        print(f"📁 Output size: {size:.2f} MB")


if __name__ == "__main__":
    process_era5_data()
