import os
import pandas as pd
import xarray as xr
from glob import glob
from config import RAW_DATA_DIR, PROCESSED_DATA_FILE


def process_single_file(file_path):
    """Read a NetCDF file and return a DataFrame."""
    try:
        ds = xr.open_dataset(file_path)
        df = ds.to_dataframe().reset_index()
        return df
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None


def process_era5_data():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)

    # Collect all NetCDF files
    nc_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*.nc")))

    if not nc_files:
        print(f"❌ No NetCDF files found in {RAW_DATA_DIR}")
        return

    print(f"📂 Found {len(nc_files)} files to process.")

    # Delete existing processed file to start fresh
    if os.path.exists(PROCESSED_DATA_FILE):
        os.remove(PROCESSED_DATA_FILE)

    # Process files in batches
    batch_size = 5
    for i in range(0, len(nc_files), batch_size):
        batch_files = nc_files[i:i + batch_size]
        batch_dfs = []

        print(f"⚙️ Processing batch {i // batch_size + 1} "
              f"({len(batch_files)} files)...")

        for f in batch_files:
            df = process_single_file(f)
            if df is not None:
                batch_dfs.append(df)

        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)

            # Write header only if file does not exist
            write_header = not os.path.exists(PROCESSED_DATA_FILE)

            batch_df.to_csv(PROCESSED_DATA_FILE,
                            mode="a",
                            header=write_header,
                            index=False)

            print(f"✅ Saved {len(batch_df)} rows to {PROCESSED_DATA_FILE}")

    print("🎉 Finished processing all files.")


if __name__ == "__main__":
    process_era5_data()
