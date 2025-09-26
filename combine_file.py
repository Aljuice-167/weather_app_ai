# combine_file.py - Patched version with safe CSV writing
import os
import pandas as pd
import xarray as xr
from glob import glob
from config import RAW_DATA_DIR, PROCESSED_DATA_FILE, GHANA_BOUNDS

# Map CDS variable names to our standard names
VARIABLE_MAP = {
    # CDS name -> Our name
    't2m': 't2m',
    'd2m': 'd2m',
    'tp': 'tp',
    'sp': 'sp',
    'u10': 'u10',
    'v10': 'v10',
    # Sometimes variables have different names
    '2m_temperature': 't2m',
    '2m_dewpoint_temperature': 'd2m',
    'total_precipitation': 'tp',
    'surface_pressure': 'sp',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10'
}


def inspect_netcdf_structure(file_path):
    """Debug function to inspect NetCDF file structure."""
    try:
        ds = xr.open_dataset(file_path, engine="netcdf4")
        print(f"\n🔍 INSPECTING: {os.path.basename(file_path)}")
        print(f"Variables: {list(ds.variables.keys())}")
        print(f"Dimensions: {list(ds.dims.keys())}")
        print(f"Coordinates: {list(ds.coords.keys())}")

        for var in ds.variables:
            if 'precip' in var.lower() or var == 'tp':
                data = ds[var].values
                print(f"📊 {var} range: {data.min():.8f} to {data.max():.8f}")
                non_zero = (data > 0).sum()
                total = data.size
                print(f"📊 {var} non-zero values: {non_zero}/{total} ({100*non_zero/total:.1f}%)")

        ds.close()
        return True
    except Exception as e:
        print(f"❌ Error inspecting {file_path}: {e}")
        return False


def process_single_file(file_path):
    """Load one NetCDF file and return cleaned DataFrame with proper variable mapping."""
    try:
        ds = None
        for engine in ["netcdf4", "h5netcdf", "scipy"]:
            try:
                ds = xr.open_dataset(file_path, engine=engine)
                break
            except:
                continue

        if ds is None:
            print(f"❌ Could not open {file_path} with any engine")
            return None

        print(f"📁 Processing: {os.path.basename(file_path)}")
        print(f"   Available variables: {list(ds.variables.keys())}")

        available_data = {}
        for orig_name, standard_name in VARIABLE_MAP.items():
            if orig_name in ds.variables:
                available_data[standard_name] = ds[orig_name]
                print(f"   ✅ Found {orig_name} -> {standard_name}")

        if not available_data:
            print(f"   ❌ No recognized variables found!")
            return None

        new_ds = xr.Dataset(available_data, coords=ds.coords)
        df = new_ds.to_dataframe().reset_index()

        # Time handling
        time_cols = ['time', 'valid_time', 'date', 'datetime']
        time_col = next((col for col in time_cols if col in df.columns), None)
        if time_col:
            df['time'] = pd.to_datetime(df[time_col], errors='coerce')
            if time_col != 'time':
                df = df.drop(columns=[time_col])
        else:
            df['time'] = pd.Timestamp('2023-01-01 00:00:00')

        # Ghana bounds
        if 'latitude' in df.columns and 'longitude' in df.columns:
            before = len(df)
            df = df[
                (df['latitude'] >= GHANA_BOUNDS['lat_min']) &
                (df['latitude'] <= GHANA_BOUNDS['lat_max']) &
                (df['longitude'] >= GHANA_BOUNDS['lon_min']) &
                (df['longitude'] <= GHANA_BOUNDS['lon_max'])
            ]
            print(f"   📍 Filtered to Ghana: {before} -> {len(df)} rows")

        df = df.dropna()

        # Clean columns
        df.columns = [str(c) for c in df.columns]
        df = df.drop_duplicates()

        # Remove unnecessary cols
        for col in ['number', 'expver', 'step', 'surface']:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Temporal features
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['hour'] = df['time'].dt.hour

        # Unit conversions
        if 'tp' in df.columns:
            df['tp'] = df['tp'] * 1000
            stats = df['tp'].describe()
            print(f"   💧 Precipitation (mm): min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")
        if 't2m' in df.columns:
            df['t2m'] = df['t2m'] - 273.15
        if 'd2m' in df.columns:
            df['d2m'] = df['d2m'] - 273.15

        new_ds.close()
        ds.close()
        return df

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_era5_data():
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    nc_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*.nc")))

    if not nc_files:
        print(f"❌ No NetCDF files found in {RAW_DATA_DIR}")
        return

    print(f"📂 Found {len(nc_files)} NetCDF files")
    inspect_netcdf_structure(nc_files[0])

    if os.path.exists(PROCESSED_DATA_FILE):
        os.remove(PROCESSED_DATA_FILE)

    total_rows = 0
    successful_files = 0

    for idx, file_path in enumerate(nc_files, start=1):
        print(f"\n⚙️ Processing {idx}/{len(nc_files)}: {os.path.basename(file_path)}")
        df = process_single_file(file_path)

        if df is None or df.empty:
            print(f"   ❌ Skipping {os.path.basename(file_path)} - no data")
            continue

        # 🚑 Safe CSV writing
        write_header = not os.path.exists(PROCESSED_DATA_FILE)
        df.to_csv(PROCESSED_DATA_FILE, mode="a", header=write_header, index=False)

        total_rows += len(df)
        successful_files += 1
        print(f"   ✅ Appended {len(df)} rows (total: {total_rows})")

    print(f"\n🎉 Processing completed!")
    print(f"📊 Successfully processed {successful_files}/{len(nc_files)} files")
    print(f"📊 Total rows: {total_rows}")
    if os.path.exists(PROCESSED_DATA_FILE):
        size = os.path.getsize(PROCESSED_DATA_FILE) / (1024 * 1024)
        print(f"📁 Output file size: {size:.2f} MB")


if __name__ == "__main__":
    process_era5_data()
