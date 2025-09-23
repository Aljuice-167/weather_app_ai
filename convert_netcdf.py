# convert_netcdf.py
import os
import glob
import pandas as pd
import xarray as xr

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import RAW_DATA_DIR, PROCESSED_DATA_FILE
except ImportError:
    from config import RAW_DATA_DIR, PROCESSED_DATA_FILE

def convert_netcdf_to_csv():
    """Convert combined NetCDF data to CSV"""
    try:
        # Get all ERA5 files
        era5_files = glob.glob(os.path.join(RAW_DATA_DIR, "era5_ghana_*.nc"))
        
        if not era5_files:
            raise FileNotFoundError(f"No ERA5 files found in {RAW_DATA_DIR}")
        
        print(f"Found {len(era5_files)} ERA5 files to combine")
        
        # Load and combine all NetCDF files
        datasets = []
        for file in era5_files:
            print(f"Loading {file}...")
            try:
                ds = xr.open_dataset(file)
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if not datasets:
            raise RuntimeError("No datasets could be loaded")
        
        # Combine all datasets along time dimension
        combined_ds = xr.concat(datasets, dim='time')
        
        # Convert to DataFrame
        df = combined_ds.to_dataframe()
        
        # Reset index if needed
        df = df.reset_index()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
        
        # Save to CSV
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"Converted NetCDF to CSV and saved to {PROCESSED_DATA_FILE}")
        
    except Exception as e:
        print(f"Error converting NetCDF to CSV: {str(e)}")
        raise

if __name__ == "__main__":
    convert_netcdf_to_csv()