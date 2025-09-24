# combine_file.py
import os
import glob
import xarray as xr
import pandas as pd

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import RAW_DATA_DIR, PROCESSED_DATA_FILE
except ImportError:
    from config import RAW_DATA_DIR, PROCESSED_DATA_FILE

def get_all_era5_files():
    """Get all ERA5 files matching the pattern"""
    return glob.glob(os.path.join(RAW_DATA_DIR, "era5_ghana_*.nc"))

def load_combined_era5_data():
    """Load and combine all ERA5 monthly files"""
    nc_files = get_all_era5_files()
    
    if not nc_files:
        raise FileNotFoundError(f"No ERA5 files found in {RAW_DATA_DIR}")
    
    print(f"Found {len(nc_files)} ERA5 files to process")
    
    # Load and combine all NetCDF files
    datasets = []
    for file in nc_files:
        print(f"Loading {file}...")
        try:
            ds = xr.open_dataset(file)
            datasets.append(ds)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if not datasets:
        raise RuntimeError("No datasets could be loaded")
    
    # Combine all datasets along time dimension
    combined_ds = xr.concat(datasets, dim='time', join='outer')
    
    return combined_ds

def process_era5_data():
    """Process the combined ERA5 data and save to CSV"""
    try:
        # Load combined data
        ds = load_combined_era5_data()
        
        # Process the data
        df = ds.to_dataframe()
        
        # Reset index if needed
        df = df.reset_index()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
        
        # Save to CSV
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")
        return True
    
    except Exception as e:
        print(f"Error processing ERA5 data: {str(e)}")
        return False

if __name__ == "__main__":
    process_era5_data()