# diagnose_netcdf.py
import os
import glob
import xarray as xr

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import RAW_DATA_DIR
except ImportError:
    from config import RAW_DATA_DIR

def diagnose_netcdf():
    """Diagnose the structure of the combined NetCDF files"""
    # Get all ERA5 files
    era5_files = glob.glob(os.path.join(RAW_DATA_DIR, "era5_ghana_*.nc"))
    
    if not era5_files:
        print(f"No ERA5 files found in {RAW_DATA_DIR}")
        return
    
    print(f"Found {len(era5_files)} ERA5 files to diagnose")
    
    # Limit to first 10 files for diagnosis to avoid memory issues
    era5_files = era5_files[:10]
    
    # Load and combine all NetCDF files
    datasets = []
    for file in era5_files:
        print(f"\nOpening file: {os.path.basename(file)}")
        try:
            ds = xr.open_dataset(file)
            datasets.append(ds)
            
            # Print info for this file
            print(f"Dimensions: {list(ds.dims)}")
            print(f"Variables: {list(ds.data_vars)}")
        except Exception as e:
            print(f"Error opening {file}: {str(e)}")
    
    if not datasets:
        print("No datasets could be loaded")
        return
    
    # Combine all datasets with explicit join parameter to fix warning
    try:
        combined_ds = xr.concat(datasets, dim='time', join='outer')
        
        print("\n=== COMBINED Dataset Info ===")
        print(combined_ds.info())
        
        print("\n=== Dimensions ===")
        for dim in combined_ds.dims:
            print(f"{dim}: {combined_ds.dims[dim]}")
        
        print("\n=== Coordinates ===")
        for coord in combined_ds.coords:
            print(f"{coord}: {combined_ds.coords[coord].values}")
        
        print("\n=== Data Variables ===")
        for var in combined_ds.data_vars:
            print(f"{var}: {combined_ds[var].dims} {combined_ds[var].shape}")
        
        # Try converting to DataFrame
        print("\n=== Converting to DataFrame ===")
        try:
            df = combined_ds.to_dataframe()
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame index: {df.index.names}")
            print(f"DataFrame columns: {list(df.columns)}")
            
            # Reset index to make coordinates columns
            df_reset = df.reset_index()
            print(f"Reset DataFrame columns: {list(df_reset.columns)}")
            
            # Check for time column
            if 'time' in df_reset.columns:
                print("\nTime column found!")
                print(f"Time values: {df_reset['time'].head()}")
            else:
                print("\nTime column not found!")
                print("Index might contain time information")
                
        except Exception as e:
            print(f"Failed to convert to DataFrame: {str(e)}")
    except Exception as e:
        print(f"Failed to combine datasets: {str(e)}")

if __name__ == "__main__":
    diagnose_netcdf()