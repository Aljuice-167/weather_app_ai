# combine_file.py
import os
import glob
import xarray as xr
import pandas as pd
import gc  # For memory management

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

def process_era5_data():
    """Process the combined ERA5 data and save to CSV"""
    try:
        # Get all ERA5 files
        nc_files = get_all_era5_files()
        
        if not nc_files:
            raise FileNotFoundError(f"No ERA5 files found in {RAW_DATA_DIR}")
        
        print(f"Found {len(nc_files)} ERA5 files to process")
        
        # Process files in batches to avoid memory issues
        batch_size = 20  # Process 20 files at a time
        all_dataframes = []
        
        for i in range(0, len(nc_files), batch_size):
            batch_files = nc_files[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(nc_files)-1)//batch_size + 1} ({len(batch_files)} files)...")
            
            # Load and combine all NetCDF files in this batch
            datasets = []
            for file in batch_files:
                print(f"Loading {os.path.basename(file)}...")
                try:
                    ds = xr.open_dataset(file)
                    datasets.append(ds)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    continue
            
            if not datasets:
                print(f"No datasets could be loaded in batch {i//batch_size + 1}")
                continue
            
            # Combine all datasets along time dimension with explicit join parameter
            combined_ds = xr.concat(datasets, dim='time', join='outer')
            
            # Convert to DataFrame
            print("Converting to DataFrame...")
            df = combined_ds.to_dataframe()
            
            # Reset index to make all coordinates columns
            df = df.reset_index()
            
            # Add to list of all dataframes
            all_dataframes.append(df)
            
            # Free memory
            del combined_ds, df
            gc.collect()
        
        if not all_dataframes:
            raise RuntimeError("No data could be processed")
        
        # Combine all dataframes
        print("Combining all batches...")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
        
        # Save to CSV
        print(f"Saving to {PROCESSED_DATA_FILE}...")
        final_df.to_csv(PROCESSED_DATA_FILE, index=False)
        
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")
        print(f"Final DataFrame shape: {final_df.shape}")
        print(f"Final DataFrame columns: {list(final_df.columns)}")
        
        return True
    
    except Exception as e:
        print(f"Error processing ERA5 data: {str(e)}")
        return False

if __name__ == "__main__":
    process_era5_data()