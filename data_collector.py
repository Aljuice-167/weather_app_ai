# data_collector.py
import os
import glob
import xarray as xr
import pandas as pd

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import RAW_DATA_DIR, PROCESSED_DATA_FILE, GHANA_BOUNDS
except ImportError:
    from config import RAW_DATA_DIR, PROCESSED_DATA_FILE, GHANA_BOUNDS

class GhanaWeatherDataCollector:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
    
    def load_raw_data(self):
        """Load and combine raw NetCDF data from multiple files"""
        # Get all ERA5 files
        era5_files = glob.glob(os.path.join(RAW_DATA_DIR, "era5_ghana_*.nc"))
        
        if not era5_files:
            raise FileNotFoundError(f"No ERA5 files found in {RAW_DATA_DIR}")
        
        print(f"Found {len(era5_files)} ERA5 files to combine")
        
        # Try multiple engines in order of preference
        engines = ['netcdf4', 'h5netcdf', 'scipy']
        for engine in engines:
            try:
                # Load and combine all NetCDF files
                datasets = []
                for file in era5_files:
                    print(f"Loading {file} with {engine} engine...")
                    ds = xr.open_dataset(file, engine=engine)
                    datasets.append(ds)
                
                # Combine all datasets along time dimension
                self.raw_data = xr.concat(datasets, dim='time')
                print(f"Successfully combined {len(datasets)} NetCDF files with {engine} engine")
                return self.raw_data
                
            except Exception as e:
                print(f"Failed to open with {engine} engine: {str(e)}")
        
        raise RuntimeError("Could not open NetCDF files with any available engine")
    
    def preprocess_data(self):
        """Convert to DataFrame and clean data"""
        if self.raw_data is None:
            self.load_raw_data()
        
        # Convert to DataFrame
        df = self.raw_data.to_dataframe()
        
        # Reset index to make all index levels columns
        df = df.reset_index()
        
        # Print column names for debugging
        print("\nDataFrame columns after reset_index:")
        print(list(df.columns))
        
        # Check for time-related columns
        time_col = None
        for col in ['time', 'valid_time', 'date', 'datetime']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            print(f"\nFound time column: {time_col}")
            # Convert to datetime
            df['time'] = pd.to_datetime(df[time_col])
            # Drop the original time column
            df = df.drop(columns=[time_col])
        else:
            print("\nNo time column found - creating dummy time")
            df['time'] = pd.Timestamp('2022-06-01 12:00:00')
        
        # Filter for Ghana boundaries
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = df[
                (df['latitude'] >= GHANA_BOUNDS["lat_min"]) &
                (df['latitude'] <= GHANA_BOUNDS["lat_max"]) &
                (df['longitude'] >= GHANA_BOUNDS["lon_min"]) &
                (df['longitude'] <= GHANA_BOUNDS["lon_max"])
            ]
        else:
            print("Warning: latitude/longitude columns not found in DataFrame")
        
        # Handle missing values
        df = df.dropna()
        
        # Add temporal features
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['hour'] = df['time'].dt.hour
        
        # Rename columns to match expected names
        column_mapping = {
            't2m': 't2m',  # Already correct
            'd2m': 'd2m',  # Already correct
            'tp': 'tp',    # Precipitation (if present)
            'sp': 'sp',    # Surface pressure (if present)
            'u10': 'u10',  # U wind component (if present)
            'v10': 'v10'   # V wind component (if present)
        }
        
        # Apply mapping for columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename(columns={old_name: new_name})
        
        # Drop unnecessary columns
        unnecessary_cols = ['number', 'expver']
        for col in unnecessary_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        self.processed_data = df
        return df
    
    def save_processed_data(self):
        """Save processed data to CSV"""
        if self.processed_data is None:
            self.preprocess_data()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
        
        self.processed_data.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"Saved processed data to {PROCESSED_DATA_FILE}")