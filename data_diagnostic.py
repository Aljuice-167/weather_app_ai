# data_diagnostic.py - Diagnose and fix data issues
import pandas as pd
import numpy as np
import os
from config import PROCESSED_DATA_FILE

def diagnose_weather_data():
    """Comprehensive diagnosis of weather data issues"""
    
    # Check if the features file exists
    features_file = PROCESSED_DATA_FILE.replace(".csv", "_features.csv")
    
    if os.path.exists(features_file):
        print(f"📊 Loading features file: {features_file}")
        data = pd.read_csv(features_file)
    else:
        print(f"📊 Loading processed file: {PROCESSED_DATA_FILE}")
        data = pd.read_csv(PROCESSED_DATA_FILE)
    
    print(f"\n🔍 DATA OVERVIEW:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check each important variable
    variables_to_check = ['tp', 't2m', 'd2m', 'sp', 'u10', 'v10']
    
    print(f"\n📈 VARIABLE ANALYSIS:")
    for var in variables_to_check:
        if var in data.columns:
            stats = data[var].describe()
            non_zero_count = (data[var] != 0).sum()
            print(f"\n{var} (Total Precipitation/Temperature/etc):")
            print(f"  Range: {stats['min']:.6f} to {stats['max']:.6f}")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Non-zero values: {non_zero_count:,} ({non_zero_count/len(data)*100:.1f}%)")
            
            # Check for potential unit issues
            if var == 'tp' and stats['max'] == 0:
                print(f"  ❌ PROBLEM: All precipitation values are zero!")
            elif var == 't2m' and stats['min'] > 200:
                print(f"  ⚠️  ISSUE: Temperature in Kelvin, need conversion")
            elif var == 't2m' and stats['max'] < 100:
                print(f"  ✅ Temperature appears to be in Celsius")
        else:
            print(f"❌ {var}: NOT FOUND in dataset")
    
    # Check time coverage
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'])
        print(f"\n📅 TIME COVERAGE:")
        print(f"  Start: {data['time'].min()}")
        print(f"  End: {data['time'].max()}")
        print(f"  Total days: {(data['time'].max() - data['time'].min()).days}")
        
        # Check for time gaps
        time_diff = data['time'].diff()
        common_interval = time_diff.mode()[0] if not time_diff.mode().empty else "Unknown"
        print(f"  Most common interval: {common_interval}")
    
    # Check geographic coverage
    if 'latitude' in data.columns and 'longitude' in data.columns:
        print(f"\n🌍 GEOGRAPHIC COVERAGE:")
        print(f"  Lat range: {data['latitude'].min():.2f} to {data['latitude'].max():.2f}")
        print(f"  Lon range: {data['longitude'].min():.2f} to {data['longitude'].max():.2f}")
        print(f"  Unique locations: {len(data.groupby(['latitude', 'longitude']))}")
    
    return data

def fix_data_issues(data):
    """Fix common data issues"""
    print(f"\n🔧 FIXING DATA ISSUES:")
    
    fixed_data = data.copy()
    
    # Fix temperature units (Kelvin to Celsius)
    if 't2m' in fixed_data.columns:
        if fixed_data['t2m'].min() > 200:  # Likely Kelvin
            print("  Converting temperature from Kelvin to Celsius...")
            fixed_data['t2m'] = fixed_data['t2m'] - 273.15
            
        if 'd2m' in fixed_data.columns and fixed_data['d2m'].min() > 200:
            print("  Converting dewpoint from Kelvin to Celsius...")
            fixed_data['d2m'] = fixed_data['d2m'] - 273.15
    
    # Check if precipitation needs unit conversion
    if 'tp' in fixed_data.columns:
        if fixed_data['tp'].max() == 0:
            print("  ❌ Cannot fix zero precipitation - check raw NetCDF files")
            print("  This might indicate:")
            print("    - Wrong variable name in NetCDF files")
            print("    - Unit conversion needed (m to mm)")
            print("    - Data extraction error")
            
            # Try to check if there's accumulated precipitation
            if 'total_precipitation' in fixed_data.columns:
                print("  Found 'total_precipitation' column, using that...")
                fixed_data['tp'] = fixed_data['total_precipitation']
        else:
            # Convert from meters to mm if needed
            if fixed_data['tp'].max() < 0.1:  # Likely in meters
                print("  Converting precipitation from meters to millimeters...")
                fixed_data['tp'] = fixed_data['tp'] * 1000
    
    # Fix engineered features if they exist
    feature_columns = ['precip_7d_avg', 'temp_7d_avg', 'soil_moisture', 'wind_speed']
    for col in feature_columns:
        if col in fixed_data.columns:
            if col.startswith('precip') and fixed_data[col].max() == 0:
                print(f"  ❌ {col} is all zeros due to precipitation issue")
            elif col.startswith('temp') and fixed_data[col].min() > 200:
                print(f"  Converting {col} from Kelvin to Celsius...")
                fixed_data[col] = fixed_data[col] - 273.15
    
    return fixed_data

def save_fixed_data(data, suffix="_fixed"):
    """Save the fixed dataset"""
    original_file = PROCESSED_DATA_FILE.replace(".csv", "_features.csv")
    if not os.path.exists(original_file):
        original_file = PROCESSED_DATA_FILE
    
    fixed_file = original_file.replace(".csv", f"{suffix}.csv")
    data.to_csv(fixed_file, index=False)
    print(f"\n💾 Fixed data saved to: {fixed_file}")
    return fixed_file

def check_raw_netcdf_files():
    """Check the original NetCDF files for precipitation data"""
    from config import RAW_DATA_DIR
    import xarray as xr
    from glob import glob
    
    print(f"\n🔍 CHECKING RAW NETCDF FILES:")
    nc_files = glob(os.path.join(RAW_DATA_DIR, "*.nc"))
    
    if not nc_files:
        print("  ❌ No NetCDF files found!")
        return
    
    # Check first file
    first_file = nc_files[0]
    print(f"  Checking: {os.path.basename(first_file)}")
    
    try:
        ds = xr.open_dataset(first_file)
        print(f"  Variables in NetCDF: {list(ds.variables.keys())}")
        
        # Check precipitation variable
        precip_vars = [v for v in ds.variables.keys() if 'precip' in v.lower() or 'tp' in v.lower() or 'rainfall' in v.lower()]
        if precip_vars:
            for pvar in precip_vars:
                data_array = ds[pvar]
                print(f"  {pvar}: shape={data_array.shape}, min={data_array.min().values:.6f}, max={data_array.max().values:.6f}")
        else:
            print("  ❌ No precipitation variables found!")
            
        ds.close()
        
    except Exception as e:
        print(f"  ❌ Error reading NetCDF: {e}")

if __name__ == "__main__":
    print("🚀 Starting weather data diagnosis...")
    
    # Step 1: Diagnose current processed data
    data = diagnose_weather_data()
    
    # Step 2: Check raw NetCDF files
    check_raw_netcdf_files()
    
    # Step 3: Fix issues if possible
    fixed_data = fix_data_issues(data)
    
    # Step 4: Save fixed data
    fixed_file = save_fixed_data(fixed_data)
    
    print(f"\n✅ Diagnosis complete!")
    print(f"Next steps:")
    print(f"1. If precipitation is still zero, check your NetCDF extraction process")
    print(f"2. Use the fixed file for training: {fixed_file}")
    print(f"3. Re-run feature engineering if needed")