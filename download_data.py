# download_data.py - Fixed version with all required variables
import cdsapi
import os
from datetime import datetime, timedelta
from config import RAW_DATA_DIR

def download_era5_data():
    """Download ERA5 reanalysis data with ALL required variables for Ghana."""
    
    # Create client
    c = cdsapi.Client()
    
    # Ensure raw data directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Ghana bounding box (expanded slightly for safety)
    area = [11, -4, 4, 2]  # North, West, South, East
    
    # ALL required variables - this is the key fix!
    variables = [
        '2m_temperature',           # t2m
        '2m_dewpoint_temperature',  # d2m  
        'total_precipitation',      # tp - THIS WAS MISSING!
        'surface_pressure',         # sp - THIS WAS MISSING!
        '10m_u_component_of_wind',  # u10 - THIS WAS MISSING!
        '10m_v_component_of_wind',  # v10 - THIS WAS MISSING!
    ]
    
    # Download data for 2023-2024 by month
    years = ['2023', '2024']
    months = ['01', '02', '03', '04', '05', '06', 
              '07', '08', '09', '10', '11', '12']
    
    for year in years:
        for month in months:
            filename = f'era5_ghana_{year}_{month}.nc'
            filepath = os.path.join(RAW_DATA_DIR, filename)
            
            # Skip if already exists
            if os.path.exists(filepath):
                print(f"✅ {filename} already exists, skipping...")
                continue
            
            print(f"📥 Downloading {filename}...")
            
            try:
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': variables,  # All 6 variables
                        'year': year,
                        'month': month,
                        'day': [
                            '01', '02', '03', '04', '05', '06',
                            '07', '08', '09', '10', '11', '12',
                            '13', '14', '15', '16', '17', '18',
                            '19', '20', '21', '22', '23', '24',
                            '25', '26', '27', '28', '29', '30', '31'
                        ],
                        'time': [
                            '00:00', '01:00', '02:00', '03:00',
                            '04:00', '05:00', '06:00', '07:00',
                            '08:00', '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00', '15:00',
                            '16:00', '17:00', '18:00', '19:00',
                            '20:00', '21:00', '22:00', '23:00'
                        ],
                        'area': area,  # Ghana bounding box
                        'format': 'netcdf',
                        'grid': [0.25, 0.25],  # 0.25-degree resolution
                    },
                    filepath
                )
                print(f"✅ Successfully downloaded {filename}")
                
            except Exception as e:
                print(f"❌ Error downloading {filename}: {str(e)}")
                # Remove partial file if it exists
                if os.path.exists(filepath):
                    os.remove(filepath)
                continue
    
    print("🎉 Data download completed!")

def verify_downloads():
    """Verify that downloaded files contain the required variables."""
    import xarray as xr
    
    nc_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.nc')]
    
    if not nc_files:
        print("❌ No NetCDF files found!")
        return False
    
    print(f"🔍 Verifying {len(nc_files)} NetCDF files...")
    
    expected_vars = ['t2m', 'd2m', 'tp', 'sp', 'u10', 'v10']
    
    for nc_file in nc_files[:3]:  # Check first 3 files
        filepath = os.path.join(RAW_DATA_DIR, nc_file)
        try:
            ds = xr.open_dataset(filepath)
            available_vars = list(ds.variables.keys())
            
            print(f"📁 {nc_file}:")
            print(f"   Variables: {available_vars}")
            
            missing_vars = [var for var in expected_vars if var not in available_vars]
            if missing_vars:
                print(f"   ❌ Missing: {missing_vars}")
            else:
                print(f"   ✅ All required variables present!")
                
            # Check data ranges
            if 'tp' in ds.variables:
                tp_data = ds['tp'].values
                print(f"   📊 Precipitation range: {tp_data.min():.6f} to {tp_data.max():.6f}")
                if tp_data.max() > 0:
                    print(f"   ✅ Precipitation data looks good!")
                else:
                    print(f"   ⚠️ All precipitation values are zero")
            
            ds.close()
            
        except Exception as e:
            print(f"   ❌ Error reading {nc_file}: {e}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting ERA5 data download...")
    download_era5_data()
    verify_downloads()
