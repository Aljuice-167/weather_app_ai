# netcdf_inspector.py - Detailed inspection of your NetCDF files
import os
import xarray as xr
from glob import glob
from config import RAW_DATA_DIR

def detailed_netcdf_inspection():
    """Perform detailed inspection of all NetCDF files."""
    
    nc_files = sorted(glob(os.path.join(RAW_DATA_DIR, "*.nc")))
    
    if not nc_files:
        print(f"❌ No NetCDF files found in {RAW_DATA_DIR}")
        return
    
    print(f"🔍 DETAILED INSPECTION OF {len(nc_files)} NetCDF FILES")
    print("=" * 60)
    
    # Check first few files in detail
    for i, file_path in enumerate(nc_files[:3]):  # Check first 3 files
        print(f"\n📁 FILE {i+1}: {os.path.basename(file_path)}")
        print("-" * 50)
        
        try:
            ds = xr.open_dataset(file_path)
            
            print("📊 BASIC INFO:")
            print(f"   Dimensions: {dict(ds.dims)}")
            print(f"   Size: {ds.nbytes / 1024 / 1024:.2f} MB")
            
            print("\n🔢 VARIABLES:")
            for var_name in ds.variables:
                var = ds[var_name]
                print(f"   {var_name}:")
                print(f"     Shape: {var.shape}")
                print(f"     Dtype: {var.dtype}")
                
                # Get data sample for analysis
                if var.size > 0 and var.ndim > 0:
                    try:
                        data = var.values
                        if data.size > 0:
                            print(f"     Range: {data.min():.6f} to {data.max():.6f}")
                            print(f"     Mean: {data.mean():.6f}")
                            
                            # Special checks for precipitation
                            if var_name in ['tp', 'total_precipitation']:
                                non_zero = (data > 0).sum()
                                print(f"     Non-zero values: {non_zero}/{data.size} ({100*non_zero/data.size:.1f}%)")
                                if non_zero > 0:
                                    print(f"     ✅ HAS PRECIPITATION DATA!")
                                else:
                                    print(f"     ❌ ALL PRECIPITATION IS ZERO!")
                        else:
                            print(f"     ⚠️ Empty data array")
                    except Exception as e:
                        print(f"     ❌ Error reading data: {e}")
                
                # Check attributes
                if hasattr(var, 'attrs') and var.attrs:
                    print(f"     Attributes: {list(var.attrs.keys())}")
                    if 'units' in var.attrs:
                        print(f"     Units: {var.attrs['units']}")
            
            print(f"\n🌍 COORDINATES:")
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                if coord.size > 0:
                    data = coord.values
                    print(f"   {coord_name}: {data.min():.2f} to {data.max():.2f} ({len(data)} points)")
            
            # Check for alternative variable names
            print(f"\n🔍 CHECKING FOR ALTERNATIVE NAMES:")
            all_vars = list(ds.variables.keys())
            
            # Common alternative names for precipitation
            precip_names = ['tp', 'total_precipitation', 'precip', 'precipitation', 'rain', 'rainfall']
            precip_found = [name for name in precip_names if name in all_vars]
            if precip_found:
                print(f"   Precipitation variables found: {precip_found}")
            else:
                print(f"   ❌ No precipitation variables found in: {all_vars}")
            
            # Check for wind components
            wind_names = ['u10', 'v10', '10m_u_component_of_wind', '10m_v_component_of_wind', 'u10m', 'v10m']
            wind_found = [name for name in wind_names if name in all_vars]
            if wind_found:
                print(f"   Wind variables found: {wind_found}")
            else:
                print(f"   ❌ No wind variables found")
            
            ds.close()
            
        except Exception as e:
            print(f"❌ Error inspecting {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary of all files
    print(f"\n📋 SUMMARY OF ALL {len(nc_files)} FILES:")
    print("-" * 50)
    
    variable_counts = {}
    for file_path in nc_files:
        try:
            ds = xr.open_dataset(file_path)
            for var in ds.variables:
                if var not in variable_counts:
                    variable_counts[var] = 0
                variable_counts[var] += 1
            ds.close()
        except:
            continue
    
    print("Variable frequency across all files:")
    for var, count in sorted(variable_counts.items()):
        print(f"   {var}: {count}/{len(nc_files)} files ({100*count/len(nc_files):.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if 'tp' not in variable_counts and 'total_precipitation' not in variable_counts:
        print("   ❌ CRITICAL: No precipitation data found in any files!")
        print("   🔧 You need to re-download data with 'total_precipitation' variable")
    
    if 'u10' not in variable_counts and '10m_u_component_of_wind' not in variable_counts:
        print("   ❌ CRITICAL: No wind data found in any files!")
        print("   🔧 You need to re-download data with wind components")
    
    required_vars = ['t2m', 'd2m', 'tp', 'sp', 'u10', 'v10']
    missing_vars = [var for var in required_vars if var not in variable_counts]
    if missing_vars:
        print(f"   ❌ Missing required variables: {missing_vars}")
        print(f"   🔧 Use the fixed download script to get all required variables")

if __name__ == "__main__":
    detailed_netcdf_inspection()
