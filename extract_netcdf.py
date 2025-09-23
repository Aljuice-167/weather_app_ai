# extract_netcdf.py
import os
import glob
import zipfile

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import RAW_DATA_DIR
except ImportError:
    from config import RAW_DATA_DIR

def extract_netcdf_from_zips():
    """Extract NetCDF files from existing ZIP downloads"""
    # Get all ZIP files
    zip_files = glob.glob(os.path.join(RAW_DATA_DIR, "era5_ghana_*.zip"))
    
    if not zip_files:
        print("No ZIP files found")
        return False
    
    success_count = 0
    
    for zip_file in zip_files:
        try:
            print(f"\nProcessing ZIP file: {os.path.basename(zip_file)}")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # List contents
                print("ZIP file contains:")
                for name in zip_ref.namelist():
                    print(f"  - {name}")
                
                # Find NetCDF file
                nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if not nc_files:
                    print("No NetCDF file found in ZIP")
                    continue
                
                # Extract
                print(f"Extracting {nc_files[0]}...")
                zip_ref.extract(nc_files[0], os.path.dirname(zip_file))
                
                # Determine the expected filename based on ZIP name
                zip_basename = os.path.basename(zip_file)
                expected_nc = zip_basename.replace('.zip', '.nc')
                
                # Rename to expected filename
                extracted_path = os.path.join(os.path.dirname(zip_file), nc_files[0])
                expected_path = os.path.join(os.path.dirname(zip_file), expected_nc)
                
                if extracted_path != expected_path:
                    os.rename(extracted_path, expected_path)
                    print(f"Renamed to: {expected_nc}")
                
                success_count += 1
                print(f"Successfully extracted to {expected_nc}")
                
        except Exception as e:
            print(f"Extraction failed for {zip_file}: {str(e)}")
    
    print(f"\nExtraction completed: {success_count} out of {len(zip_files)} files processed successfully")
    return success_count > 0

if __name__ == "__main__":
    extract_netcdf_from_zips()