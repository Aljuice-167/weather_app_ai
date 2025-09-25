# download_data.py
import os
import cdsapi
import zipfile

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import CDS_API_URL, CDS_API_KEY, RAW_DATA_FILE, GHANA_BOUNDS
except ImportError:
    from config import CDS_API_URL, CDS_API_KEY, RAW_DATA_FILE, GHANA_BOUNDS

def download_era5_data_by_chunks():
   # Define the years and months to download
    years = list(range(2020, 2025))  # 2025 is excluded, so this covers 2010–2024
    months = [f'{m:02d}' for m in range(1, 13)]

    # Create the raw data directory if it doesn't exist
    raw_dir = os.path.dirname(RAW_DATA_FILE)
    os.makedirs(raw_dir, exist_ok=True)

    # Loop through each year and month to make separate requests
    for year in years:
        for month in months:
            # Create a unique filename for each month's data
            file_name = f"era5_ghana_{year}_{month}.nc"
            file_path = os.path.join(raw_dir, file_name)

            # Skip download if the file already exists
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping download.")
                continue

            c = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY)
            
            try:
                print(f"Downloading data for {year}-{month}...")
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': [
                            'total_precipitation', 
                            '2m_temperature',
                            '2m_dewpoint_temperature'
                        ],
                        'year': str(year),
                        'month': month,
                        'day': [f'{d:02d}' for d in range(1, 32)],
                        'time': [f'{h:02d}:00' for h in range(24)],
                        'area': [
                            GHANA_BOUNDS["lat_max"], GHANA_BOUNDS["lon_min"],
                            GHANA_BOUNDS["lat_min"], GHANA_BOUNDS["lon_max"]
                        ],
                        'format': 'netcdf',
                    },
                    file_path
                )
                
                # Check for successful download and handle potential ZIP file
                if not os.path.exists(file_path):
                    raise Exception("Download failed - no file created")
                
                file_size = os.path.getsize(file_path)
                if file_size < 1000:
                    raise Exception(f"Download failed - file too small ({file_size} bytes)")
                
                print(f"Successfully downloaded {file_size} bytes for {year}-{month}")

                with open(file_path, 'rb') as f:
                    header = f.read(4)

                if header == b'PK\x03\x04':
                    print("Detected ZIP file - extracting NetCDF...")
                    temp_zip = file_path + '.zip'
                    os.rename(file_path, temp_zip)
                    
                    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                        nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                        if not nc_files:
                            raise Exception("No NetCDF file found in the ZIP archive")
                        
                        extracted_name = os.path.basename(nc_files[0])
                        extracted_path = os.path.join(raw_dir, extracted_name)
                        zip_ref.extract(nc_files[0], raw_dir)
                        
                        os.rename(extracted_path, file_path)
                    os.remove(temp_zip)
                    print(f"Successfully extracted NetCDF file to {file_path}")
                else:
                    print(f"File {file_name} is a valid NetCDF file")

            except Exception as e:
                print(f"Download failed for {year}-{month}: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                # Continue to next iteration instead of raising
                continue

if __name__ == "__main__":
    download_era5_data_by_chunks()