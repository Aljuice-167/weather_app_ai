# download_data.py - Fixed and improved
import cdsapi
import os
from datetime import datetime, timedelta
import xarray as xr
from config import RAW_DATA_DIR


def get_days_in_month(year: int, month: int):
    """Return list of valid days for a given month/year (handles leap years)."""
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    delta = (end - start).days
    return [f"{d:02d}" for d in range(1, delta + 1)]


def download_era5_data():
    """Download ERA5 reanalysis data for Ghana (2023–2024)."""

    c = cdsapi.Client()
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Ghana bounding box (North, West, South, East)
    area = [11, -4, 4, 2]

    # Required variables (short names used by CDS)
    variables = [
        "2m_temperature",           # t2m
        "2m_dewpoint_temperature",  # d2m
        "total_precipitation",      # tp
        "surface_pressure",         # sp
        "10m_u_component_of_wind",  # u10
        "10m_v_component_of_wind",  # v10
    ]

    years = ["2023", "2024"]
    months = [f"{m:02d}" for m in range(1, 13)]

    for year in years:
        for month in months:
            filename = f"era5_ghana_{year}_{month}.nc"
            filepath = os.path.join(RAW_DATA_DIR, filename)

            # Check if file already exists & is valid
            if os.path.exists(filepath):
                try:
                    xr.open_dataset(filepath).close()
                    print(f"✅ {filename} already exists and is valid, skipping...")
                    continue
                except Exception:
                    print(f"⚠️ {filename} exists but is corrupted, re-downloading...")

            print(f"📥 Downloading {filename}...")

            try:
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": variables,
                        "year": year,
                        "month": month,
                        "day": get_days_in_month(int(year), int(month)),
                        "time": [f"{h:02d}:00" for h in range(24)],
                        "area": area,
                        "format": "netcdf",
                        "grid": [0.25, 0.25],
                    },
                    filepath,
                )
                print(f"✅ Successfully downloaded {filename}")

            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)


def verify_downloads(sample_n: int = 3):
    """Verify that downloaded NetCDF files contain required variables."""

    nc_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".nc")])

    if not nc_files:
        print("❌ No NetCDF files found!")
        return False

    print(f"🔍 Verifying {len(nc_files)} NetCDF files...")

    expected_vars = ["t2m", "d2m", "tp", "sp", "u10", "v10"]

    for nc_file in nc_files[:sample_n]:  # check first few files
        filepath = os.path.join(RAW_DATA_DIR, nc_file)
        try:
            ds = xr.open_dataset(filepath)
            available_vars = list(ds.variables.keys())

            print(f"📁 {nc_file}:")
            print(f"   Variables: {available_vars}")

            missing_vars = [v for v in expected_vars if v not in available_vars]
            if missing_vars:
                print(f"   ❌ Missing: {missing_vars}")
            else:
                print(f"   ✅ All required variables present!")

            if "tp" in ds.variables:
                tp_data = ds["tp"].values
                print(
                    f"   📊 Precipitation range: {tp_data.min():.6f} → {tp_data.max():.6f}"
                )

            ds.close()

        except Exception as e:
            print(f"   ❌ Error reading {nc_file}: {e}")

    return True


if __name__ == "__main__":
    print("🚀 Starting ERA5 data download...")
    download_era5_data()
    verify_downloads()
