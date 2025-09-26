# fix_nc_files.py
import os
import shutil
import subprocess
from config import RAW_DATA_DIR

def fix_nc_files():
    """Detect and fix mislabeled NetCDF files that are actually ZIP archives."""
    nc_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".nc")]
    if not nc_files:
        print("❌ No .nc files found in raw directory.")
        return
    
    print(f"🔍 Checking {len(nc_files)} .nc files...")

    for fname in nc_files:
        fpath = os.path.join(RAW_DATA_DIR, fname)

        try:
            # Use `file` command to check file type
            result = subprocess.run(["file", fpath], capture_output=True, text=True)
            description = result.stdout.strip()

            if "Zip archive data" in description:
                print(f"⚠️  {fname} is a ZIP archive mislabeled as .nc — fixing...")

                extract_dir = fpath + "_unzipped"
                os.makedirs(extract_dir, exist_ok=True)

                # Unzip contents
                subprocess.run(["unzip", "-o", fpath, "-d", extract_dir], check=True)

                # Find extracted NetCDF or GRIB file
                extracted_files = os.listdir(extract_dir)
                nc_candidates = [f for f in extracted_files if f.endswith(".nc") or f.endswith(".grib")]

                if not nc_candidates:
                    print(f"   ❌ No .nc or .grib found inside {fname}")
                    continue

                # Take the first candidate (usually only one inside)
                extracted = os.path.join(extract_dir, nc_candidates[0])
                fixed_path = os.path.join(RAW_DATA_DIR, fname)

                # Replace the bad .nc file with the extracted one
                shutil.move(extracted, fixed_path)
                shutil.rmtree(extract_dir)

                print(f"   ✅ Fixed {fname}, now a valid {nc_candidates[0].split('.')[-1].upper()} file.")

            else:
                print(f"✅ {fname} is already a valid NetCDF.")

        except Exception as e:
            print(f"❌ Error checking {fname}: {e}")

    print("🎉 File fixing completed.")

if __name__ == "__main__":
    fix_nc_files()
