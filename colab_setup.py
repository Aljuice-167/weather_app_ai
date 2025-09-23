# colab_setup.py
import os
import sys
import subprocess

def setup_colab_environment():
    """Setup Google Colab environment with necessary packages and drive mounting"""
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        COLAB = True
    except:
        print("Not running in Google Colab")
        COLAB = False
    
    # Install required packages
    print("Installing required packages...")
    packages = [
        'cdsapi', 
        'netcdf4', 
        'xarray', 
        'pandas', 
        'numpy', 
        'scikit-learn', 
        'joblib'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"{package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Create project directories if they don't exist
    if COLAB:
        base_dir = '/content/drive/MyDrive/weather_project'
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'data', 'raw'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'data', 'processed'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'python', 'models'), exist_ok=True)
        
        # Add project path to sys.path for imports
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        
        print(f"Project directory set to: {base_dir}")
        return base_dir
    
    return os.getcwd()

if __name__ == "__main__":
    setup_colab_environment()