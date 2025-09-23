# feature_engineer.py
import pandas as pd
import numpy as np

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import PROCESSED_DATA_FILE
except ImportError:
    from config import PROCESSED_DATA_FILE

class WeatherFeatureEngineer:
    def __init__(self):
        self.data = None

    def load_data(self):
        """Load processed data"""
        self.data = pd.read_csv(PROCESSED_DATA_FILE)
        self.data['time'] = pd.to_datetime(self.data['time'])
        return self.data

    def create_features(self):
        """Create weather-related features"""
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # Ensure we have the required columns
        required_cols = ['t2m', 'd2m', 'latitude', 'longitude']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Required column {col} not found")
        
        # Add missing columns with default values if they don't exist
        if 'tp' not in df.columns:
            print("Adding default precipitation values")
            df['tp'] = 0.0  # Default no precipitation
        
        if 'sp' not in df.columns:
            print("Adding default surface pressure values")
            df['sp'] = 101325.0  # Standard atmospheric pressure
        
        if 'u10' not in df.columns:
            print("Adding default u10 wind values")
            df['u10'] = 0.0
        
        if 'v10' not in df.columns:
            print("Adding default v10 wind values")
            df['v10'] = 0.0
        
        # Sort by location and time for rolling calculations
        df = df.sort_values(['latitude', 'longitude', 'time'])
        
        # Rolling averages using groupby + transform
        if 'tp' in df.columns:
            df['precip_7d_avg'] = df.groupby(['latitude', 'longitude'])['tp'].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
        else:
            df['precip_7d_avg'] = 0.0
        
        if 't2m' in df.columns:
            df['temp_7d_avg'] = df.groupby(['latitude', 'longitude'])['t2m'].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
        else:
            df['temp_7d_avg'] = 25.0  # Default temperature
        
        # Extreme weather indicators
        if 'tp' in df.columns:
            df['heavy_rain'] = (df['tp'] > 0.05).astype(int)  # >5mm/hour
        else:
            df['heavy_rain'] = 0
        
        if 't2m' in df.columns:
            df['heat_wave'] = (df['t2m'] > 35).astype(int)     # >35°C
        else:
            df['heat_wave'] = 0
        
        # Soil moisture proxy (precipitation - evaporation)
        if 'tp' in df.columns and 't2m' in df.columns:
            df['soil_moisture'] = df['tp'] - 0.1 * df['t2m']  # Simplified calculation
        else:
            df['soil_moisture'] = 0.0
        
        # Wind speed
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
        else:
            df['wind_speed'] = 0.0
        
        # Seasonal features
        df['is_rainy_season'] = df['month'].isin([3, 4, 5, 6, 7, 8, 9, 10]).astype(int)
        
        # Lag features using groupby + transform
        if 'tp' in df.columns:
            df['precip_lag24h'] = df.groupby(['latitude', 'longitude'])['tp'].transform(
                lambda x: x.shift(1)
            )
        else:
            df['precip_lag24h'] = 0.0
        
        if 't2m' in df.columns:
            df['temp_lag24h'] = df.groupby(['latitude', 'longitude'])['t2m'].transform(
                lambda x: x.shift(1)
            )
        else:
            df['temp_lag24h'] = 25.0
        
        # Fill NaN values created by lag features
        df = df.fillna({
            'precip_lag24h': 0.0,
            'temp_lag24h': df['t2m'].mean() if 't2m' in df.columns else 25.0
        })
        
        # Reset index to ensure clean DataFrame
        df = df.reset_index(drop=True)
        
        return df

    def save_features(self, features):
        """Save feature-engineered data"""
        os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
        features.to_csv(PROCESSED_DATA_FILE, index=False)