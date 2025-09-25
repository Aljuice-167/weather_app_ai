# train_robust.py - Improved training with better label generation
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import gc

from config import (
    PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE,
    FLOOD_MODEL_FILE, RAINY_MODEL_FILE,
    TEST_SIZE, RANDOM_STATE
)

def load_and_prepare_data():
    """Load data and handle common issues"""
    
    # Try different file locations
    possible_files = [
        PROCESSED_DATA_FILE.replace(".csv", "_features_fixed.csv"),
        PROCESSED_DATA_FILE.replace(".csv", "_features.csv"),
        PROCESSED_DATA_FILE.replace(".csv", "_fixed.csv"),
        PROCESSED_DATA_FILE
    ]
    
    data = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"📂 Loading data from: {file_path}")
            data = pd.read_csv(file_path)
            break
    
    if data is None:
        raise FileNotFoundError("No processed data file found!")
    
    print(f"📊 Data shape: {data.shape}")
    
    # Fix temperature units if needed
    if 't2m' in data.columns and data['t2m'].min() > 200:
        print("🔧 Converting temperature from Kelvin to Celsius")
        data['t2m'] = data['t2m'] - 273.15
    
    if 'd2m' in data.columns and data['d2m'].min() > 200:
        print("🔧 Converting dewpoint from Kelvin to Celsius")  
        data['d2m'] = data['d2m'] - 273.15
    
    # Handle zero precipitation issue
    if 'tp' in data.columns and data['tp'].max() == 0:
        print("⚠️ All precipitation values are zero!")
        print("   Creating synthetic precipitation based on other variables...")
        
        # Create synthetic precipitation based on humidity and temperature patterns
        if 'd2m' in data.columns and 't2m' in data.columns:
            # Higher humidity + moderate temps = more likely rain
            humidity_proxy = (data['t2m'] - data['d2m'])  # Lower values = higher humidity
            temp_factor = (data['t2m'] - data['t2m'].min()) / (data['t2m'].max() - data['t2m'].min())
            
            # Create synthetic precipitation (0 to 10mm range)
            np.random.seed(42)  # For reproducibility
            data['tp'] = np.where(
                humidity_proxy < humidity_proxy.quantile(0.3),  # High humidity conditions
                np.random.exponential(2.0, len(data)) * (1 - temp_factor * 0.5),  # More rain in cooler, humid conditions
                np.random.exponential(0.5, len(data)) * temp_factor  # Light rain in other conditions
            )
            data['tp'] = np.clip(data['tp'], 0, 15)  # Reasonable precipitation range
            print(f"   Created synthetic precipitation: {data['tp'].min():.2f} to {data['tp'].max():.2f} mm")
    
    return data

def create_better_labels(data):
    """Create more meaningful weather labels"""
    print("🏷️ Creating weather labels...")
    
    # Get data statistics for thresholding
    if 'tp' in data.columns:
        tp_stats = data['tp'].describe()
        print(f"   Precipitation range: {tp_stats['min']:.2f} to {tp_stats['max']:.2f} mm")
    
    if 't2m' in data.columns:
        temp_stats = data['t2m'].describe()
        print(f"   Temperature range: {temp_stats['min']:.1f} to {temp_stats['max']:.1f} °C")
    
    # DROUGHT: Low precipitation + High temperature
    if 'tp' in data.columns and 't2m' in data.columns:
        # Use more sophisticated drought index
        precip_low = data['tp'] <= data['tp'].quantile(0.15)  # Bottom 15% precipitation
        temp_high = data['t2m'] >= data['t2m'].quantile(0.85)  # Top 15% temperature
        
        # Also consider sustained low precipitation
        if 'precip_7d_avg' in data.columns:
            sustained_dry = data['precip_7d_avg'] <= data['precip_7d_avg'].quantile(0.10)
            data['drought_label'] = (precip_low & temp_high & sustained_dry).astype(int)
        else:
            data['drought_label'] = (precip_low & temp_high).astype(int)
    else:
        data['drought_label'] = 0
    
    # FLOOD: Very high precipitation
    if 'tp' in data.columns:
        # Extreme precipitation events
        flood_threshold = data['tp'].quantile(0.95)  # Top 5% of precipitation
        data['flood_label'] = (data['tp'] >= flood_threshold).astype(int)
        
        # Enhanced with soil saturation proxy
        if 'soil_moisture' in data.columns:
            soil_saturated = data['soil_moisture'] >= data['soil_moisture'].quantile(0.90)
            data['flood_label'] = ((data['tp'] >= data['tp'].quantile(0.90)) & soil_saturated).astype(int)
    else:
        data['flood_label'] = 0
    
    # RAINY: Moderate to high precipitation
    if 'tp' in data.columns:
        # Any significant precipitation
        rain_threshold = max(data['tp'].quantile(0.6), 0.1)  # 60th percentile or 0.1mm minimum
        data['rainy_label'] = (data['tp'] >= rain_threshold).astype(int)
    else:
        data['rainy_label'] = 0
    
    # Additional labels for more interesting predictions
    
    # HEAT WAVE: Very high temperature for extended period
    if 't2m' in data.columns:
        heat_threshold = data['t2m'].quantile(0.95)  # Top 5% temperature
        data['heatwave_label'] = (data['t2m'] >= heat_threshold).astype(int)
    else:
        data['heatwave_label'] = 0
    
    # HUMID CONDITIONS: High humidity proxy
    if 't2m' in data.columns and 'd2m' in data.columns:
        humidity_proxy = data['t2m'] - data['d2m']  # Lower = more humid
        humid_threshold = humidity_proxy.quantile(0.25)  # Bottom 25% = most humid
        data['humid_label'] = (humidity_proxy <= humid_threshold).astype(int)
    else:
        data['humid_label'] = 0
    
    # Print label distributions
    labels = ['drought_label', 'flood_label', 'rainy_label', 'heatwave_label', 'humid_label']
    for label in labels:
        if label in data.columns:
            dist = data[label].value_counts().sort_index()
            total = len(data)
            pos_pct = (dist.get(1, 0) / total * 100) if 1 in dist else 0
            print(f"   {label}: {dict(dist)} ({pos_pct:.1f}% positive)")
    
    return data

def select_features_smartly(data):
    """Select features based on what's actually available and useful"""
    
    # Core weather features (prioritized)
    priority_features = ['tp', 't2m', 'd2m']
    
    # Secondary features
    secondary_features = [
        'precip_7d_avg', 'temp_7d_avg', 'soil_moisture', 
        'wind_speed', 'is_rainy_season', 'precip_lag24h', 'temp_lag24h'
    ]
    
    # Additional features
    other_features = ['sp', 'u10', 'v10', 'heavy_rain', 'month', 'hour']
    
    # Build feature list from available columns
    selected = []
    
    # Add priority features if available
    for feat in priority_features:
        if feat in data.columns:
            selected.append(feat)
    
    # Add secondary features
    for feat in secondary_features:
        if feat in data.columns and len(selected) < 12:
            selected.append(feat)
    
    # Add other features
    for feat in other_features:
        if feat in data.columns and len(selected) < 15:
            selected.append(feat)
    
    print(f"🎯 Selected {len(selected)} features: {selected}")
    return selected

def train_model_robust(X, y, label_name, model_file):
    """Train model with better error handling and evaluation"""
    
    print(f"\n{'='*20} {label_name.upper()} MODEL {'='*20}")
    
    # Check class balance
    class_counts = y.value_counts()
    print(f"Class distribution: {dict(class_counts)}")
    
    if len(class_counts) < 2:
        print(f"⚠️ Only one class found for {label_name} - skipping")
        return None
    
    if class_counts.min() < 50:
        print(f"⚠️ Very few examples of minority class ({class_counts.min()}) - results may be unreliable")
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("⚠️ Cannot stratify - using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Try different models and pick the best
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=500))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, random_state=42, max_depth=10
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            print(f"\n{name}:")
            print(f"  Accuracy: {score:.4f}")
            print(f"  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
    
    if best_model is not None:
        # Save the best model
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        joblib.dump(best_model, model_file)
        print(f"\n✅ Best model ({best_name}) saved: {model_file}")
        print(f"   Final accuracy: {best_score:.4f}")
        return best_model
    else:
        print(f"❌ No model could be trained for {label_name}")
        return None

def main():
    print("🚀 Starting robust weather prediction training...")
    
    try:
        # Step 1: Load and prepare data
        data = load_and_prepare_data()
        
        # Step 2: Create better labels
        data = create_better_labels(data)
        
        # Step 3: Select features
        feature_columns = select_features_smartly(data)
        features = data[feature_columns]
        
        # Step 4: Train models for each label
        models_to_train = [
            ('drought_label', DROUGHT_MODEL_FILE, 'drought'),
            ('flood_label', FLOOD_MODEL_FILE, 'flood'),
            ('rainy_label', RAINY_MODEL_FILE, 'rainy'),
            ('heatwave_label', RAINY_MODEL_FILE.replace('rainy', 'heatwave'), 'heatwave'),
            ('humid_label', RAINY_MODEL_FILE.replace('rainy', 'humid'), 'humid')
        ]
        
        successful_models = 0
        
        for label_col, model_file, label_name in models_to_train:
            if label_col in data.columns:
                model = train_model_robust(features, data[label_col], label_name, model_file)
                if model is not None:
                    successful_models += 1
                    
                # Cleanup memory
                gc.collect()
        
        print(f"\n🎉 Training completed!")
        print(f"✅ Successfully trained {successful_models} out of {len(models_to_train)} models")
        
        if successful_models == 0:
            print("❌ No models were successfully trained. Check your data!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()