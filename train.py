# train.py (memory-optimized for Colab)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
import gc  # Garbage collection

# Import config
from config import (
    PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE,
    FLOOD_MODEL_FILE, RAINY_MODEL_FILE,
    TEST_SIZE, RANDOM_STATE
)

# --- Ensure we train on feature-engineered dataset ---
FEATURES_FILE = PROCESSED_DATA_FILE.replace(".csv", "_features.csv")

# Memory management settings
CHUNK_SIZE = 50000  # Process data in smaller chunks
MAX_FEATURES = 10   # Limit number of features to prevent memory issues

def monitor_memory():
    """Monitor memory usage"""
    import psutil
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}% ({memory.used/1e9:.1f}GB / {memory.total/1e9:.1f}GB)")

def load_processed_data_chunked():
    """Load data in chunks to manage memory"""
    target_file = FEATURES_FILE if os.path.exists(FEATURES_FILE) else PROCESSED_DATA_FILE
    if not os.path.exists(target_file):
        raise FileNotFoundError(
            f"❌ No processed dataset found.\n"
            f"Tried: {FEATURES_FILE} and {PROCESSED_DATA_FILE}\n"
            f"➡️ Please run feature_engineer.py first."
        )
    
    print(f"📂 Loading processed data from {target_file} ...")
    
    # First, get basic info about the file
    chunk_iter = pd.read_csv(target_file, chunksize=1000)
    first_chunk = next(chunk_iter)
    total_columns = list(first_chunk.columns)
    
    # Estimate file size
    file_size_mb = os.path.getsize(target_file) / (1024 * 1024)
    print(f"📊 File size: {file_size_mb:.1f} MB")
    print(f"📊 Columns: {len(total_columns)}")
    
    # If file is too large, use sampling
    if file_size_mb > 500:  # If larger than 500MB
        print("⚠️ Large file detected. Using sampling strategy...")
        return load_sampled_data(target_file, total_columns)
    else:
        # Load normally but monitor memory
        monitor_memory()
        data = pd.read_csv(target_file)
        monitor_memory()
        return data, total_columns

def load_sampled_data(target_file, total_columns):
    """Load a representative sample of the data"""
    print("📊 Creating stratified sample...")
    
    # Read in chunks and take every nth row
    sample_ratio = 0.1  # Use 10% of data
    sampled_chunks = []
    
    chunk_iter = pd.read_csv(target_file, chunksize=CHUNK_SIZE)
    
    for i, chunk in enumerate(chunk_iter):
        # Take every nth row to maintain temporal distribution
        n = max(1, int(1 / sample_ratio))
        sampled_chunk = chunk.iloc[::n].copy()
        sampled_chunks.append(sampled_chunk)
        
        if i % 10 == 0:  # Progress update
            print(f"   Processed chunk {i+1}, sampled {len(sampled_chunk)} rows")
        
        # Memory cleanup
        del chunk
        if i % 20 == 0:
            gc.collect()
    
    # Combine sampled chunks
    data = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    gc.collect()
    
    print(f"✅ Sampled dataset size: {len(data):,} rows")
    monitor_memory()
    
    return data, total_columns

def select_best_features(data, max_features=MAX_FEATURES):
    """Select most important features to reduce dimensionality"""
    print(f"🔍 Selecting top {max_features} features...")
    
    # Priority features (most important for weather prediction)
    priority_features = [
        'tp', 't2m', 'd2m', 'precip_7d_avg', 'temp_7d_avg', 
        'soil_moisture', 'is_rainy_season', 'precip_lag24h',
        'wind_speed', 'heavy_rain'
    ]
    
    # Get available priority features
    available_priority = [f for f in priority_features if f in data.columns]
    
    # Add some additional features if we have room
    other_features = [col for col in data.columns 
                     if col not in available_priority 
                     and col not in ['drought_label', 'flood_label', 'rainy_label', 'time']
                     and data[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    selected_features = available_priority + other_features[:max_features - len(available_priority)]
    selected_features = selected_features[:max_features]
    
    print(f"📋 Selected features: {selected_features}")
    return selected_features

def add_labels_if_missing(data: pd.DataFrame) -> pd.DataFrame:
    """Add drought, flood, and rainy season labels if missing."""
    print("ℹ️ Generating labels...")
    
    # Drought label - simplified definition
    if "drought_label" not in data.columns:
        if "tp" in data.columns and "t2m" in data.columns:
            data["drought_label"] = ((data["tp"] < 0.001) & (data["t2m"] > 30)).astype(int)
        elif "precip_7d_avg" in data.columns:
            data["drought_label"] = (data["precip_7d_avg"] < 0.01).astype(int)
        else:
            data["drought_label"] = 0
    
    # Flood label - simplified definition
    if "flood_label" not in data.columns:
        if "tp" in data.columns:
            data["flood_label"] = (data["tp"] > 0.05).astype(int)  # Heavy rain threshold
        elif "precip_7d_avg" in data.columns:
            data["flood_label"] = (data["precip_7d_avg"] > 0.03).astype(int)
        else:
            data["flood_label"] = 0
    
    # Rainy label
    if "rainy_label" not in data.columns:
        if "tp" in data.columns:
            data["rainy_label"] = (data["tp"] > 0.001).astype(int)  # Any significant rain
        else:
            data["rainy_label"] = 0
    
    # Print label distribution
    for label in ['drought_label', 'flood_label', 'rainy_label']:
        if label in data.columns:
            dist = data[label].value_counts()
            print(f"   {label}: {dict(dist)}")
    
    return data

def build_lightweight_model():
    """Return a simple, memory-efficient model"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=200,  # Reduced iterations
            random_state=RANDOM_STATE,
            solver='lbfgs'  # Memory efficient solver
        ))
    ])

def train_and_evaluate(X, y, label, model_file):
    """Train lightweight model with memory management"""
    print(f"\n🔍 Training model for {label} prediction...")
    monitor_memory()
    
    # Check class distribution
    class_dist = y.value_counts()
    print(f"   Class distribution: {dict(class_dist)}")
    
    # Skip if too few positive examples
    if len(class_dist) < 2 or class_dist.min() < 10:
        print(f"⚠️ Insufficient data for {label} - skipping model training")
        return
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
            stratify=y if len(class_dist) > 1 else None
        )
    except ValueError as e:
        print(f"⚠️ Could not stratify {label} - using random split: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    
    # Train model
    model = build_lightweight_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy for {label}: {acc:.4f}")
    
    # Clean memory before saving
    del X_train, X_test, y_train, y_test, y_pred
    gc.collect()
    
    # Save model
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    joblib.dump(model, model_file)
    print(f"💾 {label.capitalize()} model saved to {model_file}")

def main():
    print("🚀 Starting memory-optimized training pipeline...")
    monitor_memory()
    
    try:
        # Step 1: Load processed features (with sampling if needed)
        data, all_columns = load_processed_data_chunked()
        
        # Step 2: Select best features to reduce memory usage
        selected_features = select_best_features(data)
        
        # Step 3: Ensure labels exist
        data = add_labels_if_missing(data)
        
        # Step 4: Prepare features and targets
        features = data[selected_features].copy()
        
        # Clean up intermediate data
        del data[selected_features]  # Remove from original dataframe
        gc.collect()
        monitor_memory()
        
        # Step 5: Train models one by one (to manage memory)
        target_models = [
            ('drought_label', DROUGHT_MODEL_FILE),
            ('flood_label', FLOOD_MODEL_FILE),
            ('rainy_label', RAINY_MODEL_FILE)
        ]
        
        for target_col, model_file in target_models:
            if target_col in data.columns:
                print(f"\n{'='*50}")
                target = data[target_col].copy()
                train_and_evaluate(features, target, target_col.replace('_label', ''), model_file)
                del target  # Clean up
                gc.collect()
            else:
                print(f"⚠️ {target_col} not found in data")
        
        print("\n🎉 Training pipeline completed successfully!")
        monitor_memory()
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        # Try emergency cleanup
        gc.collect()
        monitor_memory()

if __name__ == "__main__":
    main()