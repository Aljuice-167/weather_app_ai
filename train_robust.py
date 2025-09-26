# train_deep_learning.py - Deep learning approach for weather prediction
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import gc
import warnings
warnings.filterwarnings('ignore')

from config import (
    PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE,
    FLOOD_MODEL_FILE, RAINY_MODEL_FILE,
    TEST_SIZE, RANDOM_STATE
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


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
    
    # Handle zero precipitation issue with more realistic synthetic data
    if 'tp' in data.columns and data['tp'].max() == 0:
        print("⚠️ All precipitation values are zero!")
        print("   Creating synthetic precipitation based on seasonal/geographical patterns...")
        
        # Create more realistic synthetic precipitation
        if 'd2m' in data.columns and 't2m' in data.columns:
            # Use humidity, temperature, season, and location
            humidity_proxy = (data['t2m'] - data['d2m'])  # Temperature-dewpoint spread
            
            # Seasonal effects (Ghana has wet/dry seasons)
            if 'month' in data.columns:
                wet_season_months = [4, 5, 6, 7, 8, 9, 10]  # April to October
                seasonal_factor = data['month'].apply(lambda x: 2.0 if x in wet_season_months else 0.3)
            else:
                seasonal_factor = 1.0
            
            # Geographic effects (higher elevation = more rain)
            if 'latitude' in data.columns:
                # Northern Ghana (higher lat) is generally drier
                geo_factor = 1.5 - 0.1 * (data['latitude'] - data['latitude'].min())
            else:
                geo_factor = 1.0
            
            # Create realistic precipitation with proper distribution
            np.random.seed(42)
            base_precip = np.random.gamma(0.5, 1.0, len(data))  # Gamma distribution (realistic for precip)
            
            # Apply modifying factors
            humidity_effect = np.clip((10 - humidity_proxy) / 10, 0.1, 2.0)  # High humidity = more rain
            
            data['tp'] = base_precip * humidity_effect * seasonal_factor * geo_factor
            data['tp'] = np.clip(data['tp'], 0, 25)  # Reasonable max precipitation
            
            # Add some zero precipitation days (dry periods)
            dry_probability = 0.7  # 70% of time no rain (realistic for many climates)
            dry_mask = np.random.random(len(data)) < dry_probability
            data.loc[dry_mask, 'tp'] = 0
            
            print(f"   Created synthetic precipitation:")
            print(f"   Range: {data['tp'].min():.2f} to {data['tp'].max():.2f} mm")
            print(f"   Mean: {data['tp'].mean():.2f} mm")
            print(f"   Zero precipitation days: {(data['tp'] == 0).sum()} ({(data['tp'] == 0).mean()*100:.1f}%)")
    
    return data


def create_realistic_labels(data):
    """Create more realistic weather labels with proper thresholds"""
    print("🏷️ Creating realistic weather labels...")
    
    # Get data statistics for thresholding
    if 'tp' in data.columns:
        tp_stats = data['tp'].describe()
        print(f"   Precipitation range: {tp_stats['min']:.2f} to {tp_stats['max']:.2f} mm")
    
    if 't2m' in data.columns:
        temp_stats = data['t2m'].describe()
        print(f"   Temperature range: {temp_stats['min']:.1f} to {temp_stats['max']:.1f} °C")
    
    # DROUGHT: Use meteorological drought definition
    if 'tp' in data.columns and 't2m' in data.columns:
        # Drought = below normal precipitation for extended period + high temp
        precip_threshold = data['tp'].quantile(0.2)  # Bottom 20%
        temp_threshold = data['t2m'].quantile(0.8)   # Top 20%
        
        # Consider 7-day average if available
        if 'precip_7d_avg' in data.columns:
            drought_precip = data['precip_7d_avg'] <= data['precip_7d_avg'].quantile(0.15)
        else:
            drought_precip = data['tp'] <= precip_threshold
            
        drought_temp = data['t2m'] >= temp_threshold
        data['drought_label'] = (drought_precip & drought_temp).astype(int)
    else:
        data['drought_label'] = 0
    
    # FLOOD: High precipitation events (more conservative)
    if 'tp' in data.columns:
        # Use 95th percentile as flood threshold (extreme events)
        flood_threshold = data['tp'].quantile(0.95)
        # Also ensure it's at least 10mm (meaningful precipitation)
        flood_threshold = max(flood_threshold, 10.0)
        data['flood_label'] = (data['tp'] >= flood_threshold).astype(int)
    else:
        data['flood_label'] = 0
    
    # RAINY: Any significant precipitation (more realistic threshold)
    if 'tp' in data.columns:
        # Use 0.5mm as minimum significant precipitation
        rain_threshold = max(0.5, data['tp'].quantile(0.7))  # At least 0.5mm
        data['rainy_label'] = (data['tp'] >= rain_threshold).astype(int)
    else:
        data['rainy_label'] = 0
    
    # Print label distributions
    labels = ['drought_label', 'flood_label', 'rainy_label']
    for label in labels:
        if label in data.columns:
            dist = data[label].value_counts().sort_index()
            total = len(data)
            pos_pct = (dist.get(1, 0) / total * 100) if 1 in dist else 0
            print(f"   {label}: {dict(dist)} ({pos_pct:.1f}% positive)")
    
    return data


def create_neural_network(input_dim, label_name):
    """Create appropriate neural network architecture based on the prediction task"""
    
    model = keras.Sequential([
        # Input layer with moderate dropout
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers with decreasing size
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Use appropriate optimizer and loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def create_lstm_model(sequence_length, n_features):
    """Create LSTM model for time series weather prediction"""
    
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(25, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def prepare_sequences(data, features, target, sequence_length=24):
    """Prepare sequences for LSTM training"""
    
    # Sort by location and time
    if 'time' in data.columns:
        data_sorted = data.sort_values(['latitude', 'longitude', 'time'])
    else:
        data_sorted = data.copy()
    
    X_sequences = []
    y_sequences = []
    
    # Group by location
    if 'latitude' in data.columns and 'longitude' in data.columns:
        for (lat, lon), group in data_sorted.groupby(['latitude', 'longitude']):
            if len(group) >= sequence_length:
                X_group = group[features].values
                y_group = group[target].values
                
                for i in range(len(group) - sequence_length + 1):
                    X_sequences.append(X_group[i:(i + sequence_length)])
                    y_sequences.append(y_group[i + sequence_length - 1])  # Predict current timestep
    else:
        # If no location info, treat as single time series
        X_group = data_sorted[features].values
        y_group = data_sorted[target].values
        
        for i in range(len(data_sorted) - sequence_length + 1):
            X_sequences.append(X_group[i:(i + sequence_length)])
            y_sequences.append(y_group[i + sequence_length - 1])
    
    return np.array(X_sequences), np.array(y_sequences)


def train_deep_learning_models(X, y, label_name, model_file, data=None, feature_names=None):
    """Train both feedforward and LSTM models"""
    
    print(f"\n{'='*20} {label_name.upper()} DEEP LEARNING {'='*20}")
    
    # Check class balance
    class_counts = pd.Series(y).value_counts()
    print(f"Class distribution: {dict(class_counts)}")
    
    if len(class_counts) < 2:
        print(f"⚠️ Only one class found for {label_name} - skipping")
        return None
    
    minority_class_pct = class_counts.min() / len(y) * 100
    print(f"Minority class: {minority_class_pct:.2f}%")
    
    # Use time-based split for more realistic evaluation
    if data is not None and 'time' in data.columns:
        # Sort by time for temporal split
        time_sorted_idx = data['time'].argsort()
        split_idx = int(len(time_sorted_idx) * 0.8)
        
        train_idx = time_sorted_idx[:split_idx]
        test_idx = time_sorted_idx[split_idx:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print("Using temporal split (train on earlier data, test on later)")
    else:
        # Fallback to random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(class_counts) > 1 else None
        )
        print("Using random split")
    
    # Scale features
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models_to_try = {}
    
    # 1. Logistic Regression (baseline)
    lr_model = LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        class_weight='balanced',  # Handle imbalanced data
        C=0.1  # Add regularization
    )
    lr_model.fit(X_train_scaled, y_train)
    models_to_try['Logistic Regression'] = (lr_model, scaler, X_test_scaled)
    
    # 2. Neural Network
    nn_model = create_neural_network(X_train_scaled.shape[1], label_name)
    
    # Calculate class weights for imbalanced data
    class_weight = {0: 1.0, 1: len(y_train) / (2 * sum(y_train))} if sum(y_train) > 0 else None
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
    
    # Train neural network
    history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=1024,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0
    )
    
    models_to_try['Neural Network'] = (nn_model, scaler, X_test_scaled)
    
    # 3. LSTM (if we have time series data)
    if data is not None and len(X_train) > 1000:  # Only if enough data
        try:
            print("Preparing LSTM sequences...")
            sequence_length = 24  # 24 hours
            
            X_seq, y_seq = prepare_sequences(
                data.iloc[train_idx] if data is not None else pd.DataFrame(X_train), 
                feature_names if feature_names else list(range(X_train.shape[1])),
                f"{label_name}_label",
                sequence_length
            )
            
            if len(X_seq) > 100:  # Minimum sequences needed
                # Scale sequences
                X_seq_scaled = scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
                
                # Split sequences
                seq_split = int(len(X_seq_scaled) * 0.8)
                X_seq_train, X_seq_val = X_seq_scaled[:seq_split], X_seq_scaled[seq_split:]
                y_seq_train, y_seq_val = y_seq[:seq_split], y_seq[seq_split:]
                
                lstm_model = create_lstm_model(sequence_length, X_seq.shape[-1])
                
                lstm_history = lstm_model.fit(
                    X_seq_train, y_seq_train,
                    epochs=30,
                    batch_size=256,
                    validation_data=(X_seq_val, y_seq_val),
                    callbacks=callbacks,
                    verbose=0
                )
                
                models_to_try['LSTM'] = (lstm_model, scaler, X_test_scaled)  # Note: using regular test data for comparison
                print(f"LSTM trained on {len(X_seq)} sequences")
            
        except Exception as e:
            print(f"LSTM training failed: {e}")
    
    # Evaluate all models
    best_model = None
    best_score = 0
    best_name = ""
    best_scaler = None
    
    for name, (model, model_scaler, X_test_model) in models_to_try.items():
        try:
            if name == 'LSTM':
                # For LSTM, we'd need to prepare test sequences properly
                # For simplicity, we'll skip LSTM evaluation here
                continue
                
            y_pred = model.predict(X_test_model)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            else:
                y_pred_proba = y_pred.flatten()
                y_pred = (y_pred > 0.5).astype(int).flatten()
            
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_score = 0.5
            
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC-ROC: {auc_score:.4f}")
            
            # Print detailed classification report
            report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
            print(f"  Classification Report:\n{report}")
            
            # Use AUC as the primary metric for imbalanced data
            combined_score = 0.6 * auc_score + 0.4 * accuracy
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = model
                best_name = name
                best_scaler = model_scaler
                
        except Exception as e:
            print(f"  ❌ {name} evaluation failed: {e}")
    
    # Save the best model
    if best_model is not None:
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        
        # Save model and scaler together
        model_package = {
            'model': best_model,
            'scaler': best_scaler,
            'model_type': best_name,
            'features': feature_names if feature_names else list(range(X.shape[1]))
        }
        
        joblib.dump(model_package, model_file)
        print(f"\n✅ Best model ({best_name}) saved: {model_file}")
        print(f"   Combined score (0.6*AUC + 0.4*Acc): {best_score:.4f}")
        
        return best_model
    else:
        print(f"❌ No model could be trained for {label_name}")
        return None


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
    other_features = ['sp', 'u10', 'v10', 'month', 'hour']
    
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


def main():
    print("🚀 Starting deep learning weather prediction training...")
    
    try:
        # Step 1: Load and prepare data
        data = load_and_prepare_data()
        
        # Step 2: Create realistic labels
        data = create_realistic_labels(data)
        
        # Step 3: Select features
        feature_columns = select_features_smartly(data)
        features = data[feature_columns].values
        
        # Sample data if too large (for memory efficiency)
        if len(data) > 2000000:
            print(f"⚠️ Sampling {2000000} rows from {len(data)} for training efficiency")
            sample_idx = np.random.choice(len(data), 2000000, replace=False)
            data = data.iloc[sample_idx].reset_index(drop=True)
            features = features[sample_idx]
        
        # Step 4: Train deep learning models for each label
        models_to_train = [
            ('drought_label', DROUGHT_MODEL_FILE, 'drought'),
            ('flood_label', FLOOD_MODEL_FILE, 'flood'),
            ('rainy_label', RAINY_MODEL_FILE, 'rainy'),
        ]
        
        successful_models = 0
        
        for label_col, model_file, label_name in models_to_train:
            if label_col in data.columns:
                model = train_deep_learning_models(
                    features, 
                    data[label_col].values, 
                    label_name, 
                    model_file,
                    data=data,
                    feature_names=feature_columns
                )
                if model is not None:
                    successful_models += 1
                
                # Cleanup memory
                gc.collect()
                tf.keras.backend.clear_session()
        
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
