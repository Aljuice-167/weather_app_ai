

# train.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import config

def train_and_save_model(X, y, model_path, model_name):
    """Train RandomForest model and save it."""
    print(f"\n🚀 Training {model_name} model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print(f"   📊 Data split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

    model = RandomForestClassifier(random_state=config.RANDOM_STATE, n_estimators=100)
    model.fit(X_train, y_train)

    print("   ✅ Training complete. Evaluating...")
    y_pred = model.predict(X_test)

    print(f"\n📌 {model_name} Model Performance:")
    print(f"   - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"💾 {model_name} model saved at: {model_path}")

def main():
    print("🚀 Starting lightweight training pipeline...")

    # 1. Load data
    print(f"📂 Loading processed data from {config.PROCESSED_DATA_FILE} ...")
    df = pd.read_csv(config.PROCESSED_DATA_FILE)

    print(f"   ✅ Data loaded. Shape: {df.shape}")

    # 2. Feature selection
    feature_cols = [
        "latitude", "longitude", "temperature", "precipitation", "soil_moisture",
        "ndvi", "year", "month", "dayofyear", "sin_dayofyear", "cos_dayofyear"
    ]
    X = df[feature_cols]
    
    print("🔎 Features selected for training:")
    print("   ", feature_cols)

    # 3. Train drought model
    if "drought_label" in df.columns:
        y_drought = df["drought_label"]
        train_and_save_model(X, y_drought, config.DROUGHT_MODEL_FILE, "Drought")

    # 4. Train flood model
    if "flood_label" in df.columns:
        y_flood = df["flood_label"]
        train_and_save_model(X, y_flood, config.FLOOD_MODEL_FILE, "Flood")

if __name__ == "__main__":
    main()

