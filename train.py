# train.py (lightweight deep learning version)
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import config
from config import (
    PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE,
    FLOOD_MODEL_FILE, RAINY_MODEL_FILE,
    TEST_SIZE, RANDOM_STATE
)

# --- Ensure we train on feature-engineered dataset ---
FEATURES_FILE = PROCESSED_DATA_FILE.replace(".csv", "_features.csv")


def load_processed_data():
    """Load preprocessed + engineered features"""
    target_file = FEATURES_FILE if os.path.exists(FEATURES_FILE) else PROCESSED_DATA_FILE
    if not os.path.exists(target_file):
        raise FileNotFoundError(
            f"❌ No processed dataset found.\n"
            f"Tried: {FEATURES_FILE} and {PROCESSED_DATA_FILE}\n"
            f"➡️ Please run feature_engineer.py first."
        )
    print(f"📂 Loading processed data from {target_file} ...")
    return pd.read_csv(target_file)


def add_labels_if_missing(data: pd.DataFrame) -> pd.DataFrame:
    """Add drought, flood, and rainy season labels if missing."""
    if "drought_label" not in data.columns:
        print("ℹ️ Generating drought_label ...")
        if "precip_7d_avg" in data.columns:
            data["drought_label"] = (data["precip_7d_avg"] < 0.01).astype(int)
        else:
            raise KeyError("Missing precip_7d_avg for drought_label generation.")

    if "flood_label" not in data.columns:
        print("ℹ️ Generating flood_label ...")
        if "precip_7d_avg" in data.columns and "soil_moisture" in data.columns:
            data["flood_label"] = (
                (data["precip_7d_avg"] > 0.05) & (data["soil_moisture"] > 2)
            ).astype(int)
        else:
            raise KeyError("Missing features for flood_label generation.")

    if "rainy_label" not in data.columns:
        print("ℹ️ Generating rainy_label ...")
        if "precip_7d_avg" in data.columns:
            data["rainy_label"] = (data["precip_7d_avg"] > 0.1).astype(int)
        else:
            raise KeyError("Missing precip_7d_avg for rainy_label generation.")

    return data


def build_nn_model(input_dim: int) -> Sequential:
    """Build a smaller feedforward neural network for binary classification."""
    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(X, y, label, model_file):
    """Train deep learning model, evaluate, and save."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    # Save scaler for inference
    scaler_file = model_file.replace(".h5", "_scaler.pkl")
    joblib.dump(scaler, scaler_file)
    print(f"💾 Scaler saved to {scaler_file}")

    # Build NN
    model = build_nn_model(X_train.shape[1])

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=model_file,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Train
    print(f"\n🔍 Training Neural Network for {label} prediction...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=40,           # fewer epochs
        batch_size=16,       # smaller batch size
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ NN accuracy for {label}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    print(f"💾 {label.capitalize()} model saved to {model_file}")


def main():
    print("🚀 Starting deep learning training pipeline...")

    # Step 1: Load processed features
    data = load_processed_data()

    # Step 2: Ensure labels exist
    data = add_labels_if_missing(data)

    # Step 3: Separate features and targets
    feature_cols = [c for c in data.columns if c not in ["drought_label", "flood_label", "rainy_label"]]
    features = data[feature_cols].values.astype("float32")

    # Step 4: Train models
    train_and_evaluate(features, data["drought_label"].values, "drought", DROUGHT_MODEL_FILE.replace(".pkl", ".h5"))
    train_and_evaluate(features, data["flood_label"].values, "flood", FLOOD_MODEL_FILE.replace(".pkl", ".h5"))
    train_and_evaluate(features, data["rainy_label"].values, "rainy", RAINY_MODEL_FILE.replace(".pkl", ".h5"))

    print("\n🎉 Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
