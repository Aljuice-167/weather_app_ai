# train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from config import PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE, FLOOD_MODEL_FILE

# Extra model file for rainy season
RAINY_MODEL_FILE = "models/rainy_season_model.pkl"


def load_data():
    """Load processed data with engineered features"""
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(f"{PROCESSED_DATA_FILE} not found. Run feature_engineer.py first.")
    return pd.read_csv(PROCESSED_DATA_FILE)


def prepare_features_targets(df):
    """Separate features and targets"""
    feature_cols = [
        "t2m", "d2m", "sp", "u10", "v10", "tp",
        "precip_7d_avg", "temp_7d_avg",
        "heavy_rain", "heat_wave", "soil_moisture", "wind_speed",
        "is_rainy_season", "precip_lag24h", "temp_lag24h",
        "latitude", "longitude"
    ]

    # X = input features
    X = df[feature_cols]

    # y = labels
    y_drought = df["drought_label"]
    y_flood = df["flood_label"]
    y_rainy = df["rainy_season_label"]

    return X, y_drought, y_flood, y_rainy


def train_and_evaluate(X, y, label_name):
    """Train and evaluate a RandomForest model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n=== {label_name.upper()} MODEL REPORT ===")
    print(classification_report(y_test, y_pred))

    return model


def main():
    print("Loading processed data...")
    df = load_data()

    print("Preparing features and targets...")
    X, y_drought, y_flood, y_rainy = prepare_features_targets(df)

    # --- Train drought model ---
    drought_model = train_and_evaluate(X, y_drought, "drought")
    joblib.dump(drought_model, DROUGHT_MODEL_FILE)
    print(f"Drought model saved to {DROUGHT_MODEL_FILE}")

    # --- Train flood model ---
    flood_model = train_and_evaluate(X, y_flood, "flood")
    joblib.dump(flood_model, FLOOD_MODEL_FILE)
    print(f"Flood model saved to {FLOOD_MODEL_FILE}")

    # --- Train rainy season model ---
    rainy_model = train_and_evaluate(X, y_rainy, "rainy_season")
    joblib.dump(rainy_model, RAINY_MODEL_FILE)
    print(f"Rainy season model saved to {RAINY_MODEL_FILE}")


if __name__ == "__main__":
    main()
