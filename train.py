# train.py (Colab-friendly, with auto-sampling & feature file detection)
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

from config import (
    PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE,
    FLOOD_MODEL_FILE, RAINY_MODEL_FILE,
    TEST_SIZE, RANDOM_STATE
)

# Max rows to load into Colab memory
MAX_ROWS = 200_000


def load_processed_data():
    """Load engineered features if available, else fall back to base file."""
    features_file = PROCESSED_DATA_FILE.replace(".csv", "_features.csv")

    if os.path.exists(features_file):
        print(f"📂 Loading engineered features from {features_file} ...")
        df = pd.read_csv(features_file)
    elif os.path.exists(PROCESSED_DATA_FILE):
        print(f"📂 Loading base processed data from {PROCESSED_DATA_FILE} ...")
        df = pd.read_csv(PROCESSED_DATA_FILE)
    else:
        raise FileNotFoundError("❌ Neither features file nor base processed file found!")

    print(f"   ✅ Data loaded. Shape: {df.shape}")

    # Auto-sample if dataset is too big for Colab
    if df.shape[0] > MAX_ROWS:
        print(f"⚠️ Dataset too large ({df.shape[0]} rows). Sampling {MAX_ROWS} rows for Colab training...")
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"   🔎 Sampled dataset shape: {df.shape}")

    return df


def build_model(model_name: str):
    """Return lightweight models."""
    if model_name == "LogisticRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
        ])
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(X, y, label, model_file):
    """Train lightweight models, evaluate, and save best one."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    best_model = None
    best_score = 0
    best_name = None

    for model_name in ["LogisticRegression", "XGBoost"]:
        print(f"\n🔍 Training {model_name} for {label} prediction...")
        model = build_model(model_name)

        if model_name == "XGBoost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {model_name} accuracy for {label}: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = model_name

    print(f"\n🏆 Best model for {label}: {best_name} (accuracy {best_score:.4f})")
    joblib.dump(best_model, model_file)
    print(f"💾 {label.capitalize()} model saved to {model_file}")


def main():
    print("🚀 Starting lightweight training pipeline...")

    # Step 1: Load processed/engineered data
    data = load_processed_data()

    # Step 2: Ensure labels exist
    for label, rule in {
        "drought_label": lambda df: (df["precip_7d_avg"] < 0.01).astype(int) if "precip_7d_avg" in df.columns else None,
        "flood_label": lambda df: (
            (df["precip_7d_avg"] > 0.05) & (df["soil_moisture"] > 2)
        ).astype(int) if "precip_7d_avg" in df.columns and "soil_moisture" in df.columns else None,
        "rainy_label": lambda df: (df["precip_7d_avg"] > 0.1).astype(int) if "precip_7d_avg" in df.columns else None,
    }.items():
        if label not in data.columns:
            print(f"ℹ️ Generating {label} ...")
            series = rule(data)
            if series is None:
                raise KeyError(f"Missing required features to generate {label}")
            data[label] = series

    # Step 3: Separate features and targets
    feature_cols = [c for c in data.columns if c not in ["drought_label", "flood_label", "rainy_label"]]
    features = data[feature_cols]

    # Step 4: Train models
    train_and_evaluate(features, data["drought_label"], "drought", DROUGHT_MODEL_FILE)
    train_and_evaluate(features, data["flood_label"], "flood", FLOOD_MODEL_FILE)
    train_and_evaluate(features, data["rainy_label"], "rainy", RAINY_MODEL_FILE)

    print("\n🎉 Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
