# train.py (lightweight version)
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

# Import config
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import (
        PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE,
        FLOOD_MODEL_FILE, RAINY_MODEL_FILE,
        TEST_SIZE, RANDOM_STATE
    )
except ImportError:
    from config import (
        PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE,
        FLOOD_MODEL_FILE, RAINY_MODEL_FILE,
        TEST_SIZE, RANDOM_STATE
    )


def load_processed_data():
    """Load preprocessed features from CSV."""
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f"❌ Processed file not found: {PROCESSED_DATA_FILE}\n"
            "➡️ Please run feature_engineer.py first to generate it."
        )
    print(f"📂 Loading processed data from {PROCESSED_DATA_FILE} ...")
    return pd.read_csv(PROCESSED_DATA_FILE)


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
            # Example: rainy season when 7-day avg precipitation > 0.1
            data["rainy_label"] = (data["precip_7d_avg"] > 0.1).astype(int)
        else:
            raise KeyError("Missing precip_7d_avg for rainy_label generation.")

    return data


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

    # Step 1: Load processed features
    data = load_processed_data()

    # Step 2: Ensure labels exist
    data = add_labels_if_missing(data)

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
