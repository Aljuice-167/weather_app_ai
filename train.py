# train.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib


# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import (
        PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE, FLOOD_MODEL_FILE,
        TEST_SIZE, RANDOM_STATE
    )
except ImportError:
    from config import (
        PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE, FLOOD_MODEL_FILE,
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
    """Add drought and flood labels if they are missing."""
    if "drought_label" not in data.columns:
        print("ℹ️ Generating drought_label ...")
        # Example rule: drought if precip_7d_avg < threshold
        if "precip_7d_avg" in data.columns:
            data["drought_label"] = (data["precip_7d_avg"] < 0.01).astype(int)
        else:
            raise KeyError("Missing precip_7d_avg for drought_label generation.")


    if "flood_label" not in data.columns:
        print("ℹ️ Generating flood_label ...")
        # Example rule: flood if precipitation high + soil moisture high
        if "precip_7d_avg" in data.columns and "soil_moisture" in data.columns:
            data["flood_label"] = (
                (data["precip_7d_avg"] > 0.05) & (data["soil_moisture"] > 2)
            ).astype(int)
        else:
            raise KeyError("Missing features for flood_label generation.")


    return data




def build_model():
    """Builds model training pipelines with hyperparameter search."""
    pipelines = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE
            ))
        ]),
    }


    param_grids = {
        "RandomForest": {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, None]},
        "LogisticRegression": {"clf__C": [0.1, 1.0, 10.0]},
        "XGBoost": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 6, 10]},
    }


    return pipelines, param_grids




def train_and_evaluate(X, y, label):
    """Train multiple models, evaluate, and save the best one."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


    pipelines, param_grids = build_model()
    best_model = None
    best_score = 0
    best_name = None


    for name, pipeline in pipelines.items():
        print(f"\n🔍 Training {name} for {label} prediction...")
        grid = GridSearchCV(pipeline, param_grids[name], cv=3, n_jobs=-1, scoring="accuracy")
        grid.fit(X_train, y_train)


        y_pred = grid.best_estimator_.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {name} accuracy for {label}: {acc:.4f}")
        print(classification_report(y_test, y_pred))


        if acc > best_score:
            best_score = acc
            best_model = grid.best_estimator_
            best_name = name


    print(f"\n🏆 Best model for {label}: {best_name} (accuracy {best_score:.4f})")
    return best_model




def main():
    print("🚀 Starting training pipeline...")


    # Step 1: Load processed features
    data = load_processed_data()


    # Step 2: Ensure labels exist
    data = add_labels_if_missing(data)


    # Step 3: Separate features and targets
    features = data.drop(columns=["drought_label", "flood_label"])
    drought_target = data["drought_label"]
    flood_target = data["flood_label"]


    # Step 4: Train drought model
    drought_model = train_and_evaluate(features, drought_target, "drought")
    joblib.dump(drought_model, DROUGHT_MODEL_FILE)
    print(f"💾 Drought model saved to {DROUGHT_MODEL_FILE}")


    # Step 5: Train flood model
    flood_model = train_and_evaluate(features, flood_target, "flood")
    joblib.dump(flood_model, FLOOD_MODEL_FILE)
    print(f"💾 Flood model saved to {FLOOD_MODEL_FILE}")


    print("\n🎉 Training pipeline completed successfully!")




if __name__ == "__main__":
    main()
