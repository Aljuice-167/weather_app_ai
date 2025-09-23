# predictor.py
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

class ExtremeWeatherPredictor:
    def __init__(self):
        self.drought_model = None
        self.flood_model = None
        self.features = None
        self.target_drought = None
        self.target_flood = None

    def load_data(self):
        """Load feature-engineered data"""
        data = pd.read_csv(PROCESSED_DATA_FILE)
        
        # Define all possible features
        all_features = [
            'tp', 't2m', 'd2m', 'sp', 'u10', 'v10', 'precip_7d_avg',
            'temp_7d_avg', 'soil_moisture', 'wind_speed', 'is_rainy_season',
            'precip_lag24h', 'temp_lag24h'
        ]
        
        # Use only features that exist in the data
        available_features = [f for f in all_features if f in data.columns]
        print(f"Using features: {available_features}")
        
        self.features = data[available_features]
        
        # Create targets (simplified definitions)
        if 'precip_7d_avg' in data.columns and 't2m' in data.columns:
            data['drought'] = ((data['precip_7d_avg'] < 0.01) & (data['t2m'] > 30)).astype(int)
        else:
            data['drought'] = 0
        
        if 'precip_7d_avg' in data.columns and 'is_rainy_season' in data.columns:
            data['flood'] = ((data['precip_7d_avg'] > 0.03) & (data['is_rainy_season'] == 1)).astype(int)
        else:
            data['flood'] = 0
        
        self.target_drought = data['drought']
        self.target_flood = data['flood']

    def train_models(self):
        """Train drought and flood prediction models"""
        if self.features is None:
            self.load_data()
        
        # Split data
        X_train, X_test, y_train_drought, y_test_drought = train_test_split(
            self.features, self.target_drought, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        _, _, y_train_flood, y_test_flood = train_test_split(
            self.features, self.target_flood, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Train drought model
        self.drought_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        self.drought_model.fit(X_train, y_train_drought)
        drought_pred = self.drought_model.predict(X_test)
        print("Drought Model Report:")
        print(classification_report(y_test_drought, drought_pred))
        
        # Train flood model
        self.flood_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        self.flood_model.fit(X_train, y_train_flood)
        flood_pred = self.flood_model.predict(X_test)
        print("\nFlood Model Report:")
        print(classification_report(y_test_flood, flood_pred))

    def save_models(self):
        """Save trained models"""
        os.makedirs(os.path.dirname(DROUGHT_MODEL_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(FLOOD_MODEL_FILE), exist_ok=True)
        
        joblib.dump(self.drought_model, DROUGHT_MODEL_FILE)
        joblib.dump(self.flood_model, FLOOD_MODEL_FILE)

    def load_models(self):
        """Load pre-trained models"""
        self.drought_model = joblib.load(DROUGHT_MODEL_FILE)
        self.flood_model = joblib.load(FLOOD_MODEL_FILE)

    def predict(self, input_data):
        """Make predictions using loaded models"""
        if self.drought_model is None or self.flood_model is None:
            self.load_models()
        
        # Ensure input data has all required features
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Get the feature names the model was trained on
        feature_names = self.drought_model.feature_names_in_
        
        # Ensure input has all required features
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0.0  # Default value
        
        # Select only the features the model expects
        input_df = input_df[feature_names]
        
        drought_prob = self.drought_model.predict_proba(input_df)[:, 1]
        flood_prob = self.flood_model.predict_proba(input_df)[:, 1]
        
        return {
            'drought_probability': drought_prob[0],
            'flood_probability': flood_prob[0],
            'drought_risk': 'High' if drought_prob[0] > 0.7 else 'Medium' if drought_prob[0] > 0.4 else 'Low',
            'flood_risk': 'High' if flood_prob[0] > 0.7 else 'Medium' if flood_prob[0] > 0.4 else 'Low'
        }