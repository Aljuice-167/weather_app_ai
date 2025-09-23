# train.py
import os
import sys

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import PROCESSED_DATA_FILE
except ImportError:
    from config import PROCESSED_DATA_FILE

# Add the python directory to the path for imports
python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python')
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from data_collector import GhanaWeatherDataCollector
from feature_engineer import WeatherFeatureEngineer
from predictor import ExtremeWeatherPredictor

def main():
    """Main training pipeline"""
    print("Starting training pipeline...")
    
    # Step 1: Collect and preprocess data
    print("\nStep 1: Collecting and preprocessing data...")
    collector = GhanaWeatherDataCollector()
    collector.load_raw_data()  # This will now load and combine all files
    collector.preprocess_data()
    collector.save_processed_data()
    
    # Step 2: Feature engineering
    print("\nStep 2: Engineering features...")
    engineer = WeatherFeatureEngineer()
    features = engineer.create_features()
    engineer.save_features(features)
    
    # Step 3: Train models
    print("\nStep 3: Training models...")
    predictor = ExtremeWeatherPredictor()
    predictor.load_data()
    predictor.train_models()
    predictor.save_models()
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()