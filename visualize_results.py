# visualize_results.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from pandas.plotting import andrews_curves
import joblib
from matplotlib.gridspec import GridSpec

# Import config after setting up Colab environment
try:
    from colab_setup import setup_colab_environment
    setup_colab_environment()
    from config import PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE, FLOOD_MODEL_FILE, TEST_SIZE, RANDOM_STATE
except ImportError:
    from config import PROCESSED_DATA_FILE, DROUGHT_MODEL_FILE, FLOOD_MODEL_FILE, TEST_SIZE, RANDOM_STATE

class WeatherModelVisualizer:
    def __init__(self):
        self.data = None
        self.features = None
        self.targets = None
        self.drought_model = None
        self.flood_model = None
        self.X_train = None
        self.X_test = None
        self.y_train_drought = None
        self.y_test_drought = None
        self.y_train_flood = None
        self.y_test_flood = None
        self.drought_pred = None
        self.flood_pred = None
        self.drought_prob = None
        self.flood_prob = None
        
    def load_data_and_models(self):
        """Load processed data and trained models"""
        print("Loading data and models...")
        
        # Load data
        self.data = pd.read_csv(PROCESSED_DATA_FILE)
        
        # Define features
        feature_cols = [
            'tp', 't2m', 'd2m', 'sp', 'u10', 'v10', 'precip_7d_avg',
            'temp_7d_avg', 'soil_moisture', 'wind_speed', 'is_rainy_season',
            'precip_lag24h', 'temp_lag24h'
        ]
        
        # Use only features that exist in the data
        available_features = [f for f in feature_cols if f in self.data.columns]
        print(f"Using features: {available_features}")
        
        self.features = self.data[available_features]
        
        # Create targets
        if 'precip_7d_avg' in self.data.columns and 't2m' in self.data.columns:
            self.data['drought'] = ((self.data['precip_7d_avg'] < 0.01) & (self.data['t2m'] > 30)).astype(int)
        else:
            self.data['drought'] = 0
        
        if 'precip_7d_avg' in self.data.columns and 'is_rainy_season' in self.data.columns:
            self.data['flood'] = ((self.data['precip_7d_avg'] > 0.03) & (self.data['is_rainy_season'] == 1)).astype(int)
        else:
            self.data['flood'] = 0
        
        self.targets = self.data[['drought', 'flood']]
        
        # Split data
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train_drought, self.y_test_drought = train_test_split(
            self.features, self.targets['drought'], test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        _, _, self.y_train_flood, self.y_test_flood = train_test_split(
            self.features, self.targets['flood'], test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Load models
        self.drought_model = joblib.load(DROUGHT_MODEL_FILE)
        self.flood_model = joblib.load(FLOOD_MODEL_FILE)
        
        # Make predictions
        self.drought_pred = self.drought_model.predict(self.X_test)
        self.flood_pred = self.flood_model.predict(self.X_test)
        self.drought_prob = self.drought_model.predict_proba(self.X_test)[:, 1]
        self.flood_prob = self.flood_model.predict_proba(self.X_test)[:, 1]
        
        print("Data and models loaded successfully!")
    
    def plot_tsne(self, sample_size=5000):
        """Plot t-SNE visualization of the data"""
        print("Generating t-SNE visualization...")
        
        # Sample data for t-SNE (to reduce computation time)
        if len(self.X_test) > sample_size:
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[indices]
            y_drought_sample = self.y_test_drought.iloc[indices]
            y_flood_sample = self.y_test_flood.iloc[indices]
        else:
            X_sample = self.X_test
            y_drought_sample = self.y_test_drought
            y_flood_sample = self.y_test_flood
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(X_scaled)
        
        # Create DataFrame with t-SNE results
        tsne_df = pd.DataFrame({
            'tsne-1': tsne_results[:, 0],
            'tsne-2': tsne_results[:, 1],
            'drought': y_drought_sample.values,
            'flood': y_flood_sample.values
        })
        
        # Plot t-SNE results
        plt.figure(figsize=(16, 7))
        
        # Drought plot
        plt.subplot(1, 2, 1)
        sns.scatterplot(
            x='tsne-1', y='tsne-2',
            hue='drought',
            palette={0: 'blue', 1: 'red'},
            data=tsne_df,
            legend='full',
            alpha=0.7
        )
        plt.title('t-SNE Visualization - Drought Prediction')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        # Flood plot
        plt.subplot(1, 2, 2)
        sns.scatterplot(
            x='tsne-1', y='tsne-2',
            hue='flood',
            palette={0: 'green', 1: 'orange'},
            data=tsne_df,
            legend='full',
            alpha=0.7
        )
        plt.title('t-SNE Visualization - Flood Prediction')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/weather-project/tsne_visualization.png')
        plt.show()
        
        print("t-SNE visualization saved to Google Drive!")
    
    def plot_andrews_curves(self, sample_size=500):
        """Plot Andrew's curves for multivariate visualization"""
        print("Generating Andrew's curves...")
        
        # Sample data for Andrew's curves
        if len(self.X_test) > sample_size:
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[indices]
            y_drought_sample = self.y_test_drought.iloc[indices]
            y_flood_sample = self.y_test_flood.iloc[indices]
        else:
            X_sample = self.X_test
            y_drought_sample = self.y_test_drought
            y_flood_sample = self.y_test_flood
        
        # Create DataFrame for Andrew's curves
        andrews_df = X_sample.copy()
        andrews_df['drought'] = y_drought_sample.values
        andrews_df['flood'] = y_flood_sample.values
        
        # Plot Andrew's curves for drought
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        andrews_curves(andrews_df.sample(min(100, len(andrews_df))), 'drought', color=['blue', 'red'])
        plt.title("Andrew's Curves - Drought Prediction")
        
        # Plot Andrew's curves for flood
        plt.subplot(1, 2, 2)
        andrews_curves(andrews_df.sample(min(100, len(andrews_df))), 'flood', color=['green', 'orange'])
        plt.title("Andrew's Curves - Flood Prediction")
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/weather-project/andrews_curves.png')
        plt.show()
        
        print("Andrew's curves saved to Google Drive!")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for both models"""
        print("Generating confusion matrices...")
        
        # Compute confusion matrices
        drought_cm = confusion_matrix(self.y_test_drought, self.drought_pred)
        flood_cm = confusion_matrix(self.y_test_flood, self.flood_pred)
        
        # Plot confusion matrices
        plt.figure(figsize=(16, 7))
        
        # Drought confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(drought_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Drought', 'Drought'],
                    yticklabels=['No Drought', 'Drought'])
        plt.title('Confusion Matrix - Drought Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Flood confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(flood_cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['No Flood', 'Flood'],
                    yticklabels=['No Flood', 'Flood'])
        plt.title('Confusion Matrix - Flood Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/weather-project/confusion_matrices.png')
        plt.show()
        
        print("Confusion matrices saved to Google Drive!")
    
    def plot_roc_curves(self):
        """Plot ROC curves for both models"""
        print("Generating ROC curves...")
        
        # Compute ROC curves
        fpr_drought, tpr_drought, _ = roc_curve(self.y_test_drought, self.drought_prob)
        fpr_flood, tpr_flood, _ = roc_curve(self.y_test_flood, self.flood_prob)
        
        # Compute AUC
        auc_drought = auc(fpr_drought, tpr_drought)
        auc_flood = auc(fpr_flood, tpr_flood)
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(fpr_drought, tpr_drought, color='blue', lw=2, 
                 label=f'Drought Model (AUC = {auc_drought:.2f})')
        plt.plot(fpr_flood, tpr_flood, color='green', lw=2,
                 label=f'Flood Model (AUC = {auc_flood:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Weather Prediction Models')
        plt.legend(loc="lower right")
        
        plt.savefig('/content/drive/MyDrive/weather-project/roc_curves.png')
        plt.show()
        
        print("ROC curves saved to Google Drive!")
    
    def plot_feature_importance(self):
        """Plot feature importance for both models"""
        print("Generating feature importance plots...")
        
        # Get feature importances
        drought_importance = self.drought_model.feature_importances_
        flood_importance = self.flood_model.feature_importances_
        feature_names = self.X_test.columns
        
        # Create DataFrames for visualization
        drought_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': drought_importance
        }).sort_values('Importance', ascending=False)
        
        flood_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': flood_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importances
        plt.figure(figsize=(16, 10))
        
        # Drought feature importance
        plt.subplot(2, 1, 1)
        sns.barplot(x='Importance', y='Feature', data=drought_df.head(10), palette='Blues_d')
        plt.title('Feature Importance - Drought Prediction Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        # Flood feature importance
        plt.subplot(2, 1, 2)
        sns.barplot(x='Importance', y='Feature', data=flood_df.head(10), palette='Greens_d')
        plt.title('Feature Importance - Flood Prediction Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/weather-project/feature_importance.png')
        plt.show()
        
        print("Feature importance plots saved to Google Drive!")
    
    def plot_prediction_distribution(self):
        """Plot distribution of prediction probabilities"""
        print("Generating prediction distribution plots...")
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame({
            'Drought Probability': self.drought_prob,
            'Flood Probability': self.flood_prob,
            'Actual Drought': self.y_test_drought.values,
            'Actual Flood': self.y_test_flood.values
        })
        
        # Plot prediction distributions
        plt.figure(figsize=(16, 8))
        
        # Drought probability distribution
        plt.subplot(1, 2, 1)
        sns.histplot(data=pred_df, x='Drought Probability', hue='Actual Drought', 
                     kde=True, bins=30, palette=['blue', 'red'], alpha=0.6)
        plt.title('Distribution of Drought Prediction Probabilities')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        
        # Flood probability distribution
        plt.subplot(1, 2, 2)
        sns.histplot(data=pred_df, x='Flood Probability', hue='Actual Flood', 
                     kde=True, bins=30, palette=['green', 'orange'], alpha=0.6)
        plt.title('Distribution of Flood Prediction Probabilities')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/weather-project/prediction_distribution.png')
        plt.show()
        
        print("Prediction distribution plots saved to Google Drive!")
    
    def plot_error_analysis(self):
        """Analyze and visualize prediction errors"""
        print("Generating error analysis plots...")
        
        # Create DataFrame with predictions and actual values
        error_df = pd.DataFrame({
            'Drought Prob': self.drought_prob,
            'Flood Prob': self.flood_prob,
            'Drought Pred': self.drought_pred,
            'Flood Pred': self.flood_pred,
            'Actual Drought': self.y_test_drought.values,
            'Actual Flood': self.y_test_flood.values
        })
        
        # Identify errors
        error_df['Drought Error'] = (error_df['Drought Pred'] != error_df['Actual Drought'])
        error_df['Flood Error'] = (error_df['Flood Pred'] != error_df['Actual Flood'])
        
        # Plot error analysis
        plt.figure(figsize=(16, 12))
        
        # Drought errors by probability
        plt.subplot(2, 2, 1)
        sns.boxplot(x='Actual Drought', y='Drought Prob', hue='Drought Error', data=error_df)
        plt.title('Drought Prediction Errors by Probability')
        plt.ylabel('Predicted Probability')
        
        # Flood errors by probability
        plt.subplot(2, 2, 2)
        sns.boxplot(x='Actual Flood', y='Flood Prob', hue='Flood Error', data=error_df)
        plt.title('Flood Prediction Errors by Probability')
        plt.ylabel('Predicted Probability')
        
        # Error count by class
        plt.subplot(2, 2, 3)
        drought_errors = error_df.groupby('Actual Drought')['Drought Error'].sum()
        flood_errors = error_df.groupby('Actual Flood')['Flood Error'].sum()
        
        error_count_df = pd.DataFrame({
            'Drought': drought_errors,
            'Flood': flood_errors
        }).T
        
        error_count_df.plot(kind='bar', stacked=True)
        plt.title('Count of Prediction Errors by Class')
        plt.ylabel('Error Count')
        plt.xticks(rotation=0)
        
        # Error rate by probability bin
        plt.subplot(2, 2, 4)
        error_df['Prob Bin'] = pd.cut(error_df['Drought Prob'], bins=10)
        drought_error_rate = error_df.groupby('Prob Bin')['Drought Error'].mean()
        
        error_df['Prob Bin'] = pd.cut(error_df['Flood Prob'], bins=10)
        flood_error_rate = error_df.groupby('Prob Bin')['Flood Error'].mean()
        
        plt.plot(drought_error_rate.index.astype(str), drought_error_rate.values, 'o-', label='Drought')
        plt.plot(flood_error_rate.index.astype(str), flood_error_rate.values, 'o-', label='Flood')
        plt.title('Error Rate by Probability Bin')
        plt.xlabel('Probability Bin')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/weather-project/error_analysis.png')
        plt.show()
        
        print("Error analysis plots saved to Google Drive!")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive visualization report"""
        print("Generating comprehensive visualization report...")
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(20, 25))
        gs = GridSpec(4, 3, figure=plt.gcf())
        
        # 1. t-SNE plot (drought)
        ax1 = plt.subplot(gs[0, 0])
        if len(self.X_test) > 2000:
            indices = np.random.choice(len(self.X_test), 2000, replace=False)
            X_sample = self.X_test.iloc[indices]
            y_sample = self.y_test_drought.iloc[indices]
        else:
            X_sample = self.X_test
            y_sample = self.y_test_drought
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
        tsne_results = tsne.fit_transform(X_scaled)
        
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                        hue=y_sample, palette={0: 'blue', 1: 'red'}, 
                        alpha=0.7, ax=ax1)
        ax1.set_title('t-SNE - Drought')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        
        # 2. t-SNE plot (flood)
        ax2 = plt.subplot(gs[0, 1])
        y_sample = self.y_test_flood.iloc[indices] if 'indices' in locals() else self.y_test_flood
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                        hue=y_sample, palette={0: 'green', 1: 'orange'}, 
                        alpha=0.7, ax=ax2)
        ax2.set_title('t-SNE - Flood')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        
        # 3. ROC curves
        ax3 = plt.subplot(gs[0, 2])
        fpr_drought, tpr_drought, _ = roc_curve(self.y_test_drought, self.drought_prob)
        fpr_flood, tpr_flood, _ = roc_curve(self.y_test_flood, self.flood_prob)
        auc_drought = auc(fpr_drought, tpr_drought)
        auc_flood = auc(fpr_flood, tpr_flood)
        
        ax3.plot(fpr_drought, tpr_drought, color='blue', lw=2, 
                 label=f'Drought (AUC={auc_drought:.2f})')
        ax3.plot(fpr_flood, tpr_flood, color='green', lw=2,
                 label=f'Flood (AUC={auc_flood:.2f})')
        ax3.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves')
        ax3.legend()
        
        # 4. Confusion matrix (drought)
        ax4 = plt.subplot(gs[1, 0])
        drought_cm = confusion_matrix(self.y_test_drought, self.drought_pred)
        sns.heatmap(drought_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax4)
        ax4.set_title('Confusion Matrix - Drought')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        # 5. Confusion matrix (flood)
        ax5 = plt.subplot(gs[1, 1])
        flood_cm = confusion_matrix(self.y_test_flood, self.flood_pred)
        sns.heatmap(flood_cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax5)
        ax5.set_title('Confusion Matrix - Flood')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        
        # 6. Feature importance (drought)
        ax6 = plt.subplot(gs[1, 2])
        drought_importance = self.drought_model.feature_importances_
        feature_names = self.X_test.columns
        drought_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': drought_importance
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=drought_df.head(5), palette='Blues_d', ax=ax6)
        ax6.set_title('Top Features - Drought')
        ax6.set_xlabel('Importance')
        
        # 7. Feature importance (flood)
        ax7 = plt.subplot(gs[2, 0])
        flood_importance = self.flood_model.feature_importances_
        flood_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': flood_importance
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=flood_df.head(5), palette='Greens_d', ax=ax7)
        ax7.set_title('Top Features - Flood')
        ax7.set_xlabel('Importance')
        
        # 8. Prediction distribution (drought)
        ax8 = plt.subplot(gs[2, 1])
        pred_df = pd.DataFrame({
            'Drought Probability': self.drought_prob,
            'Actual Drought': self.y_test_drought.values
        })
        sns.histplot(data=pred_df, x='Drought Probability', hue='Actual Drought', 
                     kde=True, bins=20, palette=['blue', 'red'], alpha=0.6, ax=ax8)
        ax8.set_title('Drought Probability Distribution')
        ax8.set_xlabel('Probability')
        
        # 9. Prediction distribution (flood)
        ax9 = plt.subplot(gs[2, 2])
        pred_df = pd.DataFrame({
            'Flood Probability': self.flood_prob,
            'Actual Flood': self.y_test_flood.values
        })
        sns.histplot(data=pred_df, x='Flood Probability', hue='Actual Flood', 
                     kde=True, bins=20, palette=['green', 'orange'], alpha=0.6, ax=ax9)
        ax9.set_title('Flood Probability Distribution')
        ax9.set_xlabel('Probability')
        
        # 10. Error rate by probability bin
        ax10 = plt.subplot(gs[3, :])
        error_df = pd.DataFrame({
            'Drought Prob': self.drought_prob,
            'Flood Prob': self.flood_prob,
            'Drought Pred': self.drought_pred,
            'Flood Pred': self.flood_pred,
            'Actual Drought': self.y_test_drought.values,
            'Actual Flood': self.y_test_flood.values
        })
        
        error_df['Drought Error'] = (error_df['Drought Pred'] != error_df['Actual Drought'])
        error_df['Flood Error'] = (error_df['Flood Pred'] != error_df['Actual Flood'])
        
        error_df['Drought Bin'] = pd.cut(error_df['Drought Prob'], bins=10)
        drought_error_rate = error_df.groupby('Drought Bin')['Drought Error'].mean()
        
        error_df['Flood Bin'] = pd.cut(error_df['Flood Prob'], bins=10)
        flood_error_rate = error_df.groupby('Flood Bin')['Flood Error'].mean()
        
        ax10.plot(drought_error_rate.index.astype(str), drought_error_rate.values, 'o-', label='Drought', color='blue')
        ax10.plot(flood_error_rate.index.astype(str), flood_error_rate.values, 'o-', label='Flood', color='green')
        ax10.set_title('Error Rate by Probability Bin')
        ax10.set_xlabel('Probability Bin')
        ax10.set_ylabel('Error Rate')
        ax10.legend()
        ax10.grid(True)
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/weather-project/comprehensive_report.png', dpi=300)
        plt.show()
        
        print("Comprehensive report saved to Google Drive!")
    
    def run_all_visualizations(self):
        """Run all visualization methods"""
        self.load_data_and_models()
        self.plot_tsne()
        self.plot_andrews_curves()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_feature_importance()
        self.plot_prediction_distribution()
        self.plot_error_analysis()
        self.generate_comprehensive_report()
        
        print("\nAll visualizations completed! Check your Google Drive for the saved images.")

if __name__ == "__main__":
    visualizer = WeatherModelVisualizer()
    visualizer.run_all_visualizations()