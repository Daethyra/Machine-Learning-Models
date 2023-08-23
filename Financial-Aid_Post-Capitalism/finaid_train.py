"""Module to train a Gaussian Mixture Model for financial aid prediction using preprocessed data."""
from config import ConfigManager
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

class FinancialAidModelTrainer:
    def __init__(self):
        """Initialize the FinancialAidModelTrainer with configuration settings and selected features."""
        script_name = "finaid_train"
        self.config_manager = ConfigManager(script_name)
        self.config_manager.setup_configuration()
        self.selected_features = ['Density\n(P/Km2)', 'Agricultural Land( %)', 'CPI', 'Fertility Rate',
                                  'Unemployment rate', 'Urban_population']
        self.model_name = "FinancialAidGMM"

    def load_data(self):
        """Load the latest preprocessed data from CSV file."""
        data_path = self.config_manager.find_latest_preprocessed_file()
        try:
            if data_path:
                data = pd.read_csv(data_path, thousands=',')  # Handle thousands separator
                logging.info('Data loaded successfully from ' + data_path)
                return data
            else:
                logging.error('No preprocessed data file found.')
                return None
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data):
        """Preprocess the data by imputing missing values and selecting specific features."""
        features = data[self.selected_features]
        imputer = SimpleImputer(strategy='median')
        features_imputed = imputer.fit_transform(features)
        features_imputed = pd.DataFrame(features_imputed, columns=self.selected_features)
        return features_imputed

    def train_model(self, X_train):
        """Train the Gaussian Mixture Model using the given training data."""
        model = GaussianMixture(n_components=3)
        model.fit(X_train)
        logging.info('FinancialAidGMM model trained successfully.')
        return model

    def visualize_clusters(self, data, clusters):
        """Visualize the data clusters using Seaborn pairplot and save the plot as an image."""
        warnings.filterwarnings("ignore", message="The figure layout has changed to tight") # Ignore the specific UserWarnings related to tight_layout
        data['Cluster'] = clusters
        sns.pairplot(data, hue='Cluster', vars=self.selected_features)
        plt.tight_layout()
        datetime_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.config_manager.save_plot('images', 'clusters_visualization', datetime_str)
        logging.info(f'Visualization saved as clusters_visualization_{datetime_str}.png.')

    def save_model(self, model):
        """Save the trained Gaussian Mixture Model to a file."""
        model_path = f'data/output/models/{self.model_name}_model.pkl'
        joblib.dump(model, model_path)
        logging.info(f'Model saved as {model_path}.')

    def run(self):
        """Execute the complete training process including loading, preprocessing, training, visualizing, and saving the model."""
        data = self.load_data()
        if data is not None:
            features_imputed = self.preprocess_data(data)
            X_temp, X_test = train_test_split(features_imputed, test_size=0.20, random_state=42)
            X_train, X_val = train_test_split(X_temp, test_size=0.1875, random_state=42)
            logging.info('Data split into training, validation, and testing sets.')
            model = self.train_model(X_train)
            clusters = model.predict(features_imputed)
            self.visualize_clusters(data, clusters)
            self.save_model(model)

if __name__ == "__main__":
    trainer = FinancialAidModelTrainer()
    trainer.run()
