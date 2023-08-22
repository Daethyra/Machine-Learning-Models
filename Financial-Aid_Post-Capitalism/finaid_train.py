"""
This module is designed to train a Gaussian Mixture Model (GMM) to identify regions that require financial aid from wealthier nations.
The focus is on prioritizing societal overhaul through the endorsement of materials and labor. The process includes:

- Loading and preprocessing data from the latest CSV file.
- Splitting the data into training (65%), validation (15%), and testing (20%) sets.
- Training the GMM with the specified number of components.
- Predicting clusters and visualizing the results.
- Saving the trained model and visualization for future use.

Usage:
    Simply run this script to train the model using the specified data path and parameters. Adjust the data path, feature selection, and model hyperparameters as needed.

Author:
    Daemon 'Daethyra' Carino
"""

from config import ConfigManager
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib

# Instantiate ConfigManager
config_manager = ConfigManager()

# Configure logging using ConfigManager
config_manager.configure_logger()

# Find the latest preprocessed data file and load it using ConfigManager
data_path = config_manager.find_latest_preprocessed_file()
try:
    if data_path:
        data = pd.read_csv(data_path, thousands=',') # Handle thousands separator
        logging.info('Data loaded successfully from ' + data_path)
    else:
        logging.error('No preprocessed data file found.')
except Exception as e:
    logging.error(f"Error loading data: {e}")

# Select features (excluding non-numeric columns)
numeric_features = data.select_dtypes(include=['float64', 'int64'])
# Handling NaN values by replacing them with the median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
numeric_features = imputer.fit_transform(numeric_features)

features = numeric_features.drop(columns=['Country', 'Latitude', 'Longitude'], errors='ignore') # Handle non-numeric columns

# Split data into training, validation, and testing sets (65% training, 15% validation, 20% testing)
X_temp, X_test = train_test_split(features, test_size=0.20, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.1875, random_state=42)

logging.info('Data split into training, validation, and testing sets.')

# Create and train Gaussian Mixture Model (FinancialAidGMM)
FinancialAidGMM = GaussianMixture(n_components=3)
FinancialAidGMM.fit(X_train)
logging.info('FinancialAidGMM model trained successfully.')

# Predict clusters
clusters = FinancialAidGMM.predict(features)
data['Cluster'] = clusters
logging.info('Clusters predicted.')

# Visualize and save clusters
sns.pairplot(data, hue='Cluster', vars=features.columns)
plt.savefig('clusters_visualization.png')
plt.show()
logging.info('Visualization saved as clusters_visualization.png.')

# Save model
model_name = "FinancialAidGMM"
joblib.dump(FinancialAidGMM, f'{model_name}_model.pkl')
logging.info(f'Model saved as {model_name}_model.pkl.')
